from __future__ import annotations

import argparse
import dataclasses
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph


RUBRIC_KEYS = ["evidence", "feasibility", "risks", "clarity"]


class TranscriptEntry(TypedDict):
    round: int
    stage: str
    speaker: str
    role: str
    content: str


class IssueRecord(TypedDict):
    key: str
    description: str
    raised_by: str
    raised_round: int
    resolved_round: Optional[int]
    status: str


class DebateState(TypedDict, total=False):
    history: List[TranscriptEntry]
    round_index: int
    total_rounds: int
    open_issues: List[IssueRecord]
    signals: Dict[str, bool]
    resolved_actions: List[str]
    convergence_notes: List[str]
    consensus_reached: bool
    scores: Dict[str, int]
    final_decision: str
    judge_summary: str
    config: Dict[str, Any]


@dataclass
class AgentSpec:
    name: str
    role: str
    objective: str
    style: str


@dataclass
class DebateConfig:
    key: str
    title: str
    scenario: str
    acceptance_criteria: List[str]
    rounds: int
    temperature: float
    agent_mode: str  # "full" or "two_agent"
    include_synthesizer: bool
    include_devil: bool
    seed: int
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self)
        return data


@dataclass
class DebateResult:
    config: DebateConfig
    transcript: List[TranscriptEntry]
    scores: Dict[str, int]
    decision: str
    consensus_reached: bool
    convergence_notes: List[str]
    open_issues: List[IssueRecord]
    resolved_actions: List[str]


class LocalDebateModel:
    """Rule-guided generator for deterministic, human-readable debate turns."""

    def __init__(self, config: DebateConfig, facts: Dict[str, Any]):
        self.config = config
        self.random = random.Random(config.seed)
        self.facts = facts

    def _choice(self, options: List[str]) -> str:
        if not options:
            return ""
        if self.config.temperature < 0.5:
            return options[0]
        idx = int(self.random.random() * len(options))
        return options[idx]

    def _shuffle(self, items: List[str]) -> List[str]:
        items_copy = list(items)
        self.random.shuffle(items_copy)
        return items_copy

    def make_researcher(
        self,
        round_number: int,
        open_issues: List[IssueRecord],
        prior_feedback: List[str],
    ) -> Dict[str, Any]:
        headline_options = [
            "Bundle solar, storage, and demand response to guarantee 18% utility savings.",
            "Pair MassCEC grant with performance-based EPC contract to land <10 year payback.",
            "Stage microgrid commissioning so that tenant benefits show up in billing by month six.",
        ]
        headline = self._choice(headline_options)

        evidence_points = self._shuffle(self.facts["evidence"])[:3]
        impl_steps = self._shuffle(self.facts["implementation"])[:3]
        risk_watch = []
        outstanding_keys = sorted({issue["key"] for issue in open_issues if issue["status"] == "open"})
        if outstanding_keys:
            outstanding = ", ".join(outstanding_keys)
            risk_watch.append(f"Outstanding review items: {outstanding}")
        risk_watch.extend(self._shuffle(self.facts["baseline_risks"])[:2])

        feedback_note = ""
        if prior_feedback:
            feedback_note = "Feedback last round: " + prior_feedback[-1]

        sections = [
            f"**Round {round_number} focus:** {headline}",
            "**Scenario snapshot:** " + self.facts["snapshot"],
            "**Evidence highlights:**\n" + "\n".join(f"- {point}" for point in evidence_points),
            "**Implementation path:**\n" + "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(impl_steps)),
            "**Risk watchlist:**\n" + "\n".join(f"- {risk}" for risk in risk_watch),
        ]

        if feedback_note:
            sections.append(f"**Addressing prior critique:** {feedback_note}")

        content = "\n\n".join(sections)
        return {
            "content": content,
            "signals": {"evidence": True, "feasibility": True},
            "proposed_actions": [impl_steps[0], impl_steps[1]],
        }

    def make_critic(
        self,
        round_number: int,
        open_issues: List[IssueRecord],
    ) -> Dict[str, Any]:
        candidate_issues = self.facts["issue_bank"]
        existing_keys = {issue["key"] for issue in open_issues}
        new_issues: List[IssueRecord] = []

        for issue in candidate_issues:
            if issue["key"] in existing_keys:
                continue
            if len(new_issues) >= 2:
                break
            record: IssueRecord = {
                "key": issue["key"],
                "description": issue["description"],
                "raised_by": "Critic",
                "raised_round": round_number,
                "resolved_round": None,
                "status": "open",
            }
            new_issues.append(record)
            existing_keys.add(issue["key"])

        major_concerns = open_issues + new_issues
        major_txt = "\n".join(
            f"- {issue['key']}: {issue['description']}"
            for issue in major_concerns
        )

        clarifying = self._shuffle(self.facts["clarifying_questions"])[:2]
        risk_rating = self._choice(
            [
                "Residual risk currently sits at medium-high because monetized resilience value is still assumptive.",
                "Residual risk sits at medium thanks to solid grant backing but tenant protections need proof.",
                "Residual risk is high; storage sizing assumptions have not been validated under winter load.",
            ]
        )

        content = "\n\n".join(
            [
                f"**Round {round_number} critique:** Focusing on stubborn weaknesses.",
                "**Major concerns:**\n" + major_txt,
                "**Clarifying asks:**\n" + "\n".join(f"- {q}" for q in clarifying),
                f"**Risk posture:** {risk_rating}",
            ]
        )

        return {
            "content": content,
            "signals": {"risks": True},
            "raised_issues": new_issues,
        }

    def make_devil(
        self,
        round_number: int,
        open_issues: List[IssueRecord],
    ) -> Dict[str, Any]:
        contrarian_points = [
            "If ISO-NE enforces new dual participation rules, the revenue stack could collapse.",
            "A stalled interconnection queue could delay energization by 18 months, nullifying cost savings.",
            "Tenant trust can fray if bills lag behind savings; once lost, political capital is gone.",
        ]
        issue: IssueRecord = {
            "key": "Regulatory whiplash",
            "description": "Rule changes on community solar crediting could erode projected savings by up to 35%.",
            "raised_by": "Devil's Advocate",
            "raised_round": round_number,
            "resolved_round": None,
            "status": "open",
        }
        content = "\n\n".join(
            [
                f"**Devil's Advocate (Round {round_number}):** Stress-testing optimism.",
                "**Contrarian evidence:**\n" + "\n".join(f"- {point}" for point in self._shuffle(contrarian_points)[:2]),
                "**Worst-case storyline:** In a downside market, the co-op could face a $220k funding hole.",
                "Let's force the team to show contingency math before we pretend consensus exists.",
            ]
        )
        return {
            "content": content,
            "signals": {"risks": True},
            "raised_issue": issue,
        }

    def make_revision(
        self,
        round_number: int,
        open_issues: List[IssueRecord],
    ) -> Dict[str, Any]:
        open_now = [issue for issue in open_issues if issue["status"] == "open"]
        resolved_keys = []
        adjustments = []
        mitigations = self.facts["mitigations"]

        for issue in open_now[:2]:
            resolved_keys.append(issue["key"])
            if issue["key"] in mitigations:
                adjustments.append(mitigations[issue["key"]])
            else:
                adjustments.append(f"Documented mitigation plan for {issue['key']} with owner + due date.")

        if not adjustments:
            adjustments.append("Maintaining prior plan; no unresolved items flagged this round.")

        evidence_refresh = self._choice(
            [
                "Uploaded utility interval data (Jan-Dec 2023) to shared drive for transparency.",
                "Secured EPC letter committing to $1.92/W turnkey cap backed by performance guarantees.",
                "Validated storage dispatch model against ISO-NE winter peaks; 90% of outage use case holds.",
            ]
        )

        content = "\n\n".join(
            [
                f"**Researcher revision (Round {round_number}):** Integrating critiques into the plan.",
                "**Adjustments committed:**\n" + "\n".join(f"- {item}" for item in adjustments),
                f"**Fresh evidence:** {evidence_refresh}",
                "**Next actions:** Kick off tenant co-design workshop and lock interconnection study date.",
            ]
        )

        return {
            "content": content,
            "signals": {"feasibility": True, "evidence": True},
            "resolved_keys": resolved_keys,
            "new_actions": adjustments,
        }

    def make_synthesizer(
        self,
        round_number: int,
        open_issues: List[IssueRecord],
        resolved_actions: List[str],
    ) -> Dict[str, Any]:
        open_keys = [issue["key"] for issue in open_issues if issue["status"] == "open"]

        agreement = len(open_keys) <= 1
        tone = "We are close to consensus." if agreement else "We still have material blockers."
        outstanding = ", ".join(open_keys) if open_keys else "none"
        highlight = self._choice(
            [
                "Revision landed real movement on tenant protections.",
                "Financial engineering is sharper; still need better downside math.",
                "Stakeholder alignment reads solid, but regulatory volatility remains the swing factor.",
            ]
        )

        content = "\n\n".join(
            [
                f"**Synthesizer (Round {round_number} wrap):** {tone}",
                f"**Outstanding issues:** {outstanding}",
                "**Progress markers:**\n" + "\n".join(f"- {action}" for action in resolved_actions[-2:]),
                f"**Narrative to brief stakeholders:** {highlight}",
            ]
        )

        note = "agreement" if agreement else "open blockers remain"

        return {
            "content": content,
            "signals": {"clarity": True},
            "agreement": agreement,
            "note": note,
        }

    def make_judge(
        self,
        open_issues: List[IssueRecord],
        signals: Dict[str, bool],
        convergence_notes: List[str],
    ) -> Dict[str, Any]:
        unresolved = [issue for issue in open_issues if issue["status"] == "open"]
        resolved = [issue for issue in open_issues if issue["status"] == "resolved"]

        scores: Dict[str, int] = {}
        for key in RUBRIC_KEYS:
            base = 3
            if signals.get(key):
                base += 1
            scores[key] = min(base, 5)

        if unresolved:
            scores["risks"] = max(2, scores["risks"] - 1)
            scores["clarity"] = max(2, scores["clarity"] - 1)

        if len(resolved) >= 2:
            scores["feasibility"] = min(5, scores["feasibility"] + 1)

        avg_score = sum(scores.values()) / len(RUBRIC_KEYS)
        consensus = not unresolved and avg_score >= 3.5

        verdict = (
            "✅ Recommend go/no-go: APPROVE pilot with defined guardrails."
            if consensus
            else "⚠️ Verdict: CONDITIONAL — hold launch until risk gaps close."
        )

        rationale_parts = [
            f"Evidence score {scores['evidence']} — data packs are substantive." if signals.get("evidence") else "Evidence still thin.",
            f"Feasibility score {scores['feasibility']} — execution path mostly credible.",
            f"Risks score {scores['risks']} — unresolved items: {', '.join(issue['key'] for issue in unresolved) or 'none'}",
            f"Clarity score {scores['clarity']} — story is almost board-ready.",
        ]
        convergence_view = (
            "Convergence achieved; no open blockers remain."
            if consensus
            else "Convergence partial; remaining blockers must be addressed."
        )

        summary = "\n".join(rationale_parts + [f"**Convergence read:** {convergence_view}"])

        content = "\n\n".join(
            [
                "**Judge verdict:**",
                verdict,
                summary,
                "**Next step I require:** deliver risk register sign-off and regulatory contingency memo.",
            ]
        )

        return {
            "content": content,
            "scores": scores,
            "decision": verdict,
            "consensus": consensus,
            "convergence": convergence_view,
        }


def build_agent_specs(config: DebateConfig) -> Dict[str, AgentSpec]:
    researcher = AgentSpec(
        name="Alex Morgan",
        role="Researcher",
        objective="Assemble actionable plan that satisfies council acceptance criteria.",
        style="Structured, cites numbers, frames action items.",
    )

    critic_name = "Jordan Lee"
    critic_role = "Critic"
    if config.agent_mode == "two_agent":
        critic_role = "Critic & Judge"

    critic = AgentSpec(
        name=critic_name,
        role=critic_role,
        objective="Pressure-test the proposal, surface hard blockers, and demand proof.",
        style="Direct, risk-oriented, asks targeted questions.",
    )

    synthesizer = AgentSpec(
        name="Priya Singh",
        role="Synthesizer",
        objective="Track convergence, integrate revisions, and prep stakeholder narrative.",
        style="Facilitator tone; highlights progress and stuck points.",
    )

    judge_name = "Jordan Lee" if config.agent_mode == "two_agent" else "Morgan Kim"
    judge = AgentSpec(
        name=judge_name,
        role="Judge",
        objective="Score plan against rubric and issue final verdict.",
        style="Decisive, rubric-driven, references convergence status.",
    )

    devil = AgentSpec(
        name="Casey Vega",
        role="Devil's Advocate",
        objective="Introduce skeptical signals and stress-test assumptions.",
        style="Skeptical, contrarian, concise.",
    )

    specs: Dict[str, AgentSpec] = {
        "researcher": researcher,
        "critic": critic,
        "judge": judge,
    }

    if config.include_synthesizer:
        specs["synthesizer"] = synthesizer

    if config.include_devil:
        specs["devil"] = devil

    return specs


def build_facts() -> Dict[str, Any]:
    return {
        "snapshot": (
            "Cambridge Housing Authority seeks a 650 kW solar + 2.5 MWh storage microgrid "
            "at Riverside Homes to shave bills ~18% while providing 8-hour outage resilience."
        ),
        "evidence": [
            "Synapse 2024 load study shows average winter peak 1.32 MW; solar covers 48% daytime load.",
            "MassCEC Gap Fund has issued a draft commitment covering 35% of capital spend.",
            "Greenspark EPC bid locks in $1.95/W turnkey price with performance bonding.",
            "Tenant energy burden currently 12% of income; housing authority aims to cut to 8%.",
            "Microgrid control tested in Worcester pilot reduced outage downtime by 72%.",
        ],
        "implementation": [
            "Secure MassCEC grant agreement signature by week 6.",
            "Run tenant co-design workshops to codify bill credit policy by week 9.",
            "File interconnection application with Eversource in month 2; expedite via municipal liaison.",
            "Phase construction so the community center island mode goes live by month 8.",
            "Train onsite maintenance staff on battery O&M before commissioning.",
        ],
        "baseline_risks": [
            "Storage degradation assumptions rely on 4,000 cycles; need warranty alignment.",
            "Tenant savings must be codified in lease amendments to avoid billing disputes.",
            "Sponsor must model winter storm islanding scenario explicitly.",
        ],
        "issue_bank": [
            {
                "key": "Load-model mismatch",
                "description": "Peak winter load (1.6 MW) exceeds storage output; outage coverage unclear.",
            },
            {
                "key": "Tenant safeguards",
                "description": "Need signed policy guaranteeing minimum 15% bill reduction in first year.",
            },
            {
                "key": "Capital gap",
                "description": "Even with grants there is a $310k funding hole; what bridge financing covers it?",
            },
        ],
        "clarifying_questions": [
            "What is the fallback if MassCEC funding slips a quarter?",
            "How will tenant training handle language access for Cantonese and Spanish speakers?",
            "Who owns cybersecurity risk once the microgrid controller is online?",
            "What are the penalties if the EPC misses the production guarantee?",
        ],
        "mitigations": {
            "Tenant safeguards": "Drafted bilingual tenant benefit charter; legal review scheduled for next Tuesday.",
            "Capital gap": "Proposed using Inflation Reduction Act elective pay plus city green bond bridge.",
            "Load-model mismatch": "Resized storage to 3.1 MWh with demand response; outage coverage meets 10-hour target.",
            "Regulatory whiplash": "Structured savings sharing clause so tenants insulated from net-metering cuts.",
        },
    }


def update_signals(state: DebateState, signals: Dict[str, bool]) -> Dict[str, bool]:
    updated = dict(state["signals"])
    for key, flag in signals.items():
        if flag:
            updated[key] = True
    return updated


def build_graph(config: DebateConfig, facts: Dict[str, Any]) -> Tuple[StateGraph, DebateState, Dict[str, AgentSpec]]:
    specs = build_agent_specs(config)
    model = LocalDebateModel(config, facts)

    def researcher_node(state: DebateState) -> DebateState:
        round_number = state["round_index"] + 1
        prior_feedback = [
            entry["content"]
            for entry in state["history"]
            if entry["stage"] in ("critique", "devil")
        ]
        result = model.make_researcher(round_number, state["open_issues"], prior_feedback)
        message: TranscriptEntry = {
            "round": round_number,
            "stage": "argue",
            "speaker": specs["researcher"].name,
            "role": specs["researcher"].role,
            "content": result["content"],
        }
        return {
            "history": state["history"] + [message],
            "signals": update_signals(state, result["signals"]),
            "resolved_actions": state["resolved_actions"] + result["proposed_actions"],
        }

    def critic_node(state: DebateState) -> DebateState:
        round_number = state["round_index"] + 1
        result = model.make_critic(round_number, state["open_issues"])
        new_issues = state["open_issues"] + result["raised_issues"]
        message: TranscriptEntry = {
            "round": round_number,
            "stage": "critique",
            "speaker": specs["critic"].name,
            "role": specs["critic"].role,
            "content": result["content"],
        }
        return {
            "history": state["history"] + [message],
            "open_issues": new_issues,
            "signals": update_signals(state, result["signals"]),
        }

    def devil_node(state: DebateState) -> DebateState:
        round_number = state["round_index"] + 1
        result = model.make_devil(round_number, state["open_issues"])
        new_issue = result["raised_issue"]

        existing_keys = {issue["key"] for issue in state["open_issues"]}
        updated_issues = list(state["open_issues"])
        if new_issue["key"] not in existing_keys:
            updated_issues.append(new_issue)

        message: TranscriptEntry = {
            "round": round_number,
            "stage": "devil",
            "speaker": specs["devil"].name,
            "role": specs["devil"].role,
            "content": result["content"],
        }

        return {
            "history": state["history"] + [message],
            "open_issues": updated_issues,
            "signals": update_signals(state, result["signals"]),
        }

    def revision_node(state: DebateState) -> DebateState:
        round_number = state["round_index"] + 1
        result = model.make_revision(round_number, state["open_issues"])
        updated_issues = []
        for issue in state["open_issues"]:
            if issue["key"] in result["resolved_keys"]:
                issue = issue.copy()
                issue["status"] = "resolved"
                issue["resolved_round"] = round_number
            updated_issues.append(issue)
        message: TranscriptEntry = {
            "round": round_number,
            "stage": "revise",
            "speaker": specs["researcher"].name,
            "role": specs["researcher"].role,
            "content": result["content"],
        }
        return {
            "history": state["history"] + [message],
            "open_issues": updated_issues,
            "signals": update_signals(state, result["signals"]),
            "resolved_actions": state["resolved_actions"] + result["new_actions"],
            "round_index": state["round_index"] + 1,
        }

    def synthesizer_node(state: DebateState) -> DebateState:
        round_number = state["round_index"]
        result = model.make_synthesizer(round_number, state["open_issues"], state["resolved_actions"])
        message: TranscriptEntry = {
            "round": round_number,
            "stage": "synthesize",
            "speaker": specs["synthesizer"].name,
            "role": specs["synthesizer"].role,
            "content": result["content"],
        }
        return {
            "history": state["history"] + [message],
            "signals": update_signals(state, result["signals"]),
            "convergence_notes": state["convergence_notes"] + [result["note"]],
            "consensus_reached": result["agreement"],
        }

    def judge_node(state: DebateState) -> DebateState:
        result = model.make_judge(state["open_issues"], state["signals"], state["convergence_notes"])
        message: TranscriptEntry = {
            "round": state["round_index"],
            "stage": "verdict",
            "speaker": specs["judge"].name,
            "role": specs["judge"].role,
            "content": result["content"],
        }
        return {
            "history": state["history"] + [message],
            "scores": result["scores"],
            "final_decision": result["decision"],
            "consensus_reached": result["consensus"],
            "convergence_notes": state["convergence_notes"] + [result["convergence"]],
            "judge_summary": result["content"],
        }

    graph: StateGraph = StateGraph(DebateState)
    graph.add_node("researcher", researcher_node)
    if "critic" in specs:
        graph.add_node("critic", critic_node)
    if config.include_devil:
        graph.add_node("devil", devil_node)
    graph.add_node("revision", revision_node)
    if config.include_synthesizer:
        graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("researcher")

    def post_researcher(state: DebateState) -> str:
        return "critic" if "critic" in specs else "revision"

    def post_critic(state: DebateState) -> str:
        return "devil" if config.include_devil else "revision"

    def post_revision(state: DebateState) -> str:
        if config.include_synthesizer:
            return "synthesizer"
        if state["round_index"] < state["total_rounds"]:
            return "researcher"
        return "judge"

    def post_synth(state: DebateState) -> str:
        if state["round_index"] < state["total_rounds"]:
            return "researcher"
        return "judge"

    graph.add_conditional_edges("researcher", post_researcher, {"critic": "critic", "revision": "revision"})

    if "critic" in specs:
        next_map = {"revision": "revision"}
        if config.include_devil:
            next_map["devil"] = "devil"
        graph.add_conditional_edges("critic", post_critic, next_map)

    if config.include_devil:
        graph.add_edge("devil", "revision")

    if config.include_synthesizer:
        graph.add_conditional_edges("revision", post_revision, {"synthesizer": "synthesizer", "researcher": "researcher", "judge": "judge"})
        graph.add_conditional_edges("synthesizer", post_synth, {"researcher": "researcher", "judge": "judge"})
    else:
        graph.add_conditional_edges("revision", post_revision, {"researcher": "researcher", "judge": "judge"})

    graph.add_edge("judge", END)

    initial_state: DebateState = {
        "history": [],
        "round_index": 0,
        "total_rounds": config.rounds,
        "open_issues": [],
        "signals": {key: False for key in RUBRIC_KEYS},
        "resolved_actions": [],
        "convergence_notes": [],
        "consensus_reached": False,
        "scores": {key: 0 for key in RUBRIC_KEYS},
        "final_decision": "",
        "judge_summary": "",
        "config": config.as_dict(),
    }

    return graph, initial_state, specs


def run_debate(config: DebateConfig, output_dir: Path) -> DebateResult:
    facts = build_facts()
    graph, initial_state, specs = build_graph(config, facts)
    compiled = graph.compile(checkpointer=None)
    final_state: DebateState = compiled.invoke(initial_state)

    transcript = final_state["history"]
    scores = final_state["scores"]
    decision = final_state["final_decision"]
    consensus = final_state["consensus_reached"]
    convergence_notes = final_state["convergence_notes"]
    open_issues = final_state["open_issues"]
    resolved_actions = final_state["resolved_actions"]

    result = DebateResult(
        config=config,
        transcript=transcript,
        scores=scores,
        decision=decision,
        consensus_reached=consensus,
        convergence_notes=convergence_notes,
        open_issues=open_issues,
        resolved_actions=resolved_actions,
    )

    persist_run(result, specs, output_dir)
    return result


def persist_run(result: DebateResult, specs: Dict[str, AgentSpec], base_dir: Path) -> None:
    run_dir = base_dir / result.config.key
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_md = render_transcript_markdown(result, specs)
    (run_dir / "transcript.md").write_text(transcript_md, encoding="utf-8")

    transcript_json = {
        "config": result.config.as_dict(),
        "scores": result.scores,
        "decision": result.decision,
        "consensus_reached": result.consensus_reached,
        "convergence_notes": result.convergence_notes,
        "open_issues": result.open_issues,
        "resolved_actions": result.resolved_actions,
        "transcript": result.transcript,
    }
    (run_dir / "transcript.json").write_text(json.dumps(transcript_json, indent=2), encoding="utf-8")

    summary_lines = [
        {
            "metric": key,
            "score": value,
        }
        for key, value in result.scores.items()
    ]
    (run_dir / "scores.json").write_text(json.dumps(summary_lines, indent=2), encoding="utf-8")


def render_transcript_markdown(result: DebateResult, specs: Dict[str, AgentSpec]) -> str:
    agent_roles = ", ".join(sorted({spec.role for spec in specs.values()}))
    header = textwrap.dedent(
        f"""
        # Debate transcript — {result.config.title}

        - Variant: {result.config.key}
        - Rounds: {result.config.rounds}
        - Temperature: {result.config.temperature}
        - Agents active: {agent_roles}
        """
    ).strip()

    messages = []
    for entry in result.transcript:
        block = textwrap.dedent(
            f"""
            ---
            **Round {entry['round']} · {entry['stage'].upper()} · {entry['speaker']} ({entry['role']})**

            {entry['content']}
            """
        ).strip()
        messages.append(block)

    rubric = "\n".join(f"- {key.title()}: {value}" for key, value in result.scores.items())
    footer = textwrap.dedent(
        f"""
        ---
        **Final decision:** {result.decision}

        **Rubric scores:**\n{rubric}

        **Convergence notes:** {' | '.join(result.convergence_notes)}
        """
    ).strip()

    return "\n\n".join([header] + messages + [footer]) + "\n"


def prepare_configs() -> Dict[str, DebateConfig]:
    baseline = DebateConfig(
        key="baseline_full_lowtemp",
        title="Should Cambridge approve the Riverside Homes community microgrid?",
        scenario=(
            "City advisers and housing authority must determine whether to green-light a microgrid pilot "
            "that combines solar, storage, and demand response for Riverside Homes."
        ),
        acceptance_criteria=[
            "Quantify energy savings and payback < 10 years.",
            "Demonstrate an implementation timeline that protects tenants during construction.",
            "De-risk financial exposure if grant funding shifts.",
            "Communicate clearly enough for a city council vote.",
        ],
        rounds=2,
        temperature=0.35,
        agent_mode="full",
        include_synthesizer=True,
        include_devil=False,
        seed=21,
        notes="Baseline run with 4 agents, low temperature.",
    )

    two_agent = DebateConfig(
        key="toggle_two_agent",
        title="Compact debate: researcher vs critic-judge on Riverside microgrid",
        scenario=baseline.scenario,
        acceptance_criteria=baseline.acceptance_criteria,
        rounds=2,
        temperature=0.35,
        agent_mode="two_agent",
        include_synthesizer=False,
        include_devil=False,
        seed=22,
        notes="Toggle 1 — reduce to two agents (Researcher + Critic-Judge).",
    )

    high_temp = DebateConfig(
        key="toggle_high_temp_devil",
        title="High-temperature debate with devil's advocate pressure test",
        scenario=baseline.scenario,
        acceptance_criteria=baseline.acceptance_criteria,
        rounds=2,
        temperature=0.8,
        agent_mode="full",
        include_synthesizer=True,
        include_devil=True,
        seed=69,
        notes="Toggle 2 — raise temperature and add devil's advocate for stress test.",
    )

    return {
        baseline.key: baseline,
        two_agent.key: two_agent,
        high_temp.key: high_temp,
    }


def run_all(config_names: Optional[List[str]], output_dir: Path) -> List[DebateResult]:
    configs = prepare_configs()
    if config_names:
        missing = [name for name in config_names if name not in configs]
        if missing:
            raise ValueError(f"Unknown config keys: {', '.join(missing)}")
        selected = [configs[name] for name in config_names]
    else:
        selected = list(configs.values())

    results = []
    for config in selected:
        print(f"🔁 Running debate: {config.key} — {config.title}")
        results.append(run_debate(config, output_dir=output_dir))
        print(f"✅ Completed: {config.key}\n")

    compile_summary(results, output_dir)
    return results


def compile_summary(results: List[DebateResult], output_dir: Path) -> None:
    summary_rows = []
    for result in results:
        avg_score = sum(result.scores.values()) / len(RUBRIC_KEYS)
        summary_rows.append(
            {
                "config": result.config.key,
                "rounds": result.config.rounds,
                "agents": result.config.agent_mode,
                "temperature": result.config.temperature,
                "decision": result.decision,
                "consensus": result.consensus_reached,
                "avg_score": round(avg_score, 2),
                **{f"score_{k}": v for k, v in result.scores.items()},
                "unresolved_issues": [issue["key"] for issue in result.open_issues if issue["status"] == "open"],
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    header = ["config", "rounds", "agents", "temperature", "decision", "consensus", "avg_score"] + [
        f"score_{k}" for k in RUBRIC_KEYS
    ] + ["unresolved_issues"]

    lines = [",".join(header)]
    for row in summary_rows:
        values = [
            str(row["config"]),
            str(row["rounds"]),
            str(row["agents"]),
            str(row["temperature"]),
            row["decision"].replace(",", ";"),
            str(row["consensus"]),
            str(row["avg_score"]),
        ]
        values.extend(str(row[f"score_{k}"]) for k in RUBRIC_KEYS)
        unresolved = ";".join(row["unresolved_issues"])
        values.append(unresolved)
        lines.append(",".join(values))

    (output_dir / "summary.csv").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAS 665 multi-agent debate experiments.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Subset of config keys to run (default: run all presets).",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory to store transcripts and metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_all(args.configs, output_dir=output_dir)


if __name__ == "__main__":
    main()
