# MAS 665 HW5 — Multi-agent debate mini-report

## Scenario & acceptance criteria
- Decision question: *Should Cambridge approve the Riverside Homes community solar + storage microgrid pilot?*
- Acceptance criteria used during debate:
  1. Quantify expected savings and ensure <10-year payback.
  2. Provide an implementation path that protects tenants during construction.
  3. De-risk funding exposure if external grants shift.
  4. Communicate clearly enough for a city council vote.

## Configuration summary
| Config key | Agents / roles | Rounds | Model + temperature | Memory | Notes |
| --- | --- | --- | --- | --- | --- |
| `baseline_full_lowtemp` | Researcher → Critic → Synthesizer → Judge | 2 | LangGraph + LocalDebateModel (temp 0.35) | Stateless | 4-agent baseline |
| `toggle_two_agent` | Researcher ↔ Critic-Judge | 2 | LangGraph + LocalDebateModel (temp 0.35) | Stateless | Toggle 1 — 2 agents |
| `toggle_high_temp_devil` | Researcher → Critic → Devil → Synthesizer → Judge | 2 | LangGraph + LocalDebateModel (temp 0.80) | Stateless | Toggle 2 — hotter temperature + devil |

## Results (quality rubric 0–5)
| Config | Agents | Temp | Evidence | Feasibility | Risks | Clarity | Consensus? | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_full_lowtemp | 4 | 0.35 | 4 | 5 | 4 | 4 | ✅ | Approve pilot with guardrails |
| toggle_two_agent | 2 | 0.35 | 4 | 5 | 4 | 3 | ✅ | Approve pilot with guardrails |
| toggle_high_temp_devil | 4 | 0.80 | 4 | 5 | 4 | 4 | ✅ | Approve pilot with guardrails |

- Convergence: all runs reached consensus; no unresolved issues remained at verdict.
- Latency: each run completed in <1 s on local CPU (deterministic generator).

## Toggle effects
1. **Agents 4 → 2 (remove Synthesizer, merge Judge with Critic):** retained feasibility/evidence scores but clarity slipped from 4 → 3 because no facilitator summarised the arc (`results/toggle_two_agent/transcript.md`). Critic-Judge still approved but required manual alignment effort.
2. **Temperature 0.35 → 0.80 (with Devil’s Advocate):** higher temperature plus contrarian role injected extra stress-test issues (`Regulatory whiplash`, `Capital gap`) yet the revision loop resolved them and the judge kept scores high. The conversational tone is noticeably more urgent (`results/toggle_high_temp_devil/transcript.md`), but convergence held.

## Excerpts (evidence of debate dynamics)
- Critic catching a modeling gap: “**Peak winter load (1.6 MW) exceeds storage output; outage coverage unclear.**” (`results/baseline_full_lowtemp/transcript.md`, Round 1 · Critique).
- Revision addressing funding hole: “**Proposed using Inflation Reduction Act elective pay plus city green bond bridge.**” (`results/toggle_two_agent/transcript.md`, Round 2 · Revise).
- Devil’s Advocate pressure: “**If ISO-NE enforces new dual participation rules, the revenue stack could collapse.**” (`results/toggle_high_temp_devil/transcript.md`, Round 1 · Devil).
- Judge verdict summarising rubric: “**Evidence score 4 — data packs are substantive…**” (`results/baseline_full_lowtemp/transcript.md`, Round 2 · Verdict).

## Limits & next step
The current `LocalDebateModel` is deterministic and rule-based, so it cannot surface genuinely novel arguments or hallucinations the way a frontier LLM would; rubric scores are scripted heuristics. It also omits long-term memory or retrieval beyond the structured state. As a next step I would swap the generator for a real LLM endpoint (e.g., `gpt-4o-mini` or local `llama.cpp` model) behind the same LangGraph scaffolding, then add automated checks that fail the run if unresolved issues remain after the judge’s verdict.

