from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


def _wrap_content(lines: Iterable[str], width: int) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        wrapped.extend(textwrap.wrap(stripped, width=width) or [""])
    return wrapped


def build_excerpt(transcript_json: Path, label: str, max_width: int = 88) -> List[str]:
    data = json.loads(transcript_json.read_text(encoding="utf-8"))
    transcript = data["transcript"]
    scores = data["scores"]

    lines: List[str] = [
        label,
        "",
    ]

    # Capture two early stages and the final verdict.
    chosen = []
    for message in transcript:
        if message["round"] == 1 and message["stage"] in {"argue", "critique"}:
            chosen.append(message)
        elif message["round"] == 2 and message["stage"] in {"revise", "verdict"}:
            chosen.append(message)
        if len(chosen) >= 4:
            break

    for message in chosen:
        heading = f"Round {message['round']} · {message['stage'].upper()} · {message['speaker']}"
        lines.append(heading)
        snippet_lines = _wrap_content(message["content"].splitlines(), width=max_width)
        for snippet in snippet_lines[:4]:
            lines.append(f"  {snippet}")
        lines.append("")

    rubric_line = "Scores → " + ", ".join(f"{key.title()}: {value}" for key, value in scores.items())
    lines.append(rubric_line)
    decision_line = f"Decision → {data['decision']}"
    lines.append(decision_line)

    return lines


def render_text(lines: List[str], output_path: Path) -> None:
    font = ImageFont.load_default()
    margin = 20
    spacing = 6

    line_heights = []
    max_width = 0

    for line in lines:
        bbox = font.getbbox(line)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        line_heights.append(height)
        max_width = max(max_width, width)

    total_height = sum(h + spacing for h in line_heights) - spacing + margin * 2
    total_width = max_width + margin * 2

    image = Image.new("RGB", (total_width, total_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(image)

    y = margin
    for line, height in zip(lines, line_heights):
        draw.text((margin, y), line, fill=(26, 26, 26), font=font)
        y += height + spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    base_dir = Path("results")
    output_dir = Path("deliverables/images")

    excerpts = [
        (
            base_dir / "baseline_full_lowtemp" / "transcript.json",
            "Baseline · 4 agents · LangGraph (temp=0.35)",
            output_dir / "baseline_rounds.png",
        ),
        (
            base_dir / "toggle_high_temp_devil" / "transcript.json",
            "High temp + devil's advocate · 4 agents (temp=0.8)",
            output_dir / "high_temp_rounds.png",
        ),
        (
            base_dir / "toggle_two_agent" / "transcript.json",
            "Toggle: 2 agents (Researcher vs Critic-Judge) · LangGraph (temp=0.35)",
            output_dir / "two_agent_rounds.png",
        ),
    ]

    for transcript_path, label, image_path in excerpts:
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing transcript: {transcript_path}")
        lines = build_excerpt(transcript_path, label=label)
        render_text(lines, image_path)
        print(f"Generated {image_path}")


if __name__ == "__main__":
    main()

