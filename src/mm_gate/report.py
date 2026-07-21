"""Generate a concise Markdown report for the image-gate pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def build_report(gated_path: Path, graph_path: Path, output_path: Path) -> str:
    """Summarize gate outcomes and evidence-graph size for presentation."""
    records = [json.loads(line) for line in gated_path.read_text(encoding="utf-8").splitlines()]
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {"nodes": [], "edges": []}
    completed = [record for record in records if record.get("gate_status") == "completed"]
    decisions = Counter(record["gate"]["decision"] for record in completed)
    mean_confidence = (
        sum(record["gate"]["confidence"] for record in completed) / len(completed) if completed else 0.0
    )

    lines = [
        "# Image Gate Pilot Report",
        "",
        "## Summary",
        "",
        f"- Evaluated images: {len(records)}",
        f"- Completed gate decisions: {len(completed)}",
        f"- Keep: {decisions['keep']}",
        f"- Drop: {decisions['drop']}",
        f"- Mean gate confidence: {mean_confidence:.2f}",
        f"- Evidence KG nodes: {len(graph['nodes'])}",
        f"- Evidence KG edges: {len(graph['edges'])}",
        "",
        "## Gate Decisions",
        "",
        "| Article | Image | Type | Decision | Confidence | BLIP caption | Gate reason |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in completed:
        gate = record["gate"]
        lines.append(
            "| {article} | {image} | {image_type} | {decision} | {confidence:.2f} | {caption} | {reason} |".format(
                article=_escape(record["article_title"]),
                image=record["image_id"],
                image_type=gate["image_type"],
                decision=gate["decision"],
                confidence=gate["confidence"],
                caption=_escape(record.get("blip_caption") or ""),
                reason=_escape(gate["reason"]),
            )
        )

    report = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image-gate pilot Markdown report")
    parser.add_argument("--input", type=Path, default=Path("data/processed/gated.jsonl"))
    parser.add_argument("--graph", type=Path, default=Path("data/processed/evidence_graph.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pilot-report.md"))
    args = parser.parse_args()
    build_report(args.input, args.graph, args.output)


if __name__ == "__main__":
    main()
