"""Build a provenance-preserving evidence graph from admitted image records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_evidence_graph(gated_path: Path, raw_dir: Path, output_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Create Document-Section-Claim-Image graph, keeping evidence on every edge."""
    contexts = _load_context(raw_dir)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for line in gated_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        gate = record.get("gate")
        if record.get("gate_status") != "completed" or not gate or gate["decision"] != "keep":
            continue
        context = contexts.get(record["image_id"])
        if not context:
            continue

        document_id = f"document:{record['article_id']}"
        section_label = context["section"] or "Lead"
        section_id = f"{document_id}:section:{_slug(section_label)}"
        image_id = f"image:{record['image_id']}"
        _add_node(nodes, document_id, "Document", context["title"])
        _add_node(nodes, section_id, "Section", section_label)
        _add_node(
            nodes,
            image_id,
            "Image",
            record["image_id"],
            source_url=record["source_url"],
            local_path=record["local_path"],
            license_name=record.get("license_name"),
        )
        _add_edge(edges, document_id, "HAS_SECTION", section_id, context)
        _add_edge(edges, section_id, "HAS_IMAGE", image_id, context)

        claim_ids = gate["supported_claim_ids"] or ["lead" if section_label == "Lead" else section_label]
        for claim_label in claim_ids:
            claim_id = f"{document_id}:claim:{_slug(claim_label)}"
            _add_node(nodes, claim_id, "Claim", claim_label)
            _add_edge(edges, section_id, "HAS_CLAIM", claim_id, context)
            _add_edge(
                edges,
                image_id,
                "SUPPORTS",
                claim_id,
                context,
                evidence={
                    "wikipedia_caption": context["wikipedia_caption"],
                    "blip_caption": record.get("blip_caption"),
                    "ocr_text": record.get("ocr_text"),
                    "gate_reason": gate["reason"],
                    "gate_model": record["gate_model"],
                    "confidence": gate["confidence"],
                },
            )

    graph = {"nodes": list(nodes.values()), "edges": edges}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def _load_context(raw_dir: Path) -> dict[str, dict[str, str | None]]:
    contexts: dict[str, dict[str, str | None]] = {}
    for path in raw_dir.glob("*.json"):
        article = json.loads(path.read_text(encoding="utf-8"))
        for image in article["images"]:
            contexts[image["image_id"]] = {
                "title": article["title"],
                "section": image.get("section"),
                "wikipedia_caption": image.get("caption"),
            }
    return contexts


def _add_node(nodes: dict[str, dict[str, Any]], node_id: str, node_type: str, label: str, **attributes: Any) -> None:
    nodes.setdefault(node_id, {"id": node_id, "type": node_type, "label": label, **attributes})


def _add_edge(
    edges: list[dict[str, Any]],
    source: str,
    relation: str,
    target: str,
    context: dict[str, str | None],
    evidence: dict[str, Any] | None = None,
) -> None:
    edge_id = f"{source}|{relation}|{target}"
    if any(edge["id"] == edge_id for edge in edges):
        return
    edges.append(
        {
            "id": edge_id,
            "source": source,
            "relation": relation,
            "target": target,
            "provenance": {
                "document_title": context["title"],
                "section": context["section"],
                **({"evidence": evidence} if evidence else {}),
            },
        }
    )


def _slug(value: str) -> str:
    return "-".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence KG from gate results")
    parser.add_argument("--input", type=Path, default=Path("data/processed/gated.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/evidence_graph.json"))
    args = parser.parse_args()
    graph = build_evidence_graph(args.input, args.raw_dir, args.output)
    print(f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
