"""Document-contextual image admission gate using a local Ollama vision model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import ollama
import yaml
from pydantic import ValidationError

from mm_gate.schemas import GateDecision


def gate_images(
    captions_path: Path,
    raw_dir: Path,
    config_path: Path,
    output_path: Path,
    max_per_article: int | None = None,
) -> list[dict[str, Any]]:
    """Assess whether each image deserves admission to RAG and evidence KG."""
    model_name = _load_model_name(config_path)
    context_by_image = _load_context(raw_dir)
    records = [json.loads(line) for line in captions_path.read_text(encoding="utf-8").splitlines()]
    records = _select_records(records, max_per_article)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for record in records:
        result = {**record, "gate": None, "gate_status": None, "gate_model": model_name}
        context = context_by_image.get(record["image_id"])
        if record["caption_status"] == "skipped":
            result.update(gate_status="skipped", gate_reason="unsupported image format")
        elif not context:
            result.update(gate_status="failed", gate_reason="document context not found")
        else:
            try:
                decision = _ask_gate(model_name, Path(record["local_path"]), context, record)
                result.update(gate=decision.model_dump(), gate_status="completed")
            except (ollama.ResponseError, ValidationError, OSError, ValueError) as error:
                result.update(gate_status="failed", gate_reason=str(error))
        results.append(result)
        print(f"{record['image_id']} | {result['gate_status']}")

    with output_path.open("w", encoding="utf-8") as output:
        for record in results:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return results


def _ask_gate(
    model_name: str,
    image_path: Path,
    context: dict[str, str | None],
    record: dict[str, Any],
) -> GateDecision:
    prompt = f"""You are a strict document image-admission evaluator.

Decide whether this image should be retained for a document-grounded RAG system.

Document title: {context['title']}
Lead summary: {context['lead_summary'][:1200]}
Image section: {context['section'] or 'unknown'}
Wikipedia image caption: {context['wikipedia_caption'] or 'none'}
BLIP caption: {record.get('blip_caption') or 'none'}
OCR text: {(record.get('ocr_text') or 'none')[:500]}

Definitions:
- representative: the image directly represents the document lead topic or a central claim.
- knowledge_contribution: the image adds visual evidence such as quantitative, relational, spatial, temporal, or structural knowledge beyond the lead and its caption.
- keep: true when representative or knowledge_contribution is true; otherwise drop.
- supported_claim_ids: use "lead" and/or the image section name only when supported.

Return only JSON that conforms to the requested schema. Do not assume facts not visible in the image or supplied context."""
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        format=GateDecision.model_json_schema(),
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 160},
    )
    return GateDecision.model_validate_json(response.message.content)


def _load_context(raw_dir: Path) -> dict[str, dict[str, str | None]]:
    contexts: dict[str, dict[str, str | None]] = {}
    for path in raw_dir.glob("*.json"):
        article = json.loads(path.read_text(encoding="utf-8"))
        for image in article["images"]:
            contexts[image["image_id"]] = {
                "title": article["title"],
                "lead_summary": article["lead_summary"],
                "section": image.get("section"),
                "wikipedia_caption": image.get("caption"),
            }
    return contexts


def _load_model_name(config_path: Path) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return str(config.get("pipeline", {}).get("ollama_model", "gemma4:e4b-it-q4_K_M"))


def _select_records(records: list[dict[str, Any]], max_per_article: int | None) -> list[dict[str, Any]]:
    if max_per_article is None:
        return records
    selected: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for record in records:
        if record["caption_status"] == "skipped":
            continue
        article_id = int(record["article_id"])
        if counts[article_id] >= max_per_article:
            continue
        selected.append(record)
        counts[article_id] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run document-contextual Ollama image gate")
    parser.add_argument("--input", type=Path, default=Path("data/processed/captions.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("config/articles.example.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/gated.jsonl"))
    parser.add_argument("--max-per-article", type=int, default=None)
    args = parser.parse_args()
    gate_images(args.input, args.raw_dir, args.config, args.output, args.max_per_article)


if __name__ == "__main__":
    main()
