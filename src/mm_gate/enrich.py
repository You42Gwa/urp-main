"""OCR and keyword enrichment for downloaded document images."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, UnidentifiedImageError


DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
STOPWORDS = {
    "about", "after", "also", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "with",
}


def enrich_images(manifest_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Run English OCR over downloaded images and derive transparent keywords."""
    _configure_tesseract()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    enriched: list[dict[str, Any]] = []

    for record in records:
        enriched_record = {**record, "ocr_text": None, "keywords": [], "enrichment_status": None}
        if record["status"] != "downloaded":
            enriched_record.update(enrichment_status="skipped", enrichment_reason="image unavailable")
        else:
            enriched_record.update(_enrich_one(Path(record["local_path"])))
        enriched.append(enriched_record)
        print(f"{record['image_id']} | {enriched_record['enrichment_status']}")

    with output_path.open("w", encoding="utf-8") as output:
        for record in enriched:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return enriched


def _configure_tesseract() -> None:
    configured = os.environ.get("TESSERACT_CMD")
    executable = Path(configured) if configured else DEFAULT_TESSERACT
    if not executable.exists():
        raise FileNotFoundError(
            "Tesseract not found. Set TESSERACT_CMD or install Tesseract OCR."
        )
    pytesseract.pytesseract.tesseract_cmd = str(executable)


def _enrich_one(image_path: Path) -> dict[str, Any]:
    if image_path.suffix.lower() == ".svg":
        return {
            "enrichment_status": "skipped",
            "enrichment_reason": "SVG vector image is not supported by raster OCR",
        }
    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image.convert("RGB"), lang="eng", config="--psm 6")
        cleaned = _clean_text(text)
        return {
            "ocr_text": cleaned,
            "keywords": _keywords(cleaned),
            "enrichment_status": "completed",
        }
    except (OSError, UnidentifiedImageError, pytesseract.TesseractError) as error:
        return {"enrichment_status": "failed", "enrichment_reason": str(error)}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _keywords(text: str, limit: int = 10) -> list[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    counts = Counter(term for term in terms if term not in STOPWORDS)
    return [term for term, _ in counts.most_common(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR and keyword enrichment")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/downloads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/enrichment.jsonl"))
    args = parser.parse_args()
    enrich_images(args.manifest, args.output)


if __name__ == "__main__":
    main()
