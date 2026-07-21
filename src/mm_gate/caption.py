"""BLIP image captioning on the local CUDA GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_ID = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    """Lazily load BLIP once and generate concise factual image captions."""

    def __init__(self, cache_dir: Path) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption(self, image_path: Path) -> str:
        with Image.open(image_path) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(generated[0], skip_special_tokens=True).strip()


def caption_images(
    enrichment_path: Path,
    output_path: Path,
    cache_dir: Path = Path(".cache/huggingface"),
) -> list[dict[str, Any]]:
    """Add BLIP captions to OCR-enriched image records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in enrichment_path.read_text(encoding="utf-8").splitlines()]
    captioner: BlipCaptioner | None = None
    results: list[dict[str, Any]] = []

    for record in records:
        result = {**record, "blip_caption": None, "caption_status": None}
        if record["enrichment_status"] == "skipped":
            result.update(caption_status="skipped", caption_reason="unsupported image format")
        else:
            try:
                captioner = captioner or BlipCaptioner(cache_dir)
                result.update(blip_caption=captioner.caption(Path(record["local_path"])), caption_status="completed")
            except (OSError, UnidentifiedImageError, RuntimeError, ValueError) as error:
                result.update(caption_status="failed", caption_reason=str(error))
        results.append(result)
        print(f"{record['image_id']} | {result['caption_status']}")

    with output_path.open("w", encoding="utf-8") as output:
        for record in results:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BLIP captions for enriched images")
    parser.add_argument("--input", type=Path, default=Path("data/processed/enrichment.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/captions.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()
    caption_images(args.input, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
