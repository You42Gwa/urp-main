"""Download collected Wikimedia image files with provenance and integrity metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import time
from typing import Any

import requests

from mm_gate.collect import USER_AGENT
from mm_gate.schemas import ArticleRecord


MAX_IMAGE_BYTES = 15 * 1024 * 1024
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 3


def download_images(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    max_bytes: int = MAX_IMAGE_BYTES,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[dict[str, Any]]:
    """Download image URLs from collection records and write a JSONL manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    records: list[dict[str, Any]] = []

    for article_path in sorted(input_dir.glob("*.json")):
        article = ArticleRecord.model_validate_json(article_path.read_text(encoding="utf-8"))
        for image in article.images:
            record = _download_one(
                session,
                article,
                image.image_id,
                str(image.source_url),
                output_dir,
                max_bytes,
                request_delay,
            )
            record.update(
                {
                    "article_id": article.article_id,
                    "article_title": article.title,
                    "image_id": image.image_id,
                    "source_url": str(image.source_url),
                    "file_page_url": str(image.file_page_url) if image.file_page_url else None,
                    "license_name": image.license_name,
                    "license_url": str(image.license_url) if image.license_url else None,
                }
            )
            records.append(record)
            print(f"{article.title} | {image.image_id} | {record['status']}")

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def _download_one(
    session: requests.Session,
    article: ArticleRecord,
    image_id: str,
    source_url: str,
    output_dir: Path,
    max_bytes: int,
    request_delay: float,
) -> dict[str, Any]:
    existing = _existing_file(output_dir, article.article_id, image_id)
    if existing:
        return {
            "status": "downloaded",
            "local_path": existing.as_posix(),
            "mime_type": mimetypes.guess_type(existing.name)[0],
            "byte_size": existing.stat().st_size,
            "sha256": _sha256_file(existing),
            "reused": True,
        }

    try:
        response = _get_with_retry(session, source_url, request_delay)
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if not content_type.startswith("image/"):
            return {"status": "skipped", "reason": f"unexpected content type: {content_type or 'missing'}"}

        extension = _extension_for(content_type, source_url)
        destination = output_dir / str(article.article_id) / f"{image_id}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.part")
        digest = hashlib.sha256()
        size = 0

        with temporary.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    file.close()
                    temporary.unlink(missing_ok=True)
                    return {"status": "skipped", "reason": f"file exceeds {max_bytes} bytes"}
                file.write(chunk)
                digest.update(chunk)

        temporary.replace(destination)
        return {
            "status": "downloaded",
            "local_path": destination.as_posix(),
            "mime_type": content_type,
            "byte_size": size,
            "sha256": digest.hexdigest(),
        }
    except requests.RequestException as error:
        return {"status": "failed", "reason": str(error)}


def _extension_for(content_type: str, source_url: str) -> str:
    from_url = Path(source_url.split("?", 1)[0]).suffix.lower()
    if from_url and len(from_url) <= 5:
        return from_url
    return mimetypes.guess_extension(content_type) or ".img"


def _get_with_retry(session: requests.Session, source_url: str, request_delay: float) -> requests.Response:
    for attempt in range(MAX_RETRIES):
        time.sleep(request_delay)
        response = session.get(source_url, stream=True, timeout=45)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        response.close()
        wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else request_delay * (2 ** (attempt + 1))
        time.sleep(wait_seconds)
    raise requests.HTTPError(f"Wikimedia rate limit after {MAX_RETRIES} attempts: {source_url}")


def _existing_file(output_dir: Path, article_id: int, image_id: str) -> Path | None:
    matches = list((output_dir / str(article_id)).glob(f"{image_id}.*"))
    return next((path for path in matches if not path.name.endswith(".part")), None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 256), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download collected Wikimedia images")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/images"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/downloads.jsonl"))
    parser.add_argument("--max-bytes", type=int, default=MAX_IMAGE_BYTES)
    parser.add_argument("--request-delay", type=float, default=REQUEST_DELAY_SECONDS)
    args = parser.parse_args()
    download_images(args.input, args.output, args.manifest, args.max_bytes, args.request_delay)


if __name__ == "__main__":
    main()
