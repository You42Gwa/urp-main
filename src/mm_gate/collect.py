"""Collect English Wikipedia article and image metadata through MediaWiki APIs."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urljoin

import requests
import yaml
from bs4 import BeautifulSoup

from mm_gate.schemas import ArticleRecord, ImageRecord


API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_URL = "https://en.wikipedia.org/wiki/"
USER_AGENT = "multimodal-image-gate/0.1 (https://github.com/You42Gwa/urp-main)"


class MediaWikiClient:
    """Small MediaWiki client with stable provenance fields for each result."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def api_get(self, **params: Any) -> dict[str, Any]:
        response = self.session.get(
            API_URL,
            params={"format": "json", "formatversion": "2", **params},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def article_metadata(self, requested_title: str) -> dict[str, Any]:
        payload = self.api_get(
            action="query",
            redirects="1",
            titles=requested_title,
            prop="extracts|revisions",
            exintro="1",
            explaintext="1",
            rvprop="ids|timestamp",
            rvlimit="1",
        )
        pages = payload["query"]["pages"]
        if not pages or "missing" in pages[0]:
            raise ValueError(f"Wikipedia article not found: {requested_title}")
        return pages[0]

    def article_html(self, title: str) -> tuple[str, list[str]]:
        payload = self.api_get(action="parse", page=title, prop="text|sections")
        parse = payload["parse"]
        sections = [section["line"] for section in parse.get("sections", [])]
        return parse["text"], sections

    def image_metadata(self, file_title: str) -> dict[str, str | None]:
        payload = self.api_get(
            action="query",
            titles=file_title,
            prop="imageinfo",
            iiprop="url|mime|extmetadata",
        )
        pages = payload["query"]["pages"]
        if not pages or not pages[0].get("imageinfo"):
            return {
                "mime_type": None,
                "license_name": None,
                "license_url": None,
                "file_page_url": None,
            }

        info = pages[0]["imageinfo"][0]
        metadata = info.get("extmetadata", {})
        license_url = _normalize_url(
            metadata.get("LicenseUrl", {}).get("value") or info.get("descriptionurl")
        )
        return {
            "mime_type": info.get("mime"),
            "license_name": metadata.get("LicenseShortName", {}).get("value"),
            "license_url": license_url,
            "file_page_url": _normalize_url(info.get("descriptionurl")),
        }

    def collect_article(self, requested_title: str, max_images: int) -> ArticleRecord:
        page = self.article_metadata(requested_title)
        title = page["title"]
        html, sections = self.article_html(title)
        revision = page.get("revisions", [{}])[0]
        images = self._extract_images(html, max_images)

        return ArticleRecord(
            article_id=page["pageid"],
            title=title,
            source_url=f"{WIKI_URL}{quote(title.replace(' ', '_'))}",
            lead_summary=page.get("extract", "").strip(),
            revision_id=revision.get("revid"),
            revision_timestamp=revision.get("timestamp"),
            sections=sections,
            images=images,
        )

    def _extract_images(self, html: str, max_images: int) -> list[ImageRecord]:
        soup = BeautifulSoup(html, "html.parser")
        records: list[ImageRecord] = []
        seen_urls: set[str] = set()

        for image in soup.find_all("img"):
            src = image.get("src")
            if not src:
                continue
            source_url = urljoin("https:", src) if src.startswith("//") else urljoin(WIKI_URL, src)
            if source_url in seen_urls:
                continue

            figure = image.find_parent("figure")
            caption_tag = figure.find("figcaption") if figure else None
            caption = caption_tag.get_text(" ", strip=True) if caption_tag else None
            heading = image.find_previous(["h2", "h3", "h4"])
            section = heading.get_text(" ", strip=True) if heading else "Lead"
            file_title = self._file_title(image)
            fallback_file_page_url = (
                f"{WIKI_URL}{quote(file_title.replace(' ', '_'))}" if file_title else None
            )
            metadata = self.image_metadata(file_title) if file_title else {}
            file_page_url = metadata.pop("file_page_url", None) or fallback_file_page_url

            digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
            records.append(
                ImageRecord(
                    image_id=f"img-{digest}",
                    file_title=file_title,
                    source_url=source_url,
                    file_page_url=file_page_url,
                    caption=caption,
                    section=section,
                    **metadata,
                )
            )
            seen_urls.add(source_url)
            if len(records) >= max_images:
                break

        return records

    @staticmethod
    def _file_title(image: Any) -> str | None:
        anchor = image.find_parent("a")
        href = anchor.get("href") if anchor else None
        if not href:
            return None
        match = re.search(r"(?:\./|/wiki/)(File:[^?#]+)", unquote(href))
        return match.group(1).replace("_", " ") if match else None


def load_titles(config_path: Path) -> tuple[list[str], int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    articles = config.get("articles", [])
    titles = [item["title"] if isinstance(item, dict) else str(item) for item in articles]
    max_images = int(config.get("pipeline", {}).get("max_images_per_article", 10))
    if not titles:
        raise ValueError("Configuration contains no articles")
    return titles, max_images


def collect_to_directory(config_path: Path, output_dir: Path) -> list[Path]:
    titles, max_images = load_titles(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = MediaWikiClient()
    outputs: list[Path] = []

    for title in titles:
        article = client.collect_article(title, max_images=max_images)
        filename = f"{article.article_id}-{_safe_name(article.title)}.json"
        destination = output_dir / filename
        destination.write_text(article.model_dump_json(indent=2), encoding="utf-8")
        outputs.append(destination)
        print(f"Collected {article.title}: {len(article.images)} images")
    return outputs


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("/"):
        return urljoin("https://commons.wikimedia.org", value)
    return value if value.startswith(("https://", "http://")) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect English Wikipedia article metadata")
    parser.add_argument("--config", type=Path, default=Path("config/articles.example.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    collect_to_directory(args.config, args.output)


if __name__ == "__main__":
    main()
