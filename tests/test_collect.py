from mm_gate.collect import MediaWikiClient, _normalize_url, _safe_name
from mm_gate.caption import MODEL_ID
from mm_gate.download import _extension_for
from mm_gate.enrich import _keywords
from mm_gate.gate import _select_records
from mm_gate.kg import _slug
from mm_gate.report import _escape


def test_safe_name() -> None:
    assert _safe_name("World War II") == "world-war-ii"


def test_extract_file_title() -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup('<a href="./File:Example_image.jpg"><img src="//example.test/a.jpg"></a>', "html.parser")
    assert MediaWikiClient._file_title(soup.img) == "File:Example image.jpg"


def test_normalize_url() -> None:
    assert _normalize_url("//en.wikipedia.org/wiki/File:Example.jpg") == "https://en.wikipedia.org/wiki/File:Example.jpg"


def test_extension_for_content_type() -> None:
    assert _extension_for("image/jpeg", "https://example.test/thumbnail/320px-file") == ".jpg"


def test_keywords_excludes_stopwords() -> None:
    assert _keywords("The Apollo 11 mission and Apollo program")[:2] == ["apollo", "mission"]


def test_blip_model_id() -> None:
    assert MODEL_ID == "Salesforce/blip-image-captioning-base"


def test_gate_schema_rejects_invalid_decision() -> None:
    from pydantic import ValidationError

    from mm_gate.schemas import GateDecision

    try:
        GateDecision(
            representative=False,
            knowledge_contribution=False,
            image_type="photo",
            decision="maybe",
            confidence=0.2,
            reason="invalid test",
        )
    except ValidationError:
        return
    raise AssertionError("invalid decision should fail validation")


def test_gate_selection_is_balanced_and_skips_svg() -> None:
    records = [
        {"article_id": 1, "caption_status": "completed"},
        {"article_id": 1, "caption_status": "completed"},
        {"article_id": 1, "caption_status": "completed"},
        {"article_id": 2, "caption_status": "skipped"},
        {"article_id": 2, "caption_status": "completed"},
    ]
    selected = _select_records(records, max_per_article=2)
    assert selected == [records[0], records[1], records[4]]


def test_kg_slug() -> None:
    assert _slug("Alan Turing / Lead") == "alan-turing-lead"


def test_report_escape() -> None:
    assert _escape("A | B\nC") == "A \\| B C"
