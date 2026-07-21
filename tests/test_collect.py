from mm_gate.collect import MediaWikiClient, _normalize_url, _safe_name
from mm_gate.download import _extension_for


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
