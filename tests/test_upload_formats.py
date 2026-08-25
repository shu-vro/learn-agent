from pathlib import Path

from src.config.constants import SUPPORTED_UPLOAD_SUFFIXES


def test_supported_suffixes_cover_docling_default_formats():
    for name in (
        "paper.pdf",
        "notes.docx",
        "deck.pptx",
        "sheet.xlsx",
        "page.html",
        "readme.md",
        "scan.png",
        "talk.mp3",
    ):
        assert Path(name).suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES, name


def test_supported_suffixes_reject_unknown_and_match_path_suffix():
    assert Path("installer.exe").suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES
    # Compound Docling extensions must still match what Path.suffix returns.
    assert Path("scans.tar.gz").suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES
    assert Path("doc.dclg.xml").suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES
