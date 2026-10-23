import json
import logging
import os
from pathlib import Path

import pytest

from personal_assistant.agent.graph import Draft
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.integrations.mcp_tools import MCPTools
from personal_assistant.media.pdf_mcp import MAX_FILE_BYTES, PDFReader
from tests.conftest import ScriptedModel

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pdf_server")


@pytest.fixture
def pdf_folder(tmp_path):
    root = tmp_path / "pdf"
    root.mkdir()
    with pymupdf.open() as document:
        font = pymupdf.Font("cjk")
        for text in ("Hello PDF / Привет PDF", "Second page", "Third page", "Fourth page"):
            page = document.new_page()
            page.insert_font(fontname="fixture", fontbuffer=font.buffer)
            page.insert_text((50, 70), text, fontname="fixture")
        document.save(root / "fixture.pdf")
    return root


async def test_real_pdf_mcp_discovery_extraction_and_allowlist(harness, pdf_folder, caplog):
    settings = harness.settings.model_copy(
        update={"pdf_enabled": True, "pdf_directory": pdf_folder}
    )
    tools = MCPTools(settings)
    catalog = await tools.catalog()
    assert {item["tool"] for item in catalog} == {"list_pdf_files", "read_pdf_pages"}
    assert not tools.warnings
    assert len(json.dumps(catalog).encode()) <= 2200
    listed = await tools.read("pdf-reader", "list_pdf_files", {})
    assert "fixture.pdf" in listed["content"]
    result = await tools.read(
        "pdf-reader", "read_pdf_pages", {"file_path": "fixture.pdf", "end_page": 1}
    )
    assert result["untrusted"]
    assert "Hello PDF / Привет PDF" in result["content"]
    assert "pages 1-1 of 4" in result["content"]
    assert "Second page" not in result["content"]
    with pytest.raises(ValueError, match="allowed"):
        await tools.read("pdf-reader", "read_pdf_to_file", {"file_paths": ["fixture.pdf"]})
    with pytest.raises(ValueError, match="unavailable or failed"):
        await tools.read("pdf-reader", "read_pdf_pages", {"file_path": "../outside.pdf"})
    assert not list(pdf_folder.glob("*.txt"))
    assert "Failed to parse JSONRPC" not in caplog.text


async def test_pdf_evidence_reaches_specialist_and_reviewer(harness, pdf_folder):
    harness.settings = harness.settings.model_copy(
        update={"pdf_enabled": True, "pdf_directory": pdf_folder}
    )
    model = ScriptedModel(
        Plan(
            specialist="knowledge",
            actions=[
                Action(
                    name="mcp_read",
                    mcp_server="pdf-reader",
                    mcp_tool="read_pdf_pages",
                    mcp_arguments={"file_path": "fixture.pdf", "end_page": 1},
                )
            ],
        ),
        Draft(answer="fixture.pdf, стр. 1: Привет PDF."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model).service.chat(
        ChatRequest(text="Прочитай первую страницу fixture.pdf")
    )
    assert response.reviewed and response.review_status == "accepted"
    for _, payload, _ in model.calls[1:]:
        evidence = payload.get("operation_results", payload.get("evidence"))
        data = json.loads(evidence[0]["data"])
        assert data["untrusted"] and "Привет PDF" in data["content"]
    assert await harness.repository.list_documents() == []


def test_pdf_path_traversal_symlinks_type_and_size(pdf_folder, tmp_path):
    reader = PDFReader(pdf_folder)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes((pdf_folder / "fixture.pdf").read_bytes())
    (pdf_folder / "escape.pdf").symlink_to(outside)
    (pdf_folder / "secret.txt").write_text("synthetic private data")
    for name in ("../outside.pdf", str(outside), "escape.pdf", "secret.txt"):
        with pytest.raises(ValueError):
            reader.read(name)
    assert "escape.pdf" not in reader.list_files()
    with (pdf_folder / "large.pdf").open("wb") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="50 MiB"):
        reader.read("large.pdf")


@pytest.mark.parametrize("start,end", [(0, 1), (2, 1), (1, 4), (5, None), (3, 5)])
def test_pdf_invalid_or_excessive_page_ranges(pdf_folder, start, end):
    with pytest.raises(ValueError):
        PDFReader(pdf_folder).read("fixture.pdf", start, end)


def test_pdf_default_range_empty_directory_and_listing_pagination(pdf_folder, tmp_path):
    reader = PDFReader(pdf_folder)
    result = reader.read("fixture.pdf")
    assert "pages 1-3 of 4" in result and "Fourth page" not in result
    for index in range(25):
        (pdf_folder / f"file-{index:02}.pdf").touch()
    first = json.loads(reader.list_files())
    second = json.loads(reader.list_files(first["next_offset"]))
    assert len(first["files"]) == 20 and len(second["files"]) == 6
    assert set(first["files"]).isdisjoint(second["files"])
    assert "empty or missing" in PDFReader(tmp_path / "missing").list_files()


def test_pdf_encrypted_documents_and_bounded_excerpts(pdf_folder):
    with pymupdf.open(pdf_folder / "fixture.pdf") as document:
        document.save(
            pdf_folder / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="test"
        )
    with pytest.raises(ValueError, match="unencrypted"):
        PDFReader(pdf_folder).read("locked.pdf")
    with pymupdf.open() as document:
        for _ in range(3):
            page = document.new_page()
            page.insert_textbox(page.rect, "Long synthetic text. " * 200, fontsize=5)
        document.save(pdf_folder / "long.pdf")
    result = PDFReader(pdf_folder).read("long.pdf")
    assert len(result.encode()) < 2400 and "Bounded excerpt" in result


def test_ocr_requires_language_data_and_does_not_hide_upstream_failure(
    pdf_folder, tmp_path, monkeypatch
):
    import pdf_server

    with pytest.raises(ValueError, match="traineddata"):
        PDFReader(pdf_folder).read("fixture.pdf", ocr=True)
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    for language in ("eng", "rus"):
        (tessdata / f"{language}.traineddata").touch()

    def failed_ocr(*args, **kwargs):
        # Match the upstream source/thread so the adapter recognizes its fallback warning.
        record = logging.LogRecord(
            "root", logging.WARNING, pdf_server.__file__, 1, "OCR failed", (), None
        )
        logging.getLogger().handle(record)
        return {"text": "fallback embedded text"}

    monkeypatch.setattr(pdf_server.read_by_ocr, "fn", failed_ocr)
    with pytest.raises(ValueError, match="OCR failed"):
        PDFReader(pdf_folder, tessdata).read("fixture.pdf", ocr=True)


def test_pdf_server_name_cannot_be_overridden(harness, tmp_path):
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"pdf-reader": {"command": "/not-a-real-server"}}))
    with pytest.raises(ValueError, match="reserved"):
        MCPTools(
            harness.settings.model_copy(update={"pdf_enabled": True, "mcp_servers_file": config})
        )


@pytest.fixture
def bilingual_scan(tmp_path):
    configured = os.environ.get("TEST_PDF_TESSDATA_DIRECTORY")
    if not configured:
        pytest.skip("Set TEST_PDF_TESSDATA_DIRECTORY to test installed English/Russian OCR data")
    tessdata = Path(configured).resolve()
    assert all((tessdata / f"{code}.traineddata").is_file() for code in ("eng", "rus"))
    root = tmp_path / "scan-pdfs"
    root.mkdir()
    with pymupdf.open() as original:
        page = original.new_page()
        font = pymupdf.Font("cjk")
        page.insert_font(fontname="fixture", fontbuffer=font.buffer)
        page.insert_text((50, 100), "Hello world", fontsize=30, fontname="fixture")
        page.insert_text((50, 160), "Привет мир", fontsize=30, fontname="fixture")
        pixels = page.get_pixmap(dpi=180)
        with pymupdf.open() as scanned:
            scan = scanned.new_page()
            scan.insert_image(scan.rect, pixmap=pixels)
            assert not scan.get_text().strip()
            scanned.save(root / "scan.pdf")
    return root, tessdata


async def test_real_bilingual_scan_ocr_over_stdio(harness, bilingual_scan):
    root, tessdata = bilingual_scan
    tools = MCPTools(
        harness.settings.model_copy(
            update={
                "pdf_enabled": True,
                "pdf_directory": root,
                "pdf_tessdata_directory": tessdata,
            }
        )
    )
    result = await tools.read(
        "pdf-reader",
        "read_pdf_pages",
        {"file_path": "scan.pdf", "end_page": 1, "ocr": True, "language": "eng+rus"},
    )
    assert "method: OCR" in result["content"]
    assert "Hello world" in result["content"] and "Привет мир" in result["content"]
