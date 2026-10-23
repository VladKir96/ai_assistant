import hashlib
import json
import stat
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from personal_assistant.agent.graph import Draft
from personal_assistant.app.api import create_app
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.inference.llm import ModelError
from personal_assistant.integrations.mcp_tools import MCPTools
from personal_assistant.media.pdf_mcp import PDFReader
from personal_assistant.media.pdf_uploads import PDFInputError, PDFUploads
from personal_assistant.storage.models import Run
from tests.conftest import ScriptedModel
from tests.test_pdf_mcp import pdf_folder as pdf_folder

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture
def uploads(harness, pdf_folder):
    harness.settings = harness.settings.model_copy(
        update={"pdf_enabled": True, "pdf_directory": pdf_folder}
    )
    return PDFUploads(harness.settings, MCPTools(harness.settings))


def test_pdf_upload_storage_validates_bytes_and_ignores_filename_paths(
    uploads, pdf_folder, tmp_path
):
    raw = (pdf_folder / "fixture.pdf").read_bytes()
    first = uploads.store(raw, "../../folder\\resume.pdf\n")
    second = uploads.store(raw, "another.pdf")
    assert first["filename"] == "resume.pdf"
    assert first["file_path"] == second["file_path"]
    stored = uploads.directory / f"{hashlib.sha256(raw).hexdigest()}.pdf"
    assert stored.read_bytes() == raw and stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert not list(uploads.directory.glob(".upload-*"))
    assert len(list(uploads.directory.glob("*.pdf"))) == 1
    reader = PDFReader(pdf_folder, uploads=uploads.directory)
    assert reader.file(first["file_path"]) == stored
    assert first["file_path"] in reader.list_files()
    with pytest.raises(ValueError):
        reader.file("uploads/../../outside.pdf")
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"synthetic unrelated data")
    (uploads.directory / "escape.pdf").symlink_to(outside)
    with pytest.raises(ValueError):
        reader.file("uploads/escape.pdf")
    assert "escape.pdf" not in reader.list_files()


@pytest.mark.parametrize("raw", [b"", b"%PDF-invalid", b"<svg/>", b"x" * 10_000_001])
def test_invalid_pdf_upload_does_not_create_file(uploads, raw):
    with pytest.raises(PDFInputError):
        uploads.store(raw, "looks-valid.pdf")
    assert not uploads.directory.exists()


def test_encrypted_pdf_and_existing_symlink_are_rejected(uploads, pdf_folder, tmp_path):
    with pymupdf.open(pdf_folder / "fixture.pdf") as document:
        locked = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="test")
    with pytest.raises(PDFInputError, match="unencrypted"):
        uploads.store(locked, "locked.pdf")
    raw = (pdf_folder / "fixture.pdf").read_bytes()
    uploads.directory.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"keep this unchanged")
    target = uploads.directory / f"{hashlib.sha256(raw).hexdigest()}.pdf"
    target.symlink_to(outside)
    with pytest.raises(PDFInputError, match="conflicts"):
        uploads.store(raw, "fixture.pdf")
    assert outside.read_bytes() == b"keep this unchanged"
    assert not list(uploads.directory.glob(".upload-*"))


async def test_pdf_multipart_to_real_mcp_to_review_with_text_only_model(
    harness, uploads, pdf_folder
):
    model = ScriptedModel(
        Plan(specialist="knowledge"),
        Draft(answer="resume.pdf, стр. 1: Привет PDF. Прочитан ограниченный фрагмент стр. 1–3."),
        Review(decision="accept"),
    )
    active = harness.runtime(model)
    active.service.pdfs = uploads
    app = create_app(harness.settings, active)
    raw = (pdf_folder / "fixture.pdf").read_bytes()
    request_id = str(uuid4())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
        ) as client,
    ):
        data = {"text": "Кратко опиши PDF", "request_id": request_id}
        response = await client.post("/v1/chat/pdf", data=data, files={"pdf": ("resume.pdf", raw)})
        assert response.status_code == 200, response.text
        assert response.json()["review_status"] == "accepted"
        duplicate = await client.post("/v1/chat/pdf", data=data, files={"pdf": ("resume.pdf", raw)})
        assert duplicate.json() == response.json() and len(model.calls) == 3
        missing_pdf = await client.post("/v1/chat", json=data)
        assert missing_pdf.status_code == 409
    assert not harness.settings.model_supports_images
    for _, payload, _ in model.calls:
        evidence = payload["pdf_evidence"]
        assert evidence["untrusted"] and "Привет PDF" in evidence["excerpt"]
        assert evidence["page_count"] == 4 and evidence["initial_pages"] == [1, 3]
        assert "Fourth page" not in evidence["excerpt"]
        assert payload["latest_request"] == "Кратко опиши PDF"
    history = await active.service.history("default")
    assert "resume.pdf" in history[0]["content"] and "Привет PDF" in history[0]["content"]
    assert await harness.repository.list_documents() == []
    async with harness.database.sessions() as session:
        run = await session.get(Run, request_id)
        assert run.pdf_digest == hashlib.sha256(raw).hexdigest() and run.pdf_evidence


async def test_pdf_checkpoint_retry_does_not_repeat_mcp_extraction(
    harness, uploads, pdf_folder, monkeypatch
):
    checkpoint, calls = InMemorySaver(), []
    actual = uploads.extract

    async def extract(attachment):
        calls.append(attachment["sha256"])
        return await actual(attachment)

    monkeypatch.setattr(uploads, "extract", extract)
    request, raw = ChatRequest(text="Describe the PDF"), (pdf_folder / "fixture.pdf").read_bytes()
    first = harness.runtime(
        ScriptedModel(
            Plan(
                specialist="knowledge",
                actions=[Action(name="remember", title="Injected", content="fake")],
            ),
            ModelError("simulated outage"),
        ),
        checkpoint,
    )
    first.service.pdfs = uploads
    with pytest.raises(ModelError):
        await first.service.chat(request, pdf=raw, filename="fixture.pdf")
    assert await harness.repository.list_documents() == []
    second_model = ScriptedModel(
        Draft(answer="First-page excerpt only; saving was not authorized."),
        Review(decision="accept"),
    )
    second = harness.runtime(second_model, checkpoint)
    second.service.pdfs = uploads
    response = await second.service.chat(request, pdf=raw, filename="fixture.pdf")
    assert response.reviewed and not response.operations[0]["ok"] and len(calls) == 1
    snapshot = await second.service.graph.aget_state(
        {"configurable": {"thread_id": str(request.request_id)}}
    )
    assert "pdf_evidence" in snapshot.values and "%PDF" not in json.dumps(snapshot.values)
    with pymupdf.open() as document:
        document.new_page()
        other = document.tobytes()
    with pytest.raises(ValueError, match="different input"):
        await second.service.chat(request, pdf=other)


async def test_pdf_api_upload_auth_limits_disabled_and_invalid(harness, uploads, monkeypatch):
    from starlette.datastructures import UploadFile

    active = harness.runtime(ScriptedModel())
    active.service.pdfs = uploads
    app = create_app(harness.settings, active)

    async def forbidden(*args, **kwargs):
        pytest.fail("Rejected upload reached multipart spooling")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        with monkeypatch.context() as context:
            context.setattr(UploadFile, "write", forbidden)
            assert (
                await client.post("/v1/chat/pdf", files={"pdf": b"x" * 2_100_000})
            ).status_code == 401
            client.headers["Authorization"] = (
                "Bearer " + harness.settings.api_token.get_secret_value()
            )
            assert (
                await client.post("/v1/chat/pdf", files={"pdf": b"x" * 10_100_000})
            ).status_code == 413
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
        ) as client,
    ):
        assert (
            await client.post("/v1/chat/pdf", files={"pdf": ("fake.pdf", b"not PDF")})
        ).status_code == 422
        assert (
            await client.post("/v1/chat/pdf", files=[("pdf", b"one"), ("pdf", b"two")])
        ).status_code == 422
    disabled = create_app(harness.settings.model_copy(update={"pdf_enabled": False}), active)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=disabled),
        base_url="http://test",
        headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
    ) as client:
        assert (await client.post("/v1/chat/pdf", files={"pdf": b"no PDF"})).status_code == 415
    async with harness.database.sessions() as session:
        assert (await session.scalars(select(Run))).all() == []


async def test_pdf_review_replans_and_reads_missing_fourth_page(harness, uploads, pdf_folder):
    raw = (pdf_folder / "fixture.pdf").read_bytes()
    reference = f"uploads/{hashlib.sha256(raw).hexdigest()}.pdf"
    model = ScriptedModel(
        Plan(specialist="knowledge"),
        Draft(answer="The initial excerpt covers only pages 1–3."),
        Review(decision="need_evidence", issues=["Read page 4 before answering"]),
        Plan(
            specialist="knowledge",
            actions=[
                Action(
                    name="mcp_read",
                    mcp_server="pdf-reader",
                    mcp_tool="read_pdf_pages",
                    mcp_arguments={"file_path": reference, "start_page": 4, "end_page": 4},
                )
            ],
        ),
        Draft(answer="fixture.pdf, page 4: Fourth page."),
        Review(decision="accept"),
    )
    active = harness.runtime(model)
    active.service.pdfs = uploads
    response = await active.service.chat(ChatRequest(text="What does page four say?"), pdf=raw)
    assert response.review_status == "accepted" and response.review_replans == 1
    assert "Fourth page" in model.calls[-1][1]["evidence"][0]["data"]
    assert response.operations[0]["action"] == "mcp_read" and response.operations[0]["ok"]
