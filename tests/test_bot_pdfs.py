import httpx
import pytest
from aiogram.types import Update
from pydantic import SecretStr

from personal_assistant import bot


@pytest.mark.parametrize("enabled", [False, True])
async def test_pdf_documents_use_pdf_route_without_vision(harness, monkeypatch, enabled):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
            "pdf_enabled": enabled,
            "model_supports_images": False,
        }
    )
    posts, downloads, replies = [], [], []
    raw = b"%PDF-synthetic-download"

    async def handle(request):
        assert request.url.path == "/v1/chat/pdf"
        assert request.headers["Content-Type"].startswith("multipart/form-data;")
        posts.append(request.content)
        return httpx.Response(200, json={"text": "PDF excerpt read and reviewed"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def download(self, file, *, destination, **kwargs):
        downloads.append(file.file_id)
        destination.write(b"x" * 10_000_001 if file.file_id == "stream-overflow" else raw)
        return destination

    async def polling(dispatcher, instance, **kwargs):
        for index, (user, kind) in enumerate(
            [
                (98765, "pdf"),
                (98765, "extension-only"),
                (55555, "pdf"),
                (98765, "album"),
                (98765, "too-big"),
                (98765, "stream-overflow"),
            ]
        ):
            document = {
                "file_id": kind,
                "file_unique_id": kind,
                "file_name": "resume.pdf",
                "mime_type": "application/pdf",
            }
            if kind == "extension-only":
                document.update(file_name="RESUME.PDF", mime_type="application/octet-stream")
            if kind == "too-big":
                document["file_size"] = 10_000_001
            message = {
                "message_id": index + 1,
                "date": 0,
                "from": {"id": user, "is_bot": False, "first_name": "Test", "language_code": "ru"},
                "chat": {"id": user, "type": "private"},
                "document": document,
            }
            if kind == "pdf":
                message["caption"] = "Кратко опиши резюме"
            if kind == "album":
                message["media_group_id"] = "synthetic-album"
            await dispatcher.feed_update(
                instance, Update.model_validate({"update_id": index, "message": message})
            )

    monkeypatch.setattr(bot.Bot, "__call__", call)
    monkeypatch.setattr(bot.Bot, "download", download)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    if enabled:
        assert downloads == ["pdf", "extension-only", "stream-overflow"]
        assert len(posts) == 2 and raw in posts[0]
        assert "Кратко опиши резюме".encode() in posts[0]
        assert "первым трём страницам".encode() in posts[1]
        assert b"RESUME.PDF" in posts[1]
    else:
        assert posts == downloads == []
    assert all(reply.chat_id == 98765 for reply in replies)
