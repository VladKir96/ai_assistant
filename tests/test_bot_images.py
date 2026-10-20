import httpx
import pytest
from aiogram.types import Update
from pydantic import SecretStr

from personal_assistant import bot
from tests.test_images import image_bytes


@pytest.mark.parametrize("enabled", [False, True])
async def test_telegram_images_are_owner_only_bounded_and_use_multipart(
    harness, monkeypatch, enabled
):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
            "model_supports_images": enabled,
        }
    )
    posts, downloads, replies = [], [], []
    raw = image_bytes()

    async def handle(request):
        assert request.url.path == "/v1/chat/image"
        assert request.headers["Content-Type"].startswith("multipart/form-data;")
        posts.append(request.content)
        return httpx.Response(200, json={"text": "A red rectangle"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def fake_call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def download(self, file, *, destination, **kwargs):
        downloads.append(file.file_id)
        destination.write(b"x" * 10_000_001 if file.file_id == "too-big-stream" else raw)
        return destination

    async def polling(dispatcher, instance, **kwargs):
        for index, (owner, kind) in enumerate(
            [
                (True, "photo"),
                (True, "document"),
                (False, "photo"),
                (True, "album"),
                (True, "too-big"),
                (True, "svg"),
                (True, "too-big-stream"),
            ]
        ):
            user = 98765 if owner else 55555
            message = {
                "message_id": index + 1,
                "date": 0,
                "from": {"id": user, "is_bot": False, "first_name": "Test", "language_code": "ru"},
                "chat": {"id": user, "type": "private"},
            }
            if kind in {"document", "svg"}:
                message["document"] = {
                    "file_id": kind,
                    "file_unique_id": kind,
                    "mime_type": "image/svg+xml" if kind == "svg" else "image/png",
                }
            else:
                message["photo"] = [
                    {
                        "file_id": kind,
                        "file_unique_id": kind,
                        "width": 48,
                        "height": 32,
                        "file_size": 10_000_001 if kind == "too-big" else None,
                    }
                ]
                message["caption"] = "Что на фото?"
            if kind == "album":
                message["media_group_id"] = "synthetic-album"
            await dispatcher.feed_update(
                instance, Update.model_validate({"update_id": index, "message": message})
            )

    monkeypatch.setattr(bot.Bot, "__call__", fake_call)
    monkeypatch.setattr(bot.Bot, "download", download)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    if enabled:
        assert downloads == ["photo", "document", "too-big-stream"]
        assert len(posts) == 2
        assert "Что на фото?".encode() in posts[0]
        assert b"Describe this image." in posts[1]
        assert raw in posts[0]
    else:
        assert posts == downloads == []
    assert all(reply.chat_id == 98765 for reply in replies)
