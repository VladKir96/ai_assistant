import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiogram.types import Update
from pydantic import SecretStr

from personal_assistant.app import bot
from personal_assistant.app.api import create_app
from personal_assistant.app.audio_delivery import AudioDelivery
from personal_assistant.media import voice
from personal_assistant.media.tts import MAX_SPEECH_CHARS, detect_language, prepare_speech
from tests.conftest import ScriptedModel


@pytest.mark.parametrize(
    "text, language",
    [
        ("Hello! Your appointment is tomorrow at ten.", "en"),
        ("Привет! Завтра в Белграде будет солнечно.", "ru"),
        ("Здраво! Сутра ће у Београду бити сунчано.", "sr"),
        ("Zdravo! Sutra će u Beogradu biti sunčano.", "sr"),
        ("Danas je lepo vreme u Beogradu.", "sr"),
        ("Sutra je sastanak u deset sati.", "sr"),
        ("Ваша белешка је сачувана.", "sr"),
        ("Vaša beleška je sačuvana.", "sr"),
        ("Хорошо, заметка сохранена.", "ru"),
    ],
)
def test_real_offline_language_detection(text, language):
    pytest.importorskip("lingua")
    assert detect_language(text) == language


def test_spoken_prose_omits_code_tables_urls_and_markdown():
    spoken = prepare_speech(
        "## **Forecast**\n- Read [the source](https://example.org).\n"
        "```python\nprint('private code')\n```\n"
        "| Day | Temperature |\n| --- | --- |\n| Tomorrow | 20 |\n"
        "https://example.org/path\nThe weather is *sunny*."
    )
    assert spoken.text == "Forecast Read the source. The weather is sunny."
    assert not spoken.truncated
    assert not prepare_speech("```python\nprint('unclosed')").text
    assert not prepare_speech("🎉 ** **").text
    long = prepare_speech("A complete sentence. " * 200)
    assert len(long.text) <= MAX_SPEECH_CHARS and long.truncated
    assert long.text.endswith(".")
    assert "[[" not in prepare_speech("Hello [[phonemes]]").text


async def test_audio_preference_is_authenticated_strict_persistent_and_owner_scoped(harness):
    settings = harness.settings.model_copy(update={"telegram_owner_id": 98765})
    app = create_app(settings, harness.runtime(ScriptedModel()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client,
    ):
        url = "/v1/telegram/audio"
        assert (await client.get(url)).status_code == 401
        assert (await client.put(url, json={"enabled": True})).status_code == 401
        client.headers["Authorization"] = "Bearer " + settings.api_token.get_secret_value()
        assert (await client.get(url)).json() == {"enabled": False}
        for value in ({"enabled": "true"}, {"enabled": True, "chat_id": "other"}):
            assert (await client.put(url, json=value)).status_code == 422
        assert (await client.put(url, json={"enabled": True})).json() == {"enabled": True}
    # A new API instance reads the database, not process memory or a conversation checkpoint.
    app = create_app(settings, harness.runtime(ScriptedModel()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://api",
            headers={"Authorization": "Bearer " + settings.api_token.get_secret_value()},
        ) as client,
    ):
        assert (await client.get(url)).json() == {"enabled": True}
        assert not await harness.repository.audio_enabled("tg:55555")
        await client.put(url, json={"enabled": False})
        assert (await client.get(url)).json() == {"enabled": False}


@pytest.mark.parametrize("language", ["en", "ru", "sr"])
async def test_speech_automatically_selects_voice_and_requires_matching_config(
    harness, tmp_path, monkeypatch, language
):
    model = tmp_path / (language + ".onnx")
    model.touch()
    settings = harness.settings.model_copy(update={f"piper_{language}_model": str(model)})
    calls = []

    async def fake_command(*args, stdin=None):
        calls.append((args, stdin))
        if args[-1].endswith(".ogg"):
            Path(args[-1]).write_bytes(b"ogg-opus")  # noqa: ASYNC240

    monkeypatch.setattr(voice, "command", fake_command)
    monkeypatch.setattr(voice, "detect_language", lambda text: language)
    app = voice.create_speech_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://speech",
        headers={"Authorization": "Bearer " + settings.api_token.get_secret_value()},
    ) as client:
        assert (await client.post("/v1/audio/speech", json={"text": "Hello"})).status_code == 503
        assert calls == []
        Path(str(model) + ".json").write_text("{}")  # noqa: ASYNC240
        response = await client.post("/v1/audio/speech", json={"text": "Hello\nworld"})
        assert response.content == b"ogg-opus" and response.headers["Content-Language"] == language
        assert calls[0][1] == b"Hello world\n"
        assert str(model) in calls[0][0]
        # Neither clients nor the LLM can override automatic language detection.
        assert (
            await client.post("/v1/audio/speech", json={"text": "Hello", "language": "ru"})
        ).status_code == 422


async def test_bounded_audio_queue_does_not_block_and_off_cancels_inflight(harness):
    delivery = AudioDelivery(harness.settings)
    started, cancelled = asyncio.Event(), asyncio.Event()
    notices, audio = [], []

    async def speak(text):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def notice(text):
        notices.append(text)

    async def send(*args, **kwargs):
        audio.append(args)

    delivery.speech.speak = speak
    message = SimpleNamespace(answer=notice, answer_voice=send)
    await delivery.submit(message, "Your answer.")
    await asyncio.wait_for(started.wait(), 1)
    for _ in range(4):
        await delivery.submit(message, "Another answer.")
    assert delivery.queue.qsize() == 3 and any("queue is full" in text for text in notices)
    await asyncio.wait_for(delivery.cancel(), 1)
    assert cancelled.is_set() and not audio and delivery.queue.empty()
    await asyncio.wait_for(delivery.queue.join(), 1)


async def test_disconnected_speech_request_cancels_server_synthesis(harness, tmp_path, monkeypatch):
    from fastapi import Request

    model = tmp_path / "voice.onnx"
    model.touch()
    Path(str(model) + ".json").write_text("{}")  # noqa: ASYNC240
    settings = harness.settings.model_copy(update={"piper_en_model": str(model)})
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def fake_command(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def disconnected():
        await started.wait()
        return {"type": "http.disconnect"}

    monkeypatch.setattr(voice, "detect_language", lambda text: "en")
    monkeypatch.setattr(voice, "command", fake_command)
    monkeypatch.setattr(Request, "receive", property(lambda self: disconnected))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=voice.create_speech_app(settings)),
        base_url="http://speech",
        headers={"Authorization": "Bearer " + settings.api_token.get_secret_value()},
    ) as client:
        response = await asyncio.wait_for(
            client.post("/v1/audio/speech", json={"text": "Hello"}), 2
        )
    assert response.status_code == 499 and cancelled.is_set()


async def test_cancelling_external_speech_process_kills_and_reaps_it(monkeypatch):
    original = asyncio.create_subprocess_exec
    process, started = [], asyncio.Event()

    async def create(*args, **kwargs):
        child = await original(*args, **kwargs)
        process.append(child)
        started.set()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(voice.command(sys.executable, "-c", "import time; time.sleep(60)"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert process[0].returncode is not None
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_synthesis_failure_keeps_worker_alive_and_labels_partial_audio(harness):
    delivery = AudioDelivery(harness.settings)
    notices, audio = [], []

    async def speak(text):
        if text == "Failed answer.":
            raise ValueError("private-model-path")
        return b"ogg-opus"

    async def notice(text):
        notices.append(text)

    async def send(file, **kwargs):
        audio.append((file, kwargs))

    delivery.speech.speak = speak
    message = SimpleNamespace(answer=notice, answer_voice=send, message_id=123)
    try:
        await delivery.submit(message, "Failed answer.")
        await delivery.submit(message, "A complete sentence. " * 200)
        await asyncio.wait_for(delivery.queue.join(), 1)
        assert len(notices) == len(audio) == 1
        assert "private-model-path" not in notices[0]
        assert audio[0][1]["caption"] == "Audio contains the first part of the answer."
        assert audio[0][1]["reply_parameters"].message_id == 123
    finally:
        await delivery.cancel()


async def test_telegram_audio_mode_persists_across_new_and_bot_restart(harness, monkeypatch):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
            "voice_output_enabled": True,
            "voice_enabled": False,
        }
    )
    replies, speech_inputs, chat_inputs = [], [], []
    delivered = asyncio.Event()

    async def handle(request):
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"tts_languages": ["en", "ru", "sr"], "tts_ready": True}
            )
        if request.url.path == "/v1/telegram/audio":
            if request.method == "PUT":
                await harness.repository.set_audio_enabled(
                    "tg:98765", json.loads(request.content)["enabled"]
                )
            return httpx.Response(
                200, json={"enabled": await harness.repository.audio_enabled("tg:98765")}
            )
        if request.url.path == "/v1/audio/speech":
            speech_inputs.append(json.loads(request.content))
            return httpx.Response(200, content=b"ogg-opus")
        assert request.url.path == "/v1/chat"
        chat_inputs.append(json.loads(request.content))
        return httpx.Response(200, json={"text": "Your answer is ready."})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def send(self, method, request_timeout=None):
        replies.append(method)
        if type(method).__name__ == "SendVoice":
            delivered.set()
        return True

    batches = iter(
        [
            [
                (98765, "/audio status"),
                (55555, "/audio on"),
                (98765, "First request"),
                (98765, "/audio on"),
                (98765, "/new"),
                (98765, "Second request"),
            ],
            [(98765, "/audio status"), (98765, "Third request")],
            [(98765, "/audio off"), (98765, "Fourth request")],
        ]
    )

    async def polling(dispatcher, instance, **kwargs):
        delivered.clear()
        for index, (user, text) in enumerate(next(batches)):
            await dispatcher.feed_update(
                instance,
                Update.model_validate(
                    {
                        "update_id": index,
                        "message": {
                            "message_id": index + 1,
                            "date": 0,
                            "from": {"id": user, "is_bot": False, "first_name": "Test"},
                            "chat": {"id": user, "type": "private"},
                            "text": text,
                        },
                    }
                ),
            )
        if await harness.repository.audio_enabled("tg:98765"):
            await asyncio.wait_for(delivered.wait(), 1)

    monkeypatch.setattr(bot.Bot, "__call__", send)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    for _ in range(3):
        await bot.run_bot(settings)
    assert len(chat_inputs) == 4 and len(speech_inputs) == 2
    assert all(item == {"text": "Your answer is ready."} for item in speech_inputs)
    assert all(method.chat_id == 98765 for method in replies)
    texts = [getattr(method, "text", "") for method in replies]
    assert sum("Audio mode: on" in text for text in texts) == 2
    assert sum("Audio mode: off" in text for text in texts) == 2
    # Every generated voice follows its already delivered text.
    for index, method in enumerate(replies):
        if type(method).__name__ == "SendVoice":
            assert any(
                getattr(prior, "text", "") == "Your answer is ready." for prior in replies[:index]
            )


@pytest.mark.parametrize("kind", ["text", "voice", "photo", "pdf"])
async def test_audio_mode_applies_to_all_question_types(harness, monkeypatch, kind):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
            "voice_output_enabled": True,
            "voice_enabled": True,
            "model_supports_images": True,
            "pdf_enabled": True,
        }
    )
    delivered = asyncio.Event()
    paths = []

    async def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/telegram/audio":
            return httpx.Response(200, json={"enabled": True})
        if request.url.path == "/v1/audio/transcriptions":
            return httpx.Response(200, json={"text": "Question"})
        if request.url.path == "/v1/audio/speech":
            assert json.loads(request.content) == {"text": "Final reviewed answer."}
            return httpx.Response(200, content=b"ogg-opus")
        return httpx.Response(200, json={"text": "Final reviewed answer."})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def download(self, file, destination, **kwargs):
        destination.write(b"fixture-upload")

    async def send(self, method, request_timeout=None):
        if type(method).__name__ == "SendVoice":
            delivered.set()
        return True

    async def polling(dispatcher, instance, **kwargs):
        message = {
            "message_id": 1,
            "date": 0,
            "from": {"id": 98765, "is_bot": False, "first_name": "Test"},
            "chat": {"id": 98765, "type": "private"},
        }
        file = {"file_id": "fixture", "file_unique_id": "fixture"}
        if kind == "voice":
            message["voice"] = {**file, "duration": 1}
        elif kind == "photo":
            message["photo"] = [{**file, "width": 1, "height": 1}]
        elif kind == "pdf":
            message["document"] = {**file, "file_name": "test.pdf", "mime_type": "application/pdf"}
        else:
            message["text"] = "Question"
        await dispatcher.feed_update(
            instance, Update.model_validate({"update_id": 1, "message": message})
        )
        await asyncio.wait_for(delivered.wait(), 1)

    monkeypatch.setattr(bot.Bot, "__call__", send)
    monkeypatch.setattr(bot.Bot, "download", download)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    assert paths[-1] == "/v1/audio/speech"
    endpoint = {"photo": "/image", "pdf": "/pdf"}.get(kind, "")
    assert f"/v1/chat{endpoint}" in paths
