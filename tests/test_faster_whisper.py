import io
import json
import sys
import threading
import wave
from types import SimpleNamespace

import httpx
import pytest
from aiogram.types import Update
from pydantic import SecretStr

from personal_assistant.app import bot
from personal_assistant.media import stt
from personal_assistant.media.voice import create_speech_app


def model_directory(tmp_path):
    for name in (*stt.MODEL_FILES, "vocabulary.json"):
        (tmp_path / name).write_bytes(b"fixture")
    return tmp_path


def wav(seconds=1, rate=8000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * int(seconds * rate))
    return output.getvalue()


async def test_offline_model_cached_and_lazy_inference_runs_off_event_loop(
    harness, tmp_path, monkeypatch
):
    settings = harness.settings.model_copy(
        update={
            "stt_backend": "faster_whisper",
            "faster_whisper_model": str(model_directory(tmp_path)),
            "voice_enabled": True,
        }
    )
    main_thread, loads, calls, telemetry = threading.get_ident(), [], [], []

    class Model:
        def __init__(self, path, **kwargs):
            assert threading.get_ident() != main_thread
            assert kwargs["local_files_only"] is True
            assert kwargs["device"] == "cpu" and kwargs["compute_type"] == "int8"
            loads.append(path)

        def transcribe(self, audio, **kwargs):
            assert kwargs["language"] is None
            assert kwargs["task"] == "transcribe" and kwargs["vad_filter"]
            assert not kwargs["condition_on_previous_text"]

            def segments():
                assert threading.get_ident() != main_thread
                calls.append(True)
                yield SimpleNamespace(text=" Привет, ")
                yield SimpleNamespace(text=" помощник. ")

            return segments(), SimpleNamespace(language="ru")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(disable_telemetry_events=lambda: telemetry.append("disabled")),
    )
    monkeypatch.setattr(stt, "decode_audio", lambda audio: "waveform")
    app = create_speech_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://speech"
        ) as client,
    ):
        assert (
            await client.post("/v1/audio/transcriptions", files={"file": b"audio"})
        ).status_code == 401
        assert calls == []
        client.headers["Authorization"] = "Bearer " + settings.api_token.get_secret_value()
        for _ in range(2):
            response = await client.post("/v1/audio/transcriptions", files={"file": b"audio"})
            assert response.json() == {"text": "Привет, помощник.", "language": "ru"}
        assert len(loads) == 1 and len(calls) == 2
        assert telemetry == ["disabled"]


@pytest.mark.parametrize("language", ["en", "ru"])
def test_explicit_language_is_transcription_not_translation(
    harness, tmp_path, monkeypatch, language
):
    engine = stt.FasterWhisper(harness.settings.model_copy(update={"stt_language": language}))

    def transcribe(audio, **kwargs):
        assert kwargs["language"] == language and kwargs["task"] == "transcribe"
        return iter(
            [SimpleNamespace(text="Hello" if language == "en" else "Привет")]
        ), SimpleNamespace(language=language)

    engine.model = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(stt, "decode_audio", lambda audio: "waveform")
    assert engine.transcribe(b"audio")["language"] == language


@pytest.mark.parametrize("audio", [b"not an audio file", wav(181)])
def test_invalid_and_overlong_decoded_audio_rejected(audio):
    pytest.importorskip("av")
    with pytest.raises(ValueError):
        stt.decode_audio(audio)


def test_audio_resampled_to_mono_16khz_float32():
    pytest.importorskip("av")
    audio = stt.decode_audio(wav(seconds=1, rate=8000))
    assert len(audio) == 16000
    assert str(audio.dtype) == "float32" and audio.ndim == 1
    assert not audio.any()


def test_incomplete_model_fails_before_any_implicit_download(harness, tmp_path, monkeypatch):
    engine = stt.FasterWhisper(
        harness.settings.model_copy(update={"faster_whisper_model": str(tmp_path)})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete local models must not trigger an implicit download")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=forbidden))
    with pytest.raises(ValueError, match="speech-download"):
        engine.load()


def test_explicit_download_pins_resolved_revision_and_records_model_origin(tmp_path, monkeypatch):
    calls = []

    class Hub:
        def __init__(self, **kwargs):
            assert kwargs == {"token": False}

        def model_info(self, repo_id, revision):
            assert repo_id == stt.MODEL_REPO and revision == "main"
            return SimpleNamespace(sha="resolved-commit")

    def snapshot(repo_id, **kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == "resolved-commit" and kwargs["token"] is False
        model_directory(kwargs["local_dir"])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Hub, snapshot_download=snapshot)
    )
    assert stt.download_model(tmp_path) == tmp_path
    assert len(calls) == 1
    assert json.loads((tmp_path / "assistant-model.json").read_text()) == {
        "repo_id": stt.MODEL_REPO,
        "revision": "resolved-commit",
    }


def test_silence_does_not_produce_a_successful_empty_transcript(harness, monkeypatch):
    engine = stt.FasterWhisper(harness.settings)
    engine.model = SimpleNamespace(transcribe=lambda *args, **kwargs: (iter([]), None))
    monkeypatch.setattr(stt, "decode_audio", lambda audio: "waveform")
    with pytest.raises(ValueError, match="No speech recognized"):
        engine.transcribe(b"silence")


async def test_transcription_errors_are_sanitized_and_upload_size_is_enforced(harness, monkeypatch):
    settings = harness.settings.model_copy(update={"stt_backend": "faster_whisper"})

    def fail(self, audio):
        raise RuntimeError("private-path-or-service-details")

    monkeypatch.setattr(stt.FasterWhisper, "transcribe", fail)
    app = create_speech_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://speech",
        headers={"Authorization": "Bearer " + settings.api_token.get_secret_value()},
    ) as client:
        response = await client.post("/v1/audio/transcriptions", files={"file": b"audio"})
        assert response.status_code == 503
        assert "private-path" not in response.text
        assert (
            await client.post("/v1/audio/transcriptions", files={"file": b"a" * 10_000_001})
        ).status_code == 413


async def test_voice_message_reaches_chat_and_replies_as_text_without_tts(harness, monkeypatch):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:testing-only"),
            "telegram_owner_id": 98765,
            "voice_enabled": True,
            "voice_output_enabled": False,
        }
    )
    requests, methods = [], []

    async def handle(request):
        requests.append(request)
        if request.url.path == "/v1/audio/transcriptions":
            return httpx.Response(200, json={"text": "Какие у меня задачи?"})
        assert request.url.path == "/v1/chat"
        assert json.loads(request.content)["text"] == "Какие у меня задачи?"
        return httpx.Response(200, json={"text": "Список задач пуст."})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def download(self, file, destination, **kwargs):
        destination.write(b"ogg-input")

    async def fake_call(self, method, request_timeout=None):
        methods.append(method)
        return True

    async def polling(dispatcher, instance, **kwargs):
        for index, (user_id, duration) in enumerate([(98765, 2), (54321, 2), (98765, 181)]):
            update = Update.model_validate(
                {
                    "update_id": index,
                    "message": {
                        "message_id": index + 1,
                        "date": 0,
                        "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
                        "chat": {"id": user_id, "type": "private"},
                        "voice": {
                            "file_id": "fixture",
                            "file_unique_id": "fixture",
                            "duration": duration,
                            "file_size": 100,
                        },
                    },
                }
            )
            await dispatcher.feed_update(instance, update)

    monkeypatch.setattr(bot.Bot, "download", download)
    monkeypatch.setattr(bot.Bot, "__call__", fake_call)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    assert [request.url.path for request in requests] == ["/v1/audio/transcriptions", "/v1/chat"]
    assert any(getattr(method, "text", "") == "Список задач пуст." for method in methods)
    assert any("3 minutes" in getattr(method, "text", "") for method in methods)
