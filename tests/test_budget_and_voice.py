import copy
import json
from pathlib import Path

import httpx
import pytest

from personal_assistant.agent.graph import PLANNER, AssistantGraph
from personal_assistant.core.schemas import Plan
from personal_assistant.core.time_utils import clock_context
from personal_assistant.inference.llm import ContextLimitError, HTTPModel, fit_payload, prompt_json
from personal_assistant.media.voice import create_speech_app


def test_budget_reduction_preserves_request_and_entire_reviewed_draft():
    original = {
        "latest_request": "Summarize this",
        "draft": "A" * 1000,
        "context": "B" * 4000,
        "knowledge": [{"content": "C" * 4000}],
    }
    fitted = fit_payload("System", original, {}, 2000)
    assert fitted["latest_request"] == original["latest_request"]
    assert fitted["draft"] == original["draft"]
    assert original["context"] == "B" * 4000
    assert fitted["context_was_reduced"] is True
    with pytest.raises(ContextLimitError):
        fit_payload("System", {"draft": "A" * 10000}, {}, 2000)


@pytest.mark.parametrize("context", ["", "short", 'Заметка: "погода"\\\n' * 100])
@pytest.mark.parametrize("available", [0, 61])
def test_optional_context_fits_without_expanding_or_corrupting_unicode(context, available):
    original = {"latest_request": "Какая погода завтра?", "context": context}
    minimal = {**original, "context": "", "context_was_reduced": True}
    budget = len(("System" + prompt_json(minimal) + "{}").encode()) + available
    fitted = fit_payload("System", original, {}, budget)
    assert len(("System" + prompt_json(fitted) + "{}").encode()) <= budget
    assert context.startswith(fitted["context"])
    assert fitted["latest_request"] == original["latest_request"]
    assert original["context"] == context


async def test_russian_tomorrow_weather_with_populated_history_reaches_inference(harness):
    settings = harness.settings.model_copy(
        update={
            "weather_default_name": "Belgrade",
            "weather_default_latitude": 44.804,
            "weather_default_longitude": 20.4651,
        }
    )
    graph = object.__new__(AssistantGraph)
    graph.settings = settings
    payload = graph.payload(
        {
            "text": "Какая погода завтра?",
            "current_time": clock_context(settings.timezone)["current_time"],
            "context": {"summary": "Previous conversation " * 150, "recent_messages": []},
            "knowledge": [],
            "tasks": [],
            "events": [],
        }
    )
    payload.update(mcp_read_tools=[], job_sites=[])
    original_context = copy.deepcopy(payload["context"])
    received = []

    async def handle(request):
        if request.url.path == "/tokenize":
            return httpx.Response(404)
        body = json.loads(request.content)
        system, raw = (message["content"] for message in body["messages"])
        actual = json.loads(raw)
        assert len((system + raw + prompt_json(Plan.model_json_schema())).encode()) <= (
            settings.input_budget - 150
        )
        assert actual["latest_request"] == "Какая погода завтра?"
        assert actual["relative_dates"] == payload["relative_dates"]
        assert actual["web_capabilities"]["weather"]["default_location"] == "Belgrade"
        assert actual["context_was_reduced"]
        assert original_context["summary"].startswith(actual["context"]["summary"])
        assert "Authoritative runtime clock" in system
        received.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"specialist":"research","actions":'
                            '[{"name":"get_weather","weather_period":"tomorrow"}]}'
                        }
                    }
                ]
            },
        )

    model = HTTPModel(settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        plan = await model.generate(PLANNER, payload, Plan)
        assert plan.actions[0].weather_period == "tomorrow"
        assert len(received) == 1
        assert payload["context"] == original_context
    finally:
        await model.close()


async def test_voice_http_contract_with_mock_executables(harness, tmp_path, monkeypatch):
    whisper, piper = tmp_path / "whisper.bin", tmp_path / "russian.onnx"
    whisper.touch()
    piper.touch()
    Path(str(piper) + ".json").write_text("{}")  # noqa: ASYNC240
    settings = harness.settings.model_copy(
        update={
            "whisper_model": str(whisper),
            "piper_ru_model": str(piper),
        }
    )
    calls = []

    async def fake_command(*args, stdin=None):
        calls.append((args, stdin))
        if "-of" in args:
            Path(args[args.index("-of") + 1] + ".txt").write_text(  # noqa: ASYNC240
                "Запомни: я говорю по-русски."
            )
        elif args[-1].endswith(".ogg"):
            Path(args[-1]).write_bytes(b"synthetic-ogg")  # noqa: ASYNC240

    monkeypatch.setattr("personal_assistant.media.voice.command", fake_command)
    monkeypatch.setattr("personal_assistant.media.voice.detect_language", lambda text: "ru")
    app = create_speech_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://voice"
    ) as client:
        assert (await client.post("/v1/audio/speech", json={"text": "Hi"})).status_code == 401
        client.headers["Authorization"] = "Bearer " + settings.api_token.get_secret_value()
        response = await client.post(
            "/v1/audio/transcriptions", files={"file": ("in.ogg", b"audio")}
        )
        assert response.json()["text"].startswith("Запомни")
        response = await client.post("/v1/audio/speech", json={"text": "Привет"})
        assert response.content == b"synthetic-ogg"
        assert response.headers["content-type"] == "audio/ogg"
    assert any("auto" in args for args, _ in calls)
    assert any(str(piper) in args and stdin == "Привет\n".encode() for args, stdin in calls)
