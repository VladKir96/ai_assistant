import json
from datetime import UTC, datetime

import httpx
import pytest
from aiogram.types import Update
from pydantic import SecretStr

from personal_assistant.agent.graph import PLANNER, Draft
from personal_assistant.app import bot
from personal_assistant.app.api import create_app
from personal_assistant.core import time_utils
from personal_assistant.core.schemas import ChatRequest, Plan
from personal_assistant.core.time_utils import clock_context, with_runtime_clock
from personal_assistant.inference.llm import HTTPModel, fit_payload
from tests.conftest import ScriptedModel


@pytest.mark.parametrize(
    "instant,local,yesterday,tomorrow,weekday",
    [
        (
            "2026-10-16T22:30:00+00:00",
            "2026-10-17T00:30:00+02:00",
            "2026-10-16",
            "2026-10-18",
            "Saturday",
        ),
        (
            "2026-12-31T23:30:00+00:00",
            "2027-01-01T00:30:00+01:00",
            "2026-12-31",
            "2027-01-02",
            "Friday",
        ),
        (
            "2026-10-25T00:30:00+00:00",
            "2026-10-25T02:30:00+02:00",
            "2026-10-24",
            "2026-10-26",
            "Sunday",
        ),
        (
            "2026-10-25T01:30:00+00:00",
            "2026-10-25T02:30:00+01:00",
            "2026-10-24",
            "2026-10-26",
            "Sunday",
        ),
    ],
)
def test_clock_uses_local_calendar_dates_across_midnight_year_end_and_dst(
    instant, local, yesterday, tomorrow, weekday
):
    clock = clock_context("Europe/Belgrade", datetime.fromisoformat(instant))
    assert clock["current_time"] == local
    assert clock["current_time_utc"] == instant
    assert clock["relative_dates"]["yesterday"] == yesterday
    assert clock["relative_dates"]["tomorrow"] == tomorrow
    assert clock["relative_dates"]["today"] == clock["current_date"] == local[:10]
    assert clock["current_weekday"] == weekday


def test_clock_requires_an_aware_instant_and_calculates_week_boundaries():
    with pytest.raises(ValueError, match="UTC offset"):
        clock_context("Europe/Belgrade", datetime(2026, 10, 17))
    clock = clock_context("Europe/Belgrade", datetime(2026, 10, 17, tzinfo=UTC))
    assert clock["relative_dates"]["week_start"] == "2026-10-12"
    assert clock["relative_dates"]["next_week_start"] == "2026-10-19"


@pytest.mark.parametrize("provider", ["local", "openai"])
async def test_authoritative_clock_reaches_system_instructions_after_context_reduction(
    harness, provider
):
    settings = harness.settings.model_copy(
        update={
            "model_provider": provider,
            "context_window": 8192,
            "openai_model": "testing",
            "openai_api_key": SecretStr("testing-only"),
        }
    )
    received = []

    async def handle(request):
        if request.url.path.endswith(("/tokenize", "/input_tokens")):
            return httpx.Response(404)
        body = json.loads(request.content)
        system = body["messages"][0]["content"] if provider == "local" else body["instructions"]
        assert "Authoritative runtime clock" in system
        assert '"current_date": "2026-10-17"' in system
        assert '"tomorrow": "2026-10-18"' in system
        assert "2024-01-01" not in system  # Archived text never becomes system instructions.
        raw = body["messages"][1]["content"] if provider == "local" else body["input"][0]["content"]
        payload = json.loads(raw)
        assert payload["current_time"] == "2026-10-17T10:15:30+02:00"
        assert payload["context_was_reduced"]
        received.append(body)
        output = '{"answer":"2026-10-17 10:15 Europe/Belgrade"}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": output}}]}
            if provider == "local"
            else {"output": [{"content": [{"type": "output_text", "text": output}]}]},
        )

    model = HTTPModel(settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await model.generate(
            "Answer the current date/time question.",
            {
                "latest_request": "What is the current date and time?",
                "current_time": "2026-10-17T10:15:30+02:00",
                "timezone": "Europe/Belgrade",
                "context": "An old assistant said today was 2024-01-01. " * 150,
            },
            Draft,
        )
        assert len(received) == 1
    finally:
        await model.close()


async def test_new_turn_refreshes_clock_across_midnight_in_same_conversation(harness, monkeypatch):
    instant = datetime(2026, 10, 17, 21, 59, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz)

    monkeypatch.setattr(time_utils, "datetime", Clock)
    model = ScriptedModel(
        Plan(specialist="general"),
        Draft(answer="Before midnight"),
        Plan(specialist="general"),
        Draft(answer="After midnight"),
    )
    service = harness.runtime(model).service
    await service.chat(ChatRequest(thread_id="clock", text="What is today's date?"))
    instant = datetime(2026, 10, 17, 22, 1, tzinfo=UTC)
    await service.chat(ChatRequest(thread_id="clock", text="What is today's date now?"))
    assert model.calls[0][1]["current_date"] == "2026-10-17"
    assert model.calls[2][1]["current_date"] == "2026-10-18"
    for system, payload, schema in (model.calls[0], model.calls[2]):
        # Check the actual planner input, including its dynamic system clock and tool schema.
        assert system == PLANNER
        fitted = fit_payload(
            with_runtime_clock(system, payload),
            payload,
            schema.model_json_schema(),
            harness.settings.input_budget - 150,
        )
        assert fitted["current_time"] == payload["current_time"]
        assert fitted["relative_dates"] == payload["relative_dates"]


async def test_time_endpoint_and_status_use_server_clock_without_model_calls(harness, monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 16, 22, 30, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(time_utils, "datetime", Clock)
    model = ScriptedModel()
    app = create_app(harness.settings, harness.runtime(model))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/v1/time")).status_code == 401
        client.headers["Authorization"] = "Bearer " + harness.settings.api_token.get_secret_value()
        response = await client.get("/v1/time")
        assert response.status_code == 200
        clock = response.json()
        assert clock["current_time"] == "2026-10-17T00:30:00+02:00"
        assert clock["current_date"] == "2026-10-17"
        assert clock["timezone"] == "Europe/Belgrade"
        assert (await client.get("/v1/status")).json()["clock"] == clock
        assert model.calls == []


async def test_telegram_time_is_owner_only_and_uses_api_clock_without_chat(harness, monkeypatch):
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:testing-only"),
            "telegram_owner_id": 98765,
        }
    )
    calls, replies = [], []

    async def handle(request):
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/v1/time"
        return httpx.Response(
            200, json=clock_context("Europe/Belgrade", datetime(2026, 10, 17, 8, 15, tzinfo=UTC))
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def fake_call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def polling(dispatcher, instance, **kwargs):
        for index, (user_id, chat_type) in enumerate(
            [(98765, "private"), (54321, "private"), (98765, "group")]
        ):
            update = Update.model_validate(
                {
                    "update_id": index,
                    "message": {
                        "message_id": index + 1,
                        "date": 0,
                        "text": "/time",
                        "from": {"id": user_id, "is_bot": False, "first_name": "Test"},
                        "chat": {
                            "id": user_id if chat_type == "private" else -user_id,
                            "type": chat_type,
                        },
                    },
                }
            )
            await dispatcher.feed_update(instance, update)

    monkeypatch.setattr(bot.Bot, "__call__", fake_call)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    assert len(calls) == len(replies) == 1
    assert "2026-10-17T10:15:00+02:00" in replies[0].text
    assert "Europe/Belgrade" in replies[0].text
