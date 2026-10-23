from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from icalendar import Calendar
from sqlalchemy import func, select

from personal_assistant.agent.graph import PLANNER, Draft, authorize_mutation
from personal_assistant.app.api import create_app
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.core.time_utils import event_time
from personal_assistant.inference.llm import fit_payload
from personal_assistant.storage.models import CalendarEvent, Operation
from tests.conftest import ScriptedModel


def event_action(**changes):
    return Action(
        **{
            "name": "create_event",
            "title": "Встреча: проект",
            "content": "Discuss the assistant",
            "starts_at": "2030-11-10T10:00",
            "ends_at": "2030-11-10T11:00",
            "event_timezone": "Europe/Belgrade",
            **changes,
        }
    )


@pytest.mark.parametrize(
    "value,match",
    [
        ("2026-03-29T02:30", "does not exist"),
        ("2026-10-25T02:30", "occurs twice"),
        ("2026-07-01T10:00+01:00", "does not match"),
        ("2026-07-01", "requires both"),
    ],
)
def test_calendar_rejects_ambiguous_nonexistent_and_inconsistent_local_times(value, match):
    with pytest.raises(ValueError, match=match):
        event_time(value, "Europe/Belgrade")


def test_explicit_dst_offset_selects_the_requested_occurrence():
    first = event_time("2026-10-25T02:30+02:00", "Europe/Belgrade")
    second = event_time("2026-10-25T02:30+01:00", "Europe/Belgrade")
    assert first == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)
    assert second - first == timedelta(hours=1)


async def test_event_replay_move_cancel_and_ics_snapshot(harness):
    repository = harness.repository
    created = await repository.mutate(event_action(), "calendar:create", "test")
    assert await repository.mutate(event_action(), "calendar:create", "test") == created
    event = created["event"]
    assert event["starts_at"] == "2030-11-10T09:00:00+00:00"
    assert event["local_start"] == "2030-11-10T10:00:00+01:00"
    updated = await repository.mutate(
        Action(
            name="update_event",
            resource_id=event["id"],
            starts_at="2030-11-11T13:00",
            ends_at="2030-11-11T14:00",
        ),
        "calendar:move",
        "test",
    )
    assert updated["event"]["created_at"] == event["created_at"]
    assert updated["event"]["title"] == event["title"]
    assert updated["event"]["content"] == event["content"]
    assert updated["conflicts"] == []
    found = await repository.calendar.events("Встреча")
    assert len(found) == 1
    assert found[0]["local_start"] == "2030-11-11T13:00:00+01:00"
    exported = Calendar.from_ical(await repository.calendar.export_ics()).walk("VEVENT")
    assert len(exported) == 1
    assert str(exported[0]["summary"]) == event["title"]
    assert exported[0].decoded("dtstart") == datetime(2030, 11, 11, 12, tzinfo=UTC)
    await repository.mutate(
        Action(name="cancel_event", resource_id=event["id"]), "calendar:cancel", "test"
    )
    assert await repository.calendar.events("Встреча") == []
    assert (await repository.calendar.get(event["id"]))["status"] == "cancelled"
    assert len(await repository.calendar.events("Встреча", include_cancelled=True)) == 1
    assert Calendar.from_ical(await repository.calendar.export_ics()).walk("VEVENT") == []
    assert await repository.tasks() == []
    assert await repository.list_documents() == []


async def test_conflicts_are_reported_but_adjacent_and_cancelled_events_do_not_conflict(harness):
    first = await harness.repository.mutate(event_action(), "overlap:first", "test")
    second = await harness.repository.mutate(
        event_action(starts_at="2030-11-10T10:30", ends_at="2030-11-10T11:30"),
        "overlap:second",
        "test",
    )
    assert [event["id"] for event in second["conflicts"]] == [first["event"]["id"]]
    adjacent = await harness.repository.mutate(
        event_action(starts_at="2030-11-10T11:30", ends_at="2030-11-10T12:00"),
        "overlap:adjacent",
        "test",
    )
    assert adjacent["conflicts"] == []
    await harness.repository.mutate(
        Action(name="cancel_event", resource_id=second["event"]["id"]), "overlap:cancel", "test"
    )
    replacement = await harness.repository.mutate(
        event_action(starts_at="2030-11-10T11:00", ends_at="2030-11-10T11:30"),
        "overlap:replacement",
        "test",
    )
    assert replacement["conflicts"] == []


async def test_all_day_event_preserves_local_dates_over_dst_and_agenda_overlap(harness):
    created = await harness.repository.mutate(
        event_action(starts_at="2026-10-25", ends_at="2026-10-26", all_day=True),
        "calendar:all-day",
        "test",
    )
    event = created["event"]
    assert datetime.fromisoformat(event["ends_at"]) - datetime.fromisoformat(
        event["starts_at"]
    ) == timedelta(hours=25)
    agenda = await harness.repository.calendar.events(
        date_from=date(2026, 10, 25), date_to=date(2026, 10, 26)
    )
    assert [item["id"] for item in agenda] == [event["id"]]
    assert (
        await harness.repository.calendar.events(
            date_from=date(2026, 10, 26), date_to=date(2026, 10, 27)
        )
        == []
    )
    exported = Calendar.from_ical(await harness.repository.calendar.export_ics()).walk("VEVENT")[0]
    assert exported.decoded("dtstart") == date(2026, 10, 25)
    assert exported.decoded("dtend") == date(2026, 10, 26)


@pytest.mark.parametrize(
    "changes",
    [
        {"ends_at": "2030-11-10T09:00"},
        {"ends_at": ""},
        {"event_timezone": "Invalid/Timezone"},
        {"all_day": True},
    ],
)
async def test_invalid_event_does_not_leave_an_event_or_operation(harness, changes):
    with pytest.raises(ValueError):
        await harness.repository.mutate(event_action(**changes), "invalid:event", "test")
    async with harness.database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(CalendarEvent)) == 0
        assert await session.scalar(select(func.count()).select_from(Operation)) == 0


@pytest.mark.parametrize("text", ["Show my agenda", "Не добавляй встречу", "Add a weekly meeting"])
def test_calendar_write_requires_intent_and_does_not_drop_recurrence(text):
    with pytest.raises(ValueError):
        authorize_mutation(event_action(), text)


async def test_calendar_graph_forces_review_and_revision_never_repeats_write(harness):
    model = ScriptedModel(
        Plan(specialist="productivity", actions=[event_action()]),
        Draft(answer="Saved this in your Google Calendar."),
        Review(decision="revise", issues=["Only the private calendar was updated"]),
        Plan(specialist="productivity", actions=[event_action()]),
        Draft(answer="Saved in your private calendar at 10:00 Europe/Belgrade."),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Create a calendar event on 10 November 2030 from 10:00 to 11:00.")
    )
    assert result.review_status == "accepted"
    async with harness.database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(CalendarEvent)) == 1
        assert await session.scalar(select(func.count()).select_from(Operation)) == 1


async def test_overlap_notice_is_present_even_when_model_omits_it(harness):
    first = await harness.repository.mutate(event_action(), "notice:first", "test")
    model = ScriptedModel(
        Plan(specialist="productivity", actions=[event_action()]),
        Draft(answer="Your event is saved."),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Create another event on 10 November 2030 from 10:00 to 11:00.")
    )
    assert "Calendar overlap" in result.text
    assert first["event"]["id"] in result.text


@pytest.mark.parametrize("is_owner", [False, True])
async def test_bot_agenda_and_calendar_export_remain_private(harness, monkeypatch, is_owner):
    from aiogram.types import Update
    from pydantic import SecretStr

    from personal_assistant.app import bot

    event = (await harness.repository.mutate(event_action(), "bot:event", "test"))["event"]
    exported = await harness.repository.calendar.export_ics()
    settings = harness.settings.model_copy(
        update={"telegram_token": SecretStr("12345:testing-only"), "telegram_owner_id": 98765}
    )
    replies, requests = [], []

    async def transport(request):
        requests.append(request.url.path)
        if request.url.path == "/v1/events":
            return httpx.Response(200, json=[event])
        assert request.url.path == "/v1/calendar.ics"
        return httpx.Response(200, content=exported)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(transport), **kwargs),
    )

    async def fake_call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def polling(dispatcher, instance, **kwargs):
        sender = 98765 if is_owner else 77777
        for index, command in enumerate(("/agenda", "/calendar")):
            update = Update.model_validate(
                {
                    "update_id": index,
                    "message": {
                        "message_id": index + 1,
                        "date": 0,
                        "from": {"id": sender, "is_bot": False, "first_name": "Test"},
                        "chat": {"id": sender, "type": "private"},
                        "text": command,
                    },
                }
            )
            await dispatcher.feed_update(instance, update)

    monkeypatch.setattr(bot.Bot, "__call__", fake_call)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    if is_owner:
        assert requests == ["/v1/events", "/v1/calendar.ics"]
        assert event["local_start"] in replies[0].text
        assert replies[1].document.filename == "calendar.ics"
        assert replies[1].document.data == exported
    else:
        assert requests == []
        assert replies == []


async def test_calendar_api_requires_auth_and_exports_ics(harness):
    event = (await harness.repository.mutate(event_action(), "api:event", "test"))["event"]
    app = create_app(harness.settings, harness.runtime(ScriptedModel()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/v1/events")).status_code == 401
        assert (await client.get("/v1/calendar.ics")).status_code == 401
        client.headers["Authorization"] = "Bearer " + harness.settings.api_token.get_secret_value()
        agenda = await client.get("/v1/events?date_from=2030-11-10&date_to=2030-11-11")
        assert [item["id"] for item in agenda.json()] == [event["id"]]
        assert (await client.get(f"/v1/events/{event['id']}")).status_code == 200
        assert (await client.get("/v1/events/missing")).status_code == 404
        exported = await client.get("/v1/calendar.ics")
        assert exported.headers["content-type"].startswith("text/calendar")
        assert len(Calendar.from_ical(exported.content).walk("VEVENT")) == 1


def test_calendar_schema_and_policy_fit_the_default_input_budget(harness):
    fit_payload(
        PLANNER,
        {"latest_request": "Create an event tomorrow", "calendar_events": []},
        Plan.model_json_schema(),
        harness.settings.input_budget - 150,
    )
