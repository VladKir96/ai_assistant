import json
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest

from personal_assistant.api import create_app
from personal_assistant.graph import Draft
from personal_assistant.models import Document, Message, Run
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.time_utils import date_bounds
from tests.conftest import ScriptedModel


async def seed_dates(harness):
    # On this DST transition date, the Belgrade calendar day lasts 25 hours.
    timestamps = [
        datetime(2026, 10, 24, 21, 59, 59, tzinfo=UTC),
        datetime(2026, 10, 24, 22, tzinfo=UTC),
        datetime(2026, 10, 25, 22, 59, 59, tzinfo=UTC),
        datetime(2026, 10, 25, 23, tzinfo=UTC),
    ]
    async with harness.database.sessions.begin() as session:
        for index, timestamp in enumerate(timestamps):
            session.add(
                Document(
                    id=f"date-note-{index}",
                    kind="note",
                    title=f"Matched note {index}",
                    content=f"Synthetic content {index}",
                    created_at=timestamp,
                    updated_at=timestamp + timedelta(days=2),
                    embedding=[1.0] * 1024,
                    embedding_model="date-test",
                )
            )
    return timestamps


def test_local_day_bounds_respect_dst():
    start, end = date_bounds(date(2026, 10, 25), date(2026, 10, 26), ZoneInfo("Europe/Belgrade"))
    assert start == datetime(2026, 10, 24, 22, tzinfo=UTC)
    assert end == datetime(2026, 10, 25, 23, tzinfo=UTC)
    assert end - start == timedelta(hours=25)


async def test_created_and_edited_times_are_preserved_through_retrieval_and_export(harness):
    repository = harness.repository
    original = await repository.mutate(
        Action(name="save_note", title="Timestamp test", content="Original"), "dates:save", "test"
    )
    updated = await repository.mutate(
        Action(name="update_note", resource_id=original["id"], content="Edited"),
        "dates:edit",
        "test",
    )
    assert updated["created_at"] == original["created_at"]
    assert datetime.fromisoformat(updated["updated_at"]) > datetime.fromisoformat(
        original["updated_at"]
    )
    assert (await repository.get_document(original["id"]))["created_at"] == original["created_at"]
    assert (await repository.search("Timestamp"))[0]["created_at"] == original["created_at"]
    assert (await repository.revisions(original["id"]))[0]["snapshot"]["created_at"] == original[
        "created_at"
    ]
    exported = await repository.export_markdown()
    assert f"Created (UTC): {original['created_at']}" in exported
    assert f"Updated (UTC): {updated['updated_at']}" in exported


@pytest.mark.parametrize("semantic", [False, True])
async def test_date_filters_apply_to_keyword_and_vector_candidates(harness, semantic):
    await seed_dates(harness)
    if semantic:
        harness.settings.embedding_model = "date-test"

        async def embed(query):
            assert query.strip()
            return [1.0] * 1024

        harness.repository.embeddings.embed = embed
    query = "unmatched русская фраза" if semantic else "Matched"
    results = await harness.repository.search(
        query, date_from=date(2026, 10, 25), date_to=date(2026, 10, 26)
    )
    assert {item["id"] for item in results} == {"date-note-1", "date-note-2"}
    assert all(item["created_at"].endswith("+00:00") for item in results)
    dates_only = await harness.repository.search(
        "", date_from=date(2026, 10, 25), date_to=date(2026, 10, 26)
    )
    assert {item["id"] for item in dates_only} == {"date-note-1", "date-note-2"}
    edited = await harness.repository.search(
        "", date_from=date(2026, 10, 25), date_to=date(2026, 10, 26), date_field="updated_at"
    )
    assert edited == []


async def test_date_only_lookup_reaches_agent_with_timestamped_results(harness):
    await seed_dates(harness)
    model = ScriptedModel(
        Plan(
            specialist="knowledge",
            actions=[
                Action(
                    name="search_knowledge",
                    date_from=date(2026, 10, 25),
                    date_to=date(2026, 10, 26),
                )
            ],
        ),
        Draft(answer="Two notes were created on that date."),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Show notes created on 25 October 2026.")
    )
    assert result.review_status == "accepted"
    documents = json.loads(model.calls[1][1]["operation_results"][0]["data"])["documents"]
    assert {item["id"] for item in documents} == {"date-note-1", "date-note-2"}
    assert all("created_at" in item and "updated_at" in item for item in documents)


async def test_history_date_filter_uses_message_creation_time(harness):
    timestamps = await seed_dates(harness)
    async with harness.database.sessions.begin() as session:
        session.add(
            Run(id="dated-history", thread_id="dates", input_text="test", status="completed")
        )
        await session.flush()
        for index, timestamp in enumerate(timestamps):
            session.add(
                Message(
                    thread_id="dates",
                    run_id="dated-history",
                    role="user",
                    content=f"Message {index}",
                    created_at=timestamp,
                )
            )
    history = await harness.repository.search_history(
        "", date_from=date(2026, 10, 25), date_to=date(2026, 10, 26)
    )
    assert {item["content"] for item in history} == {"Message 1", "Message 2"}
    assert all(item["created_at"].endswith("+00:00") for item in history)


async def test_api_validates_date_ranges_and_exposes_both_timestamps(harness):
    await seed_dates(harness)
    app = create_app(harness.settings, harness.runtime(ScriptedModel()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
        ) as client,
    ):
        response = await client.get(
            "/v1/notes", params={"date_from": "2026-10-25", "date_to": "2026-10-26"}
        )
        assert response.status_code == 200
        assert {item["id"] for item in response.json()} == {"date-note-1", "date-note-2"}
        assert all("created_at" in item and "updated_at" in item for item in response.json())
        response = await client.get(
            "/v1/notes", params={"date_from": "2026-10-26", "date_to": "2026-10-25"}
        )
        assert response.status_code == 422
        assert (await client.get("/v1/notes?date_field=content")).status_code == 422


async def test_review_and_revision_share_the_requests_clock_and_calendar_limitations(harness):
    model = ScriptedModel(
        Plan(specialist="productivity", needs_review=True),
        Draft(answer="An external calendar entry was created."),
        Review(decision="revise", issues=["External calendar synchronization is unavailable"]),
        Plan(specialist="productivity"),
        Draft(answer="I can manage a private calendar; external calendars are not connected."),
        Review(decision="accept"),
    )
    await harness.runtime(model).service.chat(ChatRequest(text="What can you schedule?"))
    reference = model.calls[0][1]["current_time"]
    assert datetime.fromisoformat(reference).tzinfo is not None
    for _, payload, _ in model.calls:
        assert payload["current_time"] == reference
        assert payload["timezone"] == "Europe/Belgrade"
        assert payload["time_capabilities"]["tasks_and_reminders"]
        assert payload["time_capabilities"]["calendar_events"]
        assert not payload["time_capabilities"]["external_calendar"]
