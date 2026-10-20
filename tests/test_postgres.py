import os
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url

from personal_assistant.config import Settings
from personal_assistant.context import ContextBuilder
from personal_assistant.database import Database
from personal_assistant.graph import AssistantGraph, Draft
from personal_assistant.llm import Embeddings, ModelError
from personal_assistant.mcp_tools import MCPTools
from personal_assistant.models import CalendarEvent, Document, Message, Operation, Run
from personal_assistant.repository import Repository
from personal_assistant.research import Research
from personal_assistant.schemas import Action, ChatRequest, Plan
from personal_assistant.service import AssistantService
from tests.conftest import ScriptedModel


async def test_postgres_reindex_compare_and_swap_with_concurrent_edit():
    from tests.test_review_fixes import exercise_reindex_race

    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL to an isolated migrated PostgreSQL database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("Integration tests require a database name ending in _test")
    settings = Settings(_env_file=None, database_url=url, model_provider="local")
    database, identifier = Database(url), str(uuid4())
    try:
        await exercise_reindex_race(database, settings, identifier)
    finally:
        async with database.sessions.begin() as session:
            await session.execute(delete(Document).where(Document.id == identifier))
            await session.execute(delete(Operation).where(Operation.id == f"{identifier}:edit"))
        await database.close()


async def test_postgres_concurrent_reminder_claims_do_not_overlap():
    import asyncio
    from datetime import timedelta

    from personal_assistant.jobs import claim_due
    from personal_assistant.models import Task, now

    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL to an isolated migrated PostgreSQL database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("Integration tests require a database name ending in _test")
    database, identifiers = Database(url), [str(uuid4()), str(uuid4())]
    try:
        async with database.sessions.begin() as session:
            session.add_all(
                [
                    Task(id=value, title="Synthetic reminder", due_at=now() - timedelta(minutes=1))
                    for value in identifiers
                ]
            )
        first, second = await asyncio.gather(claim_due(database), claim_due(database))
        first_ids, second_ids = {item["id"] for item in first}, {item["id"] for item in second}
        assert not first_ids & second_ids
        assert set(identifiers) <= first_ids | second_ids
    finally:
        async with database.sessions.begin() as session:
            await session.execute(delete(Task).where(Task.id.in_(identifiers)))
        await database.close()


async def test_postgres_reminder_exhaustion_preserves_live_claims_and_recovers_expired_ones():
    from datetime import timedelta

    from personal_assistant.jobs import claim_due
    from personal_assistant.models import Task, now

    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL to an isolated migrated PostgreSQL database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("Integration tests require a database name ending in _test")
    database = Database(url)
    expired, live, retryable = [str(uuid4()) for _ in range(3)]
    instant = now()
    try:
        async with database.sessions.begin() as session:
            for identifier, attempts, age in ((expired, 5, 4), (live, 5, 1), (retryable, 4, 4)):
                session.add(
                    Task(
                        id=identifier,
                        title="Synthetic recovery test",
                        due_at=instant - timedelta(minutes=5),
                        notification_status="sending",
                        notification_attempts=attempts,
                        notification_claimed_at=instant - timedelta(minutes=age),
                    )
                )
        claimed = {item["id"] for item in await claim_due(database)}
        assert retryable in claimed and expired not in claimed and live not in claimed
        async with database.sessions() as session:
            assert (await session.get(Task, expired)).notification_status == "failed"
            assert (await session.get(Task, live)).notification_status == "sending"
            retry = await session.get(Task, retryable)
            assert retry.notification_status == "sending" and retry.notification_attempts == 5
    finally:
        async with database.sessions.begin() as session:
            await session.execute(delete(Task).where(Task.id.in_([expired, live, retryable])))
        await database.close()


async def test_postgres_vectors_and_checkpoint_recovery():
    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL to an isolated migrated PostgreSQL database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("Integration tests require a database name ending in _test")
    settings = Settings(_env_file=None, database_url=url)
    database = Database(url)
    identifier = uuid4().hex

    class SyntheticEmbeddings:
        settings = Settings(_env_file=None, embedding_model="synthetic-test-vectors")

        async def embed(self, content):
            position = 0 if "Python" in content or "питон" in content else 1
            return [1.0 if index == position else 0.0 for index in range(1024)]

    repository = Repository(database, SyntheticEmbeddings())
    request = ChatRequest(text="Save a note: durable Python assistant", thread_id="integration")
    document_ids = []
    checkpoint_url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        for index, title in enumerate(("Python engineering", "Rust systems")):
            document = await repository.mutate(
                Action(name="save_note", title=title, content=title),
                f"integration:{identifier}:{index}",
                "integration",
            )
            document_ids.append(document["id"])
        # There is no lexical match for the Russian query: this exercises pgvector ranking.
        assert (await repository.search("питон", limit=1))[0]["title"] == "Python engineering"

        first = ScriptedModel(
            Plan(
                specialist="knowledge",
                actions=[
                    Action(
                        name="save_note",
                        title="Checkpoint Python",
                        content="Persisted before failure",
                    )
                ],
            ),
            ModelError("simulated inference outage"),
        )

        def service(model, saver):
            graph = AssistantGraph(
                settings,
                model,
                repository,
                ContextBuilder(database, model),
                Research(settings),
                MCPTools(settings),
            ).compile(saver)
            return AssistantService(database, graph, settings.input_budget)

        async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as saver:
            with pytest.raises(ModelError):
                await service(first, saver).chat(request)
        second = ScriptedModel(Draft(answer="The note is saved."))
        async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as saver:
            result = await service(second, saver).chat(request)
            assert result.text == "The note is saved."
            assert len(second.calls) == 1
            await saver.adelete_thread(str(request.request_id))
        from uuid import NAMESPACE_URL, uuid5

        document_ids.append(str(uuid5(NAMESPACE_URL, f"{request.request_id}:0")))
        async with database.sessions() as session:
            assert (
                len(
                    (
                        await session.scalars(
                            select(Document).where(Document.id == document_ids[-1])
                        )
                    ).all()
                )
                == 1
            )
    finally:
        async with database.sessions.begin() as session:
            await session.execute(delete(Document).where(Document.id.in_(document_ids)))
            await session.execute(delete(Message).where(Message.run_id == str(request.request_id)))
            await session.execute(delete(Run).where(Run.id == str(request.request_id)))
        await database.close()


async def test_postgres_calendar_and_local_date_filtering():
    from datetime import UTC, date, datetime

    url = os.getenv("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set TEST_POSTGRES_URL to an isolated migrated PostgreSQL database")
    if not (make_url(url).database or "").endswith("_test"):
        pytest.fail("Integration tests require a database name ending in _test")
    settings = Settings(_env_file=None, database_url=url, embedding_backend="disabled")
    database = Database(url)
    embeddings = Embeddings(settings)
    repository = Repository(database, embeddings)
    identifier = uuid4().hex
    operation_id = f"calendar-integration:{identifier}"
    event_id = None
    try:
        created = await repository.mutate(
            Action(
                name="create_event",
                title=f"Calendar integration {identifier}",
                starts_at="2026-10-25",
                ends_at="2026-10-26",
                all_day=True,
                event_timezone="Europe/Belgrade",
            ),
            operation_id,
            "integration",
        )
        event_id = created["event"]["id"]
        loaded = await repository.calendar.get(event_id)
        assert loaded["starts_at"] == "2026-10-24T22:00:00+00:00"
        assert loaded["ends_at"] == "2026-10-25T23:00:00+00:00"
        async with database.sessions.begin() as session:
            session.add(
                Document(
                    id=identifier,
                    kind="note",
                    title=f"Date integration {identifier}",
                    content="Synthetic calendar date check",
                    created_at=datetime(2026, 10, 25, 22, 59, tzinfo=UTC),
                    updated_at=datetime(2026, 10, 26, tzinfo=UTC),
                )
            )
        notes = await repository.search(
            identifier, date_from=date(2026, 10, 25), date_to=date(2026, 10, 26)
        )
        assert [note["id"] for note in notes] == [identifier]
        assert notes[0]["created_at"].endswith("+00:00")
    finally:
        async with database.sessions.begin() as session:
            await session.execute(delete(CalendarEvent).where(CalendarEvent.id == event_id))
            await session.execute(delete(Operation).where(Operation.id == operation_id))
            await session.execute(delete(Document).where(Document.id == identifier))
        await embeddings.close()
        await database.close()
