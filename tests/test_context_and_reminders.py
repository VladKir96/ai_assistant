from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import select

from personal_assistant.bot import permitted, split_text
from personal_assistant.context import ContextBuilder
from personal_assistant.jobs import claim_due
from personal_assistant.models import Message, Run, Task, Thread, now
from personal_assistant.schemas import Summary


async def test_compaction_preserves_archive_and_persists_summary_cursor(harness):
    class Summarizer:
        def __init__(self):
            self.seen = []

        async def generate(self, system, payload, schema):
            assert schema is Summary
            self.seen.extend(item["content"] for item in payload["messages"])
            return Summary(
                summary="Decision: keep deadline 2026-11-01. Pending: select job sources."
            )

    async with harness.database.sessions.begin() as session:
        run_id = str(uuid4())
        session.add(Run(id=run_id, thread_id="long", input_text="test", status="completed"))
        await session.flush()
        for index in range(20):
            session.add(
                Message(
                    thread_id="long",
                    run_id=run_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"Message {index}: deadline 2026-11-01. " + "Useful information. " * 10,
                )
            )
    model = Summarizer()
    builder = ContextBuilder(harness.database, model)
    context = await builder.build("long")
    assert len(context["recent_messages"]) <= 6
    assert "2026-11-01" in context["summary"]
    assert len(model.seen) + len(context["recent_messages"]) == 20
    async with harness.database.sessions() as session:
        assert len((await session.scalars(select(Message))).all()) == 20
        assert (await session.get(Thread, "long")).summary_cursor > 0
    seen = len(model.seen)
    assert await builder.build("long") == context
    assert len(model.seen) == seen
    archived = await harness.repository.search_history("deadline")
    assert archived and "2026-11-01" in archived[0]["content"]


async def test_due_reminders_are_claimed_once_and_completed_tasks_are_excluded(harness):
    async with harness.database.sessions.begin() as session:
        for identifier, state in (("open-task", "open"), ("completed-task", "completed")):
            session.add(
                Task(
                    id=identifier,
                    title=identifier,
                    status=state,
                    due_at=now() - timedelta(minutes=1),
                )
            )
    claimed = await claim_due(harness.database)
    assert [item["id"] for item in claimed] == ["open-task"]
    assert await claim_due(harness.database) == []
    async with harness.database.sessions.begin() as session:
        task = await session.get(Task, "open-task")
        task.notification_claimed_at = now() - timedelta(minutes=4)
    assert [item["id"] for item in await claim_due(harness.database)] == ["open-task"]


def test_telegram_rejects_nonowner_and_groups():
    owner = SimpleNamespace(
        from_user=SimpleNamespace(id=123), chat=SimpleNamespace(id=123, type="private")
    )
    assert permitted(owner, 123)
    assert not permitted(owner, 456)
    assert not permitted(owner, 0)
    owner.chat.type = "group"
    assert not permitted(owner, 123)


def test_telegram_splits_utf16_without_losing_text():
    original = "Русский и English 🚀" * 1000
    parts = split_text(original)
    assert "".join(parts) == original
    assert all(len(part.encode("utf-16-le")) // 2 <= 3500 for part in parts)
