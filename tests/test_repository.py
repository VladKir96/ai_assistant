from datetime import timedelta

import pytest
from sqlalchemy import func, select

from personal_assistant.models import Revision, now
from personal_assistant.schemas import Action


async def test_note_revision_retrieval_export_and_deletion(harness):
    repository = harness.repository
    note = await repository.mutate(
        Action(
            name="save_note", title="Project architecture", content="Use Chroma", tags=["project"]
        ),
        "note:1",
        "test",
    )
    await repository.mutate(
        Action(name="update_note", resource_id=note["id"], content="Use PostgreSQL and pgvector"),
        "note:2",
        "test",
    )
    assert (await repository.search("pgvector"))[0]["id"] == note["id"]
    assert "PostgreSQL" in await repository.export_markdown()
    async with harness.database.sessions() as session:
        revision = await session.scalar(select(Revision))
        assert revision.snapshot["content"] == "Use Chroma"
    await repository.mutate(Action(name="delete_note", resource_id=note["id"]), "note:3", "test")
    assert await repository.search("pgvector") == []
    async with harness.database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Revision)) == 0


async def test_fact_correction_replaces_existing_record(harness):
    first = await harness.repository.mutate(
        Action(name="remember", title="preferred_language", content="English"), "fact:1", "user"
    )
    second = await harness.repository.mutate(
        Action(name="remember", title="preferred_language", content="Russian"), "fact:2", "user"
    )
    assert first["id"] == second["id"]
    assert len(await harness.repository.list_documents("fact")) == 1
    assert (await harness.repository.get_document(first["id"]))["content"] == "Russian"


async def test_invalid_reminder_time_does_not_create_task(harness):
    with pytest.raises(ValueError, match="timezone"):
        await harness.repository.mutate(
            Action(name="create_task", title="Reminder", due_at=now().replace(tzinfo=None)),
            "task:1",
            "user",
        )
    with pytest.raises(ValueError, match="future"):
        await harness.repository.mutate(
            Action(name="create_task", title="Reminder", due_at=now() - timedelta(hours=1)),
            "task:2",
            "user",
        )
    assert await harness.repository.tasks() == []
