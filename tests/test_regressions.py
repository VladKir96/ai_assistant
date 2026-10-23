import pytest
from sqlalchemy import select

from personal_assistant.agent.context import ContextBuilder
from personal_assistant.agent.graph import Draft, authorize_mutation
from personal_assistant.app.demo import run_demo
from personal_assistant.core.config import Settings
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review, Summary
from personal_assistant.storage.models import Message, Run, Thread
from tests.conftest import ScriptedModel


def test_example_environment_parses_without_treating_empty_mcp_path_as_directory():
    settings = Settings(_env_file=".env.example")
    assert settings.mcp_servers_file is None
    assert settings.embedding_dimensions == 1024


@pytest.mark.parametrize(
    "action,text",
    [
        (Action(name="save_note", content="wrong"), "Show my notes"),
        (Action(name="create_task", title="wrong"), "Show my tasks"),
        (Action(name="remember", title="wrong", content="wrong"), "Don't remember this"),
    ],
)
def test_read_only_and_negative_intent_cannot_create_records(action, text):
    with pytest.raises(ValueError):
        authorize_mutation(action, text)


async def test_large_archive_message_is_fully_summarized_in_bounded_parts(harness):
    original = "Важное решение 🚀 " * 1000

    class Summarizer:
        def __init__(self):
            self.parts = []

        async def generate(self, system, payload, schema):
            assert schema is Summary
            self.parts.extend(item["content"] for item in payload["messages"])
            assert sum(len(item["content"].encode()) for item in payload["messages"]) <= 4000
            return Summary(summary="An important decision was discussed.")

    async with harness.database.sessions.begin() as session:
        session.add(Run(id="oversize", thread_id="oversize", input_text="old", status="completed"))
        await session.flush()
        session.add(
            Message(thread_id="oversize", run_id="oversize", role="assistant", content=original)
        )
    model = Summarizer()
    result = await ContextBuilder(harness.database, model).build("oversize")
    assert "".join(model.parts) == original
    assert result["recent_messages"] == []
    async with harness.database.sessions() as session:
        assert (await session.get(Thread, "oversize")).summary_cursor == (
            await session.scalar(select(Message.id))
        )


async def test_review_receives_the_entire_draft(harness):
    draft = "Supported content. " * 300 + "Important final constraint."
    model = ScriptedModel(
        Plan(specialist="research"), Draft(answer=draft), Review(decision="accept")
    )
    response = await harness.runtime(model).service.chat(ChatRequest(text="Review this research"))
    assert model.calls[-1][1]["draft"] == draft
    assert response.text == draft


async def test_demo_ignores_network_integrations_in_environment(monkeypatch, capsys):
    monkeypatch.setenv("ASSISTANT_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("ASSISTANT_EMBEDDING_BACKEND", "http")
    monkeypatch.setenv("ASSISTANT_MCP_SERVERS_FILE", "/does/not/exist")
    await run_demo()
    assert "PASS" in capsys.readouterr().out
