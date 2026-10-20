import asyncio
import copy
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from starlette.datastructures import UploadFile

from personal_assistant.authorization import authorize_mutation
from personal_assistant.context import bounded_context
from personal_assistant.graph import PLANNER, AssistantGraph, Draft
from personal_assistant.http_guard import PrivateBodyGuard
from personal_assistant.llm import ContextLimitError, HTTPModel, ModelError
from personal_assistant.mcp_tools import MCPTools, ServerConfig
from personal_assistant.models import Document
from personal_assistant.repository import Repository
from personal_assistant.schemas import Action, ChatRequest, Plan
from personal_assistant.time_utils import clock_context
from personal_assistant.tool_execution import ToolExecutor
from personal_assistant.tools import TOOLS, TOOLS_BY_ROLE
from personal_assistant.voice import create_speech_app
from tests.conftest import ScriptedModel


@pytest.mark.parametrize(
    "template",
    [
        "Never delete note {}.",
        "Не надо удалять заметку {}.",
        "Explain what delete means for note {}.",
        "'Delete note {}'",
        "If needed, delete note {}.",
    ],
)
async def test_wrong_planner_cannot_delete_on_negation_or_quoted_discussion(harness, template):
    note = await harness.repository.mutate(
        Action(name="save_note", title="Keep", content="Protected"), "seed", "test"
    )
    model = ScriptedModel(
        Plan(specialist="knowledge", actions=[Action(name="delete_note", resource_id=note["id"])]),
        Draft(answer="No deletion."),
    )
    response = await harness.runtime(model).service.chat(
        ChatRequest(text=template.format(note["id"]))
    )
    assert (await harness.repository.get_document(note["id"]))["content"] == "Protected"
    assert not response.operations[0]["ok"]
    assert "/delete_note" in response.text


@pytest.mark.parametrize(
    "name,command",
    [
        ("delete_note", "/delete_note"),
        ("delete_note", "Please delete note"),
        ("delete_note", "Удали заметку с ID"),
        ("forget", "/forget"),
        ("cancel_event", "/cancel_event"),
    ],
)
def test_destructive_authorization_is_bound_to_exact_command_and_target(name, command):
    identifier = "11111111-1111-1111-1111-111111111111"
    action = Action(name=name, resource_id=identifier)
    authorize_mutation(action, f"{command} {identifier}")
    for text in (
        f"Never {command} {identifier}",
        f"{command} {identifier} is an example",
        f"{command} 22222222-2222-2222-2222-222222222222",
    ):
        with pytest.raises(ValueError):
            authorize_mutation(action, text)


async def test_explicit_delete_command_executes_once(harness):
    note = await harness.repository.mutate(
        Action(name="save_note", content="Delete this one"), "seed", "test"
    )
    model = ScriptedModel(
        Plan(specialist="knowledge", actions=[Action(name="delete_note", resource_id=note["id"])]),
        Draft(answer="Deleted."),
    )
    service = harness.runtime(model).service
    request = ChatRequest(text=f"/delete_note {note['id']}")
    first = await service.chat(request)
    assert (await service.chat(request)) == first
    assert first.operations[0]["ok"]
    with pytest.raises(ValueError):
        await harness.repository.get_document(note["id"])


def test_structured_context_prioritizes_recent_turns_over_russian_summary():
    context = {
        "summary": "я" * 1800,
        "recent_messages": [
            {"role": "user", "content": "LATEST_VACANCY_MARKER"},
            {"role": "assistant", "content": "Salary: 1000 EUR"},
        ],
    }
    original = copy.deepcopy(context)
    result = bounded_context(context, 2200)
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 2200
    assert result["recent_messages"] == context["recent_messages"]
    assert len(result["summary"]) < 1800
    assert result["context_was_reduced"]
    assert context == original


def test_large_recent_message_clipping_remains_valid_json_with_escaping():
    result = bounded_context(
        {"summary": "old", "recent_messages": [{"role": "user", "content": '\\"\n🚀' * 4000}]}, 2200
    )
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 2200
    assert result["recent_messages"][0]["truncated"]


async def test_lexical_ranking_keeps_older_exact_note_after_40_newer_matches(harness):
    instant = datetime(2026, 10, 19, tzinfo=UTC)
    async with harness.database.sessions.begin() as session:
        session.add(
            Document(
                id="expected",
                kind="note",
                title="Project Zephyr",
                content="Exact application details",
                updated_at=instant - timedelta(days=50),
            )
        )
        for index in range(40):
            session.add(
                Document(
                    id=f"new-{index}",
                    kind="note",
                    title=f"Project {index}",
                    content="General archive",
                    updated_at=instant,
                )
            )
    assert (await harness.repository.search("Project Zephyr", limit=1))[0]["id"] == "expected"


async def exercise_reindex_race(database, settings, identifier):
    started, resume = asyncio.Event(), asyncio.Event()
    old, new, previous = (
        [1.0] + [0.0] * 1023,
        [0.0, 1.0] + [0.0] * 1022,
        [0.0, 0.0, 1.0] + [0.0] * 1021,
    )

    class Encoder:
        def __init__(self, pause):
            self.settings = settings.model_copy(
                update={"embedding_backend": "http", "embedding_model": "new-model"}
            )
            self.pause = pause

        async def embed(self, text):
            if self.pause and f"Race {identifier}" in text:
                self.pause = False
                started.set()
                await resume.wait()
            return new if "NEW_CONTENT" in text else old

    async with database.sessions.begin() as session:
        session.add(
            Document(
                id=identifier,
                kind="note",
                title=f"Race {identifier}",
                content="OLD_CONTENT",
                embedding=previous,
                embedding_model="previous-model",
            )
        )
    task = asyncio.create_task(Repository(database, Encoder(True)).reindex())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await Repository(database, Encoder(False)).mutate(
            Action(name="update_note", resource_id=identifier, content="NEW_CONTENT"),
            f"{identifier}:edit",
            "test",
        )
        resume.set()
        await task
        async with database.sessions() as session:
            document = await session.get(Document, identifier)
            assert document.content == "NEW_CONTENT"
            assert list(document.embedding) == new
            assert document.embedding_model == "new-model"
    finally:
        resume.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def test_reindex_does_not_replace_new_vector_with_old_content(harness):
    await exercise_reindex_race(harness.database, harness.settings, "race")


async def test_interrupted_reindex_keeps_completed_batches_and_can_resume(harness):
    class Encoder:
        settings = harness.settings.model_copy(
            update={"embedding_backend": "http", "embedding_model": "new"}
        )
        calls, fail = 0, True

        async def embed(self, text):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise ModelError("Synthetic interruption")
            return [1.0] + [0.0] * 1023

    async with harness.database.sessions.begin() as session:
        session.add_all(
            [
                Document(
                    id=value,
                    kind="note",
                    title=value,
                    content="Keep original content",
                    embedding_model="old",
                )
                for value in ("a", "b")
            ]
        )
    encoder = Encoder()
    repository = Repository(harness.database, encoder)
    with pytest.raises(ModelError):
        await repository.reindex()
    async with harness.database.sessions() as session:
        assert (await session.get(Document, "a")).embedding_model == "new"
        assert (await session.get(Document, "b")).embedding_model == "old"
    encoder.fail = False
    assert await repository.reindex() == 2
    assert (await repository.get_document("b"))["content"] == "Keep original content"


async def test_unsupported_counter_uses_cached_conservative_fallback(harness):
    calls = []

    async def handle(request):
        calls.append(request)
        assert request.url.path == "/tokenize"
        return httpx.Response(404)

    model = HTTPModel(harness.settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        for _ in range(2):
            with pytest.raises(ContextLimitError):
                await model.fit_request("System", {"latest_request": "x" * 20000}, Draft)
        assert len(calls) == 1
    finally:
        await model.close()


async def test_unauthorized_upload_never_reaches_multipart_parser(harness, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Unauthorized body reached multipart file handling")

    monkeypatch.setattr(UploadFile, "write", forbidden)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_speech_app(harness.settings)),
        base_url="http://speech",
    ) as client:
        assert (
            await client.post("/v1/audio/transcriptions", files={"file": b"x" * 2_100_000})
        ).status_code == 401
        client.headers["Authorization"] = "Bearer " + harness.settings.api_token.get_secret_value()
        assert (
            await client.post("/v1/audio/transcriptions", files={"file": b"x" * 10_100_000})
        ).status_code == 413


@pytest.mark.parametrize("declared", [None, b"1"])
async def test_streamed_body_limit_handles_missing_or_false_content_length(declared):
    parsed, sent = [], []

    async def app(*args):
        parsed.append(True)

    chunks = iter(
        [
            {"type": "http.request", "body": b"x" * 80, "more_body": True},
            {"type": "http.request", "body": b"x" * 80, "more_body": False},
        ]
    )

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    headers = [(b"authorization", b"Bearer testing")]
    if declared:
        headers.append((b"content-length", declared))
    await PrivateBodyGuard(app, "testing", 100)(
        {"type": "http", "path": "/v1/audio/transcriptions", "headers": headers}, receive, send
    )
    assert not parsed
    assert sent[0]["status"] == 413


@pytest.mark.parametrize(
    "provider,window,text,catalog_size",
    [
        ("local", 8192, "Hello", 0),
        ("local", 16384, "A" * 2000, 0),
        ("local", 16384, "Hello", 2),
        ("openai", 8192, "Hello", 2),
    ],
)
async def test_complete_planner_uses_configured_provider_counter(
    harness, provider, window, text, catalog_size
):
    from pydantic import SecretStr

    settings = harness.settings.model_copy(
        update={
            "context_window": window,
            "model_provider": provider,
            "openai_model": "testing",
            "openai_api_key": SecretStr("test-only"),
        }
    )
    graph = object.__new__(AssistantGraph)
    graph.settings = settings
    payload = graph.payload(
        {
            "text": text,
            "current_time": clock_context(settings.timezone)["current_time"],
            "context": {"summary": "", "recent_messages": []},
            "knowledge": [],
            "tasks": [],
            "events": [],
        }
    )
    payload["mcp_read_tools"] = [
        {
            "server": "fixture",
            "tool": f"read_{index}",
            "description": "D" * 300,
            "arguments_schema": {"type": "object"},
        }
        for index in range(catalog_size)
    ]
    original = copy.deepcopy(payload)
    counted, generated = [], []

    async def handle(request):
        body = json.loads(request.content)
        if request.url.path.endswith(("/tokenize", "/input_tokens")):
            counted.append(body)
            if provider == "local":
                assert request.url.host == "localhost"
                return httpx.Response(200, json={"tokens": list(range(3000))})
            assert body["model"] == "testing" and "text" in body
            return httpx.Response(200, json={"input_tokens": 3000})
        generated.append(body)
        actual = json.loads(
            body["messages"][1]["content"] if provider == "local" else body["input"][0]["content"]
        )
        assert actual == original
        output = '{"specialist":"general"}'
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
        await model.generate(PLANNER, payload, Plan)
        assert counted and len(generated) == 1
        assert payload == original
    finally:
        await model.close()


@pytest.mark.parametrize(
    "data",
    [{"choices": []}, {"choices": [None]}, {"choices": [{"message": {"content": "not-json"}}]}],
)
async def test_malformed_model_responses_are_sanitized_after_retry(harness, data):
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(200, json=data)

    model = HTTPModel(harness.settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with pytest.raises(ModelError, match="invalid structured output"):
            await model.generate("Answer", {"latest_request": "Hello"}, Draft)
        assert len(calls) == 2
    finally:
        await model.close()


async def test_embedding_outage_preserves_keyword_reads_and_note_writes(harness):
    class Unavailable:
        settings = harness.settings

        async def embed(self, text):
            raise ModelError("synthetic outage")

    harness.repository.embeddings = Unavailable()
    note = await harness.repository.mutate(
        Action(name="save_note", title="Useful", content="Keyword fallback"), "seed", "test"
    )
    assert "integration_warning" in note
    assert (await harness.repository.search("Keyword"))[0]["id"] == note["id"]
    model = ScriptedModel(Plan(specialist="general"), Draft(answer="Hello."))
    result = await harness.runtime(model).service.chat(ChatRequest(text="Hello"))
    assert "keyword search" in result.text


async def test_mcp_outage_isolated_and_catalog_cached(harness, monkeypatch):
    tools = MCPTools(harness.settings)
    tools.servers = {
        name: ServerConfig(command="fixture", read_tools=["read_label"]) for name in ("good", "bad")
    }
    calls = []

    @asynccontextmanager
    async def connect(name):
        calls.append(name)
        if name == "bad":
            raise ExceptionGroup("private-secret", [OSError("private-secret")])

        async def list_tools():
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="read_label", description="Read", inputSchema={"type": "object"}
                    )
                ]
            )

        yield SimpleNamespace(list_tools=list_tools)

    monkeypatch.setattr(tools, "connect", connect)
    first = await tools.catalog()
    assert [entry["server"] for entry in first] == ["good"]
    assert await tools.catalog() == first
    assert len(calls) == 2
    assert tools.warnings and "private-secret" not in str(tools.warnings)


async def test_empty_successful_mcp_catalog_is_not_reported_as_an_outage(harness, monkeypatch):
    tools = MCPTools(harness.settings)
    tools.servers = {"empty": ServerConfig(command="fixture", read_tools=["not_advertised"])}

    @asynccontextmanager
    async def connect(name):
        async def list_tools():
            return SimpleNamespace(tools=[])

        yield SimpleNamespace(list_tools=list_tools)

    monkeypatch.setattr(tools, "connect", connect)
    assert await tools.catalog() == []
    assert await tools.catalog() == []
    assert tools.warnings == []


def test_tool_registry_matches_names_permissions_and_executor_handlers():
    from personal_assistant.tools import ToolName

    assert {item.value for item in ToolName} == set(TOOLS)
    for name, spec in TOOLS.items():
        assert set(spec.fields.split()) <= set(Action.model_fields)
        assert set(spec.required.split()) <= set(spec.fields.split())
        assert spec.mutates or callable(getattr(ToolExecutor, name, None))
        for role in spec.roles.split():
            assert name in TOOLS_BY_ROLE[role]
        assert spec.argument_schema(Action.model_json_schema())["additionalProperties"] is False
    with pytest.raises(ValueError, match="another tool"):
        Action(name="get_weather", title="Unexpected parameter").validate_arguments()
