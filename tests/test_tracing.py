import asyncio
import hashlib
import json

import httpx
import langfuse
import pytest
import pytest_asyncio
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from personal_assistant.agent.graph import Draft
from personal_assistant.app.api import create_app
from personal_assistant.core.config import Settings
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.core.tracing import CONTENT_LIMIT, Tracing, check_tracing
from personal_assistant.inference.llm import HTTPModel, ModelError
from tests.conftest import ScriptedModel


@pytest_asyncio.fixture
async def tracing(monkeypatch):
    exporter = InMemorySpanExporter()
    constructor = langfuse.Langfuse

    def local_client(**kwargs):
        # Exercise the real SDK/OTel processor, with no Langfuse account/network.
        return constructor(**kwargs, span_exporter=exporter)

    monkeypatch.setattr(langfuse, "Langfuse", local_client)
    from uuid import uuid4

    settings = Settings(
        _env_file=None,
        langfuse_enabled=True,
        langfuse_public_key="pk-test-" + uuid4().hex,
        langfuse_secret_key="sk-test-only",
        api_token="private-api-token",
    )
    tracer = Tracing(settings)
    assert tracer.client is not None
    try:
        yield tracer, exporter
    finally:
        await tracer.close()


async def spans(tracer, exporter):
    await asyncio.to_thread(tracer.client.flush)
    return exporter.get_finished_spans()


def test_disabled_does_not_initialize_sdk(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Disabled tracing initialized the SDK")

    monkeypatch.setattr(langfuse, "Langfuse", forbidden)
    tracer = Tracing(Settings(_env_file=None))
    with tracer.observation("request", input="private") as span:
        span.update(output="private")
    assert tracer.status == {"active": False, "capture_content": False}


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"langfuse_public_key": "pk-only"},
        {"langfuse_base_url": "file:///tmp/traces"},
        {"langfuse_base_url": "https://user:password@example.com"},
        {"langfuse_base_url": "https://example.com?token=secret"},
        {"langfuse_environment": "langfuse-test"},
        {"langfuse_sample_rate": 1.1},
    ],
)
def test_tracing_configuration_validation(changes):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, langfuse_enabled=True, **changes)


def test_initialization_failure_is_nonfatal(monkeypatch, caplog):
    def broken(**kwargs):
        raise RuntimeError("secret that must not be logged")

    monkeypatch.setattr(langfuse, "Langfuse", broken)
    tracer = Tracing(
        Settings(
            _env_file=None,
            langfuse_enabled=True,
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        )
    )
    with tracer.observation("request"):
        pass
    assert tracer.client is None
    assert "secret that must" not in caplog.text


async def test_trace_tree_replanning_tools_cache_and_private_content(harness, tracing):
    tracer, exporter = tracing
    model = ScriptedModel(
        Plan(
            specialist="knowledge",
            needs_review=True,
            actions=[Action(name="save_note", title="Private", content="personal content")],
        ),
        Draft(answer="Saved, guaranteed."),
        Review(decision="revise", issues=["unsupported claim"]),
        Plan(
            specialist="knowledge",
            actions=[Action(name="save_note", title="Private", content="personal content")],
        ),
        Draft(answer="Saved."),
        Review(decision="accept"),
    )
    service = harness.runtime(model, tracing=tracer).service
    request = ChatRequest(text="Save a note: personal content", thread_id="telegram:123456")
    result = await service.chat(request)
    assert result.review_replans == 1
    assert await service.chat(request) == result
    captured = await spans(tracer, exporter)
    names = [span.name for span in captured]
    assert names.count("assistant.chat") == 2
    assert names.count("graph.supervisor") == 2
    assert names.count("graph.review") == 2
    assert "graph.revise" in names
    assert "request.cache_hit" in names
    tools = [span for span in captured if span.name == "tool.save_note"]
    assert len(tools) == 2
    assert json.loads(tools[1].attributes["langfuse.observation.metadata.reused"]) is True
    parent_by_id = {span.context.span_id: span for span in captured}
    for tool in tools:
        assert parent_by_id[tool.parent.span_id].name == "graph.knowledge"
    exported = str([dict(span.attributes) for span in captured])
    assert "personal content" not in exported
    assert "telegram:123456" not in exported
    assert "unsupported claim" not in exported
    assert hashlib.sha256(request.thread_id.encode()).hexdigest() in exported
    assert str(request.request_id) in exported
    assert all(not span.events for span in captured)


@pytest.mark.parametrize("provider", ["local", "openai"])
async def test_generation_usage_and_invalid_json_retry(tracing, provider):
    tracer, exporter = tracing
    settings = Settings(
        _env_file=None,
        model_provider=provider,
        openai_model="test-model",
        openai_api_key="provider-secret",
    )
    model = HTTPModel(settings, tracing=tracer)
    count = 0

    async def handle(request):
        nonlocal count
        count += 1
        text = "not valid JSON" if count == 1 else '{"answer":"Private answer"}'
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": text}}],
                "output": [{"content": [{"type": "output_text", "text": text}]}],
                "usage": (
                    {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
                    if provider == "local"
                    else {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}
                ),
            },
        )

    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        with tracer.observation("assistant.chat"):
            result = await model.generate("Private system", {"question": "Private query"}, Draft)
        assert result.answer == "Private answer"
        captured = await spans(tracer, exporter)
        generations = [span for span in captured if span.name == "model.generate"]
        assert len(generations) == 2
        assert generations[0].attributes["langfuse.observation.level"] == "ERROR"
        assert generations[0].attributes["langfuse.observation.status_message"] == "ValidationError"
        for span in generations:
            usage = json.loads(span.attributes["langfuse.observation.usage_details"])
            assert usage == {"input": 20, "output": 8, "total": 28}
        assert "Private" not in str([dict(span.attributes) for span in captured])
        assert all(not span.events for span in captured)
    finally:
        await model.close()


async def test_opt_in_content_is_bounded_redacted_and_has_no_media(tracing):
    tracer, exporter = tracing
    tracer.capture_content = True
    with tracer.observation(
        "request",
        input={
            "text": "private-api-token",
            "authorization": "unknown-secret",
            "image": "data:image/jpeg;base64,AAAA",
            "raw": b"PDF bytes",
            "nested": {"password": "also-secret"},
        },
    ) as span:
        span.update(output="User-visible answer")
    captured = await spans(tracer, exporter)
    attributes = captured[0].attributes
    payload = json.loads(attributes["langfuse.observation.input"])
    assert payload["text"] == "[redacted]"
    assert payload["authorization"] == "[redacted]"
    assert payload["image"] == "[media omitted]"
    assert payload["nested"]["password"] == "[redacted]"
    assert "User-visible answer" in attributes["langfuse.observation.output"]
    assert len(json.dumps(tracer.content(["я" * 10000] * 64), ensure_ascii=False).encode()) < 20000
    assert CONTENT_LIMIT == 16000


async def test_failed_request_and_tool_failures_have_safe_errors(harness, tracing):
    tracer, exporter = tracing
    service = harness.runtime(
        ScriptedModel(ModelError("provider-secret/private prompt")), tracing=tracer
    ).service
    with pytest.raises(ModelError, match="provider-secret"):
        await service.chat(ChatRequest(text="private question"))
    captured = await spans(tracer, exporter)
    root = next(span for span in captured if span.name == "assistant.chat")
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert root.attributes["langfuse.observation.status_message"] == "ModelError"
    assert "private prompt" not in str([dict(span.attributes) for span in captured])
    assert all(not span.events for span in captured)


async def test_unavailable_sdk_operations_do_not_change_result(harness, caplog):
    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("private key")

        def shutdown(self):
            raise RuntimeError("private key")

    tracer = Tracing()
    tracer.client = BrokenClient()
    service = harness.runtime(
        ScriptedModel(Plan(specialist="general"), Draft(answer="Works")), tracing=tracer
    ).service
    assert (await service.chat(ChatRequest(text="Hello"))).text == "Works"
    await tracer.close()
    assert "private key" not in caplog.text


async def test_async_contexts_are_isolated_and_cancellation_propagates(tracing):
    tracer, exporter = tracing

    async def request(name):
        with tracer.observation(name):
            await asyncio.sleep(0)
            with tracer.observation(name + ".child"):
                await asyncio.sleep(0)

    await asyncio.gather(request("first"), request("second"))
    with pytest.raises(asyncio.CancelledError):
        with tracer.observation("cancelled"):
            raise asyncio.CancelledError("private cancellation text")
    captured = await spans(tracer, exporter)
    by_name = {span.name: span for span in captured}
    for name in ("first", "second"):
        assert by_name[name + ".child"].parent.span_id == by_name[name].context.span_id
        assert by_name[name + ".child"].context.trace_id == by_name[name].context.trace_id
    assert by_name["first"].context.trace_id != by_name["second"].context.trace_id
    assert by_name["cancelled"].attributes["langfuse.observation.level"] == "ERROR"
    assert all(not span.events for span in captured)


async def test_status_does_not_expose_credentials(harness):
    active = harness.runtime(ScriptedModel())
    app = create_app(harness.settings, active)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/v1/status")).status_code == 401
        result = await client.get(
            "/v1/status",
            headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
        )
        assert result.json()["tracing"] == {
            "configured": False,
            "active": False,
            "capture_content": False,
        }


async def test_sampling_zero_omits_whole_trace(tracing):
    from uuid import uuid4

    _, exporter = tracing
    tracer = Tracing(
        Settings(
            _env_file=None,
            langfuse_enabled=True,
            langfuse_public_key="pk-zero-" + uuid4().hex,
            langfuse_secret_key="sk-test",
            langfuse_sample_rate=0,
        )
    )
    try:
        with tracer.observation("unsampled"):
            with tracer.observation("unsampled.child"):
                pass
        assert not await spans(tracer, exporter)
    finally:
        await tracer.close()


async def test_span_update_and_exit_failure_preserve_application_exception(caplog):
    class BrokenSpan:
        def __enter__(self):
            return self

        def update(self, **kwargs):
            raise RuntimeError("private SDK update")

        def __exit__(self, *args):
            assert args == (None, None, None)
            raise RuntimeError("private SDK exit")

    class Client:
        def start_as_current_observation(self, **kwargs):
            return BrokenSpan()

    tracer = Tracing()
    tracer.client = Client()
    with pytest.raises(ValueError, match="original failure"):
        with tracer.observation("test") as span:
            span.update(output="private")
            raise ValueError("original failure")
    assert "private SDK" not in caplog.text


async def test_tool_failure_is_visible_without_capturing_error_text(harness, tracing):
    from personal_assistant.agent.tool_execution import ToolExecutor

    tracer, exporter = tracing
    executor = ToolExecutor(harness.repository, None, None, None, tracing=tracer)
    with tracer.observation("request"):
        result = await executor.execute(Action(name="read_note"), {}, 0)
    assert not result["ok"]
    captured = await spans(tracer, exporter)
    tool = next(span for span in captured if span.name == "tool.read_note")
    assert tool.attributes["langfuse.observation.level"] == "ERROR"
    assert tool.attributes["langfuse.observation.status_message"] == "Tool execution failed"
    assert "read_note requires" not in str(tool.attributes)


@pytest.mark.parametrize("authenticated", [True, False])
async def test_connection_check_flushes_synthetic_trace_and_closes(monkeypatch, authenticated):
    from contextlib import nullcontext

    from personal_assistant.core import tracing as module

    calls = []

    class Client:
        def auth_check(self):
            calls.append("authenticate")
            return authenticated

        def flush(self):
            calls.append("flush")

    class Tracer:
        client = Client()

        def observation(self, name, **kwargs):
            calls.append(name)
            return nullcontext()

        async def close(self):
            calls.append("close")

    monkeypatch.setattr(module, "Tracing", lambda settings: Tracer())
    settings = Settings(
        _env_file=None,
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    if authenticated:
        await check_tracing(settings)
        assert calls == ["authenticate", "assistant.tracing_check", "flush", "close"]
    else:
        with pytest.raises(RuntimeError, match="connection failed"):
            await check_tracing(settings)
        assert calls == ["authenticate", "close"]


async def test_disabled_connection_check_is_actionable():
    with pytest.raises(RuntimeError, match="Tracing is disabled"):
        await check_tracing(Settings(_env_file=None))


async def test_parallel_agent_spans_join_before_synthesis(harness, tracing):
    from tests.test_parallel_agents import TeamModel, task, team

    tracer, exporter = tracing
    model = TeamModel(
        team(task("general", "private first task"), task("productivity", "private second task"))
    )
    await harness.runtime(model, tracing=tracer).service.chat(ChatRequest(text="private request"))
    captured = await spans(tracer, exporter)
    by_name = {span.name: span for span in captured}
    root = by_name["assistant.chat"]
    for name in (
        "graph.general",
        "graph.productivity",
        "graph.parallel_prepare",
        "graph.synthesize",
    ):
        assert by_name[name].parent.span_id == root.context.span_id
    assert by_name["graph.synthesize"].start_time >= max(
        by_name["graph.general"].end_time, by_name["graph.productivity"].end_time
    )
    assert (
        json.loads(
            by_name["graph.supervisor"].attributes["langfuse.observation.metadata.agent_count"]
        )
        == 2
    )
    assert "private" not in str([dict(span.attributes) for span in captured])
