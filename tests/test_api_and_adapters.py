import json
import socket

import httpx
import pytest

from personal_assistant.api import create_app
from personal_assistant.config import Settings
from personal_assistant.graph import Draft
from personal_assistant.llm import ContextLimitError, HTTPModel
from personal_assistant.research import PublicResolver, on_site, public_url
from tests.conftest import ScriptedModel


async def test_api_requires_auth_before_model_access(harness):
    model = ScriptedModel()
    app = create_app(harness.settings, harness.runtime(model))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/v1/notes")).status_code == 401
        assert (await client.post("/v1/chat", json={"text": "Hello"})).status_code == 401
        response = await client.get(
            "/v1/status",
            headers={"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()},
        )
        assert response.status_code == 200
        assert response.json()["retrieval"] == "keyword"
        assert "token" not in response.text
    assert model.calls == []


async def test_local_adapter_uses_schema_and_never_calls_cloud(harness):
    received = []

    async def handle(request):
        if request.url.path == "/tokenize":
            return httpx.Response(404)
        received.append(request)
        body = json.loads(request.content)
        assert body["model"] == "Qwen3.8-27B-Q4_K_M"
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"answer":"Привет"}'}}
                ]
            },
        )

    model = HTTPModel(harness.settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        assert (await model.generate("Answer", {"question": "Привет"}, Draft)).answer == "Привет"
        assert received[0].url.host == "localhost"
        with pytest.raises(ContextLimitError):
            await model.generate("Answer", {"question": "я" * 20000}, Draft)
        assert len(received) == 1
    finally:
        await model.close()


async def test_openai_adapter_uses_responses_and_disables_remote_storage():
    settings = Settings(
        _env_file=None,
        model_provider="openai",
        openai_model="configured-model",
        openai_api_key="testing-only",
    )

    async def handle(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["store"] is False
        assert body["text"]["format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"answer":"Hi"}'}],
                    }
                ],
            },
        )

    model = HTTPModel(settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        assert (await model.generate("Answer", {}, Draft)).answer == "Hi"
    finally:
        await model.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1",
        "http://localhost",
        "file:///etc/passwd",
        "https://user:password@example.org",
        "https://example.org:8080",
    ],
)
def test_research_blocks_nonpublic_targets(url):
    with pytest.raises(ValueError):
        public_url(url)


async def test_dns_resolution_rejects_private_addresses(monkeypatch):
    import asyncio

    async def private(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 443))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", private)
    with pytest.raises(ValueError, match="private"):
        await PublicResolver().resolve("rebind.example.org", 443)


def test_job_source_filter_has_domain_boundaries():
    assert on_site("https://jobs.example.org/role", ["example.org"])
    assert not on_site("https://example.org.evil.test/role", ["example.org"])
