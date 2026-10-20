import json

import httpx
import pytest

from personal_assistant import research
from personal_assistant.config import Settings


def fake_duckduckgo(monkeypatch, results=None, error=None):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 20

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            if error:
                raise error
            return results or []

    monkeypatch.setattr(research, "DDGS", Client)
    return calls


@pytest.mark.parametrize("jobs", [False, True])
async def test_duckduckgo_needs_no_key_and_filters_results(monkeypatch, jobs):
    calls = fake_duckduckgo(
        monkeypatch,
        results=[
            {"href": "http://127.0.0.1/private", "title": "Local", "body": "blocked"},
            {"href": "https://example.org.evil.test/job", "title": "Wrong domain"},
            {"href": "https://jobs.example.org/role", "title": "Вакансия", "body": "Python"},
            {"href": "https://jobs.example.org/role", "title": "Duplicate", "body": "duplicate"},
            {"href": "file:///private", "body": "blocked"},
            {"href": None},
            "malformed item",
        ],
    )
    settings = Settings(
        _env_file=None,
        search_provider="duckduckgo",
        tavily_api_key="",
        duckduckgo_region="ru-ru",
        job_sites=["example.org", "careers.example.net"],
    )
    result = await research.Research(settings).search("Python работа", jobs=jobs)
    assert result["provider"] == "duckduckgo"
    assert result["status"] == "ok"
    assert len(result["results"]) == (1 if jobs else 2)
    assert all(item["untrusted"] for item in result["results"])
    assert all(item["evidence_type"] == "search_excerpt" for item in result["results"])
    assert len(calls) == 1
    assert calls[0][1]["backend"] == "duckduckgo"
    assert calls[0][1]["region"] == "ru-ru"
    assert calls[0][1]["max_results"] == 10
    if jobs:
        assert "site:example.org" in calls[0][0]
        assert "site:careers.example.net" in calls[0][0]
        assert result["results"][0]["title"] == "Вакансия"


async def test_duckduckgo_failure_is_sanitized_and_never_falls_back(monkeypatch):
    fake_duckduckgo(
        monkeypatch,
        error=research.DDGSException("rate limit for private-query-do-not-print"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Failure must not contact an alternative search provider")

    monkeypatch.setattr(research.httpx, "AsyncClient", forbidden)
    settings = Settings(
        _env_file=None, search_provider="duckduckgo", tavily_api_key="must-not-be-used"
    )
    with pytest.raises(ValueError) as caught:
        await research.Research(settings).search("private-query-do-not-print")
    assert "DuckDuckGo search failed" in str(caught.value)
    assert "private-query-do-not-print" not in str(caught.value)
    assert "must-not-be-used" not in str(caught.value)


@pytest.mark.parametrize("provider", ["duckduckgo", "tavily", "disabled"])
async def test_job_search_requires_selected_sites_before_any_network(monkeypatch, provider):
    def forbidden(*args, **kwargs):
        pytest.fail("Choose job websites before making search requests")

    monkeypatch.setattr(research, "DDGS", forbidden)
    monkeypatch.setattr(research.httpx, "AsyncClient", forbidden)
    settings = Settings(_env_file=None, search_provider=provider, job_sites=[])
    result = await research.Research(settings).search("Python vacancies", jobs=True)
    assert result["status"] == "needs_configuration"


async def test_disabled_search_never_contacts_providers(monkeypatch):
    calls = fake_duckduckgo(monkeypatch)
    settings = Settings(_env_file=None, search_provider="disabled", tavily_api_key="unused")
    result = await research.Research(settings).search("query")
    assert result["status"] == "needs_configuration"
    assert "disabled" in result["message"]
    assert calls == []
    assert research.web_capabilities(settings)["read_public_urls"]


async def test_tavily_remains_explicitly_selectable(monkeypatch):
    calls = fake_duckduckgo(monkeypatch)

    async def handle(request):
        assert request.url.host == "api.tavily.com"
        assert request.headers["Authorization"] == "Bearer testing-only"
        assert json.loads(request.content)["include_domains"] == ["example.org"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://example.org/job", "title": "Job", "content": "Python"},
                    {"url": "https://other.example/job", "title": "Other"},
                ]
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        research.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    settings = Settings(
        _env_file=None,
        search_provider="tavily",
        tavily_api_key="testing-only",
        job_sites=["example.org"],
    )
    result = await research.Research(settings).search("Python vacancies", jobs=True)
    assert result["provider"] == "tavily"
    assert [item["url"] for item in result["results"]] == ["https://example.org/job"]
    assert calls == []
