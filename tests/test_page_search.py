import json
from datetime import datetime
from unittest.mock import AsyncMock

import aiohttp
import pytest

from personal_assistant.config import Settings
from personal_assistant.graph import AssistantGraph, Draft
from personal_assistant.llm import fit_payload
from personal_assistant.research import Research
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from tests.conftest import ScriptedModel


def search_result():
    return {
        "status": "ok",
        "provider": "duckduckgo",
        "results": [
            {
                "url": f"https://example.org/{rank}",
                "title": f"Source {rank}",
                "content": "STALE SNIPPET: 99 °C",
            }
            for rank in range(1, 4)
        ],
    }


def page(url):
    return {
        "url": url,
        "retrieved_at": "2026-10-17T10:00:00+00:00",
        "truncated": False,
        "evidence_type": "page_excerpt",
        "untrusted": True,
        "content": "Test City, 2026-10-18, daily maximum: 18 °C. " * 200,
    }


async def test_search_reads_only_first_two_results_in_rank_order():
    service = Research(Settings(_env_file=None))
    service.search = AsyncMock(return_value=search_result())
    service.read = AsyncMock(side_effect=page)
    result = await service.search_and_read("weather Test City 2026-10-18")
    assert result["status"] == "ok"
    assert [call.args[0] for call in service.read.await_args_list] == [
        "https://example.org/1",
        "https://example.org/2",
    ]
    assert [item["rank"] for item in result["pages"]] == [1, 2]
    assert all(item["untrusted"] for item in result["pages"])
    assert "STALE SNIPPET" not in json.dumps(result)
    assert "not the source's update or forecast time" in result["retrieval_note"]


@pytest.mark.parametrize("all_fail", [False, True])
@pytest.mark.parametrize("failure", [aiohttp.ClientError, LookupError])
async def test_unreadable_pages_do_not_hide_other_source_or_expose_snippets(all_fail, failure):
    service = Research(Settings(_env_file=None))
    service.search = AsyncMock(return_value=search_result())

    async def read(url):
        if all_fail or url.endswith("/1"):
            raise failure("private-request-details")
        return page(url)

    service.read = AsyncMock(side_effect=read)
    result = await service.search_and_read("weather")
    assert result["pages"][0]["status"] == "unreadable"
    assert result["pages"][1]["status"] == ("unreadable" if all_fail else "ok")
    assert result["status"] == ("no_page_evidence" if all_fail else "ok")
    assert service.read.await_count == 2
    assert "STALE SNIPPET" not in json.dumps(result)
    assert "private-request-details" not in json.dumps(result)


@pytest.mark.parametrize("provider", ["disabled", "tavily"])
async def test_search_and_read_does_not_fetch_when_search_unconfigured(provider):
    service = Research(Settings(_env_file=None, search_provider=provider, tavily_api_key=""))
    service.read = AsyncMock(side_effect=AssertionError("No page requests expected"))
    assert (await service.search_and_read("weather"))["status"] == "needs_configuration"
    service.read.assert_not_awaited()


async def test_empty_search_does_not_fabricate_page_evidence():
    service = Research(Settings(_env_file=None))
    service.search = AsyncMock(return_value={**search_result(), "results": []})
    result = await service.search_and_read("weather")
    assert result["status"] == "no_page_evidence"
    assert result["pages"] == []


@pytest.mark.parametrize("encoding", ["utf-8", "invalid-encoding", "base64", "cp1251"])
async def test_page_reader_extracts_dated_weather_table_and_records_fetch_metadata(
    monkeypatch, encoding
):
    html = """<html><body><article><h1>Weather for Test City / Прогноз</h1>
    <p>Forecast valid 2026-10-18</p><table><tr><th>Time</th><th>Temperature</th></tr>
    <tr><td>12:00</td><td>18 C</td></tr></table></article></body></html>""".encode(
        "cp1251" if encoding == "cp1251" else "utf-8"
    )

    class Stream:
        async def iter_chunked(self, size):
            yield html

    class Response:
        status = 200
        charset = encoding
        headers = {"Content-Type": "text/html"}
        content = Stream()

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session(Response):
        def __init__(self, **kwargs):
            self.connector = kwargs["connector"]

        def get(self, url, **kwargs):
            return Response()

        async def __aexit__(self, *args):
            await self.connector.close()

    monkeypatch.setattr("personal_assistant.research.aiohttp.ClientSession", Session)
    result = await Research(Settings(_env_file=None)).read("https://example.org/weather")
    assert "2026-10-18" in result["content"]
    assert "Прогноз" in result["content"]
    assert "| 12:00 | 18 C |" in result["content"]
    assert datetime.fromisoformat(result["retrieved_at"]).utcoffset().total_seconds() == 0
    assert result["evidence_type"] == "page_excerpt"
    assert not result["truncated"]


async def test_both_sources_reach_draft_review_and_revision_with_bounded_context(harness):
    service = Research(harness.settings)
    service.search = AsyncMock(return_value=search_result())
    service.read = AsyncMock(side_effect=page)
    model = ScriptedModel(
        Plan(specialist="research", actions=[Action(name="search_web", query="Test City weather")]),
        Draft(answer="It is currently 18 °C."),
        Review(decision="revise", issues=["18 °C is tomorrow's maximum, not current conditions"]),
        Plan(specialist="research"),
        Draft(answer="Tomorrow's forecast maximum is 18 °C: https://example.org/1"),
        Review(decision="accept"),
    )
    response = await harness.runtime(model, research=service).service.chat(
        ChatRequest(text="What is tomorrow's forecast maximum for Test City?")
    )
    assert response.review_status == "accepted"
    assert service.search.await_count == 1  # The new plan selected no additional reads.
    assert service.read.await_count == 2
    system, payload, schema = model.calls[0]
    fit_payload(system, payload, schema.model_json_schema(), harness.settings.input_budget - 150)
    for system, payload, schema in model.calls[1:]:
        fitted = (
            payload
            if schema is Plan
            else fit_payload(
                system, payload, schema.model_json_schema(), harness.settings.input_budget - 150
            )
        )
        evidence = (
            fitted.get("operation_results", fitted.get("evidence"))
            or fitted["revision_context"]["previous_operations"]
        )
        result = json.loads(evidence[0]["data"])
        assert len(evidence[0]["data"].encode()) <= 6000
        assert [item["source_url"] for item in result["pages"]] == [
            "https://example.org/1",
            "https://example.org/2",
        ]
        assert all(item["content"] and item["truncated"] for item in result["pages"])
        assert all(item["retrieved_at"] for item in result["pages"])
        assert "STALE SNIPPET" not in json.dumps(payload)


def test_large_source_urls_are_omitted_explicitly_without_corrupting_evidence():
    result = {"status": "ok", "pages": [page("https://example.org/" + "a" * 8000)]}
    bounded = AssistantGraph.bounded_results(
        [{"action": "search_web", "ok": True, "result": result}]
    )
    evidence = json.loads(bounded[0]["data"])
    assert evidence["pages"] == []
    assert evidence["omitted_pages"] == 1
    assert len(bounded[0]["data"].encode()) <= 6000
