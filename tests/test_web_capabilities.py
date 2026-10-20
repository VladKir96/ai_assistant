import json

import pytest
from pydantic import SecretStr

from personal_assistant.config import Settings
from personal_assistant.graph import Draft
from personal_assistant.research import web_capabilities
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from tests.conftest import ScriptedModel


@pytest.mark.parametrize("search_configured", [False, True])
@pytest.mark.parametrize("sites_configured", [False, True])
@pytest.mark.parametrize("provider", ["duckduckgo", "tavily", "disabled"])
def test_search_configuration_is_separate_from_page_reading(
    search_configured, sites_configured, provider
):
    settings = Settings(
        _env_file=None,
        tavily_api_key="testing-only" if search_configured else "",
        search_provider=provider,
        job_sites=["example.org"] if sites_configured else [],
    )
    capabilities = web_capabilities(settings)
    assert capabilities["read_public_urls"]
    available = provider == "duckduckgo" or (provider == "tavily" and search_configured)
    assert capabilities["search_provider"] == provider
    assert capabilities["search_web"] == available
    assert capabilities["search_jobs"] == (available and sites_configured)
    assert not capabilities["browser_automation"]
    assert "testing-only" not in str(capabilities)


async def test_missing_search_key_is_visible_through_planning_review_and_revision(harness):
    harness.settings.tavily_api_key = SecretStr("")
    harness.settings.search_provider = "tavily"
    model = ScriptedModel(
        Plan(specialist="research", actions=[Action(name="search_web", query="a test query")]),
        Draft(answer="I cannot access the internet."),
        Review(
            decision="revise", issues=["Public URL reading is available; search is unconfigured"]
        ),
        Plan(specialist="research"),
        Draft(answer="Web search is not configured. Send a public URL and I can read it."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model).service.chat(ChatRequest(text="Find sources for this"))
    assert response.review_status == "accepted"
    result = json.loads(model.calls[1][1]["operation_results"][0]["data"])
    assert result["status"] == "needs_configuration"
    for _, payload, _ in model.calls:
        assert payload["web_capabilities"]["read_public_urls"]
        assert not payload["web_capabilities"]["search_web"]


async def test_supplied_url_can_be_read_without_search_configuration(harness):
    harness.settings.tavily_api_key = SecretStr("")

    class PublicPage:
        async def read(self, url):
            return {"url": url, "content": "Public page evidence", "untrusted": True}

    model = ScriptedModel(
        Plan(specialist="research", actions=[Action(name="read_page", url="https://example.org")]),
        Draft(answer="The page contains public page evidence."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model, research=PublicPage()).service.chat(
        ChatRequest(text="Read https://example.org")
    )
    assert response.operations[0]["ok"]
    result = json.loads(model.calls[1][1]["operation_results"][0]["data"])
    assert result["content"] == "Public page evidence"
