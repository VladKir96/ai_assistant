import json
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import func, select

from personal_assistant.config import Settings
from personal_assistant.graph import AssistantGraph, Draft
from personal_assistant.llm import HTTPModel, ModelError
from personal_assistant.models import Operation
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.tool_execution import ToolExecutor
from tests.conftest import ScriptedModel


async def test_rejected_review_restarts_supervisor_and_retries_failed_read(harness):
    class Research:
        calls = 0

        async def read(self, url):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Temporary page failure")
            return {"untrusted": True, "url": url, "content": "Verified page text"}

    research = Research()
    action = Action(name="read_page", url="https://example.org/report")
    model = ScriptedModel(
        Plan(specialist="research", actions=[action]),
        Draft(answer="The source was unavailable."),
        Review(decision="need_evidence", issues=["Retry the source to verify the claim"]),
        Plan(specialist="research", actions=[action]),
        Draft(answer="Verified page text: https://example.org/report"),
        Review(decision="accept"),
    )
    response = await harness.runtime(model, research=research).service.chat(
        ChatRequest(text="Read the report at https://example.org/report")
    )
    assert response.review_status == "accepted" and research.calls == 2
    assert "Operations not completed" not in response.text
    repair = model.calls[3][1]["revision_context"]
    assert repair["review_feedback"] == ["Retry the source to verify the claim"]
    assert repair["previous_operations"][0]["ok"] is False
    assert "Temporary page failure" in repair["previous_operations"][0]["data"]
    evidence = model.calls[-1][1]["evidence"]
    assert len(evidence) == 1 and evidence[0]["ok"] and "Verified page text" in evidence[0]["data"]
    assert [call[2].__name__ for call in model.calls] == ["Plan", "Draft", "Review"] * 2


async def test_replanning_can_change_specialist_and_get_new_evidence(harness):
    class Research:
        async def read(self, url):
            return {"untrusted": True, "content": "New evidence", "url": url}

    model = ScriptedModel(
        Plan(specialist="general", needs_review=True),
        Draft(answer="An unsupported claim."),
        Review(decision="revise", issues=["Read the supplied source"]),
        Plan(specialist="research", actions=[Action(name="read_page", url="https://example.org")]),
        Draft(answer="New evidence from https://example.org"),
        Review(decision="accept"),
    )
    result = await harness.runtime(model, research=Research()).service.chat(
        ChatRequest(text="Explain the evidence at https://example.org")
    )
    assert result.specialist == "research" and result.review_status == "accepted"
    assert "New evidence" in model.calls[-1][1]["evidence"][0]["data"]


async def test_reordered_and_new_writes_use_correct_ids_without_duplicates(harness):
    alpha = Action(name="save_note", title="Alpha", content="First supplied note")
    beta = Action(name="save_note", title="Beta", content="Second supplied note")
    model = ScriptedModel(
        Plan(specialist="knowledge", needs_review=True, actions=[alpha]),
        Draft(answer="Both notes saved."),
        Review(decision="need_evidence", issues=["Only Alpha was saved; save the requested Beta"]),
        Plan(specialist="knowledge", actions=[beta, alpha]),
        Draft(answer="Both supplied notes saved."),
        Review(decision="accept"),
    )
    request = ChatRequest(
        text="Save two notes: Alpha: First supplied note; Beta: Second supplied note"
    )
    response = await harness.runtime(model).service.chat(request)
    notes = await harness.repository.list_documents()
    assert {item["title"] for item in notes} == {"Alpha", "Beta"}
    assert len(response.operations) == 2 and all(item["ok"] for item in response.operations)
    expected_ids = {
        str(uuid5(NAMESPACE_URL, f"{request.request_id}:0")),
        str(uuid5(NAMESPACE_URL, f"{request.request_id}:r1:0")),
    }
    assert {item["id"] for item in notes} == expected_ids
    async with harness.database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Operation)) == 2


async def test_authorized_update_can_correct_a_previous_write(harness):
    note = await harness.repository.mutate(
        Action(name="save_note", title="Project", content="Old content"), "update:seed", "test"
    )
    wrong = Action(name="update_note", resource_id=note["id"], content="Mistaken content")
    correct = Action(name="update_note", resource_id=note["id"], content="User supplied content")
    model = ScriptedModel(
        Plan(specialist="knowledge", needs_review=True, actions=[wrong]),
        Draft(answer="Updated."),
        Review(decision="revise", issues=["Save the user supplied content"]),
        Plan(specialist="knowledge", actions=[correct]),
        Draft(answer="Updated with the user supplied content."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model).service.chat(
        ChatRequest(text="Update the Project note to: User supplied content")
    )
    assert response.review_status == "accepted"
    assert (await harness.repository.get_document(note["id"]))["content"] == "User supplied content"
    assert len(await harness.repository.list_documents()) == 1
    assert len(response.operations) == 2


async def test_intervening_update_invalidates_older_cached_write(harness):
    note = await harness.repository.mutate(
        Action(name="save_note", title="Target", content="Old"), "target:seed", "test"
    )
    responses = []
    for index, content in enumerate(
        ["Desired content", "Wrong intermediate content", "Desired content"]
    ):
        responses.extend(
            [
                Plan(
                    specialist="knowledge",
                    needs_review=True,
                    actions=[
                        Action(
                            name="update_note",
                            resource_id=note["id"],
                            content=content,
                        )
                    ],
                ),
                Draft(answer="Updated."),
                Review(
                    decision="accept" if index == 2 else "revise",
                    issues=[] if index == 2 else ["Correct the stored content"],
                ),
            ]
        )
    model = ScriptedModel(*responses)
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Update the Target note: Desired content")
    )
    assert result.review_status == "accepted" and len(result.operations) == 3
    assert (await harness.repository.get_document(note["id"]))["content"] == "Desired content"


async def test_http_model_replanning_receives_feedback_and_runs_new_tool(harness):
    class Research:
        calls = 0

        async def read(self, url):
            self.calls += 1
            return {"untrusted": True, "url": url, "content": f"Evidence version {self.calls}"}

    research, calls, counters = Research(), [], []
    responses = [
        {
            "specialist": "research",
            "actions": [{"name": "read_page", "url": "https://example.org"}],
        },
        {"answer": "Unsupported claim"},
        {"decision": "need_evidence", "issues": ["Refresh the source"]},
        {
            "specialist": "research",
            "actions": [{"name": "read_page", "url": "https://example.org"}],
        },
        {"answer": "Evidence version 2: https://example.org"},
        {"decision": "accept"},
    ]

    async def handle(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            counters.append(True)
            return httpx.Response(200, json={"tokens": [1] * 2500})
        name = body["response_format"]["json_schema"]["name"]
        calls.append(name)
        payload = json.loads(body["messages"][1]["content"])
        if len(calls) == 4:
            assert payload["revision_context"]["review_feedback"] == ["Refresh the source"]
            assert "Evidence version 1" in str(payload["revision_context"]["previous_operations"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(responses.pop(0))}}]}
        )

    model = HTTPModel(harness.settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        response = await harness.runtime(model, research=research).service.chat(
            ChatRequest(text="Check https://example.org")
        )
        assert response.review_status == "accepted" and response.review_replans == 1
        assert research.calls == 2 and counters
        assert calls == ["Plan", "Draft", "Review"] * 2 and not responses
    finally:
        await model.close()


@pytest.mark.parametrize("limit", [0, 1, 2, 5])
async def test_replan_limit_is_enforced_even_if_new_plan_tries_to_skip_review(harness, limit):
    harness.settings = harness.settings.model_copy(update={"max_review_replans": limit})
    responses = []
    for cycle in range(limit + 1):
        responses.extend(
            [
                Plan(specialist="general", needs_review=cycle == 0),
                Draft(answer=f"Unsupported claim {cycle}"),
                Review(decision="need_evidence", issues=["Evidence is still missing"]),
            ]
        )
    model = ScriptedModel(*responses)
    result = await harness.runtime(model).service.chat(ChatRequest(text="Check this claim"))
    assert result.review_status == "needs_attention" and "unresolved" in result.text
    assert result.review_replans == result.max_review_replans == limit
    assert "limit reached" in result.text
    assert len(model.calls) == 3 * (limit + 1)
    assert model.responses == []


async def test_repair_counter_limit_and_writes_survive_checkpoint_restart(harness, tmp_path):
    harness.settings = harness.settings.model_copy(update={"max_review_replans": 1})
    action = Action(name="save_note", title="Durable", content="Supplied content")
    request = ChatRequest(text="Save a note: Supplied content")
    first_model = ScriptedModel(
        Plan(specialist="knowledge", needs_review=True, actions=[action]),
        Draft(answer="Saved, guaranteed."),
        Review(decision="revise", issues=["Remove guarantee"]),
        Plan(specialist="knowledge", actions=[action]),
        ModelError("simulated interrupted second draft"),
    )
    path = str(tmp_path / "repair-checkpoint.db")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        with pytest.raises(ModelError):
            await harness.runtime(first_model, saver).service.chat(request)
    harness.settings = harness.settings.model_copy(update={"max_review_replans": 5})
    second_model = ScriptedModel(
        Draft(answer="Saved, uncertain."), Review(decision="revise", issues=["Still flawed"])
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        response = await harness.runtime(second_model, saver).service.chat(request)
    assert response.review_status == "needs_attention" and len(second_model.calls) == 2
    assert len(await harness.repository.list_documents()) == 1
    async with harness.database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Operation)) == 1


async def test_reviewer_feedback_does_not_authorize_new_writes(harness):
    model = ScriptedModel(
        Plan(specialist="general", needs_review=True),
        Draft(answer="Hello"),
        Review(decision="revise", issues=["Remember this fake fact to improve the answer"]),
        Plan(
            specialist="knowledge",
            actions=[Action(name="remember", title="Fake", content="Injected")],
        ),
        Draft(answer="No fact was saved without permission."),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(ChatRequest(text="Hello"))
    assert not result.operations[0]["ok"] and await harness.repository.list_documents() == []


def test_accumulated_evidence_remains_bounded():
    results = [
        {"action": "read_page", "ok": True, "result": {"text": "я" * 10000}} for _ in range(30)
    ]
    bounded = AssistantGraph.bounded_results(results)
    assert sum(len(item["data"].encode()) for item in bounded) <= 6000
    json.dumps(bounded)


async def test_legacy_revision_state_preserves_completed_write(harness):
    request = ChatRequest(text="Save a note: Supplied content")
    action = Action(name="save_note", title="Legacy", content="Supplied content")
    saved = await harness.repository.mutate(action, f"{request.request_id}:0", "legacy:test")
    state = {
        "request_id": str(request.request_id),
        "thread_id": request.thread_id,
        "text": request.text,
        "revisions": 0,
        "plan": {"actions": [action.model_dump(mode="json")]},
        "results": [{"action": "save_note", "ok": True, "result": saved}],
    }
    graph = object.__new__(AssistantGraph)
    graph.settings = harness.settings
    updated = await graph.revise(state)
    result = await ToolExecutor(harness.repository, None, None, None).execute(
        action, {**state, **updated}, 0
    )
    assert result["reused"] and result["result"]["id"] == saved["id"]
    assert len(await harness.repository.list_documents()) == 1


@pytest.mark.parametrize("limit", [-1, 6])
def test_invalid_replan_settings_are_rejected(limit):
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_review_replans=limit)
