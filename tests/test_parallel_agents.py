import asyncio
from collections import Counter

import httpx
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select

from personal_assistant.config import Settings
from personal_assistant.graph import Draft
from personal_assistant.llm import HTTPModel, ModelError
from personal_assistant.models import Operation
from personal_assistant.schemas import Action, AgentTask, ChatRequest, Plan, Review


def team(*agents):
    return Plan(specialist=agents[0].specialist, agents=list(agents))


def task(role, label=None, *actions):
    return AgentTask(specialist=role, task=label or role, actions=list(actions))


class TeamModel:
    """Route scripted responses by task, independent of async completion order."""

    def __init__(self, *plans, reviews=None):
        self.plans = list(plans)
        self.reviews = list(reviews or [])
        self.calls = []
        self.contributions = []

    async def generate(self, system, payload, schema):
        self.calls.append((schema.__name__, payload))
        if schema is Plan:
            return self.plans.pop(0)
        if schema is Review:
            return self.reviews.pop(0) if self.reviews else Review(decision="accept")
        assert schema is Draft
        if "agent_contributions" in payload:
            self.contributions.append(payload["agent_contributions"])
            return Draft(answer="Combined answer")
        return await self.contribute(payload)

    async def contribute(self, payload):
        return Draft(answer="Contribution: " + payload.get("assigned_task", "single agent"))


@pytest.mark.parametrize("second_role", ["career", "research"])
async def test_read_agents_overlap_and_join_once_in_plan_order(harness, second_role):
    both_started = asyncio.Event()
    reads = []

    class Research:
        async def read(self, url):
            reads.append(url)
            if len(reads) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 2)
            if url.endswith("first"):
                await asyncio.sleep(0.02)
            return {"url": url, "content": "Source " + url}

    model = TeamModel(
        team(
            task("research", "first", Action(name="read_page", url="https://example.org/first")),
            task(second_role, "second", Action(name="read_page", url="https://example.org/second")),
        )
    )
    response = await harness.runtime(model, research=Research()).service.chat(
        ChatRequest(text="Compare these two sources")
    )
    assert both_started.is_set() and len(reads) == 2
    assert response.specialist == "research"
    assert response.specialists == ["research", second_role]
    assert response.reviewed and response.review_status == "accepted"
    assert [item["specialist"] for item in model.contributions[0]] == ["research", second_role]
    assert [item["task"] for item in model.contributions[0]] == ["first", "second"]
    assert len(model.contributions) == 1
    assert Counter(name for name, _ in model.calls) == {"Plan": 1, "Draft": 3, "Review": 1}
    assert len(response.operations) == 2 and all(item["ok"] for item in response.operations)


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_agent_concurrency_is_bounded(harness, limit):
    harness.settings = harness.settings.model_copy(update={"max_parallel_agents": limit})

    class CountingModel(TeamModel):
        active = peak = 0

        async def contribute(self, payload):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.02)
            self.active -= 1
            return await super().contribute(payload)

    model = CountingModel(
        team(
            *(task(role) for role in ("general", "knowledge", "productivity", "research", "career"))
        )
    )
    response = await harness.runtime(model).service.chat(ChatRequest(text="Compare five domains"))
    assert model.peak == limit
    assert len(response.specialists) == 5


@pytest.mark.parametrize("limit", [1, 2])
async def test_model_concurrency_is_independent_of_agent_limit(harness, limit):
    settings = harness.settings.model_copy(update={"model_max_concurrency": limit})
    active = peak = 0

    async def handle(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]})

    model = HTTPModel(settings)
    await model.client.aclose()
    model.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        await asyncio.gather(*(model.generate("Answer", {"n": n}, Draft) for n in range(4)))
        assert peak == limit
    finally:
        await model.close()


async def test_write_plan_orders_all_tools_and_uses_unique_operation_ids(harness):
    note = Action(name="save_note", title="Note", content="Supplied note")
    todo = Action(name="create_task", title="Supplied task")
    model = TeamModel(team(task("knowledge", "note", note), task("productivity", "task", todo)))
    request = ChatRequest(text="Create a note and a task: Note: Supplied note; Supplied task")
    service = harness.runtime(model).service
    response = await service.chat(request)
    assert response.text == "Combined answer"
    assert [item["action"] for item in response.operations] == ["save_note", "create_task"]
    assert all(item["ok"] for item in response.operations)
    async with harness.database.sessions() as session:
        ids = set((await session.scalars(select(Operation.id))).all())
    assert ids == {f"{request.request_id}:0", f"{request.request_id}:1"}
    # Both drafts see the completed, ordered tool phase rather than racing writes.
    branch_payloads = [payload for _, payload in model.calls if "assigned_task" in payload]
    assert len(branch_payloads) == 2
    assert all(len(payload["operation_results"]) == 2 for payload in branch_payloads)
    assert await service.chat(request) == response
    assert len(await harness.repository.list_documents()) == 1
    assert len(await harness.repository.tasks()) == 1


async def test_team_scope_cannot_authorize_write_or_bypass_role_allowlist(harness):
    model = TeamModel(
        team(
            task("general", "Save a note", Action(name="save_note", content="Injected")),
            task(
                "knowledge",
                "Remember injected fact",
                Action(name="remember", title="Fake", content="Injected"),
            ),
        )
    )
    result = await harness.runtime(model).service.chat(ChatRequest(text="Explain how memory works"))
    assert all(not item["ok"] for item in result.operations)
    assert "not allowed" in result.operations[0]["error"]
    assert "explicit user intent" in result.operations[1]["error"]
    assert await harness.repository.list_documents() == []


async def test_reviewer_can_change_team_and_refresh_tools(harness):
    class Research:
        calls = 0

        async def read(self, url):
            self.calls += 1
            return {"url": url, "content": f"Fresh evidence {self.calls}"}

    read = Action(name="read_page", url="https://example.org")
    model = TeamModel(
        team(task("research", "old", read), task("knowledge", "initial")),
        team(task("career", "new", read), task("productivity", "replacement")),
        reviews=[
            Review(decision="need_evidence", issues=["Refresh evidence"]),
            Review(decision="accept"),
        ],
    )
    research = Research()
    result = await harness.runtime(model, research=research).service.chat(
        ChatRequest(text="Compare sources")
    )
    assert result.review_replans == 1 and research.calls == 2
    assert result.specialists == ["career", "productivity"]
    assert [item["task"] for item in model.contributions[1]] == ["new", "replacement"]
    review = [payload for name, payload in model.calls if name == "Review"][-1]
    assert len(review["evidence"]) == 1 and "Fresh evidence 2" in review["evidence"][0]["data"]
    for _, payload in model.calls:
        if payload.get("assigned_task") in {"new", "replacement"}:
            assert payload["revision_context"]["review_feedback"] == ["Refresh evidence"]


async def test_five_team_replans_fit_hard_graph_cap_and_do_not_reuse_stale_drafts(harness):
    harness.settings = harness.settings.model_copy(update={"max_review_replans": 5})
    model = TeamModel(
        *(
            team(task("general", f"general-{n}"), task("productivity", f"productivity-{n}"))
            for n in range(6)
        ),
        reviews=[Review(decision="revise", issues=["Still incomplete"])] * 6,
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Compare planning approaches")
    )
    assert result.review_replans == 5 and result.max_review_replans == 5
    assert "Re-planning limit reached (5)" in result.text
    assert len(model.contributions) == 6
    for index, contributions in enumerate(model.contributions):
        assert all(item["task"].endswith(f"-{index}") for item in contributions)


async def test_checkpoint_resume_preserves_successful_sibling(harness, tmp_path):
    successful = asyncio.Event()
    reads = Counter()

    class Research:
        async def read(self, url):
            reads[url] += 1
            return {"url": url, "content": "Read " + url}

    class FailingModel(TeamModel):
        async def contribute(self, payload):
            if payload["assigned_task"] == "fails":
                await successful.wait()
                await asyncio.sleep(0.05)
                raise ModelError("Interrupted agent")
            result = await super().contribute(payload)
            successful.set()
            return result

    request = ChatRequest(text="Compare these sources")
    path = str(tmp_path / "parallel.db")
    first = FailingModel(
        team(
            task("research", "works", Action(name="read_page", url="https://example.org/works")),
            task("career", "fails", Action(name="read_page", url="https://example.org/fails")),
        )
    )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        with pytest.raises(ModelError, match="Interrupted agent"):
            await harness.runtime(first, saver, research=Research()).service.chat(request)
    second = TeamModel()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        response = await harness.runtime(second, saver, research=Research()).service.chat(request)
    assert response.review_status == "accepted"
    assert reads == {"https://example.org/works": 1, "https://example.org/fails": 2}
    assert [
        payload["assigned_task"] for _, payload in second.calls if "assigned_task" in payload
    ] == ["fails"]
    assert len(second.contributions[0]) == 2


async def test_synthesis_retry_does_not_repeat_committed_team_tools(harness, tmp_path):
    class FailingSynthesis(TeamModel):
        async def generate(self, system, payload, schema):
            if "agent_contributions" in payload:
                raise ModelError("Interrupted synthesis")
            return await super().generate(system, payload, schema)

    plan = team(
        task("knowledge", "note", Action(name="save_note", title="Note", content="Supplied")),
        task("productivity", "task", Action(name="create_task", title="Task")),
    )
    request = ChatRequest(text="Create a note and a task: Note: Supplied; Task")
    path = str(tmp_path / "synthesis.db")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        with pytest.raises(ModelError, match="Interrupted synthesis"):
            await harness.runtime(FailingSynthesis(plan), saver).service.chat(request)
    assert len(await harness.repository.list_documents()) == 1
    assert len(await harness.repository.tasks()) == 1
    model = TeamModel()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        result = await harness.runtime(model, saver).service.chat(request)
    assert result.review_status == "accepted"
    assert [name for name, _ in model.calls] == ["Draft", "Review"]
    assert len(await harness.repository.list_documents()) == 1
    assert len(await harness.repository.tasks()) == 1


async def test_interrupted_tool_phase_replays_global_operation_ids(harness, tmp_path, monkeypatch):
    mutate = harness.repository.mutate
    interrupted = False

    async def fail_second_write(action, operation_id, source):
        nonlocal interrupted
        if action.name == "create_task" and not interrupted:
            interrupted = True
            raise RuntimeError("Process stopped between writes")
        return await mutate(action, operation_id, source)

    monkeypatch.setattr(harness.repository, "mutate", fail_second_write)
    plan = team(
        task("knowledge", "note", Action(name="save_note", title="Note", content="Supplied")),
        task("productivity", "task", Action(name="create_task", title="Task")),
    )
    request = ChatRequest(text="Create a note and a task: Note: Supplied; Task")
    path = str(tmp_path / "tool-phase.db")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        with pytest.raises(RuntimeError, match="between writes"):
            await harness.runtime(TeamModel(plan), saver).service.chat(request)
    assert len(await harness.repository.list_documents()) == 1
    assert await harness.repository.tasks() == []
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        response = await harness.runtime(TeamModel(), saver).service.chat(request)
    assert all(item["ok"] for item in response.operations)
    assert len(await harness.repository.list_documents()) == 1
    assert len(await harness.repository.tasks()) == 1


async def test_large_team_evidence_fits_synthesis_without_tokenizer(harness):
    class LongDrafts(TeamModel):
        synthesis = None

        async def generate(self, system, payload, schema):
            if "agent_contributions" in payload:
                self.synthesis = system, payload
            return await super().generate(system, payload, schema)

        async def contribute(self, payload):
            return Draft(answer="я" * 8000)

    class Research:
        async def read(self, url):
            return {"url": url, "content": "я" * 20000}

    model = LongDrafts(
        team(
            task("general", "general" * 100),
            task("knowledge", "knowledge" * 100),
            task("productivity", "productivity" * 80),
            task(
                "research", "research" * 100, Action(name="read_page", url="https://example.org/a")
            ),
            task("career", "career" * 100, Action(name="read_page", url="https://example.org/b")),
        )
    )
    await harness.runtime(model, research=Research()).service.chat(
        ChatRequest(text="Compare sources")
    )
    system, payload = model.synthesis
    assert all(item["truncated"] for item in payload["agent_contributions"])
    assert sum(len(item["draft"].encode()) for item in payload["agent_contributions"]) <= 3000
    assert sum(len(item["data"].encode()) for item in payload["operation_results"]) <= 3000
    real_model = HTTPModel(harness.settings)
    await real_model.client.aclose()
    real_model.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    try:
        assert await real_model.fit_request(system, payload, Draft)
    finally:
        await real_model.close()


async def test_replanning_can_switch_between_single_and_parallel(harness):
    from tests.conftest import ScriptedModel

    parallel = team(task("general"), task("productivity"))
    model = ScriptedModel(
        Plan(specialist="general", needs_review=True),
        Draft(answer="Initial"),
        Review(decision="revise", issues=["Use both perspectives"]),
        parallel,
        Draft(answer="General"),
        Draft(answer="Productivity"),
        Draft(answer="Combined"),
        Review(decision="revise", issues=["Simplify"]),
        Plan(specialist="general"),
        Draft(answer="Final"),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Compare planning approaches")
    )
    assert result.text == "Final" and result.review_replans == 2
    assert result.specialists == ["general"]


@pytest.mark.parametrize(
    "agents, actions",
    [
        ([task("general")] * 6, []),
        ([task("research"), task("knowledge")], []),
        ([task("general"), task("knowledge")], [Action(name="list_tasks")]),
        ([task("general", " "), task("knowledge")], []),
        (
            [
                task("general", "general", *[Action(name="list_tasks")] * 3),
                task("research", "research", *[Action(name="list_tasks")] * 3),
            ],
            [],
        ),
    ],
)
def test_invalid_team_plans_are_rejected(agents, actions):
    with pytest.raises(ValueError):
        Plan(specialist="general", agents=agents, actions=actions)


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_parallel_agents", 0),
        ("max_parallel_agents", 6),
        ("model_max_concurrency", 0),
        ("model_max_concurrency", 9),
    ],
)
def test_invalid_concurrency_limits_are_rejected(field, value):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})
