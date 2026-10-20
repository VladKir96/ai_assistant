from datetime import timedelta
from uuid import uuid4

import pytest

from personal_assistant.authorization import authorize_mutation
from personal_assistant.graph import Draft
from personal_assistant.jobs import claim_due
from personal_assistant.models import Task, now
from personal_assistant.schemas import Action, AgentTask, ChatRequest, Plan, Review
from personal_assistant.service import AssistantService
from personal_assistant.tool_execution import assign_analysis_identity, effective_results
from tests.conftest import ScriptedModel


@pytest.mark.parametrize(
    "name,text",
    [
        ("create_event", "Create a histogram of these values"),
        ("create_event", "Create a note about my meeting"),
        ("create_event", "Create a note about tasks and events"),
        ("create_event", "Создай заметку о задачах и встречах"),
        ("create_task", "Create a note about dogs and tasks"),
        ("create_event", "Schedule a task about an event"),
        ("create_event", 'Save a note: "Create an event tomorrow"'),
        ("create_event", 'Create a note "cats and create a meeting"'),
        ("create_event", 'Create a note "cats and create a meeting'),
        ("create_event", r'Create a note "cats \" and create a meeting"'),
        ("create_event", "Создай гистограмму встреч"),
        ("update_note", "Add 2 and 3"),
        ("update_event", "Change the chart colour"),
        ("save_note", "Save me time by explaining Docker"),
        ("save_note", "Note"),
        ("complete_task", "Finish explaining how Docker works"),
        ("remember", "Помнишь мой адрес?"),
        ("save_note", 'Explain the command "save a note"'),
    ],
)
def test_unrelated_verbs_and_quoted_content_do_not_authorize_writes(name, text):
    with pytest.raises(ValueError, match="explicit user intent"):
        authorize_mutation(Action(name=name, resource_id=str(uuid4())), text)


@pytest.mark.parametrize(
    "name,text",
    [
        ("save_note", "Save a note: supplied content"),
        ("save_note", "Сохрани заметку: мои данные"),
        ("create_task", "Remind me tomorrow to pay rent"),
        ("create_task", "Напомни завтра позвонить"),
        ("create_event", "Create another calendar event tomorrow"),
        ("create_event", "Запланируй встречу завтра"),
        ("remember", "Please, remember my city is Belgrade"),
        ("create_task", "Remember my city and remind me tomorrow to pay rent"),
        ("create_task", "Create a note and a task: supplied content"),
    ],
)
def test_explicit_bilingual_objects_and_multiple_requests_remain_supported(name, text):
    authorize_mutation(Action(name=name), text)


@pytest.mark.parametrize(
    "name,kind", [("update_note", "note"), ("update_event", "event"), ("complete_task", "task")]
)
def test_target_selector_is_bound_to_id_or_exact_unique_stored_title(name, kind):
    identifier = str(uuid4())
    action = Action(name=name, resource_id=identifier)
    verb = "Complete" if name == "complete_task" else "Update"
    target = {"title": "Project", "unique_title": True}
    authorize_mutation(action, f"{verb} {kind} {identifier}: new content", target)
    authorize_mutation(action, f'{verb} {kind} "Project": new content', target)
    authorize_mutation(action, f"{verb} the Project {kind} to: new content", target)
    for text in (
        f"{verb} {kind}: new content about Project {identifier}",
        f"{verb} {kind} Other with Project {identifier}",
        f"{verb} {kind} {uuid4()}: new content",
        f"{verb} {kind} project: new content",  # Exact title spelling is intentional.
    ):
        with pytest.raises(ValueError):
            authorize_mutation(action, text, target)
    with pytest.raises(ValueError):
        authorize_mutation(
            action, f"{verb} {kind} Project: content", {**target, "unique_title": False}
        )
    with pytest.raises(ValueError):
        authorize_mutation(action, f"{verb} {kind} Other", {"title": kind, "unique_title": True})


async def test_wrong_planner_cannot_overwrite_or_select_another_note(harness):
    first = await harness.repository.mutate(
        Action(name="save_note", title="Project", content="Keep"), "seed:first", "test"
    )
    second = await harness.repository.mutate(
        Action(name="save_note", title="Other", content="Keep too"), "seed:second", "test"
    )
    for text, note in (("Add 2 and 3", first), ("Update note Project: supplied content", second)):
        model = ScriptedModel(
            Plan(
                specialist="knowledge",
                actions=[Action(name="update_note", resource_id=note["id"], content="Wrong")],
            ),
            Draft(answer="No write applied."),
        )
        response = await harness.runtime(model).service.chat(ChatRequest(text=text))
        assert not response.operations[0]["ok"]
    assert (await harness.repository.get_document(first["id"]))["content"] == "Keep"
    assert (await harness.repository.get_document(second["id"]))["content"] == "Keep too"
    # A duplicate title is ambiguous even when the model supplies an existing ID.
    await harness.repository.mutate(
        Action(name="save_note", title="Project", content="Duplicate"), "seed:third", "test"
    )
    target = await harness.repository.mutation_target(
        Action(name="update_note", resource_id=first["id"])
    )
    assert not target["unique_title"]


async def test_authorized_rename_keeps_target_binding_during_review_repair(harness):
    note = await harness.repository.mutate(
        Action(name="save_note", title="Project", content="Old"), "rename:seed", "test"
    )

    def edit(content):
        return Plan(
            specialist="knowledge",
            needs_review=True,
            actions=[
                Action(name="update_note", resource_id=note["id"], title="Renamed", content=content)
            ],
        )

    model = ScriptedModel(
        edit("Wrong"),
        Draft(answer="Updated."),
        Review(decision="revise", issues=["Fix content"]),
        edit("Correct"),
        Draft(answer="Corrected."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model).service.chat(
        ChatRequest(text="Update the Project note: rename to Renamed; content Correct")
    )
    assert all(item["ok"] for item in response.operations)
    stored = await harness.repository.get_document(note["id"])
    assert stored["title"] == "Renamed" and stored["content"] == "Correct"


@pytest.mark.parametrize("kind", ["note", "task", "event"])
async def test_storage_backed_selectors_allow_explicit_bilingual_edits(harness, kind):
    if kind == "note":
        seeded = await harness.repository.mutate(
            Action(name="save_note", title="Project", content="Old"), "seed", "test"
        )
        action = Action(name="update_note", resource_id=seeded["id"], content="Correct")
        text, role = "Обнови заметку Project: Correct", "knowledge"
    elif kind == "task":
        seeded = await harness.repository.mutate(
            Action(name="create_task", title="Project"), "seed", "test"
        )
        action = Action(name="complete_task", resource_id=seeded["task"]["id"])
        text, role = "Complete task Project", "productivity"
    else:
        seeded = await harness.repository.mutate(
            Action(
                name="create_event",
                title="Project",
                starts_at="2030-11-10T10:00",
                ends_at="2030-11-10T11:00",
            ),
            "seed",
            "test",
        )
        action = Action(
            name="update_event",
            resource_id=seeded["event"]["id"],
            starts_at="2030-11-10T12:00",
            ends_at="2030-11-10T13:00",
        )
        text, role = "Перенеси событие Project на 12:00", "productivity"
    model = ScriptedModel(
        Plan(specialist=role, actions=[action], needs_review=True),
        Draft(answer="Updated."),
        Review(decision="accept"),
    )
    response = await harness.runtime(model).service.chat(ChatRequest(text=text))
    assert response.operations[0]["ok"] and response.review_status == "accepted"


async def test_expired_final_reminder_claim_reaches_failed_without_another_send(
    harness, monkeypatch
):
    from personal_assistant import jobs

    instant = now()
    async with harness.database.sessions.begin() as session:
        session.add(
            Task(
                id="last",
                title="Final attempt",
                due_at=instant - timedelta(minutes=1),
                notification_attempts=4,
            )
        )
    monkeypatch.setattr(jobs, "now", lambda: instant)
    assert [item["id"] for item in await claim_due(harness.database)] == ["last"]
    monkeypatch.setattr(jobs, "now", lambda: instant + timedelta(minutes=2))
    assert await claim_due(harness.database) == []
    async with harness.database.sessions() as session:
        assert (await session.get(Task, "last")).notification_status == "sending"
    monkeypatch.setattr(jobs, "now", lambda: instant + timedelta(minutes=4))
    assert await claim_due(harness.database) == []
    async with harness.database.sessions() as session:
        task = await session.get(Task, "last")
        assert task.notification_status == "failed" and task.notification_attempts == 5
        assert task.notification_claimed_at is None
    assert await claim_due(harness.database) == []


def analysis_result(identity, query, artifact, *, round=0, ok=True):
    return {
        "action": "run_python",
        "ok": ok,
        "round": round,
        "arguments": {"name": "run_python", "query": query, "analysis_id": identity},
        "result": {"artifacts": [{"id": artifact}] if artifact else []},
    }


@pytest.mark.parametrize("artifact,ok", [("B-fixed", True), (None, True), (None, False)])
def test_partial_chart_repair_supersedes_only_its_task_even_when_query_changes(artifact, ok):
    first = analysis_result("alpha", "Plot A", "A")
    old = analysis_result("beta", "Plot B", "B")
    replacement = analysis_result("beta", "Correct B units", artifact, round=1, ok=ok)
    results = [first, old, replacement]
    assert effective_results(results) == [first, replacement]
    assert [entry["id"] for entry in AssistantService.response_artifacts(results)] == (
        ["A", "B-fixed"] if artifact else ["A"]
    )


def test_ambiguous_analysis_repair_requires_identity_instead_of_replacing_other_tasks():
    results = [analysis_result("alpha", "Plot A", "A"), analysis_result("beta", "Plot B", "B")]
    state = {"revisions": 1, "results": results}
    with pytest.raises(ValueError, match="analysis_id"):
        assign_analysis_identity(Action(name="run_python", query="Fix a chart"), state, 0)
    same = Action(name="run_python", query="Plot B")
    assign_analysis_identity(same, state, 0)
    assert same.analysis_id == "beta"
    single = Action(name="run_python", query="Different correction query")
    assign_analysis_identity(single, {**state, "results": results[:1]}, 0)
    assert single.analysis_id == "alpha"
    with pytest.raises(ValueError, match="distinct analysis_id"):
        Plan(specialist="analysis", actions=[same, same])


def test_legacy_checkpoint_queries_can_be_repaired_without_losing_other_charts():
    first = analysis_result("alpha", "Plot A", "A")
    second = analysis_result("beta", "Plot B", "B")
    for item in (first, second):
        del item["arguments"]["analysis_id"]
    action = Action(name="run_python", query="Plot B")
    assign_analysis_identity(action, {"revisions": 1, "results": [first, second]}, 0)
    replacement = analysis_result(action.analysis_id, "Correct B", "B-fixed", round=1)
    assert [
        entry["id"] for entry in AssistantService.response_artifacts([first, second, replacement])
    ] == ["A", "B-fixed"]


async def test_parallel_charts_keep_successful_sibling_after_single_agent_repair_and_replay(
    harness,
):
    a, b, fixed = (
        {"id": char * 64, "filename": name + ".png"}
        for char, name in (("a", "A"), ("b", "B"), ("c", "B-fixed"))
    )

    class Model:
        plans = reviews = 0

        async def generate(self, system, payload, schema):
            if schema is Plan:
                self.plans += 1
                if self.plans == 1:
                    return Plan(
                        specialist="analysis",
                        agents=[
                            AgentTask(
                                specialist="analysis",
                                task=query,
                                actions=[Action(name="run_python", query=query)],
                            )
                            for query in ("Plot A", "Plot B")
                        ],
                    )
                previous = payload["revision_context"]["previous_actions"]
                assert [item["analysis_id"] for item in previous] == [
                    "analysis-0-0",
                    "analysis-0-1",
                ]
                return Plan(
                    specialist="analysis",
                    actions=[
                        Action(name="run_python", query="Fix B units", analysis_id="analysis-0-1")
                    ],
                )
            if schema is Review:
                self.reviews += 1
                return (
                    Review(decision="revise", issues=["Fix B units only"])
                    if self.reviews == 1
                    else Review(decision="accept")
                )
            assert schema is Draft
            return Draft(answer="Analysis results.")

    class Analysis:
        async def run(self, query, state):
            return {
                "status": "ok",
                "artifacts": [{"Plot A": a, "Plot B": b, "Fix B units": fixed}[query]],
            }

    active = harness.runtime(Model(), analysis=Analysis())
    request = ChatRequest(text="Plot A and B using two agents")
    response = await active.service.chat(request)
    assert [item.id for item in response.artifacts] == [a["id"], fixed["id"]]
    assert response.review_replans == 1 and response.review_status == "accepted"
    assert await active.service.chat(request) == response
