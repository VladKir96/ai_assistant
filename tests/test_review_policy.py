from personal_assistant.graph import Draft
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from tests.conftest import ScriptedModel


async def test_knowledge_summary_cannot_skip_review_when_planner_omits_flag(harness):
    await harness.repository.mutate(
        Action(name="save_note", title="Project", content="Код проекта ЛИСА-7392."),
        "review-policy:seed",
        "test",
    )
    model = ScriptedModel(
        Plan(specialist="knowledge", needs_review=False),
        Draft(answer="Код проекта — ЛИСА-7392."),
        Review(decision="accept"),
    )
    result = await harness.runtime(model).service.chat(
        ChatRequest(text="Суммируй заметку Project, сохранив код.")
    )
    assert result.reviewed
    assert result.review_status == "accepted"
    assert model.calls[-1][2] is Review
    assert model.calls[-1][1]["knowledge"][0]["title"] == "Project"
