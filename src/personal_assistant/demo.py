"""A deterministic application smoke test. It does not claim to evaluate a real model."""

from tempfile import TemporaryDirectory
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from personal_assistant.config import Settings
from personal_assistant.context import ContextBuilder
from personal_assistant.database import Database
from personal_assistant.graph import AssistantGraph, Draft
from personal_assistant.llm import Embeddings
from personal_assistant.mcp_tools import MCPTools
from personal_assistant.repository import Repository
from personal_assistant.research import Research
from personal_assistant.schemas import Action, ChatRequest, Plan
from personal_assistant.service import AssistantService


class DemoModel:
    async def generate(self, system, payload, schema):
        if schema is Plan:
            return Plan(
                specialist="knowledge",
                actions=[
                    Action(
                        name="save_note",
                        title="Assistant project",
                        content="Use English and Russian.",
                        tags=["demo"],
                    )
                ],
            )
        if schema is Draft:
            result = payload["operation_results"][0]
            assert result["ok"]
            return Draft(answer="Saved the bilingual project note / Заметка о проекте сохранена.")
        raise AssertionError(f"Unexpected demo call: {schema}")


async def run_demo():
    with TemporaryDirectory(prefix="assistant-demo-") as directory:
        settings = Settings(
            _env_file=None,
            database_url=f"sqlite+aiosqlite:///{directory}/demo.db",
            model_provider="local",
            embedding_backend="disabled",
            mcp_servers_file=None,
        )
        db = Database(settings.database_url)
        embeddings = Embeddings(settings)
        try:
            await db.create_for_tests()
            repository = Repository(db, embeddings)
            model = DemoModel()
            graph = AssistantGraph(
                settings,
                model,
                repository,
                ContextBuilder(db, model),
                Research(settings),
                MCPTools(settings),
            ).compile(InMemorySaver())
            service = AssistantService(db, graph, settings.input_budget)
            request = ChatRequest(
                request_id=uuid4(), text="Save a note: project uses English and Russian."
            )
            response = await service.chat(request)
            replay = await service.chat(request)
            assert response == replay
            assert len(await repository.list_documents("note")) == 1
            assert (await repository.search("English"))[0]["title"] == "Assistant project"
            print(response.text)
            print("PASS: LangGraph routing, note persistence, retrieval, and duplicate protection.")
            print("Deterministic demo only; no model, Telegram account, or internet was used.")
        finally:
            await embeddings.close()
            await db.close()
