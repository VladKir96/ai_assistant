from dataclasses import dataclass

import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver

from personal_assistant.artifacts import ArtifactStore
from personal_assistant.config import Settings
from personal_assistant.context import ContextBuilder
from personal_assistant.database import Database
from personal_assistant.graph import AssistantGraph
from personal_assistant.llm import Embeddings
from personal_assistant.mcp_tools import MCPTools
from personal_assistant.repository import Repository
from personal_assistant.research import Research
from personal_assistant.runtime import Runtime
from personal_assistant.service import AssistantService


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, system, payload, schema):
        self.calls.append((system, payload, schema))
        assert self.responses, f"Unexpected model call: {schema.__name__}"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, schema), (type(response), schema)
        return response


@dataclass
class Harness:
    settings: Settings
    database: Database
    repository: Repository

    def runtime(
        self, model, checkpointer=None, research=None, weather=None, tracing=None, analysis=None
    ):
        graph = AssistantGraph(
            self.settings,
            model,
            self.repository,
            ContextBuilder(self.database, model),
            research or Research(self.settings),
            MCPTools(self.settings),
            weather=weather,
            tracing=tracing,
            analysis=analysis,
        ).compile(checkpointer or InMemorySaver())
        return Runtime(
            AssistantService(
                self.database,
                graph,
                self.settings.input_budget,
                tracing=tracing,
                artifacts=ArtifactStore(self.settings.artifact_directory),
            ),
            self.repository,
            self.database,
        )


@pytest_asyncio.fixture
async def harness(tmp_path):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        api_token="test-token-with-at-least-32-characters",
        model_provider="local",
        embedding_backend="disabled",
        mcp_servers_file=None,
        pdf_upload_directory=tmp_path / "pdf-uploads",
        artifact_directory=tmp_path / "artifacts",
    )
    database = Database(settings.database_url)
    embeddings = Embeddings(settings)
    await database.create_for_tests()
    try:
        yield Harness(settings, database, Repository(database, embeddings))
    finally:
        await embeddings.close()
        await database.close()
