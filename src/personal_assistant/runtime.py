from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from personal_assistant.artifacts import ArtifactStore
from personal_assistant.config import Settings
from personal_assistant.context import ContextBuilder
from personal_assistant.database import Database
from personal_assistant.graph import AssistantGraph
from personal_assistant.llm import Embeddings, HTTPModel
from personal_assistant.mcp_tools import MCPTools
from personal_assistant.pdf_uploads import PDFUploads
from personal_assistant.repository import Repository
from personal_assistant.research import Research
from personal_assistant.service import AssistantService
from personal_assistant.tracing import Tracing
from personal_assistant.vision import VisionAnalyzer


@dataclass
class Runtime:
    service: AssistantService
    repository: Repository
    database: Database


@asynccontextmanager
async def runtime(settings: Settings):
    db = Database(settings.database_url)
    tracing = Tracing(settings)
    model = HTTPModel(settings, tracing=tracing)
    embeddings = Embeddings(settings)
    try:
        async with AsyncExitStack() as stack:
            if settings.database_url.startswith("sqlite"):
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                await db.create_for_tests()
                checkpoint = await stack.enter_async_context(
                    AsyncSqliteSaver.from_conn_string(":memory:")
                )
            else:
                await db.ping()
                checkpoint = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(
                        settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
                    )
                )
            repository = Repository(db, embeddings)
            mcp = MCPTools(settings)
            graph = AssistantGraph(
                settings,
                model,
                repository,
                ContextBuilder(db, model),
                Research(settings),
                mcp,
                tracing=tracing,
            ).compile(checkpoint)
            yield Runtime(
                AssistantService(
                    db,
                    graph,
                    settings.input_budget,
                    VisionAnalyzer(settings, model),
                    PDFUploads(settings, mcp),
                    tracing=tracing,
                    artifacts=ArtifactStore(settings.artifact_directory),
                ),
                repository,
                db,
            )
    finally:
        # Ensure each cleanup runs even if another resource fails to close.
        async with AsyncExitStack() as cleanup:
            cleanup.push_async_callback(tracing.close)
            cleanup.push_async_callback(db.close)
            cleanup.push_async_callback(embeddings.close)
            cleanup.push_async_callback(model.close)
