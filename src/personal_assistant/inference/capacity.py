"""Check the complete minimum planner request against the configured model."""

from personal_assistant.agent.graph import PLANNER, TIME_CAPABILITIES
from personal_assistant.core.config import Settings
from personal_assistant.core.schemas import Plan
from personal_assistant.core.time_utils import clock_context, with_runtime_clock
from personal_assistant.inference.llm import ContextLimitError, HTTPModel, ModelError
from personal_assistant.integrations.mcp_tools import MCPTools
from personal_assistant.integrations.research import web_capabilities


async def check_capacity(settings: Settings):
    tools, model = MCPTools(settings), HTTPModel(settings)
    try:
        payload = {
            "latest_request": "Hello",
            "context": {"summary": "", "recent_messages": []},
            "knowledge": [],
            "open_tasks": [],
            "calendar_events": [],
            **clock_context(settings.timezone),
            "time_capabilities": TIME_CAPABILITIES,
            "web_capabilities": web_capabilities(settings),
            "python_available": settings.python_enabled,
            "job_sites": settings.job_sites,
            "mcp_read_tools": await tools.catalog(),
        }
        await model.fit_request(with_runtime_clock(PLANNER, payload), payload, Plan)
    except ContextLimitError:
        raise ModelError(
            "The configured context cannot hold the planner, tool catalog and clock. "
            "Enable the provider token counter or select a larger matching model context."
        ) from None
    finally:
        await model.close()
