"""One registry for action names, parameters, permissions and planner documentation."""

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fields: str
    required: str
    roles: str
    mutates: bool = False

    def argument_schema(self, action_schema: dict) -> dict:
        return {
            "type": "object",
            "properties": {name: action_schema["properties"][name] for name in self.fields.split()},
            "required": self.required.split(),
            "additionalProperties": False,
        }


SPECS = [
    ToolSpec("save_note", "title content tags", "content", "knowledge", True),
    ToolSpec(
        "update_note", "resource_id title content tags", "resource_id content", "knowledge", True
    ),
    ToolSpec("delete_note", "resource_id", "resource_id", "knowledge", True),
    ToolSpec("remember", "title content tags", "title content", "knowledge", True),
    ToolSpec("forget", "resource_id", "resource_id", "knowledge", True),
    ToolSpec("create_task", "title content due_at", "title", "productivity", True),
    ToolSpec("complete_task", "resource_id", "resource_id", "productivity", True),
    ToolSpec("list_tasks", "", "", "general productivity research career analysis"),
    ToolSpec(
        "search_knowledge",
        "query date_from date_to date_field",
        "",
        "general knowledge productivity research career analysis",
    ),
    ToolSpec(
        "search_history",
        "query date_from date_to date_field",
        "",
        "general knowledge productivity research career analysis",
    ),
    ToolSpec("read_note", "resource_id", "resource_id", "knowledge research career analysis"),
    ToolSpec("read_page", "url", "url", "research career analysis"),
    ToolSpec("search_web", "query", "query", "research career analysis"),
    ToolSpec("search_jobs", "query", "query", "research career"),
    ToolSpec(
        "mcp_read",
        "mcp_server mcp_tool mcp_arguments",
        "mcp_server mcp_tool",
        "knowledge productivity research career analysis",
    ),
    ToolSpec(
        "create_event",
        "title content starts_at ends_at event_timezone all_day",
        "title starts_at ends_at",
        "productivity",
        True,
    ),
    ToolSpec(
        "update_event",
        "resource_id title content starts_at ends_at event_timezone all_day",
        "resource_id",
        "productivity",
        True,
    ),
    ToolSpec("cancel_event", "resource_id", "resource_id", "productivity", True),
    ToolSpec(
        "list_events",
        "query date_from date_to",
        "",
        "general productivity research career analysis",
    ),
    ToolSpec(
        "read_event", "resource_id", "resource_id", "general productivity research career analysis"
    ),
    ToolSpec(
        "get_weather",
        "query weather_period hourly temperature_unit date_from date_to",
        "",
        "research career analysis",
    ),
    ToolSpec("run_python", "query analysis_id", "query", "analysis"),
]
TOOLS = {tool.name: tool for tool in SPECS}
ToolName = StrEnum("ToolName", {name: name for name in TOOLS})
MUTATIONS = {tool.name for tool in SPECS if tool.mutates}
READ_ACTIONS = set(TOOLS) - MUTATIONS
TOOLS_BY_ROLE = {
    role: {tool.name for tool in SPECS if role in tool.roles.split()}
    for role in ("general", "knowledge", "productivity", "research", "career", "analysis")
}
