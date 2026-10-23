"""Tool execution is separate from routing, prompts, and response review."""

import hashlib
import json
from datetime import datetime

import aiohttp
import httpx

from personal_assistant.agent.authorization import TARGETED_WRITES, authorize_mutation
from personal_assistant.agent.tools import MUTATIONS
from personal_assistant.core.schemas import Action
from personal_assistant.core.tracing import Tracing
from personal_assistant.inference.llm import ModelError, clip


def analysis_identity(item: dict) -> str:
    arguments = item.get("arguments", {})
    # Old checkpoints have queries but no explicit identity. Keep those readable.
    return arguments.get("analysis_id") or (
        "legacy-" + hashlib.sha256(arguments.get("query", "").encode()).hexdigest()[:16]
    )


def effective_results(results: list[dict]) -> list[dict]:
    """Keep latest reads/analysis tasks while preserving every completed write."""

    def key(item, index):
        if item["action"] == "run_python":
            return "analysis:" + analysis_identity(item)
        return json.dumps(item["arguments"], sort_keys=True) if "arguments" in item else str(index)

    latest = {key(item, index): index for index, item in enumerate(results)}
    return [
        item
        for index, item in enumerate(results)
        if (item["ok"] and item["action"] in MUTATIONS) or latest[key(item, index)] == index
    ]


def assign_analysis_identity(action: Action, state: dict, index: int):
    if action.analysis_id:
        return
    cycle = state.get("revisions", 0)
    previous = [
        item
        for item in state.get("results", [])
        if item["action"] == "run_python" and item.get("round", 0) < cycle
    ]
    matching = {
        analysis_identity(item)
        for item in previous
        if item.get("arguments", {}).get("query") == action.query
    }
    identities = {analysis_identity(item) for item in previous}
    plan = state.get("plan", {})
    actions = plan.get("actions", []) or [
        raw for agent in plan.get("agents", []) for raw in agent["actions"]
    ]
    count = sum(raw["name"] == "run_python" for raw in actions) or 1
    if len(matching) == 1:
        action.analysis_id = matching.pop()
    elif len(identities) == 1 and count == 1:
        action.analysis_id = identities.pop()
    elif previous:
        raise ValueError(
            "Set analysis_id to the earlier task's ID for a repair, or a new ID for a new "
            "analysis; multiple earlier analyses make an omitted ID ambiguous"
        )
    else:
        reserved = {raw.get("analysis_id") for raw in actions}
        identity = f"analysis-{cycle}-{index}"
        while identity in reserved:
            identity += "-auto"
        action.analysis_id = identity


class ToolExecutor:
    def __init__(
        self, repository, research, mcp, weather, *, tracing: Tracing | None = None, analysis=None
    ):
        self.repository, self.research, self.mcp, self.weather = repository, research, mcp, weather
        self.tracing = tracing or Tracing()
        self.analysis = analysis

    async def execute(self, action: Action, state: dict, index: int) -> dict:
        with self.tracing.observation(
            "tool." + action.name,
            kind="tool",
            input=action.model_dump(mode="json", exclude_defaults=True),
            metadata={"review_round": state.get("revisions", 0), "action_index": index},
        ) as span:
            result = await self._execute(action, state, index)
            span.update(
                output=result,
                metadata={"ok": result["ok"], "reused": result.get("reused", False)},
                level="DEFAULT" if result["ok"] else "ERROR",
                status_message=None if result["ok"] else "Tool execution failed",
            )
            return result

    async def _execute(self, action: Action, state: dict, index: int) -> dict:
        try:
            target = None
            action.validate_arguments()
            if action.name == "run_python":
                assign_analysis_identity(action, state, index)
            if action.name in MUTATIONS:
                if action.name in TARGETED_WRITES:
                    # A prior authorized edit may have renamed this same resource.
                    # Preserve its application-verified selector for request recovery.
                    target = next(
                        (
                            item["authorization_target"]
                            for item in state.get("results", [])
                            if item["ok"]
                            and item["action"] == action.name
                            and item.get("arguments", {}).get("resource_id") == action.resource_id
                            and item.get("authorization_target")
                        ),
                        None,
                    )
                    if target is None:
                        target = await self.repository.mutation_target(action)
                authorize_mutation(action, state["text"], target)
                arguments = action.model_dump(mode="json", exclude_defaults=True)
                for previous in reversed(state.get("results", [])):
                    if not previous["ok"] or previous["action"] not in MUTATIONS:
                        continue
                    prior = previous.get("arguments", {})
                    if prior == arguments:
                        return {**previous, "reused": True}
                    # An intervening change to this target invalidates an older update
                    # result. Creates stay deduplicated rather than recreating resources.
                    if (action.resource_id and prior.get("resource_id") == action.resource_id) or (
                        action.name == "remember"
                        and prior.get("name") == "remember"
                        and prior.get("title") == action.title
                    ):
                        break
                cycle = state.get("revisions", 0)
                suffix = f"r{cycle}:{index}" if cycle else str(index)
                result = await self.repository.mutate(
                    action,
                    f"{state['request_id']}:{suffix}",
                    f"conversation:{state['thread_id']}/request:{state['request_id']}",
                )
            else:
                result = await getattr(self, action.name)(action, state)
            if action.name == "run_python" and result["status"] != "ok":
                return {
                    "action": str(action.name),
                    "ok": False,
                    "result": result,
                    "error": result.get("error") or "Python execution failed",
                }
            return {
                "action": str(action.name),
                "ok": True,
                "result": result,
                **({"authorization_target": target} if target else {}),
            }
        except (
            ValueError,
            LookupError,
            ModelError,
            httpx.HTTPError,
            aiohttp.ClientError,
            OSError,
            TimeoutError,
        ) as error:
            message = (
                str(error)
                if isinstance(error, (ValueError, ModelError))
                else "External service failed"
            )
            return {"action": str(action.name), "ok": False, "error": clip(message, 350)}

    async def search_knowledge(self, action, state):
        documents = await self.repository.search(
            action.query,
            date_from=action.date_from,
            date_to=action.date_to,
            date_field=action.date_field,
        )
        return {"documents": documents, "integration_warning": self.repository.retrieval_warning}

    async def search_history(self, action, state):
        if action.date_field != "created_at":
            raise ValueError("Conversation history supports created_at date filtering only")
        return {
            "messages": await self.repository.search_history(
                action.query,
                date_from=action.date_from,
                date_to=action.date_to,
            )
        }

    async def read_note(self, action, state):
        return await self.repository.get_document(action.resource_id)

    async def list_tasks(self, action, state):
        return {"tasks": await self.repository.tasks()}

    async def list_events(self, action, state):
        return {
            "events": await self.repository.calendar.events(
                action.query,
                date_from=action.date_from,
                date_to=action.date_to,
            )
        }

    async def read_event(self, action, state):
        return await self.repository.calendar.get(action.resource_id)

    async def get_weather(self, action, state):
        return await self.weather.forecast(action, datetime.fromisoformat(state["current_time"]))

    async def read_page(self, action, state):
        return await self.research.read(action.url)

    async def search_web(self, action, state):
        return await self.research.search_and_read(action.query)

    async def search_jobs(self, action, state):
        return await self.research.search(action.query, jobs=True)

    async def mcp_read(self, action, state):
        return await self.mcp.read(action.mcp_server, action.mcp_tool, action.mcp_arguments)

    async def run_python(self, action, state):
        if self.analysis is None:
            raise ValueError("Python analysis is unavailable")
        return await self.analysis.run(action.query, state)
