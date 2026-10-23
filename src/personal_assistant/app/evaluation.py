import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from personal_assistant.agent.graph import PLANNER, TIME_CAPABILITIES
from personal_assistant.core.config import Settings
from personal_assistant.core.schemas import Plan
from personal_assistant.inference.llm import HTTPModel
from personal_assistant.integrations.research import web_capabilities


async def evaluate(output: Path):
    settings = Settings()
    cases = json.loads(await asyncio.to_thread(Path("evals/planner_cases.json").read_text))
    model = HTTPModel(settings)
    results = []
    try:
        for case in cases:
            started = time.monotonic()
            result = await model.generate(
                PLANNER,
                {
                    "latest_request": case["request"],
                    "context": {},
                    "knowledge": [],
                    "open_tasks": [],
                    "calendar_events": [],
                    "time_capabilities": TIME_CAPABILITIES.copy(),
                    "mcp_read_tools": [],
                    "job_sites": [],
                    # Cases can simulate configured tools without credentials or tool calls.
                    "web_capabilities": {
                        **web_capabilities(settings),
                        **case.get("web_capabilities", {}),
                    },
                    "current_time": "2026-10-16T12:00:00+02:00",
                    "timezone": "Europe/Belgrade",
                },
                Plan,
            )
            actions = [action.name for action in result.actions]
            results.append(
                {
                    **case,
                    "actual_specialist": result.specialist,
                    "actual_actions": actions,
                    "passed": result.specialist == case["specialist"]
                    and actions == case["actions"],
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            )
    finally:
        await model.close()
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "provider": settings.model_provider,
        "model": settings.local_model
        if settings.model_provider == "local"
        else settings.openai_model,
        "scope": "Planner routing and tool selection only; no actions executed",
        "cases": results,
    }

    def write_report():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    await asyncio.to_thread(write_report)
    passed = sum(item["passed"] for item in results)
    print(f"Planner evaluation: {passed}/{len(results)} passed. Report: {output}")
    if passed != len(results):
        raise SystemExit(1)
