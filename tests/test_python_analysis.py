import asyncio
import base64
import io
import json
import os
import stat
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from personal_assistant.api import create_app
from personal_assistant.artifacts import ArtifactStore
from personal_assistant.config import Settings
from personal_assistant.graph import Draft
from personal_assistant.python_analysis import PythonAnalysis, PythonProgram, runner_token
from personal_assistant.python_runner import (
    DockerExecutor,
    create_python_runner_app,
)
from personal_assistant.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.service import AssistantService
from tests.conftest import ScriptedModel


def png():
    data = io.BytesIO()
    Image.new("RGB", (80, 50), "white").save(data, format="PNG")
    return data.getvalue()


def test_single_task_team_syntax_normalizes_without_losing_actions():
    from personal_assistant.schemas import AgentTask

    action = Action(name="run_python", query="Calculate 2+2")
    plan = Plan(
        specialist="analysis",
        agents=[AgentTask(specialist="analysis", task="Calculate", actions=[action])],
    )
    assert plan.actions == [action] and not plan.agents


def test_invalid_runner_configuration_does_not_expose_credentials():
    with pytest.raises(ValueError) as caught:
        Settings(
            _env_file=None,
            api_token="private-marker-must-not-leak",
            python_enabled=True,
            python_runner_url="file:///tmp/runner",
        )
    assert "private-marker" not in str(caught.value)


def test_artifacts_are_validated_private_immutable_and_cannot_escape(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.save("../../chart.png\n", png())
    assert artifact.filename == "chart.png"
    path = store.directory / f"{artifact.id}.png"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert store.save("another.png", png()).id == artifact.id
    assert store.read(artifact.id).startswith(b"\x89PNG")
    assert not list(store.directory.glob(".chart-*"))
    with pytest.raises(FileNotFoundError):
        store.read("../../secret")
    with pytest.raises(ValueError):
        store.save("fake.png", b"<svg/>")
    path.unlink()
    target = tmp_path / "outside.png"
    target.write_bytes(png())
    path.symlink_to(target)
    with pytest.raises(OSError):
        store.read(artifact.id)


def test_chart_metadata_is_removed(tmp_path):
    from PIL.PngImagePlugin import PngInfo

    metadata = PngInfo()
    metadata.add_text("private", "hidden metadata")
    raw = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(raw, format="PNG", pnginfo=metadata)
    store = ArtifactStore(tmp_path)
    artifact = store.save("plot.png", raw.getvalue())
    with Image.open(io.BytesIO(store.read(artifact.id))) as result:
        assert "private" not in result.info


async def test_runner_authentication_limits_and_scoped_token(harness):
    class Executor:
        calls = []

        def health(self):
            pass

        def execute(self, code):
            self.calls.append(code)
            return {"status": "ok", "stdout": "42", "artifacts": []}

    executor = Executor()
    app = create_python_runner_app(harness.settings, executor)
    assert runner_token(harness.settings) != harness.settings.api_token.get_secret_value()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runner"
        ) as client,
    ):
        assert (await client.get("/health")).status_code == 200
        assert (await client.post("/v1/execute", content=b"bad-json")).status_code == 401
        headers = {"Authorization": "Bearer " + runner_token(harness.settings)}
        assert (
            await client.post("/v1/execute", headers=headers, content=b"x" * 65537)
        ).status_code == 413
        assert (
            await client.post("/v1/execute", headers=headers, json={"code": "x" * 12001})
        ).status_code == 422
        result = await client.post("/v1/execute", headers=headers, json={"code": "print(42)"})
        assert result.json()["stdout"] == "42"
    assert executor.calls == ["print(42)"]


async def test_runner_cancellation_does_not_release_live_job_slot(harness):
    import threading

    entered, finish = threading.Event(), threading.Event()

    class Executor:
        def execute(self, code):
            entered.set()
            finish.wait(5)
            return {"status": "ok", "artifacts": []}

    app = create_python_runner_app(harness.settings, Executor())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runner"
    ) as client:
        headers = {"Authorization": "Bearer " + runner_token(harness.settings)}
        first = asyncio.create_task(client.post("/v1/execute", headers=headers, json={"code": "1"}))
        assert await asyncio.to_thread(entered.wait, 2)
        first.cancel()
        try:
            second = await client.post("/v1/execute", headers=headers, json={"code": "2"})
            assert second.status_code == 429
        finally:
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await first


async def test_code_generation_runner_and_png_validation(harness, monkeypatch):
    from personal_assistant import python_analysis as module

    settings = harness.settings.model_copy(update={"python_enabled": True})
    model = ScriptedModel(PythonProgram(code="print(6*7); plt.hist([1,2,2,3])"))

    async def handle(request):
        assert request.headers["Authorization"] == "Bearer " + runner_token(settings)
        assert json.loads(request.content)["code"].startswith("print(6*7)")
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "stdout": "42\n",
                "artifacts": [
                    {"filename": "histogram.png", "data": base64.b64encode(png()).decode()}
                ],
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    analysis = PythonAnalysis(settings, model)
    result = await analysis.run(
        "Calculate and plot",
        {
            "text": "Calculate 6*7 and plot [1,2,2,3]",
            "current_time": "2030-11-10T10:00:00+01:00",
            "timezone": "Europe/Belgrade",
        },
    )
    assert model.calls[0][1]["current_time"] == "2030-11-10T10:00:00+01:00"
    assert result["stdout"] == "42\n"
    assert len(result["artifacts"]) == 1
    assert analysis.artifacts.read(result["artifacts"][0]["id"])


async def test_disabled_analysis_never_generates_or_executes(harness):
    model = ScriptedModel()
    with pytest.raises(ValueError, match="disabled"):
        await PythonAnalysis(harness.settings, model).run("Calculate", {"text": "1+1"})
    assert not model.calls


async def test_telegram_returns_chart_photos_only_to_owner(harness, monkeypatch):
    from aiogram.types import Update
    from pydantic import SecretStr

    from personal_assistant import bot

    identifier = "a" * 64
    requests, replies = [], []
    settings = harness.settings.model_copy(
        update={
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
        }
    )

    async def handle(request):
        requests.append(request.url.path)
        if request.url.path == "/v1/chat":
            return httpx.Response(
                200,
                json={
                    "text": "42",
                    "artifacts": [
                        {"id": identifier, "filename": "chart.png", "media_type": "image/png"}
                    ],
                },
            )
        assert request.url.path == "/v1/artifacts/" + identifier
        return httpx.Response(200, content=png())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        bot.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def call(self, method, request_timeout=None):
        replies.append(method)
        return True

    async def polling(dispatcher, instance, **kwargs):
        for index, user in enumerate((98765, 11111)):
            await dispatcher.feed_update(
                instance,
                Update.model_validate(
                    {
                        "update_id": index,
                        "message": {
                            "message_id": index + 1,
                            "date": 0,
                            "from": {"id": user, "is_bot": False, "first_name": "Test"},
                            "chat": {"id": user, "type": "private"},
                            "text": "Calculate and plot",
                        },
                    }
                ),
            )

    monkeypatch.setattr(bot.Bot, "__call__", call)
    monkeypatch.setattr(bot.Dispatcher, "start_polling", polling)
    await bot.run_bot(settings)
    assert requests == ["/v1/chat", "/v1/artifacts/" + identifier]
    photos = [reply for reply in replies if type(reply).__name__ == "SendPhoto"]
    assert len(photos) == 1 and photos[0].photo.data == png()
    assert all(reply.chat_id == 98765 for reply in replies)


async def test_analysis_data_dependency_review_repair_and_chart_api(harness):
    artifact = ArtifactStore(harness.settings.artifact_directory).save("chart.png", png())
    calls = []

    class Analysis:
        async def run(self, query, state):
            calls.append(state)
            if len(calls) == 1:
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": "ZeroDivisionError",
                    "error": "Correct code",
                    "artifacts": [],
                }
            return {"status": "ok", "stdout": "42", "artifacts": [artifact.model_dump()]}

    model = ScriptedModel(
        Plan(
            specialist="analysis",
            actions=[Action(name="list_events"), Action(name="run_python", query="Calculate")],
        ),
        Draft(answer="Calculation failed"),
        Review(decision="revise", issues=["Fix the division"]),
        Plan(specialist="analysis", actions=[Action(name="run_python", query="Calculate")]),
        Draft(answer="Result: 42"),
        Review(decision="accept"),
    )
    active = harness.runtime(model, analysis=Analysis())
    request = ChatRequest(text="Calculate and plot")
    response = await active.service.chat(request)
    assert response.review_replans == 1 and response.review_status == "accepted"
    assert response.artifacts == [artifact]
    assert calls[0]["results"][0]["action"] == "list_events"
    assert calls[1]["feedback"] == ["Fix the division"]
    assert await active.service.chat(request) == response
    assert len(calls) == 2
    app = create_app(harness.settings, active)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client,
    ):
        route = "/v1/artifacts/" + artifact.id
        assert (await client.get(route)).status_code == 401
        headers = {"Authorization": "Bearer " + harness.settings.api_token.get_secret_value()}
        result = await client.get(route, headers=headers)
        assert result.headers["content-type"] == "image/png"
        assert result.content.startswith(b"\x89PNG")
        assert (await client.get("/v1/artifacts/invalid", headers=headers)).status_code == 404


def test_new_failed_chart_attempt_does_not_serve_stale_images():
    old = {"action": "run_python", "ok": True, "round": 0, "result": {"artifacts": [{"id": "old"}]}}
    assert AssistantService.response_artifacts([old])[0]["id"] == "old"
    assert not AssistantService.response_artifacts(
        [old, {"action": "run_python", "ok": False, "round": 1}]
    )


def test_docker_configuration_has_no_network_host_mounts_or_credentials(monkeypatch):
    from personal_assistant import python_runner as module

    recorded = {}

    class Container:
        id = "test-job"
        removed = False

        def start(self):
            pass

        def remove(self, force):
            self.removed = force

    container = Container()

    def create(image, **kwargs):
        recorded.update(kwargs)
        return container

    def create_exec(*args, **kwargs):
        recorded["exec_user"] = kwargs["user"]
        recorded["exec_count"] = recorded.get("exec_count", 0) + 1
        return {"Id": str(recorded["exec_count"])}

    api = SimpleNamespace(exec_create=create_exec, exec_inspect=lambda *args: {"ExitCode": 0})

    def stream(identifier, **kwargs):
        yield (b"42\n" if identifier == "1" else b'{"artifacts":[],"omitted_artifacts":0}'), None

    api.exec_start = stream
    client = SimpleNamespace(containers=SimpleNamespace(create=create), api=api, close=lambda: None)
    monkeypatch.setattr(module, "docker_client", lambda: client)
    result = DockerExecutor("trusted-image").execute("print(42)")
    assert result["status"] == "ok" and result["stdout"] == "42\n"
    assert recorded["network_mode"] == "none" and recorded["read_only"]
    assert recorded["cap_drop"] == ["ALL"]
    assert recorded["exec_user"] == "65534:65534"
    assert recorded["mem_limit"] == recorded["memswap_limit"] == "512m"
    assert recorded["pids_limit"] == 32
    assert "volumes" not in recorded and "mounts" not in recorded
    assert set(recorded["environment"]) == {
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    }
    assert container.removed


@pytest.mark.skipif(
    os.getenv("TEST_PYTHON_DOCKER") != "1",
    reason="Set TEST_PYTHON_DOCKER=1 after building Dockerfile.python",
)
def test_real_docker_calculation_chart_and_isolation(tmp_path):
    executor = DockerExecutor("personal-assistant-python:local")
    executor.health()
    result = executor.execute(
        "import os, socket\n"
        "print(6*7, os.getuid())\n"
        "assert not os.path.exists('/app/.env')\n"
        "assert not os.path.exists('/var/run/docker.sock')\n"
        "assert not any(k.startswith('ASSISTANT_') for k in os.environ)\n"
        "try:\n socket.create_connection(('1.1.1.1',443), timeout=1)\n"
        "except OSError:\n print('network blocked')\n"
        "else:\n raise AssertionError('network reachable')\n"
        "plt.hist([1,2,2,3,4]); plt.title('Тест / Test'); plt.xlabel('Value')\n"
    )
    assert result["status"] == "ok", result
    assert "42 65534" in result["stdout"] and "network blocked" in result["stdout"]
    assert len(result["artifacts"]) == 1
    chart = result["artifacts"][0]
    artifact = ArtifactStore(tmp_path).save(chart["filename"], base64.b64decode(chart["data"]))
    assert ArtifactStore(tmp_path).read(artifact.id)


@pytest.mark.skipif(
    os.getenv("TEST_PYTHON_DOCKER") != "1", reason="Enable real Docker sandbox tests"
)
@pytest.mark.parametrize(
    "code, message",
    [
        ("import time; time.sleep(100)", "timed out"),
        ("print('x'*100000)", "exceeded"),
        ("print(1/0)", "code failed"),
        ("import os; os.symlink('/etc/passwd', '/work/leak.png')", "collection failed"),
    ],
)
def test_real_docker_resource_errors_are_bounded_and_cleaned(monkeypatch, code, message):
    from personal_assistant import python_runner as module

    monkeypatch.setattr(module, "TIMEOUT", 3)
    result = DockerExecutor("personal-assistant-python:local").execute(code)
    assert result["status"] == "error" and message in result["error"], result
    client = module.docker_client()
    try:
        assert not client.containers.list(
            all=True, filters={"label": "personal-assistant=python-job"}
        )
    finally:
        client.close()
