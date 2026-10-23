import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from personal_assistant.app import cli, local_start
from personal_assistant.app.local_start import Service, StartupError, Supervisor


@pytest.fixture
def configured(harness, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("pyproject.toml", "compose.yaml", ".env", "model.gguf"):
        (tmp_path / name).touch()
    monkeypatch.setattr(local_start, "executable", lambda *args: None)
    return harness.settings.model_copy(
        update={
            "database_url": "postgresql+psycopg://assistant:test-only@localhost:5432/assistant",
            "postgres_password": SecretStr("test-only"),
            "local_model_file": tmp_path / "model.gguf",
            "llama_executable": sys.executable,
            "telegram_token": SecretStr("12345:test-only"),
            "telegram_owner_id": 98765,
        }
    )


def test_plan_includes_enabled_companions_and_custom_ports(configured, tmp_path, monkeypatch):
    speech = tmp_path / "speech"
    speech.mkdir()
    (speech / "model.bin").touch()
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: object())
    settings = configured.model_copy(
        update={
            "voice_enabled": True,
            "voice_output_enabled": False,
            "stt_backend": "faster_whisper",
            "faster_whisper_model": str(speech),
            "embedding_backend": "http",
            "embedding_api_key": configured.api_token,
            "api_url": "http://localhost:18000",
        }
    )
    services = local_start.plan(settings)
    assert [item.name for item in services] == ["inference", "embeddings", "speech", "api"]
    assert services[-1].command == ["api", "--host", "127.0.0.1", "--port", "18000"]


def test_standard_installation_selects_all_features_from_settings(configured, tmp_path):
    speech = tmp_path / "speech"
    speech.mkdir()
    (speech / "model.bin").touch()
    voice = tmp_path / "voice.onnx"
    voice.touch()
    Path(str(voice) + ".json").touch()
    pdfs = tmp_path / "pdf"
    pdfs.mkdir()
    settings = configured.model_copy(
        update={
            "voice_enabled": True,
            "voice_output_enabled": True,
            "stt_backend": "faster_whisper",
            "faster_whisper_model": str(speech),
            "piper_en_model": str(voice),
            "pdf_enabled": True,
            "pdf_directory": pdfs,
            "python_enabled": True,
            "embedding_backend": "http",
            "embedding_api_key": configured.api_token,
        }
    )
    assert [service.name for service in local_start.plan(settings)] == [
        "python-runner",
        "inference",
        "embeddings",
        "speech",
        "api",
    ]
    disabled = settings.model_copy(
        update={
            "voice_enabled": False,
            "voice_output_enabled": False,
            "pdf_enabled": False,
            "python_enabled": False,
            "embedding_backend": "disabled",
        }
    )
    assert [service.name for service in local_start.plan(disabled)] == ["inference", "api"]


def test_cloud_profile_does_not_require_local_weights(configured):
    services = local_start.plan(
        configured.model_copy(update={"model_provider": "openai", "local_model_file": None})
    )
    assert [item.name for item in services] == ["api"]


def test_tts_only_startup_needs_no_whisper_and_accepts_serbian_voice(
    configured, tmp_path, monkeypatch
):
    model = tmp_path / "serbian.onnx"
    model.touch()
    settings = configured.model_copy(
        update={"voice_output_enabled": True, "voice_enabled": False, "piper_sr_model": str(model)}
    )
    checked = []
    monkeypatch.setattr(local_start, "executable", lambda value, label: checked.append(label))
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(StartupError, match="matching .onnx.json"):
        local_start.plan(settings)
    Path(str(model) + ".json").touch()
    assert [service.name for service in local_start.plan(settings)] == [
        "inference",
        "speech",
        "api",
    ]
    assert "Whisper" not in checked
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(StartupError, match="uv sync --frozen"):
        local_start.plan(settings)


def test_enabled_pdf_requires_installed_dependencies_and_directory(
    configured, tmp_path, monkeypatch
):
    settings = configured.model_copy(
        update={"pdf_enabled": True, "pdf_directory": tmp_path / "pdf"}
    )
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(StartupError, match="uv sync --frozen"):
        local_start.plan(settings)
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(StartupError, match="mkdir -p"):
        local_start.plan(settings)
    settings.pdf_directory.mkdir()
    assert [service.name for service in local_start.plan(settings)] == ["inference", "api"]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"telegram_token": SecretStr("invalid-secret")}, "TELEGRAM_TOKEN"),
        ({"telegram_owner_id": 0}, "TELEGRAM_OWNER_ID"),
        ({"telegram_owner_id": 12345}, "TELEGRAM_OWNER_ID"),
        ({"local_model_file": None}, "GGUF"),
        ({"postgres_password": SecretStr("different-secret")}, "matching"),
        ({"database_url": "invalid-secret"}, "DATABASE_URL"),
        ({"api_url": "http://localhost:8080"}, "distinct ports"),
        ({"api_url": "http://localhost:5432"}, "distinct ports"),
    ],
)
def test_invalid_configuration_fails_without_exposing_secrets(configured, changes, match):
    with pytest.raises(StartupError, match=match) as caught:
        local_start.plan(configured.model_copy(update=changes))
    assert "invalid-secret" not in str(caught.value)
    assert "different-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost:8000",
        "http://example.org",
        "http://user:secret@localhost",
        "http://localhost/wrong",
        "http://localhost:0",
    ],
)
def test_managed_endpoints_reject_unsupported_configuration(url):
    with pytest.raises(StartupError, match="loopback HTTP") as caught:
        local_start.endpoint("api", url, "")
    assert "secret" not in str(caught.value)


def test_no_bot_and_check_are_read_only(configured, monkeypatch, capsys):
    settings = configured.model_copy(update={"telegram_token": SecretStr("")})

    def forbidden(*args, **kwargs):
        pytest.fail("--check must not start services or probe occupied ports")

    monkeypatch.setattr(local_start, "docker_ready", forbidden)
    monkeypatch.setattr(local_start, "check_ports", forbidden)
    local_start.start(settings, no_bot=True, check=True)
    assert "inference, api." in capsys.readouterr().out
    assert not Path("data").exists()


def test_python_runner_joins_native_startup_when_enabled(configured, monkeypatch):
    monkeypatch.setattr(local_start.importlib.util, "find_spec", lambda name: object())
    settings = configured.model_copy(update={"python_enabled": True})
    services = local_start.plan(settings, no_bot=True)
    assert [service.name for service in services] == ["python-runner", "inference", "api"]
    assert services[0].port == 8766


def test_duplicate_launcher_lock_is_released_without_pid_files(tmp_path):
    path = tmp_path / "startup.lock"
    with local_start.launcher_lock(path):
        with pytest.raises(StartupError, match="Another assistant start"):
            with local_start.launcher_lock(path):
                pytest.fail("Concurrent launcher acquired the same lock")
    with local_start.launcher_lock(path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_port_conflict_reports_service_before_starting_anything(monkeypatch):
    class Occupied:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            raise OSError("occupied")

    monkeypatch.setattr(local_start.socket, "socket", lambda *args: Occupied())
    with pytest.raises(StartupError, match="api: port 8000 is unavailable"):
        local_start.check_ports([Service("api", "127.0.0.1", 8000, "http://localhost:8000/health")])


def test_readiness_retries_and_checks_authenticated_api_status(tmp_path, monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/v1/status"
        assert request.headers["Authorization"] == "Bearer testing-only"
        return httpx.Response(503 if len(requests) == 1 else 200, json={"database": "ready"})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(local_start.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(local_start.time, "sleep", lambda seconds: None)
    supervisor = Supervisor(tmp_path, 5)
    supervisor.wait_ready(
        Service("api", "127.0.0.1", 8000, "http://localhost:8000/health", "testing-only")
    )
    assert len(requests) == 2


def test_readiness_timeout_is_actionable_and_sanitized(tmp_path, monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    monkeypatch.setattr(local_start.httpx, "Client", lambda **kwargs: client)
    ticks = iter([0, 0, 2])
    monkeypatch.setattr(local_start.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(local_start.time, "sleep", lambda seconds: None)
    with pytest.raises(StartupError, match="api readiness timed out") as caught:
        Supervisor(tmp_path, 1).wait_ready(
            Service("api", "127.0.0.1", 8000, "http://localhost:8000/health", "private-secret")
        )
    assert "private-secret" not in str(caught.value)


def test_macos_starts_docker_desktop_and_waits_for_engine(tmp_path, monkeypatch):
    probes = iter([1, 1, 0])
    monkeypatch.setattr(local_start.sys, "platform", "darwin")
    monkeypatch.setattr(
        local_start.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, next(probes)),
    )
    monkeypatch.setattr(local_start.time, "sleep", lambda seconds: None)
    calls = []
    supervisor = Supervisor(tmp_path, 5)
    monkeypatch.setattr(supervisor, "command", lambda name, command: calls.append(command))
    local_start.docker_ready(supervisor)
    assert calls == [["open", "-a", "Docker"]]


def test_child_failure_logs_are_private_and_console_error_omits_output(tmp_path):
    supervisor = Supervisor(tmp_path, 5)
    try:
        with pytest.raises(StartupError, match="migrate failed") as caught:
            supervisor.command(
                "migrate", [sys.executable, "-c", "raise RuntimeError('private-secret')"]
            )
        assert "private-secret" not in str(caught.value)
        assert "private-secret" in (tmp_path / "migrate.log").read_text()
        assert stat.S_IMODE((tmp_path / "migrate.log").stat().st_mode) == 0o600
    finally:
        supervisor.stop()


def test_real_shutdown_reaps_parent_and_inference_style_subprocess(tmp_path):
    marker = tmp_path / "ready"
    program = """
import signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def stop(*args):
    child.wait(timeout=5)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    supervisor = Supervisor(tmp_path, 5)
    process = supervisor.launch("inference", [sys.executable, "-c", program, str(marker)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
    finally:
        supervisor.stop()
    assert process.poll() == 0
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


@pytest.mark.parametrize("failure", ["readiness", "context"])
def test_partial_start_failure_stops_started_native_processes(configured, monkeypatch, failure):
    instances = []
    service = Service("api", "127.0.0.1", 8000, "http://localhost:8000/health")

    class FailingSupervisor(Supervisor):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)

        def command(self, name, command):
            if name == "context-check" and failure == "context":
                raise StartupError("context failed")

        def launch(self, name, command):
            return super().launch(name, [sys.executable, "-c", "import time; time.sleep(60)"])

        def wait_ready(self, service):
            if service.name == "api":
                raise StartupError("readiness failed")

    inference = Service("inference", "127.0.0.1", 8080, "http://localhost:8080/health")
    monkeypatch.setattr(local_start, "plan", lambda *args, **kwargs: [inference, service])
    monkeypatch.setattr(local_start, "check_ports", lambda *args: None)
    monkeypatch.setattr(local_start, "docker_ready", lambda *args: None)
    monkeypatch.setattr(local_start, "Supervisor", FailingSupervisor)
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(StartupError, match=f"{failure} failed"):
        local_start.start(configured, no_bot=True)
    assert all(process.poll() is not None for _, process in instances[0].children)
    if failure == "context":
        assert [name for name, _ in instances[0].children] == ["inference"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_cli_start_forwards_options_and_exits_without_traceback(configured, monkeypatch):
    monkeypatch.setattr(cli, "Settings", lambda: configured)
    monkeypatch.setattr(
        sys, "argv", ["assistant", "start", "--no-bot", "--check", "--startup-timeout", "20"]
    )
    calls = []
    monkeypatch.setattr(local_start, "start", lambda settings, **kwargs: calls.append(kwargs))
    cli.main()
    assert calls == [{"no_bot": True, "check": True, "timeout": 20}]

    def fail(*args, **kwargs):
        raise StartupError("Port unavailable; stop previous service")

    monkeypatch.setattr(local_start, "start", fail)
    with pytest.raises(SystemExit, match="Port unavailable") as caught:
        cli.main()
    assert caught.value.__suppress_context__
