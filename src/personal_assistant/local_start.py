"""Foreground supervisor for the native Mac/Linux development stack."""

import errno
import fcntl
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from personal_assistant.config import Settings

LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class StartupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Service:
    name: str
    host: str
    port: int
    health_url: str
    health_key: str = ""

    @property
    def command(self) -> list[str]:
        return [self.name, "--host", self.host, "--port", str(self.port)]


def endpoint(name: str, url: str, path: str, key: str = "") -> Service:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in LOOPBACK
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != path
        ):
            raise ValueError
        port = 80 if parsed.port is None else parsed.port
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise StartupError(
            f"{name}: native start requires a loopback HTTP URL with path '{path or '/'}'. "
            "Use the manual startup guide for external services."
        ) from None
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    authority = f"[{host}]" if ":" in host else host
    return Service(name, host, port, f"http://{authority}:{port}/health", key)


def executable(value: str, label: str):
    if not shutil.which(value):
        raise StartupError(f"{label} executable is missing; check its path in .env.")


def plan(settings: Settings, *, no_bot: bool = False) -> list[Service]:
    if not all(Path(name).is_file() for name in ("pyproject.toml", "compose.yaml", ".env")):
        raise StartupError("Run from the project directory after first-time setup (.env required).")
    if len(settings.api_token.get_secret_value()) < 24:
        raise StartupError("Set a private ASSISTANT_API_TOKEN; complete assistant init first.")
    try:
        database = make_url(settings.database_url)
    except ArgumentError:
        raise StartupError("Invalid ASSISTANT_DATABASE_URL; see docs/startup.md.") from None
    if (
        database.drivername != "postgresql+psycopg"
        or database.host not in LOOPBACK
        or (5432 if database.port is None else database.port) != 5432
        or database.username != "assistant"
        or database.database != "assistant"
        or not settings.postgres_password.get_secret_value()
        or database.password != settings.postgres_password.get_secret_value()
    ):
        raise StartupError(
            "Native start expects Compose PostgreSQL: assistant database/user on localhost:5432, "
            "with DATABASE_URL and POSTGRES_PASSWORD matching. See docs/startup.md."
        )
    executable("docker", "Docker")
    if settings.pdf_enabled:
        if any(importlib.util.find_spec(name) is None for name in ("pdf_server", "pymupdf")):
            raise StartupError("PDF reading enabled: add --extra pdf to uv run.")
        if not settings.pdf_directory.is_dir():
            raise StartupError("Create ASSISTANT_PDF_DIRECTORY (default: mkdir -p data/pdf).")
    services = []
    if settings.python_enabled:
        if importlib.util.find_spec("docker") is None:
            raise StartupError("Python analysis enabled: add --extra python-runner to uv run.")
        services.append(endpoint("python-runner", settings.python_runner_url, ""))
    if settings.model_provider == "local":
        if not settings.local_model_file or not settings.local_model_file.is_file():
            raise StartupError("Set ASSISTANT_LOCAL_MODEL_FILE to an existing GGUF file.")
        if settings.model_supports_images and (
            not settings.local_mmproj_file or not settings.local_mmproj_file.is_file()
        ):
            raise StartupError("Local vision needs an existing ASSISTANT_LOCAL_MMPROJ_FILE.")
        executable(settings.llama_executable, "llama-server")
        services.append(
            endpoint(
                "inference",
                settings.local_base_url,
                "/v1",
                settings.local_api_key.get_secret_value(),
            )
        )
    if settings.embedding_backend == "http":
        if importlib.util.find_spec("sentence_transformers") is None:
            raise StartupError("Embeddings enabled: add --extra embeddings to uv run.")
        if settings.embedding_api_key.get_secret_value() != settings.api_token.get_secret_value():
            raise StartupError("Native embeddings require EMBEDDING_API_KEY to match API_TOKEN.")
        services.append(endpoint("embeddings", settings.embedding_base_url, "/v1"))
    if settings.voice_enabled:
        if settings.stt_backend == "faster_whisper":
            if importlib.util.find_spec("faster_whisper") is None:
                raise StartupError("Speech enabled: use uv run --extra speech assistant start.")
            if not (Path(settings.faster_whisper_model) / "model.bin").is_file():
                raise StartupError(
                    "Download the faster-whisper model first; see README.md voice setup."
                )
        else:
            executable(settings.whisper_executable, "Whisper")
            executable(settings.ffmpeg_executable, "FFmpeg")
            if not settings.whisper_model or not Path(settings.whisper_model).is_file():
                raise StartupError("Configure an existing ASSISTANT_WHISPER_MODEL first.")
        if settings.voice_output_enabled:
            executable(settings.piper_executable, "Piper")
            executable(settings.ffmpeg_executable, "FFmpeg")
            if not any(
                value and Path(value).is_file()
                for value in (settings.piper_en_model, settings.piper_ru_model)
            ):
                raise StartupError("Configure a Piper voice or set VOICE_OUTPUT_ENABLED=false.")
        services.append(endpoint("speech", settings.speech_base_url, ""))
    services.append(endpoint("api", settings.api_url, "", settings.api_token.get_secret_value()))
    if not no_bot:
        from aiogram.utils.token import TokenValidationError, validate_token

        token = settings.telegram_token.get_secret_value()
        try:
            validate_token(token)
        except TokenValidationError:
            raise StartupError("Set a valid ASSISTANT_TELEGRAM_TOKEN, or use --no-bot.") from None
        if settings.telegram_owner_id <= 0 or settings.telegram_owner_id == int(
            token.partition(":")[0]
        ):
            raise StartupError(
                "Set your personal TELEGRAM_OWNER_ID; discover it with assistant bot and /id."
            )
    ports = [service.port for service in services]
    if len(ports) != len(set(ports)) or 5432 in ports:
        raise StartupError("Native services need distinct ports, separate from PostgreSQL's 5432.")
    return services


@contextmanager
def launcher_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        path.chmod(0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StartupError("Another assistant start is running in this project.") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_ports(services: list[Service]):
    for service in services:
        family = socket.AF_INET6 if ":" in service.host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((service.host, service.port))
            except OSError as error:
                if error.errno not in {None, errno.EADDRINUSE}:
                    raise StartupError(
                        f"Cannot check {service.name} port {service.port}; "
                        "check local socket permissions."
                    ) from None
                raise StartupError(
                    f"{service.name}: port {service.port} is unavailable. Stop the previous "
                    "service in its original terminal before using assistant start."
                ) from None


class Supervisor:
    def __init__(self, logs: Path, timeout: float):
        self.logs, self.timeout = logs, timeout
        self.children: list[tuple[str, subprocess.Popen]] = []

    def launch(self, name: str, command: list[str]) -> subprocess.Popen:
        target = self.logs / f"{name}.log"
        with target.open("ab") as output:
            target.chmod(0o600)
            try:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                raise StartupError(
                    f"Could not start {name}; check its executable/configuration."
                ) from None
        self.children.append((name, process))
        return process

    def check_children(self):
        for name, process in self.children:
            if process.poll() is not None:
                raise StartupError(f"{name} exited; inspect {self.logs / (name + '.log')}.")

    def command(self, name: str, command: list[str]):
        process = self.launch(name, command)
        try:
            code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise StartupError(
                f"{name} timed out; inspect {self.logs / (name + '.log')}."
            ) from None
        self.children.remove((name, process))
        if code:
            raise StartupError(f"{name} failed; inspect {self.logs / (name + '.log')}.")
        self.check_children()

    def wait_ready(self, service: Service):
        deadline = time.monotonic() + self.timeout
        headers = {"Authorization": f"Bearer {service.health_key}"} if service.health_key else {}
        path = "/v1/status" if service.name == "api" else "/health"
        url = service.health_url.removesuffix("/health") + path
        with httpx.Client(timeout=2, trust_env=False) as client:
            while time.monotonic() < deadline:
                self.check_children()
                try:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    if (
                        data.get("database") == "ready"
                        if service.name == "api"
                        else data.get("status") == "ok"
                    ):
                        print(f"  {service.name} ready on port {service.port}", flush=True)
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(0.5)
        raise StartupError(
            f"{service.name} readiness timed out; inspect {self.logs / (service.name + '.log')}."
        )

    def stop(self):
        # Each native service has its own process group, including its subprocesses.
        for _, process in reversed(self.children):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        for _, process in reversed(self.children):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def docker_ready(supervisor: Supervisor):
    def ready():
        try:
            return (
                subprocess.run(
                    ["docker", "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).returncode
                == 0
            )
        except subprocess.TimeoutExpired:
            return False

    if ready():
        return
    if sys.platform != "darwin":
        raise StartupError("Start the Docker engine, then retry assistant start.")
    print("Starting Docker Desktop; waiting for its engine...", flush=True)
    supervisor.command("docker-desktop", ["open", "-a", "Docker"])
    deadline = time.monotonic() + supervisor.timeout
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(1)
    raise StartupError("Docker Desktop did not become ready. Open Docker Desktop and retry.")


def start(settings: Settings, *, no_bot: bool = False, check: bool = False, timeout: int = 300):
    services = plan(settings, no_bot=no_bot)
    if check:
        print(
            "Configuration valid. Native services: "
            + ", ".join(service.name for service in services)
            + ("." if no_bot else ", bot.")
        )
        print("No services started; Docker readiness and occupied ports were not checked.")
        return
    with launcher_lock(Path("data/local-start.lock")):
        check_ports(services)
        logs = Path("data/logs/local") / f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
        logs.mkdir(parents=True, mode=0o700)
        supervisor = Supervisor(logs, timeout)
        previous_term = signal.getsignal(signal.SIGTERM)

        def interrupted(*_):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        print(f"Starting local assistant. Logs: {logs}", flush=True)
        try:
            docker_ready(supervisor)
            print("Starting PostgreSQL...", flush=True)
            supervisor.command("postgres", ["docker", "compose", "up", "-d", "--wait", "postgres"])
            if settings.python_enabled:
                print("Building isolated Python worker image...", flush=True)
                supervisor.command(
                    "python-image",
                    [
                        "docker",
                        "build",
                        "-f",
                        "Dockerfile.python",
                        "-t",
                        settings.python_worker_image,
                        ".",
                    ],
                )
            print("Applying database migrations...", flush=True)
            supervisor.command("migrate", [sys.executable, "-m", "personal_assistant", "migrate"])
            for service in services:
                if service.name == "api":
                    print("Checking complete planner context capacity...", flush=True)
                    supervisor.command(
                        "context-check",
                        [sys.executable, "-m", "personal_assistant", "check-context"],
                    )
                print(f"Starting {service.name}...", flush=True)
                supervisor.launch(
                    service.name, [sys.executable, "-m", "personal_assistant", *service.command]
                )
                supervisor.wait_ready(service)
            if not no_bot:
                supervisor.launch("bot", [sys.executable, "-m", "personal_assistant", "bot"])
                print(f"Telegram bot launched; polling progress: {logs / 'bot.log'}", flush=True)
            print("Local assistant running. Press Ctrl+C to stop native services.", flush=True)
            while True:
                supervisor.check_children()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping native services...", flush=True)
        finally:
            # Ignore repeated interrupts while reaping only our own process groups.
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                supervisor.stop()
            finally:
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGTERM, previous_term)
            print("Native services stopped. PostgreSQL data is retained.", flush=True)
