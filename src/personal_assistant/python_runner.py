"""Trusted Docker broker. Generated Python runs only in disposable worker containers."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import Field

from personal_assistant.config import Settings
from personal_assistant.http_guard import PrivateBodyGuard
from personal_assistant.python_analysis import runner_token
from personal_assistant.schemas import StrictModel

TIMEOUT = 20
MAX_OUTPUT = 32_000
COLLECTOR = """
import base64, json, os, stat
charts, omitted = [], 0
for entry in sorted(os.scandir('/work'), key=lambda entry: entry.name):
    if not entry.name.lower().endswith('.png'):
        continue
    fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 3000000:
            raise ValueError('Chart must be a regular PNG file below 3 MB')
        if len(charts) >= 3:
            omitted += 1
            continue
        raw = stream.read(3000001)
    if len(raw) > 3000000:
        raise ValueError('Chart exceeds the 3 MB limit')
    charts.append({'filename': entry.name[:100], 'data': base64.b64encode(raw).decode()})
print(json.dumps({'artifacts': charts, 'omitted_artifacts': omitted}))
"""


def docker_client():
    import docker

    if sys.platform == "darwin" and not os.environ.get("DOCKER_HOST"):
        result = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True,
            timeout=5,
            check=True,
            text=True,
        )
        return docker.DockerClient(base_url=result.stdout.strip(), timeout=5)
    return docker.from_env(timeout=5)


def exec_python(client, container, program: str, limit: int):
    task = client.api.exec_create(
        container.id, ["python", "-I", "-c", program], user="65534:65534", workdir="/work"
    )
    output = [bytearray(), bytearray()]
    stream = client.api.exec_start(task["Id"], stream=True, demux=True)
    try:
        for pair in stream:
            for index, value in enumerate(pair):
                if value:
                    output[index].extend(value)
                    if sum(map(len, output)) > limit:
                        raise ValueError("Python stdout/stderr exceeded its limit")
    finally:
        # Docker SDK may already have closed its socket on normal EOF.
        with suppress(OSError):
            stream.close()
    return client.api.exec_inspect(task["Id"])["ExitCode"], output


class DockerExecutor:
    def __init__(self, image: str):
        self.image = image

    def health(self):
        client = docker_client()
        try:
            client.ping()
            client.images.get(self.image)
        finally:
            client.close()

    def execute(self, code: str) -> dict:
        client = container = timer = None
        expired = threading.Event()
        output = [bytearray(), bytearray()]

        def remove():
            try:
                container.remove(force=True)
            except Exception:
                pass

        def timeout():
            expired.set()
            remove()

        try:
            client = docker_client()
            container = client.containers.create(
                self.image,
                name="assistant-python-" + uuid4().hex,
                # PID 1 is a fixed root watchdog; user code is exec'd as nobody.
                command=["python", "-I", "-c", "import time; time.sleep(60)"],
                network_mode="none",
                auto_remove=True,
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                user="0:0",
                working_dir="/",
                mem_limit="512m",
                memswap_limit="512m",
                nano_cpus=1_000_000_000,
                pids_limit=32,
                shm_size="1m",
                tmpfs={
                    "/work": "rw,noexec,nosuid,nodev,size=16m,uid=65534,gid=65534,mode=700",
                    "/tmp": "rw,noexec,nosuid,nodev,size=32m,mode=1777",
                },
                environment={
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
                labels={"personal-assistant": "python-job"},
            )
            timer = threading.Timer(TIMEOUT, timeout)
            timer.daemon = True
            timer.start()
            container.start()
            program = (
                "import resource\n"
                "resource.setrlimit(resource.RLIMIT_CPU, (15, 15))\n"
                "resource.setrlimit(resource.RLIMIT_FSIZE, (3000000, 3000000))\n"
                "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
                "import numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\n"
                "import matplotlib.pyplot as plt\n"
                f"exec(compile({code!r}, '<analysis>', 'exec'))\n"
                "if len(plt.get_fignums()) > 3:\n"
                "    import sys; print('Only three open figures are exported', file=sys.stderr)\n"
                "for i, n in enumerate(plt.get_fignums()[:3], 1):\n"
                "    plt.figure(n).savefig(f'plot-{i}.png', dpi=120, bbox_inches='tight')\n"
            )
            exit_code, output = exec_python(client, container, program, MAX_OUTPUT)
            if expired.is_set():
                raise TimeoutError
            if exit_code != 0:
                return {
                    "status": "error",
                    "stdout": output[0].decode(errors="replace"),
                    "stderr": output[1].decode(errors="replace"),
                    "artifacts": [],
                    "error": "Python code failed; inspect stderr and revise the code",
                }
            # Docker archive APIs can omit tmpfs contents. A fixed collector reads
            # bounded regular files inside the same sandbox, without host extraction.
            collected, files = exec_python(client, container, COLLECTOR, 13_000_000)
            if collected != 0:
                raise ValueError("Chart collection failed: invalid file, symlink or size")
            manifest = json.loads(files[0])
            return {
                "status": "ok",
                "stdout": output[0].decode(errors="replace"),
                "stderr": output[1].decode(errors="replace"),
                "artifacts": manifest["artifacts"],
                "omitted_artifacts": manifest["omitted_artifacts"],
            }
        except Exception as error:
            message = (
                "Python execution timed out"
                if expired.is_set() or isinstance(error, TimeoutError)
                else str(error)
                if isinstance(error, ValueError)
                else "Python sandbox unavailable; check Docker and build the worker image"
            )
            return {
                "status": "error",
                "error": message,
                "artifacts": [],
                "stdout": output[0][:4000].decode(errors="replace"),
                "stderr": output[1][-2000:].decode(errors="replace"),
            }
        finally:
            if timer:
                timer.cancel()
            if container:
                remove()
            if client:
                client.close()


class ExecutionRequest(StrictModel):
    code: str = Field(min_length=1, max_length=12000)


def create_python_runner_app(settings: Settings | None = None, executor=None):
    settings = settings or Settings()
    key = runner_token(settings)

    @asynccontextmanager
    async def lifespan(app):
        explicit = settings.python_runner_token.get_secret_value()
        if (explicit and len(explicit) < 24) or (
            not explicit and len(settings.api_token.get_secret_value()) < 24
        ):
            raise RuntimeError("Set a private API token or Python runner token")
        yield

    app = FastAPI(title="Private Python runner", lifespan=lifespan)
    app.add_middleware(PrivateBodyGuard, token=key, limit=65536)
    executor = executor or DockerExecutor(settings.python_worker_image)
    lock = asyncio.Semaphore(1)

    @app.get("/health")
    async def health():
        try:
            await asyncio.to_thread(executor.health)
        except Exception:
            raise HTTPException(503, "Docker or Python worker image unavailable") from None
        return {"status": "ok"}

    @app.post("/v1/execute")
    async def execute(request: ExecutionRequest):
        # A busy worker returns a bounded error instead of queuing arbitrary code.
        try:
            await asyncio.wait_for(lock.acquire(), timeout=1)
        except TimeoutError:
            raise HTTPException(429, "Python runner busy; retry the operation") from None
        try:
            task = asyncio.create_task(asyncio.to_thread(executor.execute, request.code))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # Keep the slot until the bounded job finishes/removes its container.
                await asyncio.shield(task)
                raise
        finally:
            lock.release()

    return app
