# Python calculations and charts

The `analysis` specialist executes small Python programs for arithmetic, statistics,
line/bar/scatter charts and histograms. NumPy (`np`) and Matplotlib (`plt`) are preloaded;
Python's standard library is available. It explains actual stdout and returns up to
three PNG charts through Telegram or the API. No vision model is needed. Analysis
always enters review and supports the existing bounded tool-repair workflow.

## Native Mac startup

Keep your existing private `.env` and add:

```dotenv
ASSISTANT_PYTHON_ENABLED=true
ASSISTANT_PYTHON_RUNNER_URL=http://localhost:8766
ASSISTANT_PYTHON_WORKER_IMAGE=personal-assistant-python:local
ASSISTANT_ARTIFACT_DIRECTORY=data/artifacts
```

Stop any running launcher with Ctrl+C, then run from the project folder:

```sh
uv run --extra speech --extra pdf --extra python-runner assistant start
```

The launcher checks the Docker client extra, starts/waits for Docker Desktop, builds
the cached worker image from `Dockerfile.python`, and starts the private runner before
the model and API. The first build needs internet for locked dependencies. NumPy and
Matplotlib are installed in the worker image, not your native Python environment.

For manual startup, build the image and run the broker in its own terminal:

```sh
docker build -f Dockerfile.python -t personal-assistant-python:local .
uv run --extra python-runner assistant python-runner
```

Then start the model/API/bot as in [startup](startup.md). Check runner readiness with
`curl --fail http://127.0.0.1:8766/health`. It requires both Docker and the worker image.
The assistant's authenticated `/v1/status` reports whether analysis is configured;
this is not a guarantee of successful execution. Missing Docker/image/runner produces
a tool failure, not an invented result. Tracing is optional and follows normal settings.

## Docker deployment

Enable the flag in `.env`, build the worker image, and add the `analysis` profile:

```sh
docker build -f Dockerfile.python -t personal-assistant-python:local .
docker compose --profile analysis --profile telegram up -d --build --wait api python-runner bot
```

Compose points the API at `http://python-runner:8766` and retains charts in the
`artifacts` volume. The broker uses a private internal control network and publishes
no port. Existing inference/speech deployment settings remain applicable. Check with
`docker compose logs --tail=50 python-runner` if its health check fails.

The **trusted broker** has the Docker socket to launch the fixed worker containers.
The API, bot, and generated-code workers do not receive it. Docker socket access is
powerful; keep the broker private. See [Docker daemon security](https://docs.docker.com/engine/security/).
Never add host networking, project mounts, credentials, privileged mode, or the Docker
socket to the worker containers.

## Execution and limits

- The supervisor selects `analysis` and `run_python(query=...)`. That assignment may
  first read permitted notes, calendar data, public pages, weather or PDF MCP excerpts.
  Put dependent data collection and calculation in the **same assignment**: parallel
  branches do not share newly retrieved data until synthesis.
- The configured model generates a validated `PythonProgram` from the original
  request, bounded evidence and source excerpts. No service credentials, raw attachments
  or host mounts are supplied to the worker; task data is embedded in its program.
- Each disposable worker has no network or host mounts, a read-only base filesystem,
  dropped capabilities and `no-new-privileges`. Code runs as UID 65534, separate from
  a fixed root watchdog. Writable `/work` and `/tmp` are bounded 16/32 MB tmpfs mounts.
- Limits: 20 seconds wall time, 15 CPU seconds, one CPU, 512 MB memory with no extra
  swap, 32 processes, 64 descriptors, and 32 KB combined stdout/stderr. Exceeding limits
  fails the job. Containers are removed on success/failure/timeout; the fixed watchdog
  stops orphan jobs after 60 seconds. The broker serializes jobs and returns a bounded
  busy response when occupied.
- The first three open figures are saved automatically; explicit PNG saves also work.
  A fixed collector reads bounded regular files inside the sandbox. Symlinks/special
  files are rejected. Each PNG is limited to 3 MB. The API verifies decoding and an
  eight-million-pixel limit, strips metadata, converts to RGB and caps dimensions at
  2000 pixels per side. No archives or model-selected host paths are extracted.
- Native charts persist in `data/artifacts`, mode 0600; Docker uses its named volume.
  SHA-256 identities make them immutable and replay-safe. No image bytes enter graph
  checkpoints or chat JSON. Files have no automatic expiry or vector indexing; `down -v`
  removes the Docker chart volume along with other volumes.

This is Docker isolation, not an AST filter or a guarantee against container-runtime/
kernel vulnerabilities. There is **no host-execution fallback**, saved notebook state,
network/package installation, arbitrary local-file access or general CSV upload support.
Supply small datasets in messages or use existing read tools. Pandas/SciPy are not
installed by this feature. Only validated PNGs are exported from jobs.

## Results, repair and delivery

Try: “Use Python to calculate the mean of 10, 20, 30, 40 and plot a histogram.” Or:
“Построй график y = x² от −5 до 5 и вычисли среднее для [2, 4, 6].”

The reviewer may ask the supervisor to regenerate/re-execute failed code within the
existing re-planning limit. Error output is available as evidence. Each Python task has
a stable `analysis_id`, assigned automatically on its first execution. Repair plans reuse
that ID even when the query changes; new independent tasks use distinct IDs. When several
earlier tasks exist, an ambiguous repair without an ID is rejected rather than guessed.
A repair replaces only its task's charts, preserving successful charts from other tasks.
A failed/no-chart replacement removes that task's old images; a prose-only repair keeps
current charts. Replies have at most three charts and report omitted attachments.
Review checks code/results, not independent visual inspection of every pixel; model
mistakes in calculations/plots are still possible.

`ChatResponse.artifacts` contains `{id, filename, media_type}`. Download with
`GET /v1/artifacts/{id}` using the assistant bearer token. IDs cannot be arbitrary paths.
Telegram fetches these files from the API and sends photos, falling back to PNG documents
if Telegram rejects the photo. The text CLI prints the answer; use Telegram/API for images.

Runner authentication uses a scoped HMAC derived from `ASSISTANT_API_TOKEN`; the runner
does not accept the assistant token directly. Optionally set an independent secret of
at least 24 characters with `ASSISTANT_PYTHON_RUNNER_TOKEN` in both configurations.
Use TLS for a runner on another machine. Langfuse content capture, when enabled, can
include supplied datasets/generated code; leave it off to exclude that private content.

## Verification

After building the worker image, run real Docker checks with:

```sh
TEST_PYTHON_DOCKER=1 uv run --extra dev --extra python-runner pytest -q tests/test_python_analysis.py
```

They cover arithmetic, charts, denied networking, absence of assistant credentials/
socket/host files, timeouts, output limits, symlink rejection and cleanup. Other tests
cover code generation, authentication, downloads, Telegram delivery, repair and replay.
These tests do not send live Telegram messages.
