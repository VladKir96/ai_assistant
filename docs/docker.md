# Run the assistant in Docker on Linux or Windows

This guide runs PostgreSQL, Redis, the assistant API, local CPU inference, and the
optional Telegram bot in Linux containers. Python, uv, and llama.cpp are installed
inside the images; none is needed on the host. The same setup can run on a Linux
server while you use Telegram from another computer or phone.

For hosted inference, the application and database still run in Docker; select the
hosted option below to omit the local model container. The native Apple Silicon
setup has its own [startup guide](startup.md).

## 1. Prepare Docker and the repository

- **Linux:** Install [Docker Engine](https://docs.docker.com/engine/install/) and
  the [Compose plugin](https://docs.docker.com/compose/install/linux/). Start the
  Docker daemon and ensure your account can run Docker commands.
- **Windows:** Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
  with the WSL 2 backend and a Linux distribution, such as Ubuntu. Enable that
  distribution under **Settings → Resources → WSL Integration** and use Linux
  containers. Run all commands below in the distribution's Bash terminal, rather
  than PowerShell or Command Prompt. See [Docker's WSL setup](https://docs.docker.com/desktop/features/wsl/).

Clone or extract this repository into a directory such as `~/Assistant`. On Windows,
keep it inside the WSL Linux filesystem, including the model files. From that directory:

```sh
cd ~/Assistant
docker info
docker compose version
mkdir -p models data/pdf data/tessdata
```

Continue only when `docker info` can contact the engine. These instructions use
Compose v2 or newer (`docker compose`, with a space), including `--wait` support.

Local inference needs enough memory for the GGUF weights, the configured context
cache, and the other services. Check Docker/WSL memory limits as well as physical
RAM. The default 27B model is large, and CPU generation can be slow. You can use a
smaller compatible instruction model, or the hosted option below. Model selection
still needs workflow evaluation; fitting in memory does not establish answer quality.

## 2. Build the application and create credentials

Build a bootstrap image without starting any services:

```sh
docker build -t personal-assistant:local .
```

If this checkout already has a private `.env`, keep it and skip initialization.
For a fresh checkout, run the existing initializer inside the image:

```sh
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD,target=/setup" \
  --workdir /setup \
  personal-assistant:local assistant init
```

The initializer reads the mounted `.env.example` and writes `.env` with random API,
database, and inference credentials, owned by your Linux/WSL account. It refuses to
overwrite an existing `.env`. The file is excluded from the Docker build context.
Do not replace an existing database password: the PostgreSQL volume retains the
password it was initialized with.

Edit `.env` in a text editor. Preserve the generated credentials, set
`ASSISTANT_TIMEZONE` to your IANA timezone, and begin with these optional services off:

```dotenv
ASSISTANT_EMBEDDING_BACKEND=disabled
ASSISTANT_MODEL_SUPPORTS_IMAGES=false
ASSISTANT_VOICE_ENABLED=false
ASSISTANT_VOICE_OUTPUT_ENABLED=false
ASSISTANT_PDF_ENABLED=false
ASSISTANT_PYTHON_ENABLED=false
```

## 3. Choose where inference runs

### Option A: Local model entirely in Docker

Download a llama.cpp-compatible instruction GGUF and place it at `models/model.gguf`,
or substitute its actual filename below. Weights are not bundled or downloaded by
Compose. The runtime must support the model architecture, chat template, and
JSON-schema output; see the [existing model validation record](model-validation.md)
for the default artifact.

Set these fields in `.env`:

```dotenv
ASSISTANT_MODEL_PROVIDER=local
ASSISTANT_LOCAL_MODEL_FILE=./models/model.gguf
ASSISTANT_LOCAL_MODEL=assistant-local
ASSISTANT_CONTEXT_WINDOW=16384
ASSISTANT_MAX_OUTPUT_TOKENS=2048
ASSISTANT_REQUEST_TIMEOUT_SECONDS=600
ASSISTANT_MODEL_MAX_CONCURRENCY=1
ASSISTANT_MODEL_SUPPORTS_IMAGES=false
```

The file path is a **host** path relative to the project directory. Compose mounts
it read-only as `/models/model.gguf`. `ASSISTANT_LOCAL_MODEL` is the alias used by both
the server and client, so it need not match the filename. Keep the generated
`ASSISTANT_LOCAL_API_KEY`; Compose supplies it to both sides. Choose context and
output budgets supported by your model and available memory.

Define this helper in your Bash terminal:

```sh
dc() { docker compose -f compose.yaml -f compose.linux.yaml --profile linux-inference "$@"; }
```

The override connects the API to `http://llama:8080/v1` on the container network
and waits for the model's health check before starting the API. There is no native
model process and no dependency on `host.docker.internal` for inference. The filename
`compose.linux.yaml` also applies to Linux containers running under Windows/WSL 2.

This profile uses the upstream CPU [llama.cpp server image](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md).
It does not configure GPU access. CUDA/other GPU deployments require a compatible
image, host drivers, and container device configuration; they are outside this baseline.

### Option B: Hosted model, application entirely in Docker

Instead of Option A, set these fields in `.env`:

```dotenv
ASSISTANT_MODEL_PROVIDER=openai
ASSISTANT_OPENAI_API_KEY=YOUR_PROVIDER_KEY
ASSISTANT_OPENAI_MODEL=YOUR_MODEL_ID
```

Choose a model supporting the Responses API and JSON-schema output. The configured
context/output budgets must fit that model. This is an explicit provider selection;
the application never switches from local to hosted inference automatically.
Selected chat context is sent to the provider; its data policies apply.

Use only the base Compose file for this option:

```sh
dc() { docker compose -f compose.yaml "$@"; }
```

No GGUF, `llama` container, or Linux inference override is needed. If switching from
an already running local Docker deployment, stop its model first using
`docker compose --profile linux-inference stop llama`.

## 4. Start and verify text chat

Use the `dc` helper for your chosen option for every command below. Define it again
when opening a new terminal. It keeps the same Compose files and inference profile
in use for startup, one-off commands, and updates.

```sh
dc config --quiet
dc up -d --build --wait --wait-timeout 900 api redis
dc ps -a
```

Compose starts PostgreSQL, runs the `migrate` service for application and LangGraph
schemas, and starts the API and Redis. With Option A it also starts `llama` and waits
for the model to load. A successfully completed `migrate` container is expected to
show `Exited (0)`. Other requested services should be running and, where configured,
healthy. Initial builds, image downloads, and model loading can take several minutes.

Check database/API health, the complete planner input budget, and an actual response:

```sh
dc exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
dc exec api assistant check-context
dc exec api assistant chat --text "Say hello briefly."
```

The health check should print `{"status":"ok"}`. The CLI inside the container uses
the configured API token automatically. `check-context` verifies model connectivity
and input capacity; the final command exercises generation and the assistant workflow.
Do not enable Telegram until these succeed. A healthy API alone does not prove that
model requests work. API documentation is at <http://127.0.0.1:8000/docs> on the host.

## 5. Add Telegram

Create a bot through Telegram's BotFather and put its complete token in
`ASSISTANT_TELEGRAM_TOKEN` in `.env`. To discover your personal user ID, leave
`ASSISTANT_TELEGRAM_OWNER_ID=0` temporarily and run:

```sh
dc run --rm --no-deps bot
```

Send `/id` in a private chat with that bot. Save the returned personal ID as
`ASSISTANT_TELEGRAM_OWNER_ID`, then press Ctrl+C to stop the temporary bot.
Start its background container:

```sh
dc --profile telegram up -d --build --wait bot
dc --profile telegram logs --tail=50 bot
```

Send `/start`, `/notes`, and an ordinary question. Both sender and private chat must
match the configured owner. Stop any other process polling the same bot token.
The bot uses outbound connections; using Telegram from your phone does not require
publishing the API to the internet.

## Optional features

### Reminders

With Telegram credentials configured, start the Celery worker and scheduler:

```sh
dc --profile reminders up -d --build worker scheduler
```

They use Redis and the same PostgreSQL data. Keep one scheduler running.

### PDF reading and uploads

Set `ASSISTANT_PDF_ENABLED=true` in `.env`, then recreate the application containers
as described below. The image already includes the PDF dependencies. Put reference
PDFs in `data/pdf`; keep both bind-mounted directories readable by container UID
10001. Uploaded PDFs use the writable `pdf-uploads` named volume. Optional OCR needs
language files in `data/tessdata`; see [PDF setup](pdf.md#optional-ocr-for-scanned-pdfs).

### Python calculations and charts

Set `ASSISTANT_PYTHON_ENABLED=true`, build the isolated worker image, and start the
private broker together with the updated API:

```sh
docker build -f Dockerfile.python -t personal-assistant-python:local .
dc --profile analysis up -d --build --wait api python-runner
```

Recreate the bot too if you use Telegram. Only the trusted broker receives the Docker
socket; generated-code workers have no network, host mounts, or assistant credentials.
See [Python analysis](python-analysis.md#docker-deployment) for limits and verification.

### Vision, speech, and semantic search

For local vision, use the [vision overlay](images.md#docker) with a compatible model
and matching projector; include that overlay in your `dc` helper for later commands.
Hosted vision requires a capable model and the explicit image-support flag.

Speech and embedding companions are not defined in the supplied Compose stack, and
the base image does not install their optional runtime dependencies or model weights.
Keep voice input/output and HTTP embeddings disabled for this baseline. Enabling them
requires separately provisioned companion services and reachable container URLs; see
[voice setup](voice.md) and [semantic search](../README.md#semantic-memory-and-rag).

## Everyday use, configuration changes, and updates

After restarting Docker or your computer, return to the project directory, define
the appropriate `dc` helper, and run the startup command again. Run the relevant
optional-feature commands for services you enabled. Containers also have a restart
policy, but check health and logs after an unexpected shutdown.

For a `.env` change, recreate the affected containers so they receive the new values:

```sh
dc up -d --force-recreate --wait --wait-timeout 900 api
dc --profile telegram up -d --force-recreate bot
```

Run the second command only if you use Telegram. Recreate enabled reminder or analysis
services too when changing settings they consume. `restart` alone keeps the old
container environment. If the model path, alias, key, or context changed under Option A,
recreate `llama` along with `api`.

For a source update, finish active requests, stop the application services, update
your checkout, then rebuild and apply migrations before restarting:

```sh
dc --profile '*' stop api bot worker scheduler python-runner
# Update your checkout here, preserving .env, model files, and database volumes.
dc build api migrate
dc run --rm migrate
dc up -d --build --wait --wait-timeout 900 api redis
```

Then start the optional services you previously enabled. Keep one API process.

Inspect logs and stop the whole stack with:

```sh
dc logs --tail=100 api
dc --profile telegram logs --tail=100 bot
dc --profile '*' stop
```

For local model diagnostics use `dc logs --tail=100 llama`. Logs are available through
Docker, rather than the native launcher's `data/logs/local` directory.

Named volumes retain PostgreSQL, Redis, uploaded PDFs, and generated artifacts across
container replacement. Keep `.env`, host model/PDF/OCR files, and backups too. `stop`
retains containers and data; `down` removes containers/networks but retains named
volumes. **`down -v` deletes named volumes, including the assistant database.**
See [backups](../README.md#export-and-backups); use `dc` in place of `docker compose`
for that guide's database backup command and include uploads/artifacts separately.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Cannot connect to Docker | Start the Linux daemon or Docker Desktop; on Windows, check WSL integration and Linux-container mode. |
| Missing model / bind source does not exist | Set `ASSISTANT_LOCAL_MODEL_FILE` to an existing readable GGUF file, relative to the project; use Linux paths in WSL. |
| API waits for `llama` or model is unhealthy | Inspect `dc logs --tail=100 llama`; check model/runtime compatibility, Docker memory limits, and context size. |
| API reports connection refused to a host model | Use the Option A helper so `compose.linux.yaml` supplies the container service URL. |
| PostgreSQL authentication fails after changing `.env` | Restore the password matching the existing volume; regenerating `.env` does not change it. |
| `check-context` fails or chat times out | Confirm model readiness and supported context/output budgets; CPU inference may need a longer request timeout. |
| Telegram polling conflict / commands ignored | Stop duplicate bot instances; verify the complete token and your personal owner ID, then recreate the bot. |
| Port already allocated | Stop the conflicting host/native service; the stack publishes localhost ports 5432, 6379, 8000, and, with Option A, 8080. |

The published ports bind to `127.0.0.1`. On a remote server, Telegram can be used
without exposing them; use an SSH tunnel for local API access. A public HTTP deployment
needs a separately configured authenticated TLS endpoint.
