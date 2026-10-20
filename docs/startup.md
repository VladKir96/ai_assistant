# Start the assistant on an Apple Silicon Mac

Use this guide for the default setup: PostgreSQL in Docker Desktop, with Qwen,
the assistant API, and the Telegram bot running directly on your Mac. Start with
text messages; add speech, semantic search, and reminders after text chat works.

Run all project commands from the directory containing `pyproject.toml`,
`compose.yaml`, and your private `.env`. The examples use `~/Assistant`, which is
`/Users/vladislav/Assistant` on the current machine. Change that path if you clone
the repository somewhere else. Run commands one block at a time and check the
result before continuing.

## Daily startup with one command

After completing first-time setup below, run from the project directory:

```sh
uv run assistant start
```

With faster-whisper enabled in `.env`, use this version to ensure the speech extra
is installed:

```sh
uv run --extra speech assistant start
```

With PDF reading enabled, use `uv run --extra speech --extra pdf assistant start`.
Create `data/pdf` first, then upload PDFs as Telegram files or put local PDFs there;
follow [PDF setup](pdf.md). The PDF server
is launched on demand through stdio; it needs no separate terminal.

This replaces the separate model, API, speech/embedding, and bot terminals. Stop those
previously started native processes, including the Telegram bot, before switching.
Leave an existing Compose PostgreSQL container running; the launcher checks it through
`docker compose up -d --wait postgres`.

The launcher:

1. Validates credentials, owner ID, model files, optional dependencies, local service
   URLs, and port assignments before starting services.
2. Opens Docker Desktop on macOS when necessary and waits for the engine.
3. Starts/waits for PostgreSQL and applies application and LangGraph migrations.
4. Starts native inference for a local model profile; hosted profiles skip this process.
5. Starts speech when `VOICE_ENABLED=true` and embeddings when `EMBEDDING_BACKEND=http`.
6. Checks the complete planner context against the running model before starting the
   API (`assistant check-context`), including its schema, clock, and available MCP tools.
7. Waits for each HTTP service, including authenticated API/database readiness, then
   launches the Telegram bot. Its polling/authentication progress is in `bot.log`;
   a launched process alone does not prove Telegram connectivity.

It prints the current log directory, for example
`data/logs/local/20261018-120000-12345/`. Each service has its own log file, including
`inference.log`, `api.log`, `bot.log`, and any enabled companion logs. Files have private
permissions and are ignored by Git. Inspect them locally when a service fails; logs
may contain private data, so avoid publishing them unedited.

Keep the launcher terminal open. **Ctrl+C** stops the native processes it started,
including the inference subprocess. A failed or timed-out service also triggers
cleanup. Docker Desktop and PostgreSQL stay running; data and volumes are retained.
To stop PostgreSQL afterward, use `docker compose stop postgres`.

Useful variants:

```sh
uv run --extra speech assistant start --check
uv run --extra speech assistant start --no-bot
uv run --extra speech assistant start --startup-timeout 600
```

`--check` checks static configuration only; it does not probe Docker, occupied ports,
Telegram credentials over the network, or actual model readiness. The readiness
timeout defaults to 300 seconds per service. For semantic embeddings, also include
`--extra embeddings`; their existing companion may download its model on first use.
These commands do not download chat or speech weights or configure missing credentials.

For manual startup or after switching models, run `uv run assistant check-context`
once inference is ready. Local llama.cpp uses its tokenizer; hosted Responses providers
use their input-token counter. If a compatible endpoint lacks a counter, conservative
byte accounting remains available and may require a larger matching model context.
An unsupported small configuration is reported before the launcher starts the API/bot.

This launcher targets the repository's Compose-managed `assistant` PostgreSQL database
on localhost:5432 and native HTTP services bound to loopback addresses. It reads ports
from `.env` URLs, with `/v1` paths for inference/embeddings and no path for API/speech.
External endpoints, custom database layouts, Redis, and reminder workers/schedulers
use the manual instructions. Only one launcher may run per project; occupied native
ports cause an error rather than reusing a possibly stale process.

## What must be running

| Component | Purpose | Ready when | Keep running? |
| --- | --- | --- | --- |
| Docker Desktop | Runs PostgreSQL | `docker info` shows a working Server section | Yes |
| PostgreSQL | Notes, facts, tasks, calendar, conversations, graph checkpoints | Compose reports `healthy` | Yes, in the background |
| Native model server | Generates assistant responses | Its terminal reports `model loaded` and `listening` on port 8080 | Yes |
| Assistant API | Runs the agents and accesses storage | `/health` returns `{"status":"ok"}` | Yes |
| Telegram bot | Receives your messages and forwards them to the API | Polling starts; private `/start` gets a reply after owner setup | Yes |

The normal dependency order is **Docker Desktop → PostgreSQL → migrations →
model server → API → bot**. Starting the bot alone does not start these dependencies.
The `/id` setup command is an exception: it needs only a valid bot token and access
to Telegram, so you can discover your user ID before starting the database or model.

Redis, the reminders worker/scheduler, embeddings, and speech are optional services.
They are unnecessary for the initial text-chat check.

## First-time setup

Complete this section once. If dependencies and `.env` already exist, keep them
and check their settings rather than generating new credentials.

### 1. Check the required software

Install Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/), Docker Desktop for
Apple Silicon, and a native Apple Silicon llama.cpp runtime that supports your model.

```sh
cd ~/Assistant
uv --version
docker --version
docker compose version
uv sync --frozen --extra dev --extra mcp
```

`docker --version` confirms that the Docker client exists. It does **not** confirm
that the engine is running. Start Docker Desktop:

```sh
open -a Docker
```

Wait for Docker Desktop to show that its engine is running, then check:

```sh
docker info
```

Continue when the command exits successfully and shows a Server section. If it
cannot connect to `docker.sock`, Docker is not ready; fix that before running Compose.

### 2. Create or keep your private configuration

If `.env` does not exist yet:

```sh
uv run assistant init
```

The command generates private API, database, and inference credentials. If `.env`
already exists, skip it; the command deliberately refuses to overwrite that file.
Keep this same configuration across restarts. An existing PostgreSQL volume retains
its original password; generating another `.env` does not change that password.

For the first text-only start, keep these settings in `.env`:

```dotenv
ASSISTANT_EMBEDDING_BACKEND=disabled
ASSISTANT_VOICE_ENABLED=false
```

Edit `.env` as a configuration file; the commands load it automatically. There is
no need to activate a virtual environment or `source .env` when using `uv run`.
Exported `ASSISTANT_*` variables take precedence over values in `.env`.

### 3. Configure your model and native executable

Model weights and runtime executables are separate downloads. For the Qwen artifact
used in this checkout, the file layout is:

```text
Assistant/
├── models/
│   └── Qwen3.8-27B-GGUF/
│       └── Qwen3.8-27B-UD-Q4_K_M.gguf
└── data/
    └── runtime/
        └── llama-b11669/
            └── llama-server
```

These files are local downloads and are excluded from Git. A fresh clone will not
contain them. See [the validated runtime and model artifact](model-validation.md)
for the runtime release link, checksum, and configuration used on this Mac.

Set `ASSISTANT_LLAMA_EXECUTABLE` in `.env` to your actual `llama-server` executable.
If it is installed on your shell's PATH, the value can be `llama-server`. For the
layout above, a project-relative value also works:

```dotenv
ASSISTANT_LLAMA_EXECUTABLE=data/runtime/llama-b11669/llama-server
```

Select the matching model profile:

```sh
uv run assistant models
uv run assistant models --use qwen-local
```

The second command checks that this profile's GGUF exists and updates the model
settings in `.env`; it does not launch a service. For a different file or model,
edit/add a profile in [model-profiles.toml](../model-profiles.toml) and select its
name. The `local_model_file` must match the real path, including any renamed folder.
See [switching models](../README.md#switch-models) for the full procedure.

### 4. Set up your Telegram identity

Create a bot with [BotFather](https://t.me/BotFather), then put its complete token
in `ASSISTANT_TELEGRAM_TOKEN` in `.env`. The token contains a numeric prefix, a colon,
and a secret suffix, with no spaces. Keep both sides of the colon.

| Setting | What belongs there |
| --- | --- |
| `ASSISTANT_TELEGRAM_TOKEN` | The complete secret token issued by BotFather |
| `ASSISTANT_TELEGRAM_OWNER_ID` | Your personal numeric Telegram user ID |

The token's numeric prefix identifies the **bot**. Your owner ID identifies **you**.
To discover your personal ID through your own bot:

1. Set `ASSISTANT_TELEGRAM_OWNER_ID=0` in `.env` and save the file.
2. Run `uv run assistant bot`. Keep this terminal open.
3. In Telegram, open the private chat with the bot you created and send `/id`.
4. Copy the number from the reply into `ASSISTANT_TELEGRAM_OWNER_ID` in `.env`.
5. Press **Ctrl+C** in the bot terminal to stop it. Start it again during the
   everyday startup sequence below so it reads the saved owner ID.

In setup mode, `/id` works while assistant commands remain restricted. The command
returns only the requesting user's own ID. In normal operation, use your personal
account in a private chat with the bot. Token or owner-ID edits require a bot restart.

## Everyday startup

Open four terminal tabs. In **each tab**, first run `cd ~/Assistant`. Commands for
the model, API, and bot remain in the foreground: their terminal does not return to
a prompt while they are running. Leave each running and use the next tab.

If a component is already running, check it and use that instance. Start one API,
one model server, and one polling bot for this setup.

### Terminal A: database, migrations, and checks

Open Docker Desktop and wait for its engine, then run:

```sh
cd ~/Assistant
docker info
docker compose up -d --wait postgres
docker compose ps postgres
```

**Ready:** PostgreSQL is `Up` and `healthy`. The `-d` flag keeps it in the background;
this terminal remains available for commands. The first start may download the image.
If startup fails, inspect `docker compose logs --tail=50 postgres` and resolve the
error before continuing.

Apply migrations on the first start and after application updates:

```sh
uv run assistant migrate
```

**Ready:** the command finishes with `Application and LangGraph migrations applied.`
It is safe to rerun when there are no new migrations. A failed migration must be
resolved before starting the API.

### Terminal B: local Qwen server

```sh
cd ~/Assistant
uv run assistant inference
```

**Ready:** the log reports `model loaded` and `listening on http://127.0.0.1:8080`.
Wait for both; file loading and GPU initialization take time. Leave this terminal
running. This command uses the executable, model file, alias, and context size from
your saved configuration.

If it reports a missing executable or GGUF, return to the model configuration step.
If port 8080 is already in use, check the existing server before launching another.

### Terminal C: assistant API

```sh
cd ~/Assistant
uv run assistant api
```

**Ready:** Uvicorn reports `Application startup complete` and listens on port 8000.
Leave this terminal running.

Return to **Terminal A** to verify database access:

```sh
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

**Expected:** `{"status":"ok"}`. This checks the API and database; it does not test
the model. Verify the full assistant path with a simple text request:

```sh
uv run assistant chat --thread startup-check --text "Reply with a short greeting."
```

**Ready:** the command prints an assistant response. A response can take several
seconds or longer because a request may require multiple model calls. This diagnostic
is ordinary chat and is saved in conversation history. If it fails, check the API
and model terminals before adding Telegram to the workflow.

The API documentation is available at <http://127.0.0.1:8000/docs>. `/health` is
public locally; `/v1/*` endpoints require your application bearer token. The chat
CLI supplies that token from `.env` automatically.

### Terminal D: Telegram bot

After the CLI chat check passes and your personal owner ID is saved:

```sh
cd ~/Assistant
uv run assistant bot
```

**Ready:** polling starts without a token, network, or conflict error. Keep this
terminal open. A running bot may be quiet while it waits for messages.

In the bot's **private Telegram chat**, check these in order:

| Send | Expected result | What it verifies |
| --- | --- | --- |
| `/id` | Your personal numeric user ID | Telegram connection and identity discovery |
| `/start` | Help/welcome reply | The bot accepts you as its configured owner |
| `/time` | Date, time, UTC offset, timezone, weekday | API clock without model inference |
| `/notes` | Saved notes or `No items.` | Bot → API → database |
| `/agenda` | Upcoming events or an empty list | Private calendar access |
| `Hello! Please reply briefly.` | An assistant response | Bot → API → model → reply |

If `/id` works but `/start` does not, compare the returned ID with your saved
`ASSISTANT_TELEGRAM_OWNER_ID`, then stop and restart the bot after correcting it.
If `/notes` works but ordinary chat fails, focus on the model server and API logs.

The private calendar is part of the base application and needs no external account.
After an application update, run `uv run assistant migrate` before restarting the API
and bot; calendar storage was added in migration `0002`. `/calendar` downloads active
events as an ICS snapshot. See [dates and calendar usage](../README.md#dates-times-and-the-private-calendar)
for event creation, moving/cancellation, local-date note queries, and current limitations.

## Troubleshooting by symptom

For web search after text chat works, keep `ASSISTANT_SEARCH_PROVIDER=duckduckgo` in
`.env` and restart the API after any edits. This provider needs no API key; dependency
installation includes `ddgs`. The assistant supports multilingual search requests.
`ASSISTANT_DUCKDUCKGO_REGION=wt-wt` uses worldwide results. DuckDuckGo may block or
rate-limit automated requests; a reported search failure does not disable reading a
supplied public URL. For Tavily, select `tavily` explicitly and supply its key. Job
search additionally requires your chosen domains in `ASSISTANT_JOB_SITES`.

| Symptom | Check and recovery |
| --- | --- |
| `Cannot connect to the Docker daemon` / `docker.sock` | Open Docker Desktop; wait until `docker info` succeeds, then retry the PostgreSQL command. |
| `Set LOCAL_MODEL_FILE` or missing GGUF | Check the file path in your model profile, including renamed directories; select the profile again. An explicit `--model-file` overrides the saved path for that launch. |
| `llama-server` not found | Set `ASSISTANT_LLAMA_EXECUTABLE` to the actual native executable; start from the project directory. |
| `Address already in use` | A process already owns the port. Inspect it with `lsof -nP -iTCP:8080 -sTCP:LISTEN` (or port 8000); use the intended running service or stop it in its original terminal before restarting. |
| Database connection refused during migrations | Check Docker Desktop and `docker compose ps postgres`; wait for PostgreSQL to become healthy. |
| Database password authentication failed | Check that `.env` still has the credentials used to initialize the existing database. Editing the password in `.env` alone does not change the database user's password. |
| API `/health` fails | Read Terminal C; confirm PostgreSQL is healthy and migrations completed. |
| Wrong date/time or wrong “today” | Send `/time`. If it is wrong, check automatic date/time in macOS System Settings and `ASSISTANT_TIMEZONE=Europe/Belgrade`, then restart the API. If `/time` is correct but chat is wrong, keep the exact request/reply for a model/planner regression check. |
| `/time` is unknown or `/v1/time` returns 404 | Restart the API and bot after this code update: Ctrl+C in their original terminals, then `uv run assistant api` and `uv run assistant bot` in separate terminals. For containerized services, rebuild/recreate the app containers. |
| `calendar_events` relation/table is missing | Run `uv run assistant migrate` from the project directory against the same database as the API, then restart the API. |
| `Token is invalid` / invalid token format | Copy the complete BotFather token, including its numeric prefix and colon, into `ASSISTANT_TELEGRAM_TOKEN`; restart the bot. |
| Telegram says `Unauthorized` despite valid token format | Check that the token is the current token for this bot. A local format check does not verify it with Telegram. |
| `/id` replies but `/start` is silent | Save your personal ID from `/id`, rather than the bot ID, and restart the bot. Use a private chat. |
| Even `/id` is silent | Confirm polling is running, check the bot terminal for connection/authentication errors, and verify that you opened the bot matching your token. This check does not need Qwen or PostgreSQL. |
| Telegram reports a `getUpdates` conflict | Stop duplicate polling instances, including a bot container if you started one. Run a single bot process for this token. |
| `/start` works but `/notes` says service unavailable | Start/check the API and database; confirm the bot and API use the same `ASSISTANT_API_TOKEN` and appropriate API URL. |
| CLI chat or Telegram ordinary chat fails | Check Terminal B for model readiness and Terminal C for the request error. The model alias, endpoint, and inference key must match the API configuration. |
| Context-budget error on a short message | Restart the updated API and run `uv run assistant check-context` with inference ready. The check includes instructions, schema, clock, and MCP tools. If the endpoint lacks a token counter, its conservative fallback may require a larger context configured in both the profile and server. No history deletion is needed. |
| Configuration edits seem to have no effect | Restart affected processes. Check whether exported environment variables override `.env`. Recreate Docker app containers when their environment changes. |
| Voice is disabled or unavailable | Follow the README voice setup, start `uv run --extra speech assistant speech`, wait for startup, check `curl -fsS http://127.0.0.1:8765/health`, then restart the bot with voice enabled. For input with text replies, set `ASSISTANT_VOICE_OUTPUT_ENABLED=false`; no Piper setup is needed. |
| faster-whisper reports missing/incompatible weights | Run `uv run --extra speech assistant speech-download`; point `ASSISTANT_FASTER_WHISPER_MODEL` at the complete CTranslate2 directory, not the chat GGUF or a whisper.cpp file. On this Mac use `cpu` and `int8`, then restart speech. |
| Search says it is unconfigured | Check `ASSISTANT_SEARCH_PROVIDER`. DuckDuckGo needs no key; Tavily needs its key. Restart the API after edits. Job search also needs selected job domains. |
| DuckDuckGo search fails or is rate-limited | Try later or supply a public URL. The app reports the failure and does not silently switch search providers. |
| Weather still uses search or denies access | Restart the API to load `get_weather`. Check `ASSISTANT_WEATHER_ENABLED=true`; `/v1/status` reports `web_capabilities.weather`. Open-Meteo works independently of DuckDuckGo/Tavily. |
| Weather asks for a location | Configure both `ASSISTANT_WEATHER_DEFAULT_LATITUDE` and `ASSISTANT_WEATHER_DEFAULT_LONGITUDE` with a descriptive `ASSISTANT_WEATHER_DEFAULT_NAME`, then restart the API; or name a city/country in the request. |
| Open-Meteo unavailable / dates not covered | Retry later after a provider failure. Forecasts extend up to 16 days, including today; the tool does not substitute another date or serve expired data when refresh fails. |

When sharing an error for troubleshooting, include the failed command and relevant
terminal error, while keeping `.env`, bot tokens, and authorization headers private.
Avoid using token-bearing Telegram API URLs in shared commands or browser history.

## Stop and restart

Stop foreground processes with **Ctrl+C** in this order: bot, API, model server.
Then stop the database if you want to shut down the whole stack:

```sh
docker compose stop postgres
```

Stopping a container preserves its database volume. Keep the volume and your existing
`.env` for the next start; `docker compose down -v` would remove the stored database.
After a Mac restart, follow the everyday startup sequence again. Saving configuration
or closing a terminal does not automatically launch the dependent services.

## Add optional services after text chat works

| Feature | Additional setup | Running processes |
| --- | --- | --- |
| Semantic memory/RAG | Install embeddings extra, configure HTTP embeddings, then reindex | Local embeddings companion on port 8081, plus the base stack |
| Voice input | Install the speech extra and download faster-whisper; configure CPU/INT8 and enable voice | Speech companion on port 8765 and restarted bot; text replies work without TTS |
| Voice output | Install FFmpeg/Piper and configured voices; enable voice output | Speech companion plus bot with `ASSISTANT_VOICE_OUTPUT_ENABLED=true` |
| PDF reading/uploads | Install the PDF extra, create `data/pdf`, enable PDF in `.env`; OCR needs language files | Managed stdio server; uploads accepted as Telegram documents; [setup](pdf.md) |
| Image understanding | Select a vision-capable model/profile; local inference needs its matching mmproj | Existing model service; [setup](images.md) |
| Reminder delivery | Configure Telegram owner, Redis, and reminder scheduling | Redis, Celery worker, and Celery Beat scheduler |

See [semantic memory](../README.md#semantic-memory-and-rag),
[voice setup](../README.md#local-voice), and [reminders](../README.md#reminders)
for their commands. Each optional foreground companion needs its own terminal.

## Optional Python calculations and charts

Follow [Python analysis setup](python-analysis.md), enable `ASSISTANT_PYTHON_ENABLED`,
and include `--extra python-runner` in the startup command. The launcher builds the
worker image and starts its private broker automatically. Docker is required even
when inference runs natively. No code runs directly on your Mac as a fallback.

## Optional tracing

To inspect request latency, model usage, tool failures and review cycles, follow
[Langfuse setup](observability.md). Tracing is disabled by default and does not change
the startup command. After configuring project keys, run `uv run assistant tracing-check`
and restart the API. Prompt and document content remain excluded unless you opt in.

## Alternative: run the API and bot in Docker

Follow [Docker deployment](../README.md#docker-deployment) instead of starting a second
native API/bot. On a Mac, keep native Metal inference running, with the configured
private inference key and `--host 0.0.0.0`, so containers can reach it through
`host.docker.internal`. The default native `127.0.0.1` binding is for native clients.

Use `docker compose --profile telegram logs --tail=50 bot` for bot errors and
`docker compose logs --tail=50 api` for API errors. After editing `.env`, recreate
the app containers so they receive the updated environment:

```sh
docker compose --profile telegram up -d --build --force-recreate api bot
```

`docker compose restart` alone does not reload environment variables. Keep exactly
one polling bot and one API instance. The same private `/id` owner setup applies to
the containerized bot.
