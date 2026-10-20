# Personal Assistant

A private multilingual assistant built with Python, LangGraph, Telegram, and
PostgreSQL/pgvector. Default local inference is **Qwen3.8-27B Q4_K_M** through
llama.cpp, with a native Metal backend on an Apple Silicon Mac.

This first working iteration includes supervisor routing, five specialist nodes,
conditional response review, persistent notes/facts/tasks/calendar events, conversation compaction,
search and page reading, optional MCP tools, image input for vision models,
and optional voice and embeddings services.
It is designed for one owner and one API process.

The model weights and live credentials are not bundled. Automated checks exercise the
application with controlled model responses; they do not establish live Qwen quality.

## Multi-agent workflow

The LangGraph supervisor selects one specialist for simple requests, or assigns
**up to five independent agent tasks per round**. Several instances of the same role
can work on different tasks. The entire round has at most five tool actions.
Independent read-only tasks run concurrently; their contributions are combined into
one answer and reviewed before delivery.

```mermaid
flowchart TD
    request["Telegram, CLI or HTTP API request"] --> input{"Image attached?"}
    input -->|No| context["Prepare context: clock, compact history, relevant memory"]
    input -->|Yes| vision["Validate image/model support; cache bounded visual observation"]
    vision --> context
    context --> supervisor["Supervisor: choose one agent or a bounded team; plan actions"]
    supervisor --> preparation["Team with writes: execute ALL tools in agent/action order first"]

    subgraph specialists["One agent or parallel agent instances: permitted tools and scoped drafting"]
        general["General: conversation and explanations"]
        knowledge["Knowledge: notes, facts and retrieval"]
        productivity["Productivity: tasks, reminders and calendar"]
        research["Research: web pages, search and weather"]
        career["Career: job search and vacancy assessment"]
        analysis["Analysis: isolated Python calculations and PNG charts"]
    end

    preparation --> general
    preparation --> knowledge
    preparation --> productivity
    preparation --> research
    preparation --> career
    preparation --> analysis
    general --> join{"Team request?"}
    knowledge --> join
    productivity --> join
    research --> join
    career --> join
    analysis --> join
    join -->|Yes: wait for all contributions| synthesis["Synthesize one answer from bounded drafts and evidence"]
    join -->|No| gate{"Review required?"}
    synthesis --> reviewer

    gate -->|Yes| reviewer["Reviewer: check answer against request and evidence"]
    gate -->|No| finish["Finalize answer and archive conversation"]
    reviewer -->|Rejected; budget remains| revise["Increment repair counter; retain feedback and evidence"]
    revise --> supervisor
    reviewer -->|Accepted or re-planning limit reached| finish
    finish --> reply["Reply to user"]
```

Review is mandatory for team answers, research/career answers, knowledge reads,
calendar operations, image/PDF input, and web, weather or MCP tool use; other drafts
can request review.
Rejected reviews return to the supervisor with the previous draft, feedback and tool
evidence. It can choose another specialist, fetch missing evidence, retry reads/failed
operations, or select no tools for a prose correction. Every repair round is reviewed.

The default is **two re-planning rounds after the initial attempt**. Set
`ASSISTANT_MAX_REVIEW_REPLANS=0..5` in `.env` and restart to change it; zero disables
automatic repairs. The request keeps its counter/limit across checkpoint recovery.
Identical successful writes return their recorded results; new authorized corrections
use distinct operation IDs. Tool feedback never grants write permission. At the limit,
the assistant returns its latest draft with an explicit unresolved-review notice.
`ChatResponse` includes `review_replans`, `max_review_replans`, and `specialists`
(the final round's participating agent roles, including repeated roles). The legacy
`specialist` field names the primary role.

`ASSISTANT_MAX_PARALLEL_AGENTS=3` limits concurrently active agents (1–5). The supervisor
can assign up to five tasks; excess tasks wait for a slot. Agent execution and inference
concurrency are separate: `ASSISTANT_MODEL_MAX_CONCURRENCY=1` preserves the default local
llama.cpp single-slot setup while independent tool/network work overlaps. Raise inference
concurrency only when the configured model server has capacity for it. Restart the API
after changing either setting; no new service is required.

For example: “Use two agents in parallel: read this report at [URL], and independently
list my tasks. Combine the results.” Ordinary requests can still use a single agent.

Agents work from the same request snapshot, rather than consuming other agents' drafts.
The synthesizer joins their current-round results in assignment order. If any planned
tool writes, **all** tools in that round run sequentially before drafts fan out; this
avoids racing edits and preserves read/write order. Existing user authorization and
mutation replay protection still apply. Work requiring a newly generated ID or content
must wait for another planning step; the planner must not invent those dependencies.

All agents share the configured local or hosted model. PostgreSQL stores notes,
facts, tasks, calendar events, conversation archives, and graph checkpoints; optional
pgvector embeddings support semantic retrieval. See [architecture](docs/architecture.md)
for tool permissions, memory, and recovery details.

## Run the offline demo

Install Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/), then run from this directory:

```sh
uv sync --frozen --extra dev --extra mcp
uv run assistant demo
```

The demo uses a temporary SQLite database and deterministic responses to exercise the
actual LangGraph workflow, note persistence, retrieval, and duplicate protection.
It makes no network calls and leaves no personal data or services behind.

## Start on your Mac

**Start with the [step-by-step startup guide](docs/startup.md).** It covers first-time
setup, the exact terminal layout, readiness checks, Telegram identity setup, stopping,
and recovery from common errors. Run commands from the project directory; keep your
existing `.env` and database volume across restarts.

After first-time setup, start the native stack from **one terminal**:

```sh
uv run assistant start
```

For faster-whisper voice input, include its optional dependencies:

```sh
uv run --extra speech assistant start
```

The launcher opens Docker Desktop on macOS if needed, waits for PostgreSQL, applies
migrations, then starts local inference, enabled speech/embedding companions, the API,
and Telegram. It checks readiness before starting dependent services. All native
processes stay attached to this launcher; **Ctrl+C stops them together**. PostgreSQL
continues running in Docker and keeps its data. Per-service logs are under the
timestamped `data/logs/local/` directory printed at startup.

Stop any individually launched model, API, speech/embedding, and bot processes before
switching to this command. Occupied service ports or another launcher cause an
actionable error. `--check` validates configuration without starting anything;
`--no-bot` starts the stack without Telegram. See the startup guide for optional
dependencies, timeouts, and custom-service limitations.

The default Mac setup runs PostgreSQL in Docker Desktop and the model server, API,
and Telegram bot directly on the host. For manual startup, use this order:

| Step | Where | Command | Continue when |
| --- | --- | --- | --- |
| 1 | Docker Desktop / setup terminal | `open -a Docker`, then `docker info` | Docker Server is available |
| 2 | Setup terminal | `docker compose up -d --wait postgres` | Database is healthy |
| 3 | Setup terminal, first start or after updates | `uv run assistant migrate` | Migrations complete successfully |
| 4 | Separate model terminal | `uv run assistant inference` | Model loaded and listening on port 8080 |
| 5 | Separate API terminal | `uv run assistant api` | Application startup complete |
| 6 | Setup terminal | `curl --fail --silent --show-error http://127.0.0.1:8000/health` | `{"status":"ok"}` |
| 7 | Setup terminal | `uv run assistant chat --text "Say hello briefly."` | An assistant reply |
| 8 | Separate bot terminal | `uv run assistant bot` | Private `/start`, `/notes`, and ordinary chat work |

This is a checklist for an already configured installation. For a fresh checkout,
complete the guide's dependency installation, private configuration, model file/runtime,
and Telegram `/id` steps first. The model, API, and bot commands stay in the foreground;
leave those terminals open. Starting the bot alone does not start its dependencies.

The inference launcher uses the saved model profile, GPU offloading, and one inference
slot. An explicit `--model-file` overrides the saved GGUF path for that launch. Record
the artifact checksum and runtime version when benchmarking; see the
[validated local Qwen setup](docs/model-validation.md).

API documentation: <http://127.0.0.1:8000/docs>. Private endpoints require the bearer
token from `.env`; the CLI supplies it automatically. If a request fails after a
write, retry identical input/thread with the `--request-id` printed by the CLI.

Notes/facts persist across conversations. Use an explicit request such as
`Remember my home city is Belgrade` to store a personal fact. Plain chat
does not automatically become permanent personal memory.

Deletion and cancellation require a separate command containing the exact resource ID:

```text
/delete_note <ID>
/forget <ID>
/cancel_event <ID>
```

Replace `<ID>` with the note, fact, or event ID returned by the assistant. These work
in Telegram or as CLI/API chat messages. Negated requests, quoted examples, and
discussion of deletion never authorize these operations. Creation and editing accept
explicit requests, such as `Save a note: ...`,
`Remind me tomorrow ...`, or `Create a calendar event ...`. Put the operation and object
before a colon/newline; text after it is content, not additional permission.
Edits and task completion must identify the item by its ID or exact unique stored title,
for example `Update the Project note: ...`.
Title matching is case-sensitive; duplicate titles require an ID. An ambiguous request
does not authorize a write, even when the model proposes one.

## Switch models

The agents, tools, notes, and conversation storage are independent of the chat model.
Keep named model presets in [model-profiles.toml](model-profiles.toml). Each preset
selects its provider, model name/file, context size, output limit, timeout, and thinking
setting. Credentials remain in `.env`.

```sh
uv run assistant models
uv run assistant models --use qwen-local
```

To add another local model, download a compatible GGUF into `models/`, copy the
commented local preset, and set its actual model file and supported context size.
Then select that profile with `--use NAME`. This updates only model settings in `.env`,
preserves credentials and storage settings, and also works with Docker Compose.
Use `--profiles-file PATH` for a separate presets file. Exported environment variables
still take precedence over `.env`.

Finish any active request before switching. Stop the current inference server and API,
then start them again so both load the same profile:

```sh
uv run assistant inference
uv run assistant api
```

Run those commands in separate terminals. For a containerized API, recreate it with
`docker compose up -d --build --force-recreate api`. A Linux inference container also
needs recreation with the existing inference profile/Compose override. For a cloud
profile, set its API key in `.env`, select the profile, and restart the API; the local
inference process is unnecessary. Profiles never start services or send requests.

The local runtime must support the model architecture, chat template, and JSON-schema
output. A GGUF file alone does not establish compatibility or agent quality. After a
switch, run `uv run assistant evaluate --output data/new-model-evaluation.json` for
the multilingual planner checks, then test representative assistant workflows.

Changing the chat model retains notes, facts, tasks, and conversations. It does not
require reindexing; changing the separate embedding model does. Voice models are also
configured separately from these chat-model profiles.

## Telegram

Create a bot through Telegram's BotFather. Put its token and your numeric Telegram user
ID in `.env` as `ASSISTANT_TELEGRAM_TOKEN` and `ASSISTANT_TELEGRAM_OWNER_ID`.
Do not put these values into source files, screenshots, issues, or this README.

`ASSISTANT_TELEGRAM_TOKEN` must contain the complete token from BotFather: a numeric
prefix, a colon, and the secret suffix, with no spaces. A username or only the secret
suffix will fail the local format check before any network request. Your personal
numeric user ID belongs in `ASSISTANT_TELEGRAM_OWNER_ID`, separately from the token.

To discover your user ID, leave `ASSISTANT_TELEGRAM_OWNER_ID=0`, start the bot, and send
`/id` in its private chat. Set the returned number in `.env` and restart the bot.
The numeric prefix of the bot token is the **bot's** ID and must not be used as your
owner ID. The `/id` command returns only the sender's own ID; assistant commands and
stored data remain restricted to the configured owner.

```sh
uv run assistant bot
```

`/id` returns the sender's own ID in a private chat. All assistant commands and data
require the configured owner's private chat. Commands: `/start`, `/help`, `/new`,
`/notes`, `/tasks`, `/export`, `/agenda` (next seven days), `/calendar` (ICS export).
The assistant supports multilingual free-form requests. `/new` starts a conversation
while retaining notes/facts; the bot's currently selected thread
is process-local and resets to its default thread after a bot restart.

## Semantic memory and RAG

Keyword retrieval works immediately. To add multilingual semantic search, install and
start the local embedding companion (the first start downloads BGE-M3 weights):

```sh
uv sync --frozen --extra dev --extra mcp --extra embeddings
uv run assistant embeddings
```

Set these values in `.env`, restart the API, and reindex existing notes:

```dotenv
ASSISTANT_EMBEDDING_BACKEND=http
ASSISTANT_EMBEDDING_BASE_URL=http://localhost:8081/v1
ASSISTANT_EMBEDDING_MODEL=BAAI/bge-m3
```

`assistant init` sets the embedding API key to the application token. Set
`ASSISTANT_EMBEDDING_DEVICE=mps` to test Apple GPU embedding inference, or keep `cpu`
for portability. The database expects 1024 dimensions. Changing models requires reindexing.

```sh
uv run assistant reindex
```

If embeddings are unavailable, notes still save and keyword retrieval remains available;
the response reports the fallback. Reindex after recovery to add missing vectors.
Reindexing works in bounded batches and retries notes changed during encoding. Completed
updates remain committed if interrupted; rerun the command to finish safely.

## Dates, times, and the private calendar

Notes and personal facts retain their original `created_at` and latest `updated_at`.
Both timestamps are exposed in retrieval/API results and Markdown exports. The date
an event is described as occurring inside a note remains separate from those metadata
timestamps. Conversation retrieval and compaction also retain message timestamps.

The API captures the system clock in your configured IANA timezone for every new
request. Local date, weekday, yesterday/today/tomorrow, and Monday-based week boundaries
are calculated in Python. This clock is included in the model's system instructions
and shared by planning, drafting, review, and revision; old conversations cannot set
the current date. Configure:

```dotenv
ASSISTANT_TIMEZONE=Europe/Belgrade
```

After this update, restart both the API and bot. Send **`/time`** in the owner's private
Telegram chat for the API's date/time, UTC offset, timezone, and weekday without using
the language model. Authenticated `GET /v1/time` returns the same clock metadata;
`GET /v1/status` also includes a `clock` object for diagnostics. A chat response uses
the reference time captured at the start of its request, so it can be seconds behind
the time at delivery. Replaying a completed request ID returns its original answer.

The clock comes from the machine running the API, including when running in Docker.
Keep the Mac's automatic date/time enabled. If `/time` is wrong, check the Mac's clock,
`ASSISTANT_TIMEZONE`, and whether the API was restarted; if `/time` is correct but a
natural-language answer is wrong, that is a model/planning issue rather than a missing
clock. No website or language model is used to establish the system time.

Storage timestamps use UTC. Note/history date ranges are interpreted as local calendar
dates in this timezone, with an inclusive start and exclusive end. They respect local
day boundaries and daylight-saving changes. Example requests:

- `Show notes I created yesterday.`
- `Show notes updated last week.`
- `When did I create this note, and when was it last edited?`

The model translates natural-language date requests into tool arguments; SQL filters
the timestamps independently of keyword/vector similarity. Results are bounded rather
than an exhaustive unlimited archive export. The `/v1/notes` and `/v1/memories` API
endpoints also accept `date_from`, `date_to` (YYYY-MM-DD), and `date_field` (`created_at`
or `updated_at`).

The **built-in private calendar** stores events in PostgreSQL, separately from tasks
and notes. It supports timed/all-day events, agenda queries, moving/editing events,
cancellation, and overlap warnings. Timed events have an explicit start, end, and IANA
timezone; missing or ambiguous details require clarification. Nonexistent DST times
are rejected; repeated times need an explicit offset to choose the intended occurrence.
All-day event end dates are exclusive. Overlaps are allowed and reported with an
application-generated notice, even if the model omits them from its prose.

After upgrading, run migrations before restarting the API and bot:

```sh
uv run assistant migrate
```

Try these in the owner's private Telegram chat:

- `Create a calendar event called Interview tomorrow from 10:00 to 11:00.`
- `What is on my calendar tomorrow?`
- `Move calendar event <ID> to 11 November 2030, 13:00–14:00 Europe/Belgrade.`
- `/cancel_event <ID>`
- `/agenda` to list the next seven days, or `/calendar` to download an ICS snapshot.

Replace `<ID>` with the actual event ID returned by the agenda. Event title searches
can find events outside the default seven-day window; use date ranges for specific days.
API reads: `/v1/events`, `/v1/events/{id}`, and `/v1/calendar.ics` (all authenticated).

You can also export active events from the terminal:

```sh
uv run assistant calendar-export --output data/calendar.ics
```

The file is an importable snapshot with stable event IDs, not a live calendar sync;
later edits/cancellations are not automatically reflected in another app. Recurrence,
calendar import/sync, meeting invitations, and automatic event notifications are not
implemented. A calendar entry does not schedule a reminder: ask explicitly for a
separate reminder task and run the reminder services below when notifications are needed.

## Research and career workflows

Web search defaults to **DuckDuckGo through the third-party `ddgs` library**, with
no API key or account required. Set your preferred provider in `.env`:

```dotenv
ASSISTANT_SEARCH_PROVIDER=duckduckgo
ASSISTANT_DUCKDUCKGO_REGION=wt-wt
```

`wt-wt` uses worldwide results; `us-en` and `ru-ru` are regional alternatives. Queries
can be multilingual. Only the DuckDuckGo backend is selected; the library's
multi-provider `auto` mode is not used. Search queries are sent to the selected
provider; private notes are not automatically attached to search requests.

This is an unofficial web-search integration rather than DuckDuckGo's official
Instant Answer API, which [does not provide full web search results](https://github.com/duckduckgo/duckduckgo-publisher/blob/master/share/site/duckduckgo/api.tx).
DuckDuckGo can block or rate-limit automated requests. Failures are reported explicitly;
the app does not silently switch to another search provider. A search excerpt remains
partial evidence and is not equivalent to reading the linked page.

For general web searches, the assistant now **opens the first two usable result URLs**
and passes their extracted page text (including HTML tables) to the answer and reviewer.
It preserves search rank, source links, fetch timestamps, and truncation flags. Failed
page reads are reported separately; snippets are not substituted as page evidence.
The first two results are a starting point, not a guarantee of source quality.

Weather questions use the dedicated Open-Meteo integration described below rather
than search snippets or the first two result pages.

To use Tavily, explicitly set `ASSISTANT_SEARCH_PROVIDER=tavily` and supply
`ASSISTANT_TAVILY_API_KEY`. To turn off search, set `ASSISTANT_SEARCH_PROVIDER=disabled`.
Public HTML/plain-text URL reading works independently of the search provider/key.

Restart the API to load this page-reading change or new search settings: stop its
terminal process with Ctrl+C, then run `uv run assistant api` again. The Telegram bot
uses that API; it does not need a restart for this change.
The authenticated `/v1/status` endpoint reports `search_provider`, `search_configured`, and
`web_capabilities`, which are also supplied to the supervisor, specialist,
and reviewer. These describe configured support; individual sites can still fail or
require JavaScript/login that the current page reader does not support.

Job sources are intentionally empty until you choose them. Configure bare hostnames:

```dotenv
ASSISTANT_JOB_SITES=["jobs.example.org","careers.example.com"]
```

The career agent can search selected domains and compare source excerpts against your
known preferences. Site-specific connectors, browser automation for JavaScript/login
pages, vacancy tracking, scheduled search digests, and application submission are future
work. The assistant does not bypass website access restrictions.

## Weather forecasts

The read-only `get_weather` tool uses [Open-Meteo](https://open-meteo.com/en/docs)
for current model conditions, daily forecasts, and hourly data. No account or API key
is required for this personal, non-commercial use. Open-Meteo's [free-service terms](https://open-meteo.com/en/terms)
include request limits; commercial use requires a suitable service plan. Answers
automatically include Open-Meteo/CC BY 4.0 attribution and GeoNames credit when geocoding
is used. No additional Python dependency is needed: the client uses async HTTP/JSON.

Set a default location in `.env` to avoid repeating it. For example, Belgrade:

```dotenv
ASSISTANT_WEATHER_ENABLED=true
ASSISTANT_WEATHER_DEFAULT_NAME=Belgrade, Serbia
ASSISTANT_WEATHER_DEFAULT_LATITUDE=44.804
ASSISTANT_WEATHER_DEFAULT_LONGITUDE=20.4651
```

The public configuration template leaves coordinates empty. Supply both coordinates
or neither; without a default the assistant asks for a location. Naming another city
overrides the default through Open-Meteo geocoding. A place can include a country/region,
such as `London, GB` or `Paris, France`. Ambiguous matches are returned as
candidates for clarification, rather than silently choosing between similar cities.
Explicit `latitude,longitude` requests also work. Timezone alone never sets a location.

Restart the API after this update or after changing defaults: Ctrl+C in its existing
terminal, then `uv run assistant api`. No migration, new model, or bot restart is needed
for weather. Containerized APIs need their image rebuilt and container recreated.
Try:

- `What is the weather right now?`
- `What is tomorrow's weather? Include minimum and maximum temperatures.`
- `Show the hourly forecast for tomorrow.`
- `What is tomorrow's weather in London, UK, in Fahrenheit?`

Forecasts support up to 16 days, including today; `week` means the next seven days.
Explicit date ranges use an inclusive `date_from` and exclusive `date_to`. Hourly output
is capped at 48 rows and can be shortened further by the context budget; omissions are
reported explicitly. Hourly “current” requests cover the next 24 hours. Celsius, km/h,
and mm are the defaults; Fahrenheit temperature requests are supported.

Current/daily/hourly values include their units and valid times; fetch time and cache
age are separate. Forecasts are cached in memory for ten minutes, place lookups for
24 hours, with bounded cache size and at most three attempts for transient failures.
Expired data is not served when refresh fails, and snippets are never used as fallback.
Current conditions more than two hours behind or fifteen minutes ahead of the request
clock are rejected, even after a fresh fetch; an incorrect OS clock can trigger this check.
Search can be disabled while weather remains enabled. Only requested/default coordinates
or an explicit place name are sent to Open-Meteo, not conversation archives or notes.

Hourly Unix timestamps are converted with the returned IANA timezone, including DST.
Open-Meteo's daily aggregation uses the fixed response UTC offset; if a selected period
crosses a DST change, the evidence includes that limitation. Model conditions and
forecasts are not direct station observations. Missing values remain unknown, not zero.

## Reminders

Create a task with an explicit reminder time, for example `Remind me tomorrow at 10:00
to review my project`. The planner receives the current time in `Europe/Belgrade` by
default; change `ASSISTANT_TIMEZONE` if needed. Ambiguous reminders should ask a question.

Reminder delivery requires Telegram configuration plus Redis, one worker, and one scheduler:

```sh
docker compose up -d --wait redis
uv run celery -A personal_assistant.jobs:celery_app worker --pool=solo --loglevel=INFO
```

In another terminal:

```sh
mkdir -p data
uv run celery -A personal_assistant.jobs:celery_app beat --loglevel=INFO --schedule=data/celerybeat
```

Tasks persist while processes are stopped; reminders are delivered only when the required
services run. Telegram delivery is at least once and may duplicate after a crash.
Inspect failed delivery state through `/v1/tasks`. An expired fifth delivery claim
becomes `failed` on the next worker poll; it cannot remain `sending` indefinitely.

## Python calculations and charts

The `analysis` specialist executes small Python programs using NumPy and Matplotlib in
disposable Docker sandboxes. It calculates statistics and creates charts/histograms,
then returns up to three PNG attachments through Telegram or the API. No vision model
is needed. Code has no network, host files or credentials; time/memory/output are bounded.
Analysis is reviewed, and failed code can be regenerated within the existing repair
limit. This role also participates in parallel teams.

Enable `ASSISTANT_PYTHON_ENABLED=true`, then start with:

```sh
uv run --extra speech --extra pdf --extra python-runner assistant start
```

The launcher builds the worker image and starts its private broker. See
[Python analysis setup and limits](docs/python-analysis.md) for native/Docker deployment,
examples, downloads, and the trusted broker's Docker access.

## Local voice

Voice input supports **faster-whisper** and the existing whisper.cpp backend.
For [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3),
install the optional dependencies and download the complete CTranslate2 model directory
(about 3.1 GB, once):

```sh
uv sync --frozen --extra dev --extra mcp --extra speech
uv run --extra speech assistant speech-download
```

Configure `.env` for voice input with text replies:

```dotenv
ASSISTANT_STT_BACKEND=faster_whisper
ASSISTANT_FASTER_WHISPER_MODEL=models/faster-whisper-large-v3
ASSISTANT_FASTER_WHISPER_DEVICE=cpu
ASSISTANT_FASTER_WHISPER_COMPUTE_TYPE=int8
ASSISTANT_FASTER_WHISPER_CPU_THREADS=8
ASSISTANT_STT_LANGUAGE=auto
ASSISTANT_VOICE_ENABLED=true
ASSISTANT_VOICE_OUTPUT_ENABLED=false
ASSISTANT_SPEECH_BASE_URL=http://localhost:8765
```

On Apple Silicon, use CPU/INT8: CTranslate2 supports ARM64 CPUs and NVIDIA CUDA GPUs,
[not Apple Metal/MPS](https://opennmt.net/CTranslate2/hardware_support.html).
The model's name does not establish that it is faster than whisper.cpp/Metal on a Mac.
`auto` detects the input language; `en` or `ru` can fix it explicitly. Audio is transcribed
in its original language. Runtime uses local files only; it never downloads replacement
weights or sends audio to a cloud recognition service. ONNX Runtime telemetry is disabled.

Start the speech companion in its own terminal and wait for `Application startup complete`:

```sh
uv run --extra speech assistant speech
```

Check it in another terminal:

```sh
curl -fsS http://127.0.0.1:8765/health
```

For faster-whisper, expect `status: ok`, `stt_backend: faster_whisper`, and
`model_loaded: true`. Then stop/restart `uv run assistant bot` to load the voice settings,
and send a voice message in your private chat. Input is limited to three minutes/10 MB;
decoded audio duration is checked independently of Telegram metadata. The speech HTTP
timeout defaults to 600 seconds and can be changed with `ASSISTANT_SPEECH_TIMEOUT_SECONDS`.

Switch recognition models by downloading another **CTranslate2 Whisper** repository and
changing `ASSISTANT_FASTER_WHISPER_MODEL` to its directory, then restarting speech:

```sh
uv run --extra speech assistant speech-download --repo-id Systran/faster-whisper-medium --output models/faster-whisper-medium
```

`--revision <commit>` pins a specific revision; each download records its resolved commit
in `assistant-model.json`. These models are separate from chat-model GGUFs and whisper.cpp
weights. To use whisper.cpp, set `ASSISTANT_STT_BACKEND=whisper_cpp` and configure
`ASSISTANT_WHISPER_EXECUTABLE` / `ASSISTANT_WHISPER_MODEL`; this backend needs FFmpeg.

For optional audio replies, install [Piper](https://github.com/OHF-Voice/piper1-gpl) and
FFmpeg, download Piper voices for the configured `en`/`ru` speech outputs with their
matching `.onnx.json` files, configure their existing settings, and set
`ASSISTANT_VOICE_OUTPUT_ENABLED=true`.
TTS reads the first 1,500 characters and labels partial audio. Faster-whisper input alone
needs neither Piper, whisper-cli, nor a system FFmpeg executable; PyAV decodes the audio.

The companion is HTTP-based, so a Docker bot can use the native Mac service: start speech
with `--host 0.0.0.0` for container access; Compose uses `host.docker.internal:8765`.
The default application Docker image does not install the optional speech extra or weights.
Synthetic multilingual Ogg/Opus transcription has been checked with the real large-v3
model. Real personal voice accuracy and live Telegram delivery still need a user check.

## Images with vision models

Send one Telegram photo or JPEG/PNG/WebP file with a caption, or use:

```sh
uv run assistant chat --image screenshot.png --text "What does this screenshot show?"
```

The selected model must support images and have `model_supports_images=true` in its
profile (or `ASSISTANT_MODEL_SUPPORTS_IMAGES=true` in `.env`). Local llama.cpp also
needs the model's matching `local_mmproj_file`; startup passes it as `--mmproj` and
image requests verify the running server's vision capability. Existing text profiles
remain disabled for images. See [image/model setup](docs/images.md).

The limit is one static image, 10 MB and 20 megapixels. Images are oriented, stripped
of metadata, resized to at most 1280 pixels per edge and analyzed once. Only their
hash and bounded model observation are retained for history/retries. Image answers
receive review; the reviewer checks the observation, not the original pixels.
This update needs migration `0003`; `assistant start` applies it automatically.

## Optional MCP integrations

### Local PDF reader

The managed `pdf-reader` server uses
[mcp-pdf-reader 0.1.7](https://pypi.org/project/mcp-pdf-reader/) through a read-only
adapter. It lists PDFs and extracts bounded page excerpts, with optional OCR using the
configured language data. Files stay local; extracted excerpts go to your configured
chat model, including a hosted provider if selected. Send a PDF as a **Telegram file/document**, optionally with
a question in its caption, or put a local PDF in `data/pdf`. Uploads need no vision model.
PDF URL downloads are not implemented. See [PDF setup and usage](docs/pdf.md).

```sh
uv sync --frozen --extra dev --extra mcp --extra speech --extra pdf
mkdir -p data/pdf data/tessdata
```

Set `ASSISTANT_PDF_ENABLED=true` in your private `.env`, then restart with:

```sh
uv run --extra speech --extra pdf assistant start
```

Ask, for example: `Read the first page of resume.pdf` or
`List the PDFs available to you`. Each read allows up to three pages of a PDF up to
50 MiB; excerpts include filename/page provenance. Long-document indexing is separate.
Telegram/API uploads allow one PDF up to 10 MB. They are retained in `data/pdf-uploads`
(a named volume in Docker), with safe content-based names for follow-up reads. A
captionless upload summarizes a bounded excerpt from the first three pages; it does
not analyze the entire PDF. Migration `0004` is applied by `assistant start`.
No separate server terminal or `data/mcp.json` entry is needed for this integration.

### Other trusted servers

Built-in notes/tasks do not require MCP. For a trusted local stdio MCP server, create
an untracked `data/mcp.json`, using absolute executable paths and explicitly allowed tools:

```json
{
  "my_service": {
    "command": "/absolute/path/to/server-executable",
    "args": [],
    "read_tools": ["search_items"],
    "environment_variables": ["MY_SERVICE_TOKEN"]
  }
}
```

Set `ASSISTANT_MCP_SERVERS_FILE=data/mcp.json`. Only named environment variables are
forwarded. The model cannot install servers or expand the allowlist. `read_tools` is
your declaration of trust, not a sandbox; it must not include tools with write effects.
Remote MCP transports and external write approvals are not implemented yet.
For Docker, install the chosen server inside the image and mount its configuration
read-only; a native macOS stdio executable cannot run inside a Linux container.

## Tracing and monitoring with Langfuse

Optional Langfuse tracing records request and graph-step latency, model generations
and provider-reported tokens, tool outcomes, and reviewer/supervisor repair cycles.
It works with local and hosted models, native startup, and the Docker API image.
Tracing is disabled by default; prompt, note, and document content is excluded unless
you explicitly enable content capture.

Configure project keys and an endpoint in `.env`, run `uv run assistant tracing-check`,
then restart the API. See [Langfuse setup, privacy, and monitoring](docs/observability.md)
for the complete configuration and dashboard workflow. A Langfuse backend is separate
from the assistant; it can be cloud-hosted or self-hosted.

## Docker deployment

This is an alternative to the native API/bot startup above. Stop any native API or
bot before starting its container, and use a single inference server. Complete the
same `.env`, model, and Telegram identity setup first. Open Docker Desktop and wait
until `docker info` succeeds.

Keep native Metal inference in its own terminal, using the configured model path:

```sh
uv run assistant inference --host 0.0.0.0
```

Wait for the model to load and listen on port 8080. In a separate terminal, start
the API and database; Compose applies migrations through its `migrate` service:

```sh
mkdir -p data/pdf data/tessdata
docker compose up -d --build --wait api
curl --fail --silent --show-error http://127.0.0.1:8000/health
uv run assistant chat --text "Say hello briefly."
```

Once these checks pass, start the bot and check its private `/start` command:

```sh
docker compose --profile telegram up -d bot
docker compose --profile telegram logs --tail=50 bot
```

For reminders, also start:

```sh
docker compose --profile reminders up -d worker scheduler
```

The native inference launcher requires a configured private key when binding beyond
localhost. Compose reaches native services through `host.docker.internal`. If using
native speech/embeddings with containerized clients, likewise start their companions
with `--host 0.0.0.0`; their endpoints require the application token. Default published
database/API/Redis ports bind only to localhost.

For a portable **CPU inference** Linux deployment:

```sh
docker compose -f compose.yaml -f compose.linux.yaml --profile linux-inference up -d --build
```

Set `ASSISTANT_LOCAL_MODEL_FILE` to your GGUF path. This profile does not enable CUDA;
GPU Linux deployment needs a corresponding llama.cpp image/device configuration.
Ordinary Linux containers on macOS do not provide direct Metal access.

Stop services with `docker compose stop`; persistent database volumes remain intact.
Use one API process. Configuration is loaded at process startup. After editing `.env`,
recreate affected app containers with `up -d --force-recreate`; `docker compose restart`
alone keeps the previous environment. See [startup troubleshooting](docs/startup.md#troubleshooting-by-symptom).

## Cloud inference

Explicitly set `ASSISTANT_MODEL_PROVIDER=openai`, `ASSISTANT_OPENAI_API_KEY`, and
`ASSISTANT_OPENAI_MODEL`. This uses the Responses API with `store=false`. Selected
context will be sent to that provider; `store=false` is not a promise of zero provider
retention. There is no automatic local-to-cloud fallback. Other providers can be added
behind the `LanguageModel` interface.

## Tests and model evaluation

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The PostgreSQL test is skipped unless `TEST_POSTGRES_URL` is set. To run it in isolation:

```sh
docker compose -f compose.test.yaml up -d --wait
ASSISTANT_DATABASE_URL=postgresql+psycopg://assistant:testing-only@localhost:55432/assistant_test uv run assistant migrate
TEST_POSTGRES_URL=postgresql+psycopg://assistant:testing-only@localhost:55432/assistant_test uv run pytest
docker compose -f compose.test.yaml stop
```

GitHub Actions runs lint, format checks, migrations, the database test, and the offline demo.
Once a real inference server is available, run the multilingual planner benchmark:

```sh
uv run assistant evaluate --output data/planner-evaluation.json
```

This sends only synthetic cases to the selected provider and executes no tools. It
measures routing/tool selection and latency, not complete answer or reviewer quality.
Cases may supply synthetic web-capability flags to test configured search or URL-only
reading without installing credentials or making external tool calls.
No real-model evaluation result is claimed before running it.

## Export and backups

```sh
uv run assistant export --output data/my-notes.md
mkdir -p backups
docker compose exec -T postgres pg_dump -U assistant -d assistant -Fc > backups/assistant.dump
```

Markdown export includes active notes and facts. A database backup includes conversations,
revisions, tasks, operations, and checkpoints as well. Back up regularly to your chosen
private destination; automatic backup scheduling is not configured. Do not commit dumps.

Deleting a note or fact removes its live record, vector, and note revisions. It does not
erase original conversations, historical operation/checkpoint data, or backups. Full
privacy erasure and retention automation require additional implementation.

See [architecture](docs/architecture.md) for boundaries and tradeoffs. Project code uses
the MIT license; third-party software and model weights retain their own licenses.
