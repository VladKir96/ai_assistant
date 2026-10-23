# Architecture

This is a single-owner application with one API process. The Python package is a
modular monolith; PostgreSQL, Redis, inference, and optional integrations have
separate process/container lifecycles. Running several API workers is not supported:
the application currently serializes owner requests with a process-local lock.

For native development, `assistant start` supervises the local processes from one
foreground terminal. It validates configuration and port availability, starts/waits for
Compose PostgreSQL, applies migrations, and launches native inference, enabled companions,
the API, and optionally Telegram. HTTP readiness gates dependency startup. A project lock
prevents concurrent launchers; child failure or interruption stops the owned native
process groups. PostgreSQL remains Docker-managed. See [startup](startup.md) for usage
and the scope of configuration/readiness checks.

## Package layout

All runtime feature libraries are standard project dependencies. `uv run assistant start`
uses `.env` to select speech, PDF reading, MCP, embeddings, and Python analysis. The
installation includes their libraries even when a feature is disabled; disabled
companions are not started. Model weights and external native executables remain
explicit setup inputs. The application Docker image also includes FFmpeg.

Modules under `src/personal_assistant` are grouped by responsibility:

| Package | Contents |
| --- | --- |
| `app` | API, Telegram, CLI, startup, runtime wiring, jobs, demo, and evaluation |
| `agent` | Planning graph, conversation context, service, tools, and authorization |
| `core` | Settings, schemas, time utilities, HTTP guards, and tracing |
| `storage` | Database, ORM models, repositories, calendar storage, and artifacts |
| `integrations` | Web research, weather, MCP clients, and sandboxed Python analysis |
| `inference` | Model clients, profiles, context capacity checks, and embedding server |
| `media` | Images, vision, PDFs, speech recognition, and speech synthesis |

The package root contains only package initialization and the
`python -m personal_assistant` entry point. CLI commands remain unchanged. Direct module
references use the subpackage path, such as
`personal_assistant.app.jobs:celery_app` for Celery.

## Request execution

1. Authenticate the API request; Telegram separately checks both sender and private chat ID.
2. Look up the request UUID. Completed requests return the recorded result; conflicting
   reuse of a UUID is rejected.
   Image requests include the original-byte SHA-256 in this identity check. Uploads
   are decoded/normalized with byte/pixel limits and metadata removal. A capability-gated
   visual call produces a bounded untrusted observation, cached on the run for retry.
   Only this evidence enters graph state; original pixels/base64 are never stored in
   checkpoints/history. Completed turns archive the observation. Image answers always
   require review; the reviewer checks the observation rather than the original pixels.
3. Build compact conversation context and retrieve relevant notes/facts.
4. The LangGraph supervisor produces a validated single-specialist plan or 2–5 independent
   agent assignments, with at most five actions across the whole round. Each assignment
   includes a role, a scoped task and its actions. Roles may repeat; the primary role
   must be represented. Team plans cannot also include top-level actions.
5. Single-agent execution retains its direct tool/draft path. Team execution uses
   LangGraph `Send` to fan out specialist nodes, with a default three active agents.
   Read-only tasks execute their own ordered tools concurrently. If any action writes,
   `parallel_prepare` executes **all** tools in assignment/action order before fan-out,
   then checkpoints the results. Agent drafts run from that completed tool snapshot.
   Every action still passes the role allowlist and original-user-message authorization.
   Only a dedicated reducer channel accepts concurrent updates; each assignment writes
   a separate index/round-tagged result. Other shared state has a single writer.
   `synthesize` waits for all branches, merges their results in assignment order, checks
   round/index/role identity, and creates one draft. It uses at most 3 KB of agent draft
   excerpts plus 3 KB of tool text, with separate clipped task labels. Team size does
   not multiply the synthesis evidence budget.
6. All team answers, research/career answers, knowledge reads, and explicitly selected
   drafts enter the reviewer node.
   The graph enforces review for knowledge answers even if the model omits its review flag;
   knowledge operations consisting solely of mutations use the direct confirmation path.
   Rejected answers return through the checkpointed `revise` counter node to the
   supervisor. It sees the previous draft, review defects, bounded tool evidence and
   action arguments, and can choose a different specialist or run tools again. A
   prose-only correction uses an empty action list. Every repair round requires review.
7. Deliver the reviewed draft. Failed operations and unresolved review issues receive
   an application-generated notice, independent of the draft's wording.
8. Save the response and original messages in PostgreSQL.

The specialist roles are knowledge, productivity, research, career, analysis, and general conversation.
They have distinct prompts and enforced tool allowlists. Several instances of a role can
work on separate subtasks in one round, for example two research agents reading different
sources. Subtask text is a scope, never additional authorization or trusted evidence.
Agents do not recursively spawn more agents. There are no dependencies between branches;
work needing another branch's generated content/IDs must be deferred to later planning.

`MAX_PARALLEL_AGENTS` (1–5, default 3) bounds active specialist work. HTTP inference has
its own `MODEL_MAX_CONCURRENCY` semaphore (1–8, default 1). A single-slot local server
still serializes generations; network/tools can overlap independently. Higher model
concurrency requires a server with adequate slots/context/memory. These limits apply
inside one owner's request; multiple API processes are still unsupported.

`agent/tools.py` is the shared registry for action names, allowed/required parameters,
mutation classification, and role permissions. Pydantic validates parameter types;
the executor rejects missing required or unrelated parameters. Tool handlers live in
`agent/tool_execution.py`. Destructive writes require an independent standalone command
bound to one UUID: `/delete_note`, `/forget`, or `/cancel_event`. Negation, quotations,
hypotheticals, and discussion do not match that command grammar.

Other writes require an explicit note/task/calendar object in the imperative header;
content after a colon/newline cannot authorize additional operations. Edits and task
completion must name the target by UUID or its exact unique stored title. The application
resolves names and uniqueness from storage, independently of model-supplied action titles.
A successful edit retains this verified selector in request state, so renaming an item
does not break authorization during a subsequent review repair of that same item.

## Python analysis

The `analysis` specialist can collect permitted read-only evidence and invoke
`run_python`. A separate model call generates a small Python program; an authenticated
broker executes it in a disposable network-disabled Docker worker. The worker has no
host mounts or credentials. The API validates/stores PNG outputs and returns metadata;
Telegram downloads them through an authenticated artifact endpoint. All analysis drafts
require review and can return to planning for a code retry. See [setup and limits](python-analysis.md).
The Docker socket belongs only to the trusted broker, never the API or generated code.
Stable `analysis_id` values scope retries and artifact replacement to individual tasks.
Current evidence and attachments exclude superseded attempts without dropping other
completed analyses; old checkpoints use deterministic identities derived from their queries.

## Durable state and recovery

LangGraph uses its PostgreSQL checkpointer. A graph thread corresponds to one request
UUID, allowing a failed request to resume even after other conversations take place.
Conversational threads are separate archive identifiers. Each mutation and its operation
ledger entry commit in one database transaction. Replaying a failed specialist node
returns the recorded mutation result instead of creating another note or task.

Review repair defaults to two re-plans, configurable through `MAX_REVIEW_REPLANS`
(0–5). The counter and initial limit are saved in graph state and survive retries;
context preparation is not rerun, so interpretation of relative dates remains tied to
the original request clock. A separate 48-step LangGraph recursion cap covers the
largest supported team workflow (up to 37 supersteps with five repairs). The first
accepted review stops the loop; exhaustion returns the latest draft with an
application-generated notice. Status/response metadata
exposes the configured limit and actual number of re-plans.

Tool records accumulate across rounds. New reads actually execute, and newer results
replace earlier identical read attempts in bounded evidence. Successful writes with
identical validated arguments return their previous result instead of writing twice,
even if the planner reorders actions. An intervening write to a target invalidates an
older update result. Distinct/corrected writes use round-qualified operation IDs;
initial-round IDs retain their existing format for recovery compatibility. The original
user message remains the sole authorization source; reviewer feedback cannot authorize
an additional write. Persistent operation records protect partial specialist-node
replays if model generation fails after a commit. Full attempt records remain available
while exposed evidence has a shared 6 KB text budget and clipped action summaries.

Team action indices are global across assignments, preventing ledger-key collisions
between agents. Ordered tool preparation passes completed results to the next action,
so identical completed mutations are reused within a round as well as across repairs.
Successful parallel siblings are checkpointed even when another branch raises; retry
resumes failed work without re-running successful siblings. A synthesis retry reuses
the completed contributions and tool phase. Agent-output storage keeps at most five
assignment entries; synthesis rejects stale entries from another round.

All model calls are read-only HTTP operations. There is one shared model client and
one inference semaphore; agents do not require separate copies of Qwen. The application
never falls back from local to hosted inference automatically.

Named model profiles select the chat provider, file/alias, context and output budgets,
timeout, and thinking mode. The CLI validates a profile and atomically updates only
those fields in the private `.env`; credentials and storage configuration are excluded
from profiles. Processes load configuration at startup, so switching requires restarting
the API and, when applicable, local inference. Embedding and speech models have their
own configuration and lifecycles.

Each request captures a timezone-aware system-clock instant before context preparation.
`clock_context` derives the local date/weekday, UTC equivalent, and relative calendar
dates in the configured IANA timezone. Planning, drafting, review, and revision share
the same snapshot. The model adapter adds an authoritative clock block to system
instructions before enforcing the context budget; optional-context reduction cannot
discard it. Historical conversation and page dates cannot override that reference.
New requests refresh the snapshot; retries/resumed requests retain their original
reference, and completed request IDs replay their stored answer.

Authenticated `/v1/time` and the Telegram owner's `/time` command bypass inference and
return the API clock directly. `/v1/status` exposes the same clock for diagnostics.
These rely on the host OS clock, not an internet time lookup. Timestamp metadata is
separate from event dates mentioned inside documents. Local date ranges become UTC
boundaries in application code; lexical and vector queries apply filters before ranking.

## Memory and retrieval

- **Notes:** editable Markdown text, tags, source, timestamps, stable UUIDs, revisions.
- **Facts:** explicit user-requested memories, using a stable key as the title. A correction
  under the same key updates the record. Casual conversation is not automatically
  converted into personal facts.
- **Tasks:** independent records with completion state and optional timezone-aware due time.
- **Calendar:** independent timed/all-day events with original timezone, UTC start/end,
  creation/edit timestamps, cancellation state, and overlap queries. Calendar mutations
  share the same transactional operation ledger as notes/tasks. Overlaps are reported
  deterministically but do not block saving. Calendar operations enter response review.
- **Conversation archive:** original user and assistant messages, retrieved by a dedicated
  keyword history tool. Assistant statements remain labeled as assistant statements.
- **Thread summary:** bounded, derived context with an archive cursor. Original text remains
  available. Large messages are processed in bounded parts before the cursor advances.

Notes/facts use keyword search plus optional 1024-dimensional embeddings. Reciprocal rank
fusion merges keyword and cosine-distance rankings. Embeddings carry their model identifier;
changing embedding models requires reindexing. PostgreSQL performs exact vector search,
appropriate for an initial personal corpus. No ANN index is required at this scale.

The embedding service is optional and disabled by default. The API reports `keyword`
until it is explicitly enabled. Cross-language semantic retrieval requires the embedding
service and still needs real multilingual evaluation. The unit/integration tests use synthetic
vectors and do not establish the quality of BGE-M3 or Qwen.

Lexical relevance is ranked in SQL before the candidate limit, so newer partial matches
do not exclude an older exact match. Reindexing uses bounded batches, computes vectors
outside database transactions, and conditionally saves against the original title,
content and edit timestamp. Concurrent edits trigger a fresh snapshot and retry.
Committed work survives interruption; rerun maintenance to complete a partial reindex.

Embedding outages fall back to keyword retrieval and allow notes to save without vectors;
responses report this and `/v1/status` exposes `keyword_fallback` plus the warning.
The embedding client backs off after failures. MCP discovery is bounded to five seconds
per server and runs concurrently, caching success for five minutes and failures for
30 seconds. Unavailable servers or oversized catalogs produce notices and omit affected
tools; they do not prevent other tools or ordinary conversation from working.


Current notes are indexed as individual documents, with embedding input limited to 16 KB
of UTF-8 text. Optional PDF/OCR reads use a managed stdio adapter around mcp-pdf-reader:
folder-confined PDF paths, explicit OCR language data, three-page ranges, 50 MiB files,
and bounded excerpts with filename/page provenance. Its two read-only tools use the
existing MCP allowlist, untrusted-evidence handling and reviewer. Package write tools
are not exposed. No assistant secrets are forwarded to that server.
Telegram PDF documents enter through the authenticated multipart PDF route, with actual
byte/type checks and a 10 MB upload limit. Original PDFs are retained separately using
content-based names; an uploads/ namespace adds access through the same two MCP tools.
The API caches initial first-three-page evidence on the run and provides it to the
planner, specialist and mandatory reviewer. Captions alone authorize note/fact writes.
Matching retries reuse extraction/checkpoints; history retains file references/excerpts.
Docker gives the API a writable upload volume while reference PDFs remain read-only.
Long-document indexing and chunk-level retrieval citations remain future work.

## Context limits

The default context is 16,384 tokens, reserving 2,048 for output and another 2,048
for templates/overhead. Small inputs use a cheap conservative UTF-8 byte bound. Larger
inputs are counted by the configured provider: llama.cpp's `/tokenize` for local models
or `/responses/input_tokens` for hosted Responses providers. Local counting includes
system text, payload and schema; the separate template reserve remains necessary.
Unsupported counters use the byte bound and emit a diagnostic; counter failures are
reported without sending the request to another provider. No tokenizer package or
implicit model download is needed. Sources: [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting).

`assistant check-context` validates a complete minimum planner request, including
clock, capabilities, schema and the available MCP catalog, against the running model.
The native launcher performs this check before starting the API. Model/profile changes
require matching server settings; static profile validation cannot count a model that
is not running yet. The standalone check also works for manually started services.

Conversation context stays structured. Recent turns receive space before the old
summary, including for multibyte Unicode text and escaped content. Further reduction discards
old summaries and incidental retrieval before recent turns. A marker records reduction.
The latest request, full reviewed draft, and supplied tool evidence are retained; an
irreducible oversized request fails explicitly. Messages have a 12,000-character schema
limit, with actual model capacity checked on the complete request rather than a separate
4 KB service limit. Original messages remain in the archive.

## Integrations and permissions

Private `/v1/` routes authenticate in ASGI middleware before JSON or multipart parsing.
Request bodies are counted as chunks arrive and bounded before forwarding to the parser,
including when Content-Length is absent or false. API/embedding JSON bodies have a
256 KiB cap; speech permits 10 MB audio plus 64 KiB of multipart overhead and checks the
audio-file limit separately. The bounded middleware buffer trades a limited extra copy
for avoiding unauthorized multipart spooling and unbounded parser input.

Built-in tools manage application data directly. Optional MCP servers run over stdio
and must be configured by the owner, with each accessible tool explicitly listed.
`read_tools` is an owner declaration, not a security sandbox: only install servers you
trust. The model cannot add servers, execute arbitrary commands, or enable new tools.

Research fetches HTML/plain text, validates and pins public DNS addresses, revalidates
redirects, and limits downloads. It does not execute page instructions. Dynamic browser
automation, authenticated websites, external calendar/email writes, and application submissions
are not implemented in this version. Search excerpts are labeled as partial evidence.
Job search additionally filters results against owner-selected domains.

Search providers are explicitly selected: DuckDuckGo through `ddgs` is the default
keyless option, Tavily is an optional keyed provider, and `disabled` turns search off.
The DuckDuckGo adapter runs synchronous search in a thread, selects only its DuckDuckGo
backend, and serializes searches within the research service. Both adapters normalize
bounded, deduplicated results as untrusted search excerpts and filter job results
against configured domains. Provider failures never trigger automatic fallback.

`search_web` then opens the first two normalized public URLs concurrently through the
same protected page reader. Their extracted text/tables, rank, original/final URL,
UTC fetch timestamp, and truncation flag replace search snippets as answer evidence.
One failed page does not hide the other; both failures produce `no_page_evidence`.
The evidence formatter shares its byte budget between pages and preserves valid JSON,
marking shortened content or omitted pages explicitly. Draft/review share the current
bounded evidence. A rejected review can trigger a supervisor plan for another search
or page read; previous evidence remains available and refreshed reads supersede older
attempts with the same arguments. `search_jobs` continues
to return domain-filtered search excerpts until a vacancy page is explicitly read.
Weather policy requires matching location, valid forecast period, units, and value type;
source ranking and retrieval time alone do not establish reliability or freshness.

## Structured weather

`get_weather` is an allowed research/career read action and always enters review. Its
async Open-Meteo client uses fixed forecast/geocoding endpoints, independently of the
web-search provider. City queries override optional default coordinates; dominant
geocoding matches resolve automatically, while ambiguous candidates require clarification.
Defaults are explicit owner configuration rather than inferred from the user's timezone.

The forecast request retrieves current/daily/hourly data with Unix timestamps. Relative
periods resolve against the request's reference clock in the forecast location's IANA
timezone. The formatter preserves source units, dates, current validity time, fetch
time, cache age, missing values, and daily-aggregation UTC offset. Hourly conversion is
DST-aware; daily aggregation follows Open-Meteo's fixed-offset semantics and adds a
time note when a selected period crosses a DST change. Unsupported dates do not cause
substitution of another day's forecast. WMO codes map to explicit condition descriptions.

An in-memory LRU cache holds at most 128 entries: forecast TTL is ten minutes and
geocoding TTL is 24 hours. Transient transport/429/5xx failures receive at most three
attempts with bounded backoff. Errors are sanitized; expired data and search snippets
are not fallback weather evidence. HTTP clients are scoped to requests and closed.
Hourly arrays share columns to reduce context use; up to 48 rows are exposed, with
further context trimming preserving valid JSON/alignment and explicit omission counts.
The final graph node appends required source/license attribution independently of model
wording. No new dependency, database table, scheduled service, or API credential is needed.

## Speech

The Telegram bot sends authenticated audio uploads to a separate speech companion.
`stt_backend` selects whisper.cpp/FFmpeg or the included faster-whisper/CTranslate2
adapter. Faster-whisper loads a complete local model directory once at service startup,
disables ONNX Runtime telemetry, and uses local-only model loading. Explicit CLI model
downloads resolve and record a repository commit; runtime never downloads missing weights.
PyAV decodes/resamples input to mono 16 kHz with a hard three-minute sample limit;
compressed uploads are additionally capped at 10 MB. Recognition/VAD runs in a worker
thread under the companion's lock, including lazy segment iteration, so native inference
does not block its async event loop. Automatic language detection preserves the spoken
language; fixed `en`/`ru`/`sr` recognition can also be selected.

On this Mac, faster-whisper uses CPU/INT8; CTranslate2 does not expose Metal/MPS.
Speech and embedding libraries are included in the default app Docker image.
Containerized bots can access a companion through the configured speech URL; model
files and companion containers still need provisioning.
Voice input and audio replies have separate switches. Text replies work without Piper;
enabling audio replies also requires Piper voices and FFmpeg. The whisper.cpp backend
remains available with its existing executable/model settings.

Piper output and language detection are included in the standard installation and
run independently of voice input. `/audio on|off|status` controls a PostgreSQL chat preference keyed by the owner's
Telegram ID, outside LangGraph conversation state. The bot sends reviewed text first,
then submits cleaned prose to one background audio worker with three waiting slots.
Cancellation discards queued audio and closes the current speech request; the speech
server watches for disconnects, cancels synthesis, and kills/reaps child processes.
There is no extra reasoning agent or LLM call for speech.

The companion uses offline statistical language detection on the answer to select an
English, Russian, or Serbian Piper voice. Serbian Cyrillic and Latin scripts share
the Serbian route; Latin detection includes Lingua's Bosnian/Croatian models. One
predominant language is selected per reply. TTS and STT have separate locks. Each Piper
invocation loads its voice, then FFmpeg converts its WAV output to Telegram Ogg/Opus.
Audio reads at most 1,500 cleaned characters; a caption marks shortened output. See
[voice setup and limits](voice.md).

## Reminders and delivery

Celery Beat polls every 30 seconds; a worker claims due tasks using PostgreSQL row locks
and leases, then sends a reminder to the configured Telegram owner. Completed tasks are
excluded, old leases can be reclaimed, and attempts are bounded. The worker/scheduler
profile must be running, and overdue reminders are discovered when the machine wakes.
Expired final-attempt claims are finalized as `failed` before new due tasks are claimed,
so a crash cannot leave an exhausted notification permanently marked `sending`.

Database mutations are replay-safe. External Telegram delivery is **at least once**:
a crash between successful delivery and recording success can produce a duplicate.
Telegram has no application-provided idempotency key for this operation. A reviewer
cannot reverse external side effects and is not the permission enforcement layer.

## Privacy and deletion

API tokens, model keys, personal content, and Telegram URLs containing credentials are
not intentionally logged. Operational logs include request UUID, role, review status,
duration, and error category. Notes, histories, and checkpoints remain private database
content and are included in full database backups.

Deleting a note/fact removes its active record, embedding, and note revisions. It does
**not** erase original conversations, old operation results, LangGraph checkpoints, or
backups. There is no complete privacy-erasure command in this initial version. Do not
interpret `forget` as deletion of every historic copy.

Before exposing this service beyond a personal machine, add the relevant deployment
authentication, transport protection, retention policies, and distributed request locking.
The bundled Compose configuration publishes services on localhost.
