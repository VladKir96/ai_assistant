# Verification record

Standard feature installation verified on 2026-10-23:

- `uv run assistant start` now includes speech, TTS, PDF/MCP, embeddings, and Python
  runner libraries by default. `.env` selects enabled services. Old feature extras
  remain empty compatibility aliases; the developer test/lint extra is separate.
- `.env.example` and the local private `.env` were grouped into numbered sections,
  with feature switches first and model/binary requirements explained beside settings.
  Every existing private value and effective setting was preserved; missing keys
  were filled with documented defaults. Private file permissions remain 0600.
- Plain `uv run --frozen assistant start --check` passed against the local settings
  without starting services. The full suite passed **435 tests, 12 skipped**; the
  five explicit Docker sandbox tests passed separately. Lint/format, lock consistency,
  documentation links/shell syntax, CI YAML/shell syntax, and the offline demo passed.
- The default Linux ARM64 application image built with CPU PyTorch and FFmpeg.
  All feature libraries imported with container networking disabled; FFmpeg encoded
  a synthetic Ogg/Opus sample, and the image's offline assistant demo passed.
  No live model requests, Telegram delivery, or production migrations were performed.

Older records below describe their original commands. Use the current startup guides
for the single-command startup instead of historical feature-extra examples.

Non-Mac Docker setup documentation checked on 2026-10-23:

- Added a standalone Linux and Windows/WSL 2 guide covering containerized credential
  initialization, local CPU or hosted inference, migrations, text-chat checks, Telegram,
  optional features, updates, logs, and persistent data. Host Python/uv and a native
  llama.cpp executable are not required by this setup.
- The Linux inference override now waits for the model health check before starting
  the API, with a longer health-check window for model loading. Compose configuration
  validation passed for local text, Telegram, reminders/analysis, vision, hosted text,
  and hosted optional-service configurations using temporary dummy credentials.
- Shell syntax, local documentation links/anchors, Ruff lint/format, and whitespace
  checks passed. The initializer was exercised in a temporary directory: random
  credentials, private file permissions, and refusal to overwrite `.env` were checked.
- Docker's daemon was not running during these checks. No container build/startup,
  live inference, Telegram delivery, or Windows/WSL execution was performed for this
  documentation change; configuration validation is not an end-to-end deployment test.

Piper Telegram audio mode verified on 2026-10-22:

- **426 passed, 6 skipped** with speech/TTS extras and isolated PostgreSQL enabled.
  Skips are the explicit OCR-data check and five optional Python Docker worker checks.
  Ruff lint/format, lock consistency, documentation links, whitespace checks, and the
  offline demo passed. CI now also installs the optional TTS extra in its speech job.
- Migration `0005` passed on the isolated test database; `alembic check` reported no
  pending schema operations. Concurrent PostgreSQL upserts and a fresh database
  connection verified preference persistence. The test database was stopped afterward.
- Dispatcher tests cover private-owner access, `/audio on|off|status`, persistence across
  `/new` and bot restarts, text-first output, and text/voice/image/PDF questions.
  Other checks cover offline script/language detection, cleaned/truncated prose, bounded
  queues, failure recovery, cancelled requests, server disconnect handling, and actual
  child-process termination/reaping. TTS-only startup requires no Whisper weights.
- Real Piper 1.8.0 and Lingua 2.2.0 synthesized English, Russian, Serbian Latin, and
  Serbian Cyrillic samples through the speech HTTP app. Each response had the expected
  detected language and decoded as nonempty Ogg/Opus (roughly 2.8–3.4 seconds per sample).
  Conversion used a temporary imageio-ffmpeg binary; no system FFmpeg was installed.
  Voices are in ignored `models/piper`; samples/results are in `data/tts-validation`.
- Speech quality was not assessed by listening, and live Telegram delivery was not
  exercised. The community Serbian voice remains subject to its author's prosody caveat.
  No private `.env` change or production migration was made; follow [voice setup](voice.md)
  and restart the services to enable the feature.

Final code-review fixes verified on 2026-10-20:

- Writes now require explicit note/task/calendar intent; edits/completion bind to an
  exact UUID or unique stored title. Regression checks cover unrelated verbs, quoted/
  escaped/incomplete instructions, multilingual commands, wrong targets, duplicate titles,
  and a rename followed by review repair. Authorization metadata comes from storage.
- Python tasks have stable analysis identities. Partial repairs preserve other tasks'
  charts and replace only the repaired task's evidence/artifacts, including failed or
  empty replacements. Team-to-single-agent repair, cached replay and old checkpoints
  are covered. Planner guidance was shortened to preserve fallback context-budget checks.
- Expired final reminder claims become `failed`; active leases and retryable claims
  remain correct. SQLite and isolated PostgreSQL checks verify recovery and exhaustion.
- Unknown/non-text page charsets fall back to UTF-8; decoding failures remain local to
  a page/tool. Tests cover Cyrillic text, legacy encodings and a failed search source
  alongside a usable source.
- CI now installs the Python runner extra, builds `Dockerfile.python`, and enables the
  real Docker execution/isolation tests. Workflow YAML was validated locally.
- Final suite: **406 passed, 1 skipped** (OCR language data not configured for this run).
  Real Docker checks and isolated PostgreSQL were enabled. Migrations/schema consistency,
  Ruff lint/format, offline lock consistency, documentation links and offline demo passed.
  The test database was stopped afterward. No live model or Telegram delivery was tested.

Python analysis agent verified on 2026-10-20:

- Added an `analysis` role and allowlisted `run_python` tool. The configured model
  generates a bounded program; a private authenticated broker executes it only in a
  disposable Docker worker. NumPy 2.5.4 and Matplotlib 3.11.2 are locked in an isolated
  worker dependency group; Docker SDK 7.2.0 is the optional native `python-runner` extra.
- Worker and application images built on Linux ARM64. A broker running inside the
  application image, with a Docker socket and no network, successfully executed a
  separate unprivileged worker: `7 * 9 = 63`, one PNG line chart. The temporary broker
  and job were removed. Report: `data/python-docker-smoke.json`.
- Real Docker tests verify arithmetic, PNG creation, UID 65534, no assistant environment
  credentials/socket/project files, denied networking, timeout/output limits, symlink
  rejection and container cleanup. Other tests cover bounded authenticated requests,
  cancellation/slot ownership, code generation/data dependencies, review/re-execution,
  cached responses, metadata stripping, private downloads, Telegram photo delivery
  with a mocked dispatcher, configuration-secret redaction, and native startup wiring.
- Live Qwen3.8-27B UD-Q4_K_M generated and executed a mean calculation and histogram:
  `Mean: 25.0` for `[10,20,30,40]`, one valid artifact, accepted review, 32.08 seconds.
  A Russian calculation/chart request selected `analysis/run_python`. A one-item team
  wrapper initially caused validation failures; equivalent single-agent plans now
  normalize to the legacy direct path. No new authority/tool permissions are inferred.
- The generated histogram was visually inspected for labels and data. Report and
  fixture image: `data/python-agent-smoke.json`, `data/python-agent-live.png`.
  These use synthetic data, a temporary SQLite database, a temporary private runner
  and local inference. No personal notes, external web fetch, cloud inference,
  Langfuse export, or live Telegram message was involved. Temporary services stopped.
- Native and Docker setup, broker authority, hard limits, PNG retention and authenticated
  downloads are documented in `docs/python-analysis.md`. The API/bot never receive the
  Docker socket; generated code receives neither credentials nor host mounts. There is
  no native execution fallback. The feature remains opt-in; private `.env` was not edited.
- Ruff lint/format, offline lockfile consistency, Compose validation, documentation
  links and the planner context-budget regression passed. PostgreSQL integration and
  OCR language-data tests require their separate environment setup; no migration is needed.
- Final full run with real sandbox tests enabled: **354 passed, 5 skipped** (four
  isolated PostgreSQL checks and one explicit OCR language-data check). Temporary
  worker containers and the test broker were removed. The main API/bot were not started.

Parallel agent orchestration verified on 2026-10-20:

- The supervisor can emit 2–5 agent assignments, including repeated roles, with five
  tool actions total. Native LangGraph `Send` branches have separate indexed reducer
  entries and join at one synthesis node. Single-agent plans remain compatible.
- Defaults are three active agents and one simultaneous model generation. Read-only
  branch tools overlap; any write plan executes all tools in a deterministic phase
  before agent drafts. Global action indices preserve operation ledger identity;
  unchanged writes are reused across actions/repairs. Team answers always enter review,
  and the existing five-repair maximum fits the new 48-superstep hard cap.
- The full suite passed **337 tests, 5 skipped**. Skips require the isolated PostgreSQL
  database (four tests) and explicit OCR tessdata setting (one). New checks cover
  actual I/O overlap, repeated roles, both concurrency limits, one ordered join,
  allowlists/authorization, unique write IDs, interrupted tool preparation, durable
  sibling/synthesis recovery, bounded synthesis context, team changes, single/team
  transitions, maximum repairs, and real-SDK Langfuse branch tracing.
- Ruff lint/format, offline lockfile consistency, documentation links, and the offline
  demo passed. No dependency or database migration is needed for orchestration.
- A live native Qwen3.8-27B UD-Q4_K_M check passed an English two-agent workflow with
  research/read_page and productivity/list_tasks, synthesis and accepted review. The
  workflow took 40.44 seconds with one inference slot. A Russian request produced a
  valid two-agent plan in 9.54 seconds. The first live attempt ignored team routing;
  a concrete plan-format example at the beginning of the supervisor prompt fixed
  these checked cases. This is a smoke check, not a general routing-quality benchmark.
- Live inference used synthetic webpage evidence and a temporary SQLite database:
  no external page fetch, private data, Telegram message, cloud inference, or Langfuse
  export was involved. The temporary model server was shut down afterward. The local
  report is `data/parallel-agent-smoke.json`. Docker was not rebuilt and isolated
  PostgreSQL integration was not run for this change. Restart the API to load the workflow.

Langfuse integration verified on 2026-10-20:

- Added and locked Langfuse Python SDK 5.0.2 with OpenTelemetry dependencies. Native
  runtime wiring and the standard Docker build dependency set include the SDK.
- The full local suite passed: **310 passed, 5 skipped**. Four PostgreSQL integration
  tests need the isolated test database; one OCR test needs its explicit tessdata
  environment setting. Docker Desktop was not running, so no Docker rebuild or
  PostgreSQL integration run was performed for this change.
- New tests use the real Langfuse SDK with an in-memory exporter and synthetic
  model responses. They verify parent/child spans, review re-planning, mutation reuse,
  cached requests, both provider usage formats, structured-output retries, sampling,
  asynchronous context isolation/cancellation, content exclusion/redaction, safe
  error handling, connection-check behavior, and authenticated status reporting.
- Ruff lint/format, offline lockfile consistency, CLI registration, and local
  documentation links passed. Tracing remains disabled unless configured explicitly.
- No personal prompts were sent to Langfuse, no project keys were configured, and
  no live Langfuse dashboard/ingestion or real-model inference was verified. The
  connection checker authenticates and flushes a synthetic observation after the user
  supplies their deployment and project credentials; dashboard delivery still needs
  confirmation there. No application services were started or private `.env` edited.

Bounded reviewer/supervisor re-planning verified on 2026-10-19:

- Rejected reviews now route through the checkpointed revision counter back to the
  supervisor, which sees review defects, previous draft, bounded tool evidence and
  prior action arguments. It can select another specialist, execute new/repeated reads,
  retry failed operations or choose no tools for a prose correction. All repair rounds
  require another review, even if the planner omits its review flag.
- `ASSISTANT_MAX_REVIEW_REPLANS` defaults to two additional rounds and accepts 0–5.
  Counter/initial limit survive checkpoint recovery; a 32-step framework limit covers
  the maximum supported graph. Accepted reviews terminate immediately; exhausted
  requests return the latest draft with an explicit limit/unresolved-review notice.
  Responses expose `review_replans` and `max_review_replans`; status exposes the limit.
- Identical successful mutations reuse their recorded results across reordered plans.
  Authorized changed writes receive round-qualified operation IDs; initial IDs remain
  compatible with existing runs. An intervening target update invalidates an older
  cached update. Original user text remains the authorization source. Legacy pending
  revision checkpoints recover argument metadata from their stored plan to preserve
  write replay protection after upgrading.
- Fresh evidence supersedes older identical read attempts, while completed writes
  remain available for planning/audit. Evidence text stays within a shared 6 KB budget;
  full checkpoint records remain available. Repair drafting avoids duplicating the
  complete previous operation evidence in two payload fields. Base first-round planner
  instructions remain unchanged for existing context-budget regressions.
- The final complete suite passed 292 tests, including isolated PostgreSQL, real
  multilingual PDF OCR and a real MCP fourth-page read triggered by rejected review.
  Regressions cover read retry, specialist changes, reordered/new writes, corrective
  updates, cached-update invalidation, zero/default/maximum limits, feedback injection,
  HTTP-model/token-counter payloads, durable restart limits and legacy state recovery.
  Ruff lint/format, unchanged 217-package offline lock and native startup check passed.
- Built `personal-assistant:replanning-check`. A read-only non-root Docker smoke run
  against isolated PostgreSQL rejected an initial PDF answer, restarted the supervisor,
  executed a real MCP page-4 read, accepted the new answer and replayed the completed
  request from cache. Model/counter responses were synthetic; no live Qwen/cloud call
  or Telegram delivery was tested. Report: `data/replanning-verification.json`.
- Smoke records/checkpoints, container and temporary upload volume were removed; the
  isolated test database was stopped afterward. No dependency/schema migration or
  personal configuration/data change is required. Restart the API/bot or native launcher
  to load the workflow; README's agent diagram and architecture documentation are updated.

Telegram PDF uploads verified on 2026-10-19:

- Added owner/private-chat PDF document handling, MIME/filename routing, caption/default
  questions, album rejection and advertised/actual download size limits. The bot sends
  one PDF to authenticated multipart `/v1/chat/pdf`; PDF support is independent of vision.
  API auth/body checks precede spooling, and repeated/extra fields are rejected.
- Validated actual PDF bytes, page count and encryption before storage. Uploads are
  limited to 10 MB, stored atomically with mode 0600 and content-based filenames in
  `data/pdf-uploads`; client filenames are display metadata only. Repeated bytes reuse
  the file, and existing symlinks/conflicts cannot overwrite unrelated files.
- The existing MCP adapter now has a separate confined upload root exposed through
  `uploads/<hash>.pdf`. Its two read-only tools list/read reference PDFs and uploads;
  catalog bounds, page limits, optional OCR and write-tool exclusions remain intact.
  Empty embedded-text excerpts explicitly suggest OCR rather than claiming recognition.
- Uploaded PDFs obtain a bounded initial excerpt of pages 1–3 through real MCP before
  planning. Metadata/evidence reaches all relevant model stages and forces review.
  The caption alone authorizes writes. Run identity includes the PDF digest; cached
  extraction/checkpoints survive retry, and history retains filename/reference/excerpt.
  Original PDF bytes stay in file storage rather than database/checkpoint records.
- Migration `0004` and Alembic metadata consistency passed against isolated PostgreSQL.
  The complete suite passed 275 tests, including real multilingual PDF OCR and all four
  PostgreSQL integration tests. Ruff lint/format, offline lock consistency (217 packages),
  native `start --check`, and Compose configuration validation passed.
- Built `personal-assistant:pdf-uploads-check`. A read-only non-root smoke container
  with a writable disposable upload volume verified upload/auth, real MCP extraction,
  cached retry, text-only model operation and follow-up page 4. Model responses were
  synthetic; no live model/cloud request or real Telegram delivery was performed.
  The smoke volume/container and test records/checkpoints were removed. Isolated
  PostgreSQL was stopped afterward. Report: `data/pdf-upload-verification.json`.
- Native API/bot were not started and personal records/settings were unchanged. Restart
  with `uv run --extra speech --extra pdf assistant start` to apply migrations and load
  the new handlers. Docker uses a persistent named `pdf-uploads` volume initialized for
  UID 10001; reference PDFs/tessdata keep their read-only mounts. Retention, backup,
  native/Docker storage differences and excerpt limits are documented in `docs/pdf.md`.

Image input integration verified on 2026-10-19:

- Added owner-only Telegram photos/image files, CLI `chat --image`, and authenticated
  multipart `/v1/chat/image`. JPEG/PNG/WebP bytes are validated, EXIF-oriented,
  metadata-stripped, flattened on white and resized to at most 1280 pixels per edge.
  Limits: one static image, 10 MB, 20 megapixels; duplicate/extra multipart fields
  are rejected. API authentication/body bounds precede multipart spooling.
- Model profiles now include `model_supports_images`, `local_mmproj_file` and a visual
  token reserve. Switching to a text profile clears image configuration. Native
  inference passes `--mmproj`; local image calls require authenticated runtime
  `/props` evidence of vision support. Hosted image request formatting follows official
  OpenAI documentation. Text-only budgeting and base planner prompts are unchanged.
- A dedicated visual observation call sends image bytes only once. Bounded untrusted
  evidence reaches planner, specialist and mandatory review without granting write
  authority. Archive evidence is added after preparation to avoid duplicating it in
  current-turn history. Review evaluates that observation, not the original pixels.
- Migration `0003` adds nullable image digest/observation fields on runs. Matching
  retries reuse cached analysis and checkpoints; different/missing images under the
  same request ID are rejected. Original bytes/base64 are not stored in database,
  checkpoints or history. Migration and Alembic metadata check passed against isolated
  PostgreSQL; the production database was not migrated during verification.
- Final tests cover image formats/orientation/transparency/limits, capability gates,
  local/hosted request formats, token reserve, end-to-end HTTP model/graph/review flow,
  injection authorization, archive/checkpoint retry, API auth/validation, Telegram
  ownership/download bounds, profile switching and projector launch arguments.
  The final complete suite passed 264 tests, including four isolated PostgreSQL tests
  and actual multilingual PDF OCR. Ruff lint/format and the 217-package offline lock check
  passed; the current private native configuration passed `assistant start --check`.
- Built `personal-assistant:images-check`; a read-only non-root smoke container verified
  the multipart image API, auth, repeated-field rejection, review and cached retries
  against isolated PostgreSQL using synthetic model/counter responses. The optional
  Linux vision-projector Compose overlay also validated. Test records/checkpoints and
  smoke containers were removed; the isolated test database was stopped afterward.
- No live vision model or hosted API was invoked, and no real Telegram messages were
  sent. No matching local projector is installed; the current model remains unchanged
  with image input disabled until a capable model/projector is explicitly configured.
  See `docs/images.md`. Native `assistant start` applies the required migration.

PDF MCP integration verified on 2026-10-19:

- Installed/locked mcp-pdf-reader 0.1.7, compatible FastMCP 2 and PyMuPDF 1 through
  the optional PDF extra. Native configuration now enables the managed PDF server;
  `data/pdf` exists and official English/Russian tessdata_fast files are installed
  in `data/tessdata`. Only PDF keys in the private `.env` were updated.
- Real stdio discovery and calls verified multilingual embedded-text extraction, filename
  listing, page provenance and read-only allowlisting. Generated synthetic PDFs cover
  traversal/symlink escapes, wrong file type, oversized/encrypted documents, invalid
  page ranges, pagination, excerpt bounds and missing/failed OCR. Upstream import
  diagnostics go to stderr, preserving the MCP JSON-RPC transport.
- A synthetic image-only PDF containing English and Russian was recognized through
  the actual upstream OCR function over MCP. Language files use tessdata_fast revision
  `87416418657359cb625c412a48b6e1d6d41c29bd`; hashes and download commands are in
  `docs/pdf.md`. No personal PDFs, Telegram messages or hosted model calls were used.
- A scripted LangGraph request verified PDF evidence reaches the specialist/reviewer
  as untrusted data, triggers review and does not create a note automatically.
  PDF-specific descriptions live in the optional MCP catalog; the base planner prompt
  remains unchanged, preserving existing fallback context-budget regressions.
- Full suite with `TEST_PDF_TESSDATA_DIRECTORY=data/tessdata`: 234 passed, four
  PostgreSQL integration tests skipped (database code/schema unchanged). Ruff lint,
  formatting and offline lock consistency passed; the lock contains 216 packages.
- Built `personal-assistant:pdf-check` and verified real text extraction plus multilingual
  scanned OCR inside it as UID 10001, with no network, a read-only root filesystem,
  read-only synthetic PDF/language-data mounts and temporary `/tmp`. Smoke container
  removed itself; local result is `data/pdf-verification.json`.
- Telegram document uploads, PDF URL downloads and PDF indexing are not implemented.
  Copy files to `data/pdf`, then restart with
  `uv run --extra speech --extra pdf assistant start`. No database migration is needed.

Code-review fixes verified on 2026-10-19:

- Fixed destructive authorization with standalone, action/UUID-bound commands. Regression
  tests drive deliberately wrong plans through the graph and confirm negated, quoted,
  explanatory and hypothetical deletion requests preserve the note. Explicit commands
  execute once and remain protected by request/operation replay records.
- Added a shared tool registry for names, parameter schemas, required fields and role
  permissions; extracted authorization and tool handlers from the orchestration graph.
  Runtime checks reject unrelated parameters before tools execute.
- Replaced byte-only rejection of larger requests with configured-provider counting.
  Small requests retain a conservative byte bound; unsupported counters fall back with
  a diagnostic. Startup validates the complete minimum planner request before API/bot
  launch. Request, reviewed draft and tool evidence remain intact under reduction.
- Real local Qwen tokenizer checks passed for all three reproduced budget failures:
  a five-byte request under an 8,192-token configuration (2,780 counted tokens against
  a 3,946 input allowance), a 2,000-byte request under the default configuration (3,029
  tokens), and a short request with two MCP tools (3,121 tokens). Counts include schema
  text and retain a separate template/output reserve. The server itself ran at 16,384;
  the first check validates the smaller declared allowance, not a separate 8K server.
  The real `assistant check-context` command also passed.
- A synthetic Russian tomorrow-weather request completed through local planning, the
  real Open-Meteo tool, drafting and accepted review in 29.05 seconds. Used a temporary
  SQLite database, no private-note retrieval, and no Telegram messages or cloud calls.
- Conversation context now remains structured and prioritizes recent turns before an
  older Russian summary. Lexical relevance is ranked in SQL before limiting candidates;
  the regression preserves an older exact match behind 40 newer partial matches.
- Reindexing batches document IDs, encodes outside transactions, and conditionally saves
  against the original title/content/edit time. Controlled concurrent model migration
  and note editing preserve the correct new vector in both SQLite and PostgreSQL.
  Interruption/resume tests verify partial completed work and original content survive.
- Private API, speech and embedding routes authenticate before body parsing. Bounded
  chunk counting rejects absent/false Content-Length overflow before parsing/spooling.
  Tests verify unauthorized uploads and oversized authorized uploads never reach the
  multipart file writer. Model response corruption is sanitized after bounded retries.
- Embedding outages preserve keyword reads and note saves, with explicit warnings and
  diagnostics. MCP catalogs cache successful discovery, isolate server failures with
  short timeouts/backoff, distinguish healthy empty catalogs, and bound exposed schemas.
- Final normal suite: 220 passed, four PostgreSQL tests skipped without their environment
  variable. All four tests passed separately against the isolated PostgreSQL database,
  including concurrent edit/reindex and disjoint reminder claims: 224 distinct tests
  passed overall. Ruff lint/format and the unchanged 176-package offline lock check passed.
- The rebuilt Docker image passed live health (200), unauthorized status (401), and
  authorized database/status (200) checks against isolated PostgreSQL. Docker's credential
  helper delayed the initial build; a temporary anonymous CLI configuration also built
  successfully without editing saved credentials. The smoke container, test database,
  and temporary inference process were stopped afterward; personal records/settings
  were not modified. Docker Desktop remains open.
- CI now has base/speech dependency variants and a deployment image build plus API
  health/authentication smoke checks. The hosted counter path uses mocked responses;
  no live hosted-model call, Telegram delivery, or GitHub-hosted CI run was performed.
  No new dependencies or database migration are required for these fixes.

Local reports: `data/review-fixes-live.json` and `data/review-fixes-docker.json`.
Restart the native stack to load all changes. Destructive chat commands are documented
in README; existing `.env`, model profiles, and stored notes remain applicable.

Single-command native launcher checked on 2026-10-18:

- Added `assistant start`, configuration-only `--check`, `--no-bot`, and a configurable
  per-service readiness timeout. Starts Docker Desktop when needed on macOS, waits for
  Compose PostgreSQL, applies migrations, then launches configured native services.
  Speech/embeddings follow their existing settings; hosted chat profiles skip inference.
- The current private configuration passed the read-only CLI check: inference, speech,
  API, and Telegram are configured. No credentials were printed or changed, no existing
  services were stopped, and no second model or Telegram polling process was started.
- Pytest: 188 passed, two dedicated PostgreSQL integration tests skipped. Ruff lint/format
  and offline lock consistency passed. No dependencies or database schema changed.
- New tests cover startup plans, optional services/custom ports, invalid configuration
  and secret redaction, project locking, occupied-port errors, authenticated API readiness
  and retries/timeouts, macOS Docker startup orchestration, private child logs, and cleanup
  after partial startup failure. Real synthetic Python parent/child processes verify that
  inference-style subprocess groups are terminated and reaped without leaving children.
- Full live stack startup, Docker Desktop activation, model loading, embedding downloads,
  and Telegram polling were not rerun through the new launcher. Existing user services
  were left running. Bot launch is reported separately from confirmed Telegram connectivity;
  polling/authentication progress remains in its private log.

Context-budget fix checked on 2026-10-17:

- Reproduced `Request exceeds the context budget; split or shorten the task` for
  “What is the weather tomorrow?” (translated example) with Belgrade configured and populated conversation context.
  The previous budgeter retained a 400-byte history excerpt plus a marker even when
  the planner instructions, schema, and runtime metadata left less room than that.
- Optional history now fits the actual serialized space, including Unicode and JSON
  escaping, with no minimum excerpt size. Compact JSON reduces prompt overhead.
  Current requests, complete reviewed drafts, authoritative clock fields, and stored
  history are preserved. Context-window and output settings were not increased.
- Pytest: 163 passed, two PostgreSQL integration tests skipped because the dedicated
  database was not enabled. Ruff lint and format checks passed. Regressions cover
  the Russian weather request with configured default/history, empty and short context,
  escaped Unicode near the limit, immutable inputs, and genuinely oversized drafts.
- A complete local-Qwen/Open-Meteo check passed for that exact request with synthetic
  prior messages in a temporary SQLite database. The planner reduced its history
  excerpt from 2,199 to 639 bytes and fit the 12,138-byte input allowance. Draft and
  reviewer retained weather evidence; the returned minimum/maximum temperatures matched
  the source and review accepted the answer. The turn took 37.65 seconds and original
  messages remained in the archive. No Telegram delivery or personal database was used.

Local report: `data/context-budget-smoke.json`. Restart the API and resend the request
to load this fix; no model-server or bot restart is required for this change.

Open-Meteo weather integration checked on 2026-10-17:

- Ruff lint/format, offline lock consistency, the offline demo, and local documentation
  link checks passed. Pytest reported 156 passed and two PostgreSQL integration tests
  skipped; the dedicated database was not enabled. No dependency, model, or schema
  change was needed for weather.
- The read-only `get_weather` action uses the existing async HTTP client for the free
  non-commercial forecast/geocoding APIs, independently of web search. Explicitly
  configured the private default as Belgrade, Serbia (44.804, 20.4651), as requested.
  Other settings and credentials were retained. The public template leaves defaults empty.
- Tests cover current/daily/hourly data, units/Fahrenheit, paired coordinates, explicit
  city overrides, ambiguous candidates, missing/disabled locations, cache expiry without
  stale fallback, bounded retries/sanitized failures, missing/malformed values, current
  validity time, location-local relative dates, both DST offsets, column alignment and
  context trimming, complete planner budgets, review evidence, and final attribution.
- Direct live API checks passed for Belgrade current conditions and tomorrow's hourly
  forecast, plus a London override. The second Belgrade read reused cached data while
  preserving its original retrieval timestamp. Model-valid time is distinct from fetch time.
- Four complete live local-Qwen checks passed: Russian current conditions including
  apparent temperature; English default-location daily minimum/maximum for tomorrow;
  London daily minimum/maximum; and temperatures at 09:00, 12:00, and 15:00 tomorrow.
  Selected temperatures matched the provider results and all four answers passed review.
  Checked turns took 27.17–41.68 seconds. Tests used an empty temporary database, with
  web search disabled, and did not send Telegram messages or modify personal records.
- The initial London model check failed its successful-tool-evidence assertion. Its
  diagnostic detail was not retained by the first smoke script, so the cause is unknown.
  A targeted retry generated `London, GB`, obtained a successful result, and passed
  value/review checks. The later script records plan, status, operations, and answer.
- Hourly timestamps use the returned IANA timezone. Daily source aggregates retain the
  provider's fixed response offset and include a DST limitation note when relevant.
  Current data more than two hours old or fifteen minutes ahead of the request clock
  is rejected; fetch/cache age alone is not proof of current validity.
- The planner suite now contains 24 cases; its full expanded live benchmark was not
  run. Live Telegram delivery, a rebuilt Docker image, and comparison against station
  observations were not tested. Forecasts are model output, not observed-weather proof.

Local reports: `data/open-meteo-smoke.json`, `data/open-meteo-qwen-smoke.json`, and
`data/open-meteo-last-trace.json`. Restart the API to load the new action/configuration;
no bot or speech-service restart is required for weather.

Faster-whisper voice input checked on 2026-10-17:

- Installed/locked `faster-whisper` 1.2.1, CTranslate2 4.8.3, PyAV 19.0.1, and
  ONNX Runtime 1.31.0 through the optional `speech` extra. Offline lock consistency,
  Ruff lint/format, and the offline demo passed. Pytest reported 136 passed and two
  PostgreSQL integration tests skipped; the dedicated test database was not enabled.
- Downloaded `Systran/faster-whisper-large-v3` to the ignored model directory, pinned
  to resolved commit `edaa852ec7e145841d8ffdb056a99866b5f0a478`. The complete local
  directory and `assistant-model.json` provenance record are present. Chat weights
  and chat-model configuration were not changed.
- Real English and Russian synthetic speech from native macOS voices was converted
  to Ogg/Opus and submitted through the speech API with local-only loading enabled.
  Both languages were detected correctly and both expected sentences were transcribed.
  The checked clips lasted 3.84 and 4.24 seconds; the latest in-process run took 5.83
  and 6.33 seconds respectively with CPU/INT8 and eight CPU threads. Model loading
  took 1.23 seconds in that run. These short synthetic checks are not a general
  speech-accuracy or latency benchmark, or a comparison against whisper.cpp/Metal.
- Tests cover cached offline loading, worker-thread inference including segment
  iteration, original-language transcription, decoded duration/upload size limits,
  resampling, empty/error handling, download revision pinning, telemetry disabling,
  and the owner-only voice → transcription → chat → text-reply dispatcher path.
- Updated the private `.env` to faster-whisper CPU/INT8, automatic language detection,
  voice input enabled, and audio replies disabled because Piper is not configured.
  Other configuration and credentials were retained; private file permissions are 0600.
- Started the native speech service on 127.0.0.1:8765. A real HTTP health check returned
  faster-whisper with the model loaded; an unauthenticated upload returned 401. The
  bot's actual `SpeechClient` successfully transcribed the synthetic English Ogg file
  in 5.76 seconds over HTTP, using the existing application credential.
- The existing bot process still needs a restart to load its new voice settings.
  Live voice delivery/reply in Telegram, personal recordings, CUDA inference, TTS,
  and a speech-enabled Docker image were not tested. The default app image still
  uses the separate native companion for speech.

Local reports: `data/speech-live-smoke.json` and `data/speech-http-smoke.json`.
Synthetic audio fixtures are in ignored `data/stt-smoke/`. The native service can be
started again with `uv run --extra speech assistant speech` after it is stopped.

Runtime clock handling checked on 2026-10-17:

- Pytest: 125 passed, two PostgreSQL integration tests skipped because the dedicated
  test database was not enabled. Ruff lint/format passed. No dependency or schema change.
- Tests cover local midnight/year rollover, both occurrences of the autumn DST hour,
  calendar-relative dates/week boundaries, fresh clocks in consecutive turns, clock
  preservation during context reduction, system instructions for local/cloud adapters,
  authenticated API clock reads without inference, and owner-only Telegram `/time`.
- A planner-budget regression was caught while adding clock metadata. Repeated prompt
  instructions were shortened; the complete planner input with dynamic clock, real
  tool schema, capabilities, and conversation context now fits the default budget.
- Local Qwen answered synthetic English and Russian current-date/time questions
  correctly both before and after this change, despite an outdated date in sample
  history. The user's exact original failure has therefore not been reproduced.
- Three additional live Qwen planner checks passed: Russian “notes created yesterday”
  generated the correct local date range, an English meeting “tomorrow” generated the
  correct start/end dates, and a current-date/time question required no tools. No
  calendar writes, private-note reads, or Telegram messages were performed in those checks.
- The running API reported `Europe/Belgrade`, private-calendar support, and automatic
  page-reading support. Its `/v1/time` returned 404 because this new endpoint is not
  loaded until the API restarts. The new Telegram command also requires a bot restart.
  Live Telegram delivery, a full expanded planner benchmark, and Docker rebuilding
  were not tested for this change.

Synthetic reports are in `data/clock-smoke-before.json`, `data/clock-smoke-after.json`,
and `data/relative-date-smoke.json`. The API remains dependent on a correct host OS
clock; no date is hardcoded in the production clock path.

Search-to-page grounding checked on 2026-10-17:

- Ruff lint/format and the offline demo passed. Pytest reported 115 passed and two
  PostgreSQL integration tests skipped; the dedicated test database was not enabled
  for this change. No dependencies or database schema were changed.
- Tests cover opening only the first two results, rank preservation, independent page
  failures, no snippet substitution, unconfigured/empty searches, real HTML-table
  extraction from a fixture, fetch metadata, and bounded evidence shared by drafting,
  review, and revision. Both pages survive ordinary context reduction; oversized
  source metadata is omitted explicitly rather than producing broken evidence JSON.
- A live public query, `London UK weather forecast`, returned AccuWeather and
  WeatherWorld pages as its first two usable results. Both returned HTTP 403 to the
  page reader. The same reader successfully extracted example.com, confirming this
  was site-specific access failure rather than general loss of connectivity. The
  combined search correctly returned `no_page_evidence`, without snippet weather data.
- Local Qwen passed all three targeted planner checks: English and Russian weather
  questions with an explicit city selected research/search; an unspecified location
  selected research with no actions for clarification. The full planner suite now has
  23 cases; its full expanded live benchmark was not run.
- Live forecast accuracy, a complete weather answer in Telegram, and a rebuilt Docker
  image were not tested. Blocked or JavaScript-only weather pages remain unsupported;
  a structured weather integration would be needed for more dependable coverage.

Local reports: `data/weather-pages-smoke.json` and `data/weather-planner-smoke.json`.
Restart the API to load the new search-to-page flow and weather policy.

Model profile switching checked on 2026-10-17: Ruff lint/format passed; pytest reported
46 passed and one PostgreSQL integration test skipped (its dedicated database URL was
not supplied for this run). New checks cover local/cloud switching, credential and
storage preservation, quoted paths, private file permissions, environment overrides,
and rejecting missing weights/credentials without changing configuration. No additional
live model or cloud inference was run for this change.

Telegram setup diagnostics on 2026-10-17 confirmed successful `getMe` and
`getWebhookInfo` requests, an available API/database, and an incorrect owner ID equal
to the bot's ID. The stopped native model server was restarted and logged readiness
on port 8080. Added private `/id` discovery and a startup warning; mocked dispatcher
checks confirm that ID discovery does not grant access to assistant data. The suite
now reports 55 passed and one PostgreSQL integration test skipped; Ruff lint/format
passed. A full Telegram assistant reply still requires the correct personal owner ID.

Web-capability reporting checked on 2026-10-17: 61 tests passed, one PostgreSQL
integration test skipped; Ruff lint/format passed. Regression checks distinguish
unconfigured search from URL reading and supply the capability flags throughout
planning, drafting, review, and revision. Two supplied-URL cases were added to the
planner suite, which now has 14 cases; the expanded live-model benchmark has not
been run. No live page fetch or Telegram response was tested for this change.

DuckDuckGo integration checked on 2026-10-17: `ddgs` 9.16.0 and `primp` 2.0.1 are
locked and installed. Ruff lint/format, offline lockfile consistency, and the offline
demo passed; pytest reported 77 passed and one PostgreSQL integration test skipped.
Tests cover keyless provider selection, explicit DuckDuckGo-only backend selection,
result normalization/deduplication, job-domain filtering, sanitized failures without
provider fallback, disabled search, and the existing Tavily path.

A live English query returned five results, including official LangGraph pages.
The Russian query failed with both worldwide and `ru-ru` regions; the adapter returned
its explicit failure message. The underlying cause was not identified, so reliable
multilingual search is not established. Raw public-query reports are stored locally in
`data/duckduckgo-smoke.json`. No full Telegram research answer or rebuilt Docker image
was checked for this integration. Restart the API to load the new provider settings.

Dates and the private calendar checked on 2026-10-17:

- All 108 tests passed with the isolated PostgreSQL database enabled. Ruff lint/format,
  offline lockfile consistency, the offline demo, and documentation link/shell checks passed.
- Alembic migration `0002` passed on the test database; `alembic check` reported no
  pending schema operations. The same additive calendar migration was applied to the
  normal assistant database. No synthetic calendar records were added to that database.
- Note creation/edit timestamps survive retrieval, revisions, and export. Date filters
  are applied to lexical/vector candidates and conversation history. Tests cover UTC
  offsets and the 25-hour Belgrade calendar day at the autumn DST transition.
- Calendar tests cover replay-safe create/update/cancel, all-day dates, UTC/local times,
  ambiguous/nonexistent DST times, overlap reporting, owner-only Telegram agenda/export,
  and ICS round-trip parsing with `icalendar` 7.3.0.
- A real Qwen workflow in a temporary SQLite database passed Russian event creation,
  English agenda lookup, English rescheduling, and Russian cancellation. Each operation
  entered review. Saved UTC times and final cancellation state matched expectations;
  no task was created. Measured turns took 14.35–22.29 seconds in this run.
- The first live attempt omitted a required event field. The application rejected it
  without saving. Clearer field descriptions and a planner example fixed the checked
  workflow. These are development smoke checks, not an independent accuracy benchmark.
- The planner suite now contains 20 cases; the full expanded live benchmark has not
  been run. Native Telegram delivery of the new calendar commands and import into an
  external calendar app remain untested; the dispatcher/ICS contracts are tested locally.

Raw synthetic model plans and operation outcomes are in the ignored local file
`data/calendar-live-smoke.json`. API and bot processes need a restart to load the new
tools and commands. Recurring events, external sync, and automatic event notifications
are not implemented; reminder tasks continue to use the separate reminder services.

Checked on 2026-10-16 on an Apple Silicon Mac.

| Check | Result |
| --- | --- |
| Ruff lint | Passed |
| Ruff format check | Passed |
| Pytest including isolated PostgreSQL integration | 40 passed |
| Offline demo through `uv run assistant demo` | Passed |
| Docker image build with frozen dependencies | Passed on Linux ARM64 |
| Compose application startup and migrations | Passed |
| Database schema versus SQLAlchemy metadata | No pending upgrade operations |
| API health, authenticated status, notes endpoint | HTTP 200 |
| Public webpage extraction from example.com | Passed |
| Real stdio MCP fixture discovery/call/allowlist | Passed |
| Qwen inference and multilingual planner evaluation | Passed: 12/12 planner cases with the local UD-Q4_K_M file |
| Live note/fact persistence, replay, cross-thread recall, and summary review | Passed against an isolated PostgreSQL database |
| BGE-M3 retrieval quality | Not run: weights not downloaded; vector plumbing tested with synthetic vectors |
| Telegram account integration and reminder delivery | Not run: bot credentials not configured |
| Real STT/TTS accuracy and latency | Not run: speech models/executables not configured; HTTP/executable contract mocked in tests |
| Cloud provider live inference | Not run: credentials not configured; HTTP contract tested |

Tests cover transaction replay, persistent checkpoint recovery, note revisions,
memory correction, retrieval and deletion, bounded review, full-draft review input,
context compaction without dropping archive text, input limits, authentication,
Telegram owner checks, public-network restrictions, MCP allowlists, and reminder claims.

Most automated tests use controlled model outputs. Additional live Qwen checks now
cover a small multilingual planner suite and synthetic note/memory workflows; they are
not a general accuracy or safety benchmark. See [local model validation](model-validation.md)
for the exact artifact, measured timings, initial failures, and resulting fixes.
