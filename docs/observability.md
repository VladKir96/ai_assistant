# Langfuse tracing and monitoring

Langfuse shows the execution tree, latency, model usage, tool failures, and review
cycles for assistant requests. The integration uses the Python SDK directly because
our model adapter uses HTTP rather than LangChain model classes. It does not install
a callback that uploads the entire LangGraph state.

## Enable tracing

1. Create a project in [Langfuse Cloud](https://cloud.langfuse.com), or use an existing
   self-hosted Langfuse deployment. Copy the project's public and secret API keys.
   These are separate from your assistant, Telegram, and model-provider credentials.
2. Add the following settings to your existing private `.env`:

   ```dotenv
   ASSISTANT_LANGFUSE_ENABLED=true
   ASSISTANT_LANGFUSE_PUBLIC_KEY=pk-lf-your-project-key
   ASSISTANT_LANGFUSE_SECRET_KEY=sk-lf-your-project-key
   ASSISTANT_LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ASSISTANT_LANGFUSE_ENVIRONMENT=development
   ASSISTANT_LANGFUSE_SAMPLE_RATE=1.0
   ASSISTANT_LANGFUSE_CAPTURE_CONTENT=false
   ```

   Use the base URL shown by your Langfuse deployment/region. SDK 5 requires a
   compatible server: self-hosted Langfuse **v4.59.0 or newer**. See the official
   [SDK compatibility table](https://langfuse.com/docs/observability/sdk/overview).
3. Install the updated dependencies and check the connection:

   ```sh
   uv sync --frozen
   uv run assistant tracing-check
   ```

   This authenticates with Langfuse, emits a synthetic `assistant.tracing_check`
   observation, and flushes the SDK queue. It needs no running model, database, or
   Telegram bot and sends no personal request. With sampling below 1, the test trace
   may be sampled out. Authentication and a completed flush do not prove successful
   ingestion; confirm the observation appears in the dashboard. Delivery may take time.
4. Restart the native assistant:

   ```sh
   uv run assistant start
   ```

   If the launcher is already running, stop it with Ctrl+C first. For Docker,
   rebuild and recreate the API container instead:

   ```sh
   docker compose up -d --build --force-recreate api
   docker compose exec api assistant tracing-check
   ```

5. Send an ordinary request through Telegram or the API, then open Traces in
   Langfuse. Filter for `assistant.chat` and your configured environment. Inspect
   the root observation's `request_id` metadata to correlate an API request.

No Langfuse account or server is needed while tracing is disabled, which is the
default. Setting credentials alone does not enable tracing. This repository includes
the client integration, **not** a self-hosted Langfuse server stack. For that option,
follow the official [Docker Compose deployment guide](https://langfuse.com/self-hosting/deployment/docker-compose).
If Langfuse runs on your Mac and the API runs in Docker, use its reachable
`http://host.docker.internal:PORT` address; `localhost` inside the API container refers
to that container. On a shared Docker network, use the Langfuse service hostname.

## What is recorded

An example execution tree:

```text
assistant.chat
├── pdf.extract                  # Initial upload extraction, if applicable
├── graph.context
├── graph.supervisor
│   └── model.request
│       └── model.generate        # Generation observation, model and token usage
├── graph.parallel_prepare        # Team plans; ordered tools here if any writes exist
├── graph.research
│   ├── tool.get_weather
│   └── model.request
│       └── model.generate
├── graph.knowledge               # A sibling agent in a team request
├── graph.synthesize              # Joins completed team contributions into one draft
├── graph.review
├── graph.revise                  # When the reviewer requests another attempt
├── graph.supervisor
├── ...
└── graph.finish
```

- **Requests:** duration including time waiting for the single-user request lock,
  request ID, attachment type, final specialist, review status, repair count and
  failed operation count. Each incoming attempt gets its own trace; matching request
  IDs link retries. Completed replays contain `request.cache_hit` without new inference.
- **Graph steps:** duration, review round, selected specialist, agent/action counts,
  assignment index and review decisions. Sibling specialist spans share the request
  parent; synthesis follows their completion. Checkpoint resumes record only steps
  actually executed. Agent limits and model concurrency are separate, so overlapping
  agent spans do not necessarily mean simultaneous inference.
- **Tools:** duration, success/failure and whether a completed mutation was reused.
  MCP calls made through graph tools are included; the external MCP process is not
  independently instrumented.
- **Model requests:** budgeting/queueing duration and a generation observation for
  every HTTP generation attempt, including invalid-JSON and HTTP retries. Each
  generation records model/provider, schema, attempt number and provider-reported
  input/output/total tokens when available. No token estimates are invented.
- **Failures:** tool failures and raised exceptions are marked as errors. Only the
  exception class is recorded automatically, without exception text or stack traces.

Conversation threads are grouped as Langfuse sessions using a SHA-256 pseudonym of
the thread ID. Raw Telegram IDs/thread IDs are not sent. This grouping is not a
guarantee of anonymity. All requests belong to the fixed user label `owner`.

Use the dashboard to inspect slow traces, failed tool/generation observations,
repeated reviews and token consumption by model. Langfuse can calculate inference
costs when pricing is configured for the model; a custom local model has no automatic
monetary cost estimate. This integration does not measure CPU/RAM/GPU utilization,
provide uptime probes, or trace speech-companion/embedding HTTP calls independently.
The existing `/health` endpoint remains the database-backed service health check.

## Content and credentials

By default, **no prompts, answers, note contents, retrieved history, tool arguments,
PDF excerpts, or image observations** are recorded as input/output. Technical metadata
and token counts remain visible. No full graph state is sent, and binary attachments
are never passed to Langfuse by this integration.

For debugging, opt in with `ASSISTANT_LANGFUSE_CAPTURE_CONTENT=true` and restart the
API. This sends bounded request text, model prompts/payloads/responses and tool
arguments/results to your configured Langfuse deployment. These can contain your
private notes, PDF text, search queries and personal information. Strings are clipped
to 4,000 UTF-8 bytes, each input/output is bounded to about 16 KB, and long values are
explicitly excerpted. It is not a complete transcript or an exact prompt archive.

Configured `SecretStr` credentials, common credential fields, bearer credentials and
data URLs are redacted; raw byte values are omitted. This is defensive filtering,
not a universal detector of secrets embedded in arbitrary text. Keep content capture
off unless you want private text stored by that Langfuse deployment. `.env` stays
excluded from Docker build context and should never be committed.

## Availability and troubleshooting

- Missing keys or an invalid endpoint with tracing enabled cause a configuration
  error. An initialized SDK failing at runtime does not block assistant requests.
- The SDK batches export in the background. The assistant does not authenticate,
  flush, or wait for Langfuse on each request. It flushes and closes tracing resources
  at runtime shutdown. Forced process termination can lose queued observations.
- `/v1/status` requires the assistant bearer token and reports `tracing.configured`,
  `active`, and `capture_content`. `active` means the SDK initialized, **not** that
  Langfuse is reachable or all traces have been delivered. No Langfuse keys appear there.
- If the dashboard is empty, run `assistant tracing-check`, check endpoint/region,
  credentials, environment filters, sampling, API logs and outbound connectivity.
  Export failures can drop telemetry; they are not added to the assistant database
  for later replay. Setting the sample rate to 0 records no traces.
- Disable with `ASSISTANT_LANGFUSE_ENABLED=false` and restart/recreate the API.
