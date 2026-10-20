# Local Qwen validation

Checked on 2026-10-16. Model weights and runtime binaries remain under ignored local
directories; no credentials or private notes are included in this report.

## Configuration

| Item | Observed/configured value |
| --- | --- |
| Model file | `models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| File size | 15.33 GiB |
| Locally measured SHA-256 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Runtime | llama.cpp 0.7.0-dev, build b11669, commit 620d2225c |
| Runtime source | [Official Apple Silicon release](https://github.com/ggml-org/llama.cpp/releases/tag/b11669) |
| Runtime location | `data/runtime/llama-b11669/llama-server` |
| Detected GPU | Apple M5 Pro, Metal |
| Context / inference slots | 16,384 tokens / 1 |
| Model output budget / thinking | 2,048 tokens / disabled |
| Retrieval during live tests | Keyword; embedding service disabled |

The checksum identifies the tested local file; it is not an independently checked
publisher signature. The UD quantization is the user's supplied artifact and differs
from the originally discussed standard Q4_K_M distribution.

`ASSISTANT_LLAMA_EXECUTABLE`, `ASSISTANT_LOCAL_MODEL_FILE`, and the matching model alias
are configured in the private `.env`. Start the same setup from the project directory:

```sh
uv run assistant inference
```

## Planner evaluation

The initial ten-case suite passed 8/10. A Russian note lookup and an underspecified
reminder were assigned to general conversation. The read tools themselves were valid,
but the selected specialist did not match the intended domain ownership.

After making domain routing and ambiguous-reminder handling explicit in the planner
instructions, the suite passed **12/12**, including two additional note/task requests.
This is a small regression suite used during prompt development, not an independent
estimate of model accuracy.

Warm planner latency had a median of **2.73 seconds**, with a **1.47–4.76 second** range.
Server logs for the initial short planner outputs showed approximately **15.5–16.3
generated tokens/second**. Prompt reuse, output length, device load, and task complexity
affect these measurements; they do not establish full-assistant throughput.

Raw local reports: `data/qwen-planner-evaluation.json` (initial) and
`data/qwen-planner-evaluation-tuned.json` (after routing clarification).

## Live application workflow

Synthetic content was sent through a separate API on port 8001 backed by the isolated
`assistant_test` PostgreSQL database. No test notes or memories were written to the
normal personal database.

| Check | Result | Observed latency |
| --- | --- | --- |
| Save a Russian note containing a synthetic project code | Passed | 15.77 s |
| Replay the same request UUID | Recorded response returned without another write | Not benchmarked |
| Ask in English for that note's exact code in another thread | Passed, review accepted | 15.36 s |
| Save a Russian fact with a unique synthetic key | Passed | 11.38 s |
| Recall the fact in a new conversation | Passed, review accepted | 11.63 s |
| Summarize the saved note, retaining the exact code | Passed, review accepted | 18.47 s |

The first live run exposed an omitted review flag on a note summary. The graph now
requires knowledge-reading answers to enter review even when the planner omits its
flag. Requests containing only knowledge mutations retain the direct confirmation
path. An automated regression test covers this case, and the live workflow passed
after applying the fix.

The synthetic note/fact records were removed after the checks. Synthetic conversation
and checkpoint records may remain in the isolated test database. Detailed local output
is in `data/qwen-live-smoke-reviewed.json`; the initial run is in `data/qwen-live-smoke.json`.

## Scope

These checks validate real structured generation, graph execution, persistence, replay,
basic multilingual responses, source recall, and review routing for a few controlled tasks.
They do not validate broad research accuracy, long conversation summarization quality,
MCP decision quality, semantic cross-language retrieval, Telegram delivery, or speech.
Those capabilities need separate evaluations with their actual services and models.
