# Images and model capabilities

Image understanding works through Telegram, CLI and the authenticated HTTP API, using
the selected local/hosted model. No automatic cloud fallback, image generation/editing,
URL fetching, albums or video are included.

## Model setup

Install updated dependencies and apply migration `0003` before starting the API:

```sh
uv sync --frozen
uv run assistant migrate
```

All feature libraries are included by default. The database must be available for manual
`migrate`; `assistant start` applies migrations automatically.

For local inference, download a vision-language GGUF supported by your llama.cpp version
and its **matching** multimodal projector GGUF. Put both in `models/`. Compatibility
requires the projector from the same model export; its filename alone proves nothing.
See [llama.cpp multimodal setup](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
This feature does not download chat/projector weights.

Uncomment/customize `local-vision` in `model-profiles.toml`, including:

```toml
model_supports_images = true
local_mmproj_file = "models/vision/mmproj.gguf"
vision_token_reserve = 4096
```

Select the profile and restart:

```sh
uv run assistant models --use local-vision
uv run assistant start
```

The launcher requires an existing projector; `assistant inference` passes `--mmproj`
automatically. Every local image analysis also authenticates to `/props` and requires
`modalities.vision=true`, rejecting stale text-only servers before sending image bytes.
Other local compatible endpoints without this llama.cpp capability check are unsupported.

For direct `.env` setup, the equivalent fields are:

```dotenv
ASSISTANT_MODEL_SUPPORTS_IMAGES=true
ASSISTANT_LOCAL_MMPROJ_FILE=models/vision/mmproj.gguf
ASSISTANT_VISION_TOKEN_RESERVE=4096
```

For hosted inference, choose a model supporting image input and JSON-schema Responses
output; enable `model_supports_images` in its profile. Hosted capability is explicitly
configured, since a model name alone cannot establish support. The adapter sends base64
`input_image` content with `store=false`, following
[official OpenAI documentation](https://developers.openai.com/api/docs/guides/images-vision).
Your configured provider receives the normalized image and caption; its policies apply.

Switching to a text-only profile resets the image flag and projector path. `/v1/status`
reports configured support and limits, not a successful runtime vision test.

## Usage

Send one Telegram photo or JPEG/PNG/WebP file with a caption such as `What is in this picture?`.
A file preserves screenshot quality before preprocessing. Without a caption, the bot
asks for a description in your Telegram account language (EN/RU). Albums are rejected.
Both sender and private chat must match the owner before any download.

```sh
uv run assistant chat --image screenshot.png --text "Explain this error message"
```

Retries require the same `--request-id`, text, thread and original image bytes. A
different/missing image under the same request ID is rejected.

The authenticated API accepts `POST /v1/chat/image` with multipart fields:

| Field | Meaning |
| --- | --- |
| `image` | One JPEG, PNG or WebP file; actual bytes are decoded/validated |
| `text` | Optional caption, default `Describe this image.` |
| `thread_id` | Optional conversation ID, default `default` |
| `request_id` | Optional UUID, reused for an unchanged retry |

The result is the existing `ChatResponse`; `/v1/chat` remains the text JSON route.
Disabled/unverified vision returns 415, invalid images 422, upload/context overflow
413. Authentication and total-body limits run before multipart parsing. Provider
failures are reported without switching models or claiming an image was understood.

## Limits, evidence and retention

One static image, at most 10 MB and 20 megapixels. JPEG/PNG/WebP are supported; animated
formats, SVG and HEIC are not. Decoded bytes determine the format, not the filename or
MIME claim. EXIF orientation is applied, metadata removed, transparency flattened on
white, and the longest edge reduced to at most 1280 pixels. Resizing/JPEG encoding can
lose small text/detail; crop a relevant region yourself and resend if necessary.

The image goes only to the first visual-observation call. Base64 never enters the
conversation token budget. Text budgeting reserves `vision_token_reserve` for image
embeddings; this allowance is not an exact image-token count. Model-specific image
processing can differ, and the backend can still reject overflow. Increase the reserve
and matching model context when necessary. Text-only budgeting remains unchanged.

The observation has bounded description, visible text and uncertainty fields
(500/1000/300 UTF-8 bytes). It reaches the supervisor, specialist and reviewer as
untrusted evidence. Only the user's caption authorizes writes, subject to existing
rules; image text cannot authorize tool actions. Every image answer requires review,
which checks the observation rather than independently re-inspecting pixels.

Original bytes/base64 are not retained in PostgreSQL, graph checkpoints or history.
Multipart uploads may temporarily spool to disk and are closed after the request.
The raw-image SHA-256 and observation are cached on the run, avoiding repeat visual
analysis after a completed observation. After success, the conversation archive stores
the observation as untrusted data for textual follow-up/retrieval. This is not image
indexing and cannot recover omitted detail; reattach the original for another look.

## Docker

The application image includes Pillow. Rebuild it for this update. For API/bot
containers using native Metal inference, keep the normal deployment and start native
inference with the vision profile/projector.

For Linux inference, an optional overlay mounts the projector read-only:

```sh
docker compose -f compose.yaml -f compose.linux.yaml -f compose.vision.yaml --profile linux-inference up -d --build --wait api llama
```

Set the model path, matching `ASSISTANT_LOCAL_MMPROJ_FILE` and image flag first.
The overlay sets `LLAMA_ARG_MMPROJ=/models/mmproj.gguf`. Keep the normal PDF directories
present (`mkdir -p data/pdf data/tessdata`). Follow existing Linux hardware guidance;
native Metal inference remains the Mac default.
