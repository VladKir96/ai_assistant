# Telegram text and audio responses

Piper synthesizes speech locally. The bot sends the final reviewed answer as text first,
then generates a Telegram Ogg/Opus voice message in the background. It does not make
another LLM call or add an agent. Audio is off by default for every chat.

## Controls

- `/audio on`: text plus voice for subsequent assistant answers, including text,
  voice, image, and PDF questions.
- `/audio off`: text-only; cancels the current audio request and discards queued audio.
- `/audio status`: show the saved mode.

Only the configured owner in their private chat can use these controls. The preference
is stored in PostgreSQL and survives `/new`, bot restarts, and container replacement.
Help, command acknowledgements, reminders, and error messages remain text-only.

The speech companion automatically detects the language of the spoken answer, rather
than using the incoming question or Telegram account language. Supported routes are
English, Russian, and Serbian, including Serbian Latin and Cyrillic. There is no manual
language setting. Detection uses offline [Lingua](https://github.com/pemistahl/lingua-py);
its Bosnian/Croatian Latin models also feed the Serbian route because its Serbian model
uses Cyrillic. A reply uses one voice for its predominant language. Very short ambiguous
phrases and mixed-language replies can be misclassified; separate voices per sentence
are not implemented.

## Install and download voices

From the project root:

```sh
uv sync --frozen
brew install ffmpeg
mkdir -p models/piper
uv run python -m piper.download_voices --download-dir models/piper en_US-lessac-medium ru_RU-denis-medium
```

All Python speech dependencies are included in the normal installation. Piper output
does not require a Whisper model. On Linux, install FFmpeg using your distribution's
package manager. A custom FFmpeg/Piper executable can be configured using the existing
executable settings.

For Serbian, download just these two files from the
[community voice repository](https://huggingface.co/phantom9623/piper-serbian-tts):

```sh
curl -fL https://huggingface.co/phantom9623/piper-serbian-tts/resolve/main/sr_Marko_medium.onnx -o models/piper/sr_Marko_medium.onnx
curl -fL https://huggingface.co/phantom9623/piper-serbian-tts/resolve/main/sr_Marko_medium.onnx.json -o models/piper/sr_Marko_medium.onnx.json
```

The author notes prosody limitations; listen to both scripts before relying on this
voice. Do not substitute `sr_RS-serbski_institut-medium` based only on its Serbian label:
its [model card](https://huggingface.co/rhasspy/piper-voices/blob/main/sr/sr_RS/serbski_institut/medium/MODEL_CARD)
references a [Lower Sorbian dataset](https://github.com/marytts/serbski-institut-dsb-data).
Voice weights have their own licenses; check each model card when distributing voices.
The application never downloads replacement weights during chat.

Each model requires its matching `.onnx.json` beside the `.onnx` file. Configure `.env`:

```dotenv
ASSISTANT_VOICE_OUTPUT_ENABLED=true
ASSISTANT_PIPER_EN_MODEL=models/piper/en_US-lessac-medium.onnx
ASSISTANT_PIPER_RU_MODEL=models/piper/ru_RU-denis-medium.onnx
ASSISTANT_PIPER_SR_MODEL=models/piper/sr_Marko_medium.onnx
ASSISTANT_PIPER_EXECUTABLE=piper
ASSISTANT_FFMPEG_EXECUTABLE=ffmpeg
ASSISTANT_SPEECH_BASE_URL=http://localhost:8765
```

`ASSISTANT_VOICE_ENABLED` controls voice **input** independently. It may remain `false`
for text questions with spoken answers. `ASSISTANT_VOICE_OUTPUT_ENABLED=false` disables
output globally, even if the saved chat preference is on.

To change voices later, replace the corresponding model path and restart the speech
service. Model files remain ignored by Git. No prompts or agent configuration need changing.

## Start and verify

The API, speech service, and bot must use the same `.env` and private API token. The
API also needs the correct Telegram owner ID to scope the saved preference.

Stop previously started services, then run:

```sh
uv run assistant start --check
uv run assistant start
```

The same command handles input, output, or both, according to `.env`. The launcher
runs the database migrations automatically. With manual startup, run
`uv run assistant migrate`, start `uv run assistant speech`, start the API,
and restart the bot. TTS-only startup does not load Whisper weights.

Check:

```sh
curl -fsS http://127.0.0.1:8765/health
```

Expect `tts_ready: true` and `tts_languages` to include `en`, `ru`, and `sr` when all
model/config pairs and executables are present. This checks dependencies and file
availability; synthesis verifies that the model actually loads. `/audio on` checks this
readiness before saving the preference. In Telegram send `/audio on`, ask a short
question, and check that the text arrives before the voice message. Repeat in each language and in both
Serbian scripts. `/audio off` restores text-only output, including after voice input.

For a containerized bot, the existing Compose configuration points to native Mac speech
at `http://host.docker.internal:8765`; start the companion with `--host 0.0.0.0` as
described in the README. The default application image does not contain TTS dependencies
or voices. A custom speech image needs the `tts` extra, FFmpeg, read-only mounted voices,
and model paths valid **inside that image**; a Mac executable cannot run in Linux Docker.

## Limits and failure behavior

Audio excludes fenced code blocks, Markdown formatting, tables, and URLs. It reads at
most 1,500 cleaned characters; a caption labels shortened answers. The text retains the
complete answer. Piper runs once per response, loading the selected voice each time.
Speech input and output have separate locks so synthesis does not hold the transcription
lock. They still share the host's CPU resources.

There is one audio worker, up to three waiting replies, and a 10 MB response limit. A
full queue skips that audio with a notice. Jobs are cancelled at shutdown; audio is not
replayed after restart. Missing voices, failed synthesis, or unavailable services produce
a short notice and preserve the text. An unavailable language never silently uses a
different language's voice. Each external synthesis/conversion process has a timeout
and is killed and reaped on cancellation.
