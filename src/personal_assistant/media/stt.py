"""Local, offline speech recognition; model downloads are a separate explicit command."""

import io
import json
from itertools import chain
from pathlib import Path

from personal_assistant.core.config import Settings

MODEL_REPO = "Systran/faster-whisper-large-v3"
MODEL_FILES = ("config.json", "model.bin", "tokenizer.json", "preprocessor_config.json")
SAMPLE_RATE = 16000
MAX_SECONDS = 180


def validate_model(path: Path):
    required = [path / name for name in MODEL_FILES]
    vocabulary = [path / name for name in ("vocabulary.json", "vocabulary.txt")]
    if not all(file.is_file() and file.stat().st_size for file in required) or not any(
        file.is_file() and file.stat().st_size for file in vocabulary
    ):
        raise ValueError(
            "Configure ASSISTANT_FASTER_WHISPER_MODEL with a complete CTranslate2 model directory. "
            "Run `uv run assistant speech-download` first. "
            "A whisper.cpp .bin or llama.cpp GGUF is not compatible."
        )


def decode_audio(audio: bytes):
    """Decode only up to the duration limit, even for highly compressed uploads."""
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks, samples = [], 0
    try:
        with av.open(io.BytesIO(audio), mode="r") as container:
            for frame in chain(container.decode(audio=0), [None]):
                if frame is not None:
                    frame.pts = None
                for output in resampler.resample(frame):
                    samples += output.samples
                    if samples > SAMPLE_RATE * MAX_SECONDS:
                        raise ValueError("Audio exceeds the 3-minute duration limit")
                    chunks.append(output.to_ndarray().reshape(-1))
    except (av.error.FFmpegError, IndexError) as error:
        raise ValueError("Audio could not be decoded") from error
    if not samples:
        raise ValueError("Audio is empty")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


class FasterWhisper:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def load(self):
        if self.model is not None:
            return
        path = Path(self.settings.faster_whisper_model).expanduser().resolve()
        validate_model(path)
        try:
            import onnxruntime
            from faster_whisper import WhisperModel
        except ImportError:
            raise ValueError("Speech dependencies are missing; run `uv sync --frozen`.") from None
        onnxruntime.disable_telemetry_events()
        self.model = WhisperModel(
            str(path),
            device=self.settings.faster_whisper_device,
            compute_type=self.settings.faster_whisper_compute_type,
            cpu_threads=self.settings.faster_whisper_cpu_threads,
            num_workers=1,
            local_files_only=True,
        )

    def transcribe(self, audio: bytes) -> dict:
        self.load()
        waveform = decode_audio(audio)
        segments, info = self.model.transcribe(
            waveform,
            task="transcribe",
            language=None if self.settings.stt_language == "auto" else self.settings.stt_language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        # Inference is lazy: consume the generator in this worker, not on the event loop.
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ValueError("No speech recognized")
        return {"text": text, "language": info.language}


def download_model(output: Path, repo_id: str = MODEL_REPO, revision: str = "main"):
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise ValueError("Speech dependencies are missing; run `uv sync --frozen`.") from None
    info = HfApi(token=False).model_info(repo_id, revision=revision)
    snapshot_download(
        repo_id,
        revision=info.sha,
        local_dir=output,
        token=False,
        allow_patterns=[*MODEL_FILES, "vocabulary.*", "README.md"],
    )
    validate_model(output)
    (output / "assistant-model.json").write_text(
        json.dumps({"repo_id": repo_id, "revision": info.sha}, indent=2), encoding="utf-8"
    )
    return output
