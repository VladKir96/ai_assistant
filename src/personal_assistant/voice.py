import asyncio
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from personal_assistant.config import Settings
from personal_assistant.http_guard import PrivateBodyGuard
from personal_assistant.stt import FasterWhisper


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    language: str = "en"


async def command(*args: str, stdin: bytes | None = None):
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.communicate(stdin), timeout=180)
    except (TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise ValueError("Speech executable failed; check model paths and dependencies")


def create_speech_app(settings: Settings | None = None):
    settings = settings or Settings()
    if len(settings.api_token.get_secret_value()) < 24:
        raise ValueError("Set a private API token before starting the speech service")
    faster = FasterWhisper(settings) if settings.stt_backend == "faster_whisper" else None

    @asynccontextmanager
    async def lifespan(app):
        if faster:
            await asyncio.to_thread(faster.load)
        yield

    app = FastAPI(title="Local speech companion", lifespan=lifespan)
    # Allow multipart headers in addition to the 10 MB audio-file limit.
    app.add_middleware(
        PrivateBodyGuard, token=settings.api_token.get_secret_value(), limit=10_065_536
    )
    bearer = HTTPBearer(auto_error=False)
    lock = asyncio.Lock()

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if not credentials or not secrets.compare_digest(
            credentials.credentials, settings.api_token.get_secret_value()
        ):
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stt_backend": settings.stt_backend,
            "model_loaded": faster.model is not None if faster else False,
        }

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(authenticate)])
    async def transcribe(file: UploadFile):
        audio = await file.read(10_000_001)
        if len(audio) > 10_000_000:
            raise HTTPException(status_code=413, detail="Audio exceeds 10 MB")
        if not faster and not await asyncio.to_thread(Path(settings.whisper_model).is_file):
            raise HTTPException(status_code=503, detail="Configure a multilingual Whisper model")
        try:
            async with lock:
                if faster:
                    return await asyncio.to_thread(faster.transcribe, audio)
                with TemporaryDirectory(prefix="assistant-voice-") as directory:
                    root = Path(directory)
                    (root / "input.audio").write_bytes(audio)
                    await command(
                        settings.ffmpeg_executable,
                        "-y",
                        "-i",
                        str(root / "input.audio"),
                        "-t",
                        "180",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        str(root / "input.wav"),
                    )
                    await command(
                        settings.whisper_executable,
                        "-m",
                        settings.whisper_model,
                        "-f",
                        str(root / "input.wav"),
                        "-l",
                        settings.stt_language,
                        "-otxt",
                        "-np",
                        "-of",
                        str(root / "transcript"),
                    )
                    text = (root / "transcript.txt").read_text().strip()
                    if not text:
                        raise ValueError("No speech recognized")
                    return {"text": text}
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            raise HTTPException(status_code=503, detail="Local transcription failed") from error

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    async def speak(request: SpeechRequest):
        if request.language not in {"en", "ru"}:
            raise HTTPException(status_code=422, detail="Supported languages: en, ru")
        model = settings.piper_ru_model if request.language == "ru" else settings.piper_en_model
        if not await asyncio.to_thread(Path(model).is_file):
            raise HTTPException(status_code=503, detail="Configure a Piper voice for this language")
        try:
            async with lock:
                with TemporaryDirectory(prefix="assistant-voice-") as directory:
                    root = Path(directory)
                    await command(
                        settings.piper_executable,
                        "-m",
                        model,
                        "-f",
                        str(root / "speech.wav"),
                        stdin=request.text.encode(),
                    )
                    await command(
                        settings.ffmpeg_executable,
                        "-y",
                        "-i",
                        str(root / "speech.wav"),
                        "-c:a",
                        "libopus",
                        str(root / "speech.ogg"),
                    )
                    return Response((root / "speech.ogg").read_bytes(), media_type="audio/ogg")
        except (OSError, ValueError, TimeoutError) as error:
            raise HTTPException(status_code=503, detail="Local speech synthesis failed") from error

    return app


class SpeechClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def client(self):
        return httpx.AsyncClient(
            base_url=self.settings.speech_base_url,
            timeout=self.settings.speech_timeout_seconds,
            headers={"Authorization": f"Bearer {self.settings.api_token.get_secret_value()}"},
        )

    async def transcribe(self, audio: bytes) -> str:
        async with self.client() as client:
            response = await client.post(
                "/v1/audio/transcriptions", files={"file": ("voice.ogg", audio, "audio/ogg")}
            )
            response.raise_for_status()
            return response.json()["text"]

    async def speak(self, text: str, language_hint: str) -> bytes:
        language = "ru" if re.search(r"[А-Яа-яЁё]", language_hint) else "en"
        async with self.client() as client:
            response = await client.post(
                "/v1/audio/speech", json={"text": text, "language": language}
            )
            response.raise_for_status()
            return response.content
