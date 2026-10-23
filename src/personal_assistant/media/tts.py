"""Local text preparation and automatic language selection for Piper voices."""

import importlib.util
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from personal_assistant.core.config import Settings

MAX_SPEECH_CHARS = 1500


@dataclass(frozen=True)
class SpokenText:
    text: str
    truncated: bool


def prepare_speech(text: str) -> SpokenText:
    # Avoid reading code, URLs, Markdown markers, and table formatting aloud.
    text = re.sub(r"```[^\n]*\n.*?(?:```|\Z)|~~~[^\n]*\n.*?(?:~~~|\Z)", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?m)^\s*\|.*\|\s*$", " ", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)", "", text)
    text = re.sub(r"[`*_~\[\]]", "", text)
    # Piper accepts raw phoneme markup; brackets above also neutralize that notation.
    text = " ".join(text.split())
    truncated = len(text) > MAX_SPEECH_CHARS
    if truncated:
        prefix = text[:MAX_SPEECH_CHARS]
        boundary = max(prefix.rfind(". "), prefix.rfind("! "), prefix.rfind("? "))
        if boundary >= MAX_SPEECH_CHARS // 2:
            text = prefix[: boundary + 1]
        else:
            text = prefix.rsplit(" ", 1)[0] if " " in prefix else prefix
    if not any(character.isalpha() or character.isdigit() for character in text):
        text = ""
    return SpokenText(text, truncated)


@lru_cache(maxsize=1)
def language_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:
        raise ValueError("TTS dependencies are missing; run `uv sync --frozen`.") from None
    # Lingua's Serbian model is Cyrillic. Croatian/Bosnian provide the Latin-script
    # statistics for the same voice route within our supported en/ru/sr language set.
    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.RUSSIAN, Language.SERBIAN, Language.CROATIAN, Language.BOSNIAN
    ).build()


def detect_language(text: str) -> str:
    language = language_detector().detect_language_of(text)
    if language is None:
        raise ValueError("Could not determine the response language")
    code = language.iso_code_639_1.name.lower()
    return "sr" if code in {"hr", "bs"} else code


def voice_models(settings: Settings) -> dict[str, str]:
    return {
        "en": settings.piper_en_model,
        "ru": settings.piper_ru_model,
        "sr": settings.piper_sr_model,
    }


def voice_ready(model: str) -> bool:
    return bool(model and Path(model).is_file() and Path(model + ".json").is_file())


def speech_status(settings: Settings) -> dict:
    languages = [
        language for language, model in voice_models(settings).items() if voice_ready(model)
    ]
    return {
        "tts_languages": languages,
        "tts_ready": bool(
            languages
            and importlib.util.find_spec("lingua") is not None
            and shutil.which(settings.piper_executable)
            and shutil.which(settings.ffmpeg_executable)
        ),
    }
