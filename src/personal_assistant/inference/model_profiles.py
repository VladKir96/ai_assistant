import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from personal_assistant.core.config import Settings


class ProfileError(ValueError):
    pass


class ModelProfile(BaseModel):
    """Model settings only: credentials and personal storage never belong in a profile."""

    model_config = ConfigDict(extra="forbid")

    model_provider: Literal["local", "openai"]
    context_window: int = Field(ge=8192)
    max_output_tokens: int = Field(ge=256)
    request_timeout_seconds: int = Field(default=180, ge=5)
    enable_thinking: bool = False
    local_model: str = ""
    local_model_file: str = ""
    model_supports_images: bool = False
    local_mmproj_file: str = ""
    vision_token_reserve: int = Field(default=4096, ge=1024, le=16384)
    local_base_url: str = "http://localhost:8080/v1"
    openai_model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    @model_validator(mode="after")
    def validate_profile(self):
        model = self.local_model if self.model_provider == "local" else self.openai_model
        if not model.strip():
            raise ValueError("The selected provider requires a model name")
        if self.max_output_tokens + 2048 >= self.context_window:
            raise ValueError("Context must leave room for input and template overhead")
        if self.model_supports_images and (
            self.vision_token_reserve + 1024 > self.context_window - self.max_output_tokens - 2048
        ):
            raise ValueError("Vision context must leave room for the image reserve and text")
        for value in self.model_dump().values():
            if isinstance(value, str) and any(part in value for part in ("\n", "\r", "${")):
                raise ValueError("Profile values must be single-line literals")
        return self


def load_profiles(path: Path) -> dict[str, ModelProfile]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        profiles = data["profiles"]
        if set(data) != {"profiles"} or not isinstance(profiles, dict) or not profiles:
            raise ValueError("Expected named profiles")
        return {name: ModelProfile.model_validate(value) for name, value in profiles.items()}
    except (OSError, KeyError, ValueError, TypeError) as error:
        # Validation errors may contain arbitrary input; never print their raw values.
        raise ProfileError(
            "Cannot load model profiles. Check the TOML file and required model settings; "
            "credentials belong in .env."
        ) from error


def select_profile(name: str, profiles_file: Path, env_file: Path = Path(".env")):
    profiles = load_profiles(profiles_file)
    if name not in profiles:
        raise ProfileError("Unknown model profile. Run 'assistant models' to list profiles.")
    if not env_file.is_file() or env_file.is_symlink():
        raise ProfileError("Expected a regular .env file; run 'assistant init' first.")
    profile = profiles[name]
    if profile.model_provider == "local" and profile.local_model_file:
        if not (env_file.parent / profile.local_model_file).is_file():
            raise ProfileError("The profile's GGUF file does not exist; download it first.")
    if profile.model_provider == "local" and profile.model_supports_images:
        if (
            not profile.local_mmproj_file
            or not (env_file.parent / profile.local_mmproj_file).is_file()
        ):
            raise ProfileError("The vision profile requires an existing matching mmproj GGUF file.")
    try:
        # Validate the complete configuration, including the existing cloud key if needed.
        Settings(_env_file=env_file, **profile.model_dump())
    except ValidationError as error:
        raise ProfileError(
            "The profile cannot run with the current configuration. Check its budgets and "
            "provider credentials in .env; no settings were changed."
        ) from error

    updates = {}
    for key, value in profile.model_dump().items():
        if isinstance(value, bool):
            encoded = str(value).lower()
        elif isinstance(value, str):
            encoded = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        else:
            encoded = str(value)
        updates[f"ASSISTANT_{key.upper()}"] = encoded
    original = env_file.read_text(encoding="utf-8")
    lines = []
    seen = set()
    for line in original.splitlines(keepends=True):
        match = re.match(r"^\s*(?:export\s+)?(ASSISTANT_\w+)\s*=", line)
        key = match[1] if match else None
        if key in updates:
            lines.append(f"{key}={updates[key]}\n")
            seen.add(key)
        else:
            lines.append(line)
    content = "".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    content += "".join(f"{key}={value}\n" for key, value in updates.items() if key not in seen)

    descriptor, temporary = tempfile.mkstemp(prefix=".env.model-", dir=env_file.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, env_file)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return profile
