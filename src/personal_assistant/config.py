from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        hide_input_in_errors=True,
    )

    api_token: SecretStr = SecretStr("")
    postgres_password: SecretStr = SecretStr("")
    database_url: str = "postgresql+psycopg://assistant:assistant@localhost:5432/assistant"
    redis_url: str = "redis://localhost:6379/0"
    api_url: str = "http://localhost:8000"
    model_provider: Literal["local", "openai"] = "local"
    local_base_url: str = "http://localhost:8080/v1"
    local_model: str = "Qwen3.8-27B-Q4_K_M"
    local_model_file: Path | None = None
    model_supports_images: bool = False
    local_mmproj_file: Path | None = None
    vision_token_reserve: int = Field(default=4096, ge=1024, le=16384)
    local_api_key: SecretStr = SecretStr("")
    llama_executable: str = "llama-server"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    context_window: int = Field(default=16384, ge=8192)
    max_output_tokens: int = Field(default=2048, ge=256)
    request_timeout_seconds: int = Field(default=180, ge=5)
    enable_thinking: bool = False
    max_review_replans: int = Field(default=2, ge=0, le=5)
    max_parallel_agents: int = Field(default=3, ge=1, le=5)
    model_max_concurrency: int = Field(default=1, ge=1, le=8)
    python_enabled: bool = False
    python_runner_url: str = "http://localhost:8766"
    python_runner_token: SecretStr = SecretStr("")
    python_worker_image: str = Field(
        default="personal-assistant-python:local", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_./:-]{0,199}$"
    )
    artifact_directory: Path = Path("data/artifacts")
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = Field(
        default="development", pattern=r"^[a-z0-9_-]+$", max_length=100
    )
    langfuse_sample_rate: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    langfuse_capture_content: bool = False
    telegram_token: SecretStr = SecretStr("")
    telegram_owner_id: int = 0
    timezone: str = "Europe/Belgrade"
    embedding_backend: Literal["disabled", "http"] = "disabled"
    embedding_base_url: str = "http://localhost:8081/v1"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = Field(default=1024, ge=1024, le=1024)
    embedding_device: str = "cpu"
    search_provider: Literal["duckduckgo", "tavily", "disabled"] = "duckduckgo"
    duckduckgo_region: str = Field(default="wt-wt", min_length=2, max_length=20)
    tavily_api_key: SecretStr = SecretStr("")
    job_sites: list[str] = Field(default_factory=list)
    weather_enabled: bool = True
    weather_default_name: str = ""
    weather_default_latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    weather_default_longitude: float | None = Field(
        default=None, ge=-180, le=180, allow_inf_nan=False
    )

    @field_validator("weather_default_latitude", "weather_default_longitude", mode="before")
    @classmethod
    def optional_coordinate(cls, value):
        return None if value == "" else value

    stt_backend: Literal["whisper_cpp", "faster_whisper"] = "whisper_cpp"
    faster_whisper_model: str = "models/faster-whisper-large-v3"
    faster_whisper_device: Literal["cpu", "cuda"] = "cpu"
    faster_whisper_compute_type: Literal[
        "int8", "int8_float32", "float32", "float16", "int8_float16", "default"
    ] = "int8"
    faster_whisper_cpu_threads: int = Field(default=8, ge=1, le=64)
    stt_language: Literal["auto", "en", "ru"] = "auto"
    whisper_executable: str = "whisper-cli"
    whisper_model: str = ""
    piper_executable: str = "piper"
    piper_en_model: str = ""
    piper_ru_model: str = ""
    ffmpeg_executable: str = "ffmpeg"
    voice_enabled: bool = False
    voice_output_enabled: bool = True
    speech_timeout_seconds: int = Field(default=600, ge=30, le=3600)
    speech_base_url: str = "http://localhost:8765"
    mcp_servers_file: Path | None = None
    pdf_enabled: bool = False
    pdf_directory: Path = Path("data/pdf")
    pdf_upload_directory: Path = Path("data/pdf-uploads")
    pdf_tessdata_directory: Path | None = None

    @field_validator(
        "mcp_servers_file",
        "local_model_file",
        "local_mmproj_file",
        "pdf_tessdata_directory",
        mode="before",
    )
    @classmethod
    def optional_file(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_runtime(self):
        ZoneInfo(self.timezone)
        if self.python_enabled:
            from urllib.parse import urlsplit

            endpoint = urlsplit(self.python_runner_url)
            if (
                endpoint.scheme not in {"http", "https"}
                or not endpoint.hostname
                or endpoint.username
                or endpoint.password
                or endpoint.query
                or endpoint.fragment
                or endpoint.path.rstrip("/")
            ):
                raise ValueError("Python runner URL must be an HTTP(S) origin without credentials")
        if self.langfuse_enabled:
            from urllib.parse import urlsplit

            endpoint = urlsplit(self.langfuse_base_url)
            if self.langfuse_environment.startswith("langfuse"):
                raise ValueError("Langfuse environment cannot start with langfuse")
            if (
                endpoint.scheme not in {"http", "https"}
                or not endpoint.hostname
                or endpoint.username
                or endpoint.password
                or endpoint.query
                or endpoint.fragment
            ):
                raise ValueError(
                    "Langfuse base URL must be an HTTP(S) URL without credentials/query"
                )
            if not (
                self.langfuse_public_key.get_secret_value()
                and self.langfuse_secret_key.get_secret_value()
            ):
                raise ValueError("Langfuse tracing requires both public and secret keys")
        if (self.weather_default_latitude is None) != (self.weather_default_longitude is None):
            raise ValueError("Configure both default weather coordinates, or leave both empty")
        if self.max_output_tokens + 2048 >= self.context_window:
            raise ValueError("Context must leave room for input, output, and template overhead")
        if self.model_supports_images and self.vision_token_reserve + 1024 > self.input_budget:
            raise ValueError("Vision needs room for image tokens plus at least 1024 text tokens")
        if self.model_provider == "openai" and (
            not self.openai_api_key.get_secret_value() or not self.openai_model
        ):
            raise ValueError(
                "OpenAI mode requires an explicit API key and model; no automatic fallback"
            )
        for domain in self.job_sites:
            if "/" in domain or ":" in domain or not domain.strip():
                raise ValueError("JOB_SITES must contain hostnames, e.g. jobs.example.org")
        return self

    @property
    def input_budget(self) -> int:
        # Token allowance. Model adapters reserve output/template overhead and count
        # with the configured provider; unsupported counters use a safe byte bound.
        return self.context_window - self.max_output_tokens - 2048
