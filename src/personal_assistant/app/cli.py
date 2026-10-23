import argparse
import asyncio
import logging
import os
import secrets
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import uvicorn

from personal_assistant.core.config import Settings


def initialize():
    target = Path(".env")
    if target.exists():
        raise SystemExit(".env already exists; it was not modified")
    template = Path(".env.example")
    if not template.is_file():
        raise SystemExit("Run init from the project repository")
    token, password = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    content = template.read_text().replace(
        "ASSISTANT_API_TOKEN=change-me", f"ASSISTANT_API_TOKEN={token}"
    )
    content = content.replace(
        "ASSISTANT_POSTGRES_PASSWORD=change-me", f"ASSISTANT_POSTGRES_PASSWORD={password}"
    )
    content = content.replace("assistant:change-me@", f"assistant:{password}@")
    content = content.replace(
        "ASSISTANT_LOCAL_API_KEY=\n", f"ASSISTANT_LOCAL_API_KEY={secrets.token_urlsafe(32)}\n"
    )
    content = content.replace(
        "ASSISTANT_EMBEDDING_API_KEY=\n", f"ASSISTANT_EMBEDDING_API_KEY={token}\n"
    )
    descriptor = target.open("x", encoding="utf-8")
    try:
        target.chmod(0o600)
        descriptor.write(content)
    finally:
        descriptor.close()
    print("Created private .env with random credentials. No models or services were started.")


async def migrate():
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from personal_assistant.core.config import Settings

    settings = Settings()
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    ) as saver:
        await saver.setup()


async def manage(command: str, output: Path):
    from personal_assistant.inference.llm import Embeddings
    from personal_assistant.storage.database import Database
    from personal_assistant.storage.repository import Repository

    settings = Settings()
    db, embeddings = Database(settings.database_url), Embeddings(settings)
    try:
        repository = Repository(db, embeddings)
        if command == "reindex":
            print(f"Indexed {await repository.reindex()} documents.")
        else:
            content = (
                await repository.calendar.export_ics()
                if command == "calendar-export"
                else (await repository.export_markdown()).encode("utf-8")
            )

            def write_export():
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as stream:
                    output.chmod(0o600)
                    stream.write(content)

            await asyncio.to_thread(write_export)
            print(f"Exported {'calendar' if command == 'calendar-export' else 'notes'} to {output}")
    finally:
        await embeddings.close()
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Private personal assistant")
    parser.add_argument(
        "command",
        choices=[
            "init",
            "start",
            "check-context",
            "tracing-check",
            "python-runner",
            "api",
            "bot",
            "migrate",
            "speech",
            "speech-download",
            "embeddings",
            "reindex",
            "export",
            "demo",
            "chat",
            "inference",
            "evaluate",
            "models",
            "calendar-export",
        ],
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--image", type=Path, help="Attach one image to a chat request")
    parser.add_argument("--use", help="Activate a named model profile (models command)")
    parser.add_argument("--profiles-file", type=Path, default=Path("model-profiles.toml"))
    parser.add_argument("--text")
    parser.add_argument("--thread", default="cli")
    parser.add_argument("--request-id", type=UUID)
    parser.add_argument("--repo-id", default="Systran/faster-whisper-large-v3")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--no-bot", action="store_true", help="Start local services without Telegram"
    )
    parser.add_argument("--check", action="store_true", help="Validate startup configuration only")
    parser.add_argument(
        "--startup-timeout", type=int, default=300, help="Readiness timeout per service"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    # HTTP libraries can otherwise log credential-bearing Telegram URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.command == "init":
        initialize()
    elif args.command == "start":
        from pydantic import ValidationError

        from personal_assistant.app.local_start import StartupError, start

        if args.startup_timeout < 1:
            raise SystemExit("--startup-timeout must be positive")
        try:
            start(Settings(), no_bot=args.no_bot, check=args.check, timeout=args.startup_timeout)
        except (StartupError, ValueError) as error:
            message = (
                "Invalid .env settings; check configuration values without sharing credentials."
                if isinstance(error, (ValidationError, ValueError))
                else str(error)
            )
            raise SystemExit(message) from None
    elif args.command == "check-context":
        from personal_assistant.inference.capacity import check_capacity
        from personal_assistant.inference.llm import ModelError

        try:
            asyncio.run(check_capacity(Settings()))
        except ModelError as error:
            raise SystemExit(str(error)) from None
        print("Planner, tool catalog and clock fit the configured model input allowance.")
    elif args.command == "tracing-check":
        from pydantic import ValidationError

        from personal_assistant.core.tracing import check_tracing

        try:
            asyncio.run(check_tracing(Settings()))
        except ValidationError:
            raise SystemExit(
                "Invalid Langfuse settings; check .env without sharing keys."
            ) from None
        except RuntimeError as error:
            raise SystemExit(str(error)) from None
        print(
            "Langfuse authentication passed; test trace flushed. Check the dashboard for delivery."
        )
    elif args.command == "models":
        from personal_assistant.inference.model_profiles import (
            ProfileError,
            load_profiles,
            select_profile,
        )

        try:
            if args.use:
                profile = select_profile(args.use, args.profiles_file)
                print(f"Selected model profile: {args.use} ({profile.model_provider}).")
                if profile.model_provider == "local":
                    print("Restart the API and local inference server to apply it.")
                else:
                    print("Restart the API to apply it; local inference can be stopped.")
                print("Notes, conversations, credentials, and voice settings are retained.")
            else:
                for name, profile in load_profiles(args.profiles_file).items():
                    model = (
                        profile.local_model
                        if profile.model_provider == "local"
                        else profile.openai_model
                    )
                    print(
                        f"{name}: {profile.model_provider}, {model}, "
                        f"{profile.context_window} context"
                    )
        except ProfileError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "inference":
        settings = Settings()
        model_file = args.model_file or settings.local_model_file
        if not model_file or not model_file.is_file():
            raise SystemExit("Set LOCAL_MODEL_FILE or supply an existing GGUF with --model-file")
        vision_args = []
        if settings.model_supports_images:
            if not settings.local_mmproj_file or not settings.local_mmproj_file.is_file():
                raise SystemExit("Local vision requires an existing ASSISTANT_LOCAL_MMPROJ_FILE")
            vision_args = ["--mmproj", str(settings.local_mmproj_file.resolve())]
        key = settings.local_api_key.get_secret_value()
        if args.host not in {"localhost", "127.0.0.1", "::1"} and len(key) < 24:
            raise SystemExit(
                "Set a private LOCAL_API_KEY before binding inference beyond localhost"
            )
        environment = os.environ.copy()
        environment["LLAMA_API_KEY"] = key
        subprocess.run(
            [
                settings.llama_executable,
                "-m",
                str(model_file.resolve()),
                "--alias",
                settings.local_model,
                "--host",
                args.host,
                "--port",
                str(args.port or 8080),
                "-c",
                str(settings.context_window),
                "-ngl",
                "99",
                "--parallel",
                "1",
                "--jinja",
                *vision_args,
            ],
            env=environment,
            check=True,
        )
    elif args.command == "chat":
        if not args.text and not args.image:
            raise SystemExit("Supply a message with --text or attach --image")
        import httpx

        from personal_assistant.core.schemas import ChatRequest

        settings = Settings()
        request = ChatRequest(
            thread_id=args.thread,
            text=args.text or "Describe this image.",
            request_id=args.request_id or uuid4(),
        )
        try:
            data = request.model_dump(mode="json")
            options = {
                "timeout": 1200,
                "headers": {"Authorization": f"Bearer {settings.api_token.get_secret_value()}"},
            }
            if args.image:
                from personal_assistant.media.images import MAX_IMAGE_BYTES

                if not settings.model_supports_images:
                    raise SystemExit(
                        "Enable image support for a capable model first; see docs/images.md"
                    )
                with args.image.open("rb") as stream:
                    image = stream.read(MAX_IMAGE_BYTES + 1)
                if len(image) > MAX_IMAGE_BYTES:
                    raise SystemExit("Image exceeds the 10 MB limit")
                response = httpx.post(
                    f"{settings.api_url}/v1/chat/image",
                    data=data,
                    files={"image": ("image", image, "application/octet-stream")},
                    **options,
                )
            else:
                response = httpx.post(f"{settings.api_url}/v1/chat", json=data, **options)
            response.raise_for_status()
            print(response.json()["text"])
        except httpx.HTTPError:
            raise SystemExit(
                "Chat failed. Check API/model services. Retry identical input with --request-id "
                + str(request.request_id)
            ) from None
    elif args.command == "api":
        uvicorn.run(
            "personal_assistant.app.api:create_app",
            factory=True,
            host=args.host,
            port=args.port or 8000,
            access_log=False,
        )
    elif args.command == "python-runner":
        from personal_assistant.integrations.python_runner import create_python_runner_app

        uvicorn.run(
            create_python_runner_app(Settings()),
            host=args.host,
            port=args.port or 8766,
            access_log=False,
        )
    elif args.command == "speech-download":
        from personal_assistant.media.stt import download_model

        try:
            target = download_model(
                args.output or Path("models/faster-whisper-large-v3"), args.repo_id, args.revision
            )
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(f"Speech model downloaded to {target}. No services or Telegram messages started.")
    elif args.command == "bot":
        from personal_assistant.app.bot import BotConfigurationError, run_bot

        try:
            asyncio.run(run_bot(Settings()))
        except BotConfigurationError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "migrate":
        from alembic import command
        from alembic.config import Config

        command.upgrade(Config("alembic.ini"), "head")
        asyncio.run(migrate())
        print("Application and LangGraph migrations applied.")
    elif args.command in {"export", "reindex", "calendar-export"}:
        default_output = (
            "data/calendar.ics" if args.command == "calendar-export" else "data/notes.md"
        )
        asyncio.run(manage(args.command, args.output or Path(default_output)))
    elif args.command == "demo":
        from personal_assistant.app.demo import run_demo

        asyncio.run(run_demo())
    elif args.command == "evaluate":
        from personal_assistant.app.evaluation import evaluate

        asyncio.run(evaluate(args.output or Path("data/planner-evaluation.json")))
    else:
        module = (
            "personal_assistant.media.voice:create_speech_app"
            if args.command == "speech"
            else ("personal_assistant.inference.embedding_server:create_embedding_app")
        )
        uvicorn.run(
            module,
            factory=True,
            host=args.host,
            port=args.port or (8765 if args.command == "speech" else 8081),
            access_log=False,
        )
