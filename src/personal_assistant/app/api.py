import logging
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, StrictBool

from personal_assistant.app.runtime import Runtime, runtime
from personal_assistant.core.config import Settings
from personal_assistant.core.http_guard import PrivateBodyGuard
from personal_assistant.core.schemas import ChatRequest, ChatResponse
from personal_assistant.core.time_utils import clock_context
from personal_assistant.inference.llm import ContextLimitError, ModelError
from personal_assistant.integrations.research import web_capabilities
from personal_assistant.media.images import MAX_IMAGE_BYTES, ImageInputError, VisionUnavailable
from personal_assistant.media.pdf_uploads import MAX_PDF_UPLOAD_BYTES, PDFInputError, PDFUnavailable
from personal_assistant.storage.artifacts import MAX_ARTIFACT_BYTES

logger = logging.getLogger(__name__)


class AudioPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


def create_app(
    settings: Settings | None = None, supplied_runtime: Runtime | None = None
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        token = settings.api_token.get_secret_value()
        if len(token) < 24 or token == "change-me":
            raise RuntimeError("Set a private API token; run `assistant init` first")
        if supplied_runtime:
            app.state.runtime = supplied_runtime
            yield
        else:
            async with runtime(settings) as active:
                app.state.runtime = active
                yield

    app = FastAPI(title="Personal Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        PrivateBodyGuard,
        token=settings.api_token.get_secret_value(),
        limit=262_144,
        path_limits={
            "/v1/chat/image": MAX_IMAGE_BYTES + 65_536,
            "/v1/chat/pdf": MAX_PDF_UPLOAD_BYTES + 65_536,
        },
    )
    bearer = HTTPBearer(auto_error=False)

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if not credentials or not secrets.compare_digest(
            credentials.credentials, settings.api_token.get_secret_value()
        ):
            raise HTTPException(status_code=401, detail="A valid bearer token is required")

    auth = [Depends(authenticate)]

    @app.get("/health")
    async def health():
        try:
            await app.state.runtime.database.ping()
        except Exception as error:
            raise HTTPException(status_code=503, detail="Database is unavailable") from error
        return {"status": "ok"}

    @app.get("/v1/status", dependencies=auth)
    async def status():
        await app.state.runtime.database.ping()
        return {
            "database": "ready",
            "model_provider": settings.model_provider,
            "model": settings.local_model
            if settings.model_provider == "local"
            else settings.openai_model,
            "retrieval": (
                "keyword_fallback" if app.state.runtime.repository.retrieval_warning else "hybrid"
            )
            if settings.embedding_backend == "http"
            else "keyword",
            "retrieval_warning": app.state.runtime.repository.retrieval_warning,
            "search_configured": web_capabilities(settings)["search_web"],
            "search_provider": settings.search_provider,
            "web_capabilities": web_capabilities(settings),
            "job_sites": settings.job_sites,
            "timezone": settings.timezone,
            "clock": clock_context(settings.timezone),
            "calendar": "private",
            "max_review_replans": settings.max_review_replans,
            "python_analysis": {
                "configured": settings.python_enabled,
                "max_artifacts": 3,
                "max_artifact_bytes": MAX_ARTIFACT_BYTES,
            },
            "orchestration": {
                "max_agents_per_round": 5,
                "max_parallel_agents": settings.max_parallel_agents,
                "model_max_concurrency": settings.model_max_concurrency,
                "max_actions_per_round": 5,
                "write_plans": "ordered_tools_before_parallel_drafts",
            },
            "tracing": {
                "configured": settings.langfuse_enabled,
                **app.state.runtime.service.tracing.status,
            },
            "images": {
                "configured": settings.model_supports_images,
                "max_bytes": MAX_IMAGE_BYTES,
                "formats": ["JPEG", "PNG", "WebP"],
                "local_runtime_check": "llama.cpp /props modalities.vision",
            },
            "pdf_uploads": {"configured": settings.pdf_enabled, "max_bytes": MAX_PDF_UPLOAD_BYTES},
        }

    @app.get("/v1/time", dependencies=auth)
    async def current_time():
        return clock_context(settings.timezone)

    def telegram_chat_id():
        if settings.telegram_owner_id <= 0:
            raise HTTPException(409, "Configure ASSISTANT_TELEGRAM_OWNER_ID first")
        return f"tg:{settings.telegram_owner_id}"

    @app.get("/v1/telegram/audio", dependencies=auth, response_model=AudioPreference)
    async def audio_preference():
        return {"enabled": await app.state.runtime.repository.audio_enabled(telegram_chat_id())}

    @app.put("/v1/telegram/audio", dependencies=auth, response_model=AudioPreference)
    async def set_audio_preference(preference: AudioPreference):
        await app.state.runtime.repository.set_audio_enabled(telegram_chat_id(), preference.enabled)
        return preference

    @app.get("/v1/artifacts/{artifact_id}", dependencies=auth)
    async def artifact(artifact_id: str):
        import asyncio

        try:
            data = await asyncio.to_thread(app.state.runtime.service.artifacts.read, artifact_id)
        except (OSError, ValueError):
            raise HTTPException(404, "Artifact not found") from None
        return Response(
            data,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{artifact_id}.png"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=auth)
    async def chat(request: ChatRequest):
        return await execute_chat(request)

    @app.post("/v1/chat/image", response_model=ChatResponse, dependencies=auth)
    async def image_chat(
        raw_request: Request,
        image: Annotated[UploadFile, File()],
        text: Annotated[str, Form(min_length=1, max_length=12000)] = "Describe this image.",
        thread_id: Annotated[
            str, Form(min_length=1, max_length=100, pattern=r"^[\w:-]+$")
        ] = "default",
        request_id: Annotated[UUID | None, Form()] = None,
    ):
        try:
            if not settings.model_supports_images:
                raise HTTPException(
                    status_code=415, detail="Image input is disabled for the selected model"
                )
            form = await raw_request.form()
            if set(form) - {"image", "text", "thread_id", "request_id"} or any(
                len(form.getlist(field)) != 1 for field in form
            ):
                raise HTTPException(
                    status_code=422, detail="Send one image and no repeated or extra fields"
                )
            raw = await image.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")
            return await execute_chat(
                ChatRequest(text=text, thread_id=thread_id, request_id=request_id or uuid4()), raw
            )
        finally:
            await image.close()

    @app.post("/v1/chat/pdf", response_model=ChatResponse, dependencies=auth)
    async def pdf_chat(
        raw_request: Request,
        pdf: Annotated[UploadFile, File()],
        text: Annotated[
            str, Form(min_length=1, max_length=12000)
        ] = "Briefly summarize the first three pages of this PDF.",
        thread_id: Annotated[
            str, Form(min_length=1, max_length=100, pattern=r"^[\w:-]+$")
        ] = "default",
        request_id: Annotated[UUID | None, Form()] = None,
    ):
        try:
            if not settings.pdf_enabled:
                raise HTTPException(
                    status_code=415, detail="PDF uploads are disabled; enable the PDF integration"
                )
            form = await raw_request.form()
            if set(form) - {"pdf", "text", "thread_id", "request_id"} or any(
                len(form.getlist(field)) != 1 for field in form
            ):
                raise HTTPException(
                    status_code=422, detail="Send one PDF and no repeated or extra fields"
                )
            raw = await pdf.read(MAX_PDF_UPLOAD_BYTES + 1)
            if len(raw) > MAX_PDF_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="PDF upload exceeds the 10 MB limit")
            return await execute_chat(
                ChatRequest(text=text, thread_id=thread_id, request_id=request_id or uuid4()),
                pdf=raw,
                filename=pdf.filename or "document.pdf",
            )
        finally:
            await pdf.close()

    async def execute_chat(
        request: ChatRequest,
        image: bytes | None = None,
        *,
        pdf: bytes | None = None,
        filename: str = "document.pdf",
    ):
        started = time.monotonic()
        try:
            attachments = {}
            if image is not None:
                attachments["image"] = image
            if pdf is not None:
                attachments.update(pdf=pdf, filename=filename)
            result = await app.state.runtime.service.chat(request, **attachments)
            logger.info(
                "chat_completed request_id=%s specialist=%s review=%s duration_ms=%d",
                request.request_id,
                result.specialist,
                result.review_status,
                round((time.monotonic() - started) * 1000),
            )
            return result
        except ContextLimitError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except (VisionUnavailable, PDFUnavailable) as error:
            raise HTTPException(status_code=415, detail=str(error)) from error
        except (ImageInputError, PDFInputError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ModelError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            error_id = str(uuid4())
            logger.error("chat_failed error_id=%s type=%s", error_id, type(error).__name__)
            raise HTTPException(
                status_code=500, detail=f"Request failed; reference {error_id}"
            ) from error

    async def filtered_documents(kind, date_from, date_to, date_field):
        try:
            return await app.state.runtime.repository.list_documents(
                kind, date_from=date_from, date_to=date_to, date_field=date_field
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/notes", dependencies=auth)
    async def notes(
        date_from: date | None = None,
        date_to: date | None = None,
        date_field: Literal["created_at", "updated_at"] = "created_at",
    ):
        return await filtered_documents("note", date_from, date_to, date_field)

    @app.get("/v1/memories", dependencies=auth)
    async def memories(
        date_from: date | None = None,
        date_to: date | None = None,
        date_field: Literal["created_at", "updated_at"] = "created_at",
    ):
        return await filtered_documents("fact", date_from, date_to, date_field)

    @app.get("/v1/documents/{resource_id}", dependencies=auth)
    async def document(resource_id: str):
        try:
            return await app.state.runtime.repository.get_document(resource_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error

    @app.get("/v1/tasks", dependencies=auth)
    async def tasks():
        return await app.state.runtime.repository.tasks()

    @app.get("/v1/events", dependencies=auth)
    async def events(
        query: str = "",
        date_from: date | None = None,
        date_to: date | None = None,
        include_cancelled: bool = False,
    ):
        try:
            return await app.state.runtime.repository.calendar.events(
                query, date_from=date_from, date_to=date_to, include_cancelled=include_cancelled
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/events/{resource_id}", dependencies=auth)
    async def calendar_event(resource_id: str):
        try:
            return await app.state.runtime.repository.calendar.get(resource_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/v1/calendar.ics", dependencies=auth)
    async def export_calendar():
        return Response(
            await app.state.runtime.repository.calendar.export_ics(),
            media_type="text/calendar",
            headers={"Content-Disposition": 'attachment; filename="calendar.ics"'},
        )

    @app.get("/v1/documents/{resource_id}/revisions", dependencies=auth)
    async def revisions(resource_id: str):
        return await app.state.runtime.repository.revisions(resource_id)

    @app.get("/v1/export", response_class=PlainTextResponse, dependencies=auth)
    async def export():
        content = await app.state.runtime.repository.export_markdown()
        return PlainTextResponse(
            content, headers={"Content-Disposition": 'attachment; filename="assistant-notes.md"'}
        )

    @app.get("/v1/history/{thread_id}", dependencies=auth)
    async def history(thread_id: str):
        return await app.state.runtime.service.history(thread_id)

    return app
