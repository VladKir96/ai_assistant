import asyncio
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from personal_assistant.core.config import Settings
from personal_assistant.core.http_guard import PrivateBodyGuard


class EmbeddingRequest(BaseModel):
    input: str = Field(min_length=1, max_length=16000)
    model: str


def create_embedding_app(settings: Settings | None = None):
    settings = settings or Settings()
    if len(settings.api_token.get_secret_value()) < 24:
        raise ValueError("Set a private API token before starting embeddings")

    @asynccontextmanager
    async def lifespan(app):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding dependencies are missing; run `uv sync --frozen`."
            ) from error
        app.state.encoder = await asyncio.to_thread(
            SentenceTransformer,
            settings.embedding_model,
            device=settings.embedding_device,
            cache_folder="data/models/embeddings",
        )
        yield

    app = FastAPI(title="Local multilingual embeddings", lifespan=lifespan)
    app.add_middleware(PrivateBodyGuard, token=settings.api_token.get_secret_value(), limit=262_144)
    bearer = HTTPBearer(auto_error=False)
    lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    async def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ):
        if not credentials or not secrets.compare_digest(
            credentials.credentials, settings.api_token.get_secret_value()
        ):
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.post("/v1/embeddings", dependencies=[Depends(authenticate)])
    async def embed(request: EmbeddingRequest):
        if request.model != settings.embedding_model:
            raise HTTPException(status_code=422, detail="Requested embedding model is not loaded")
        async with lock:
            result = await asyncio.to_thread(
                app.state.encoder.encode, [request.input], normalize_embeddings=True
            )
        return {
            "model": request.model,
            "data": [{"object": "embedding", "index": 0, "embedding": result[0].tolist()}],
        }

    return app
