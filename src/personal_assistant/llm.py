import asyncio
import copy
import json
import logging
from time import monotonic
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from personal_assistant.config import Settings
from personal_assistant.images import PreparedImage, VisionUnavailable
from personal_assistant.time_utils import with_runtime_clock
from personal_assistant.tracing import Tracing

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)
CONTEXT_MESSAGE = (
    "Required input exceeds context budget; shorten the task or check model/tokenizer configuration"
)


class ModelError(RuntimeError):
    pass


class ContextLimitError(ValueError):
    pass


def clip(value: str, byte_limit: int) -> str:
    return value.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")


def prompt_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def reduced_payloads(payload: dict):
    """Prefer recent turns and tool evidence over old summaries and incidental retrieval."""
    result = copy.deepcopy(payload)
    result["context_was_reduced"] = True
    context = result.get("context")
    if isinstance(context, dict) and context.get("summary"):
        context["summary"] = ""
        yield result
    for key in ("open_tasks", "calendar_events", "knowledge"):
        while result.get(key):
            result[key].pop()
            yield result
    if isinstance(context, dict):
        messages = context.get("recent_messages", [])
        while len(messages) > 1:
            messages.pop(0)
            yield result
        if messages:
            while messages[0].get("content"):
                messages[0]["content"] = messages[0]["content"][: len(messages[0]["content"]) // 2]
                messages[0]["truncated"] = True
                yield result
    elif isinstance(context, str):
        while result["context"]:
            result["context"] = result["context"][: len(result["context"]) // 2]
            yield result


def fit_payload(system: str, payload: dict, definition: dict, budget: int) -> dict:
    """Shrink optional retrieved context, never the current request or draft under review."""
    result = copy.deepcopy(payload)

    def size():
        return len((system + prompt_json(result) + prompt_json(definition)).encode())

    if size() <= budget:
        return result
    for result in reduced_payloads(payload):
        if size() <= budget:
            return result
    raise ContextLimitError(CONTEXT_MESSAGE)


class LanguageModel(Protocol):
    async def generate(self, system: str, payload: dict, schema: type[T]) -> T: ...


class HTTPModel:
    def __init__(self, settings: Settings, *, tracing: Tracing | None = None):
        self.settings = settings
        self.tracing = tracing or Tracing()
        self.client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self.semaphore = asyncio.Semaphore(settings.model_max_concurrency)
        self.token_counter_supported: bool | None = None

    async def input_size(self, system: str, payload: dict, definition: dict, name: str) -> int:
        content = prompt_json(payload)
        text = system + content + prompt_json(definition)
        # A byte bound is cheap and safe for small requests. Larger requests use the
        # configured provider's counter, never another model or a cloud fallback.
        if (
            len(text.encode()) <= self.settings.input_budget - 150
            or self.token_counter_supported is False
        ):
            return len(text.encode())
        if self.settings.model_provider == "local":
            url = self.settings.local_base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
            body = {"content": text, "add_special": True, "parse_special": False}
            key = self.settings.local_api_key.get_secret_value()
        else:
            request, endpoint, key = self._request(system, content, definition, name)
            url = endpoint + "/input_tokens"
            body = {field: request[field] for field in ("model", "instructions", "input", "text")}
        try:
            response = await self.client.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=15,
            )
            if response.status_code in {400, 404, 405, 501}:
                self.token_counter_supported = False
                logger.warning(
                    "Configured endpoint has no token counter; using conservative byte accounting"
                )
                return len(text.encode())
            response.raise_for_status()
            data = response.json()
            count = (
                len(data["tokens"])
                if self.settings.model_provider == "local" and isinstance(data["tokens"], list)
                else data["input_tokens"]
            )
            if type(count) is not int or count <= 0:
                raise ValueError("Invalid token count")
            self.token_counter_supported = True
            return count
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
            raise ModelError(
                "Configured model token counter failed; check the endpoint or retry"
            ) from None

    async def fit_request(
        self, system: str, payload: dict, schema: type[BaseModel], image_tokens: int = 0
    ) -> dict:
        definition = schema.model_json_schema()
        budget = self.settings.input_budget - 150 - image_tokens
        if await self.input_size(system, payload, definition, schema.__name__) <= budget:
            return copy.deepcopy(payload)
        for reduced in reduced_payloads(payload):
            if await self.input_size(system, reduced, definition, schema.__name__) <= budget:
                return reduced
        raise ContextLimitError(CONTEXT_MESSAGE)

    async def ensure_vision(self):
        if not self.settings.model_supports_images:
            raise VisionUnavailable("Image input is disabled for the selected model")
        if self.settings.model_provider != "local":
            return
        # Check the running server, not only the configured profile. Some text-only
        # llama.cpp versions fail badly on images without a matching projector.
        try:
            response = await self.client.get(
                self.settings.local_base_url.rstrip("/").removesuffix("/v1") + "/props",
                headers={
                    "Authorization": f"Bearer {self.settings.local_api_key.get_secret_value()}"
                },
                timeout=10,
            )
            response.raise_for_status()
            supported = response.json().get("modalities", {}).get("vision") is True
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            raise VisionUnavailable(
                "Cannot verify local vision support; check llama.cpp /props and the mmproj file"
            ) from None
        if not supported:
            raise VisionUnavailable(
                "The running local model has no vision support; load its matching mmproj"
            )

    async def generate(
        self, system: str, payload: dict, schema: type[T], *, image: PreparedImage | None = None
    ) -> T:
        with self.tracing.observation(
            "model.request", metadata={"schema": schema.__name__, "has_image": image is not None}
        ):
            return await self._generate(system, payload, schema, image=image)

    async def _generate(
        self, system: str, payload: dict, schema: type[T], *, image: PreparedImage | None = None
    ) -> T:
        if image:
            await self.ensure_vision()
        definition = schema.model_json_schema()
        system = with_runtime_clock(system, payload)
        payload = await self.fit_request(
            system,
            payload,
            schema,
            image_tokens=self.settings.vision_token_reserve if image else 0,
        )
        content = prompt_json(payload)
        async with self.semaphore:
            for attempt in range(2):
                instruction = system
                if attempt:
                    instruction += " Return exactly one JSON object matching the supplied schema."
                body, endpoint, key = self._request(
                    instruction, content, definition, schema.__name__, image=image
                )
                try:
                    return await self._attempt(
                        body, endpoint, key, schema, instruction, payload, attempt
                    )
                except (
                    ValidationError,
                    KeyError,
                    TypeError,
                    ValueError,
                    IndexError,
                    AttributeError,
                ) as error:
                    if attempt:
                        raise ModelError("Model returned invalid structured output") from error
                except httpx.HTTPError as error:
                    retryable = not isinstance(error, httpx.HTTPStatusError) or (
                        error.response.status_code in {429, 500, 502, 503, 504}
                    )
                    if attempt or not retryable:
                        # Do not include response bodies, prompts, headers, or URLs in errors.
                        raise ModelError(
                            "Inference endpoint failed; check its configuration"
                        ) from error
                    await asyncio.sleep(0.5)
        raise ModelError("No model response")

    async def _attempt(self, body, endpoint, key, schema, instruction, payload, attempt):
        settings = self.settings
        with self.tracing.observation(
            "model.generate",
            kind="generation",
            model=settings.local_model
            if settings.model_provider == "local"
            else settings.openai_model,
            input={"system": instruction, "payload": payload},
            model_parameters={"max_output_tokens": settings.max_output_tokens},
            metadata={
                "provider": settings.model_provider,
                "schema": schema.__name__,
                "attempt": attempt + 1,
            },
        ) as span:
            response = await self.client.post(
                endpoint,
                json=body,
                headers={"Authorization": f"Bearer {key}"} if key else {},
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage")
            if isinstance(usage, dict):
                counts = {
                    "input": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "output": usage.get("completion_tokens", usage.get("output_tokens")),
                    "total": usage.get("total_tokens"),
                }
                counts = {key: val for key, val in counts.items() if type(val) is int and val >= 0}
                if counts:
                    span.update(usage_details=counts)
            if settings.model_provider == "local":
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ModelError("Model output was truncated; increase the output budget")
                raw = choice["message"]["content"]
            else:
                if data.get("status") == "incomplete":
                    raise ModelError("Model response was incomplete")
                raw = "".join(
                    item["text"]
                    for output in data.get("output", [])
                    for item in output.get("content", [])
                    if item.get("type") == "output_text"
                )
            span.update(output=raw)
            return schema.model_validate_json(raw)

    def _request(
        self,
        system: str,
        content: str,
        definition: dict,
        name: str,
        *,
        image: PreparedImage | None = None,
    ):
        settings = self.settings
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        if settings.model_provider == "local":
            if image:
                messages[1]["content"] = [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": image.data_url()}},
                ]
            return (
                {
                    "model": settings.local_model,
                    "messages": messages,
                    "max_tokens": settings.max_output_tokens,
                    "temperature": 0.2,
                    "chat_template_kwargs": {"enable_thinking": settings.enable_thinking},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": name, "schema": definition, "strict": False},
                    },
                },
                f"{settings.local_base_url.rstrip('/')}/chat/completions",
                settings.local_api_key.get_secret_value(),
            )
        return (
            {
                "model": settings.openai_model,
                "instructions": system,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": content},
                            {
                                "type": "input_image",
                                "image_url": image.data_url(),
                                "detail": "auto",
                            },
                        ]
                        if image
                        else content,
                    }
                ],
                "max_output_tokens": settings.max_output_tokens,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "schema": definition,
                        "strict": False,
                    }
                },
            },
            f"{settings.openai_base_url.rstrip('/')}/responses",
            settings.openai_api_key.get_secret_value(),
        )

    async def close(self):
        await self.client.aclose()


class Embeddings:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=60)
        self.unavailable_until = 0.0

    async def embed(self, content: str) -> list[float] | None:
        if self.settings.embedding_backend == "disabled":
            return None
        if monotonic() < self.unavailable_until:
            raise ModelError("Embedding service temporarily unavailable")
        key = self.settings.embedding_api_key.get_secret_value()
        try:
            response = await self.client.post(
                f"{self.settings.embedding_base_url.rstrip('/')}/embeddings",
                json={"input": clip(content, 16000), "model": self.settings.embedding_model},
                headers={"Authorization": f"Bearer {key}"} if key else {},
            )
            response.raise_for_status()
            vector = response.json()["data"][0]["embedding"]
            import math

            if len(vector) != 1024 or not all(math.isfinite(float(x)) for x in vector):
                raise ValueError("Invalid embedding")
            return [float(x) for x in vector]
        except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as error:
            self.unavailable_until = monotonic() + 30
            raise ModelError("Embedding service failed; no cloud fallback was attempted") from error

    async def close(self):
        await self.client.aclose()
