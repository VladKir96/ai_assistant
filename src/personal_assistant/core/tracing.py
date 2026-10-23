"""Explicit Langfuse instrumentation, with no automatic capture of graph state or errors."""

import asyncio
import hashlib
import json
import logging
import re
from contextlib import contextmanager, nullcontext
from functools import wraps

from personal_assistant.core.config import Settings

logger = logging.getLogger(__name__)
CONTENT_LIMIT = 16_000
SENSITIVE_KEY = re.compile(r"token|secret|password|authorization|api.?key|data_url", re.I)


class Observation:
    def __init__(self, tracer, span=None):
        self.tracer, self.span = tracer, span

    def update(self, **attributes):
        if self.span is None:
            return
        try:
            for key in ("input", "output"):
                if key in attributes:
                    if self.tracer.capture_content:
                        attributes[key] = self.tracer.content(attributes[key])
                    else:
                        attributes.pop(key)
            self.span.update(**attributes)
        except Exception:
            self.tracer.warn()


class Tracing:
    def __init__(self, settings: Settings | None = None):
        self.client = None
        self.provider = None
        self.capture_content = bool(settings and settings.langfuse_capture_content)
        self._warned = False
        self._secrets = (
            tuple(
                value.get_secret_value()
                for value in settings.__dict__.values()
                if hasattr(value, "get_secret_value") and value.get_secret_value()
            )
            if settings
            else ()
        )
        if not settings or not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

            # A private provider avoids exporting unrelated libraries' spans or
            # altering the process-wide OpenTelemetry provider.
            self.provider = TracerProvider(
                sampler=ParentBased(TraceIdRatioBased(settings.langfuse_sample_rate))
            )
            self.client = Langfuse(
                public_key=settings.langfuse_public_key.get_secret_value(),
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_base_url.rstrip("/"),
                environment=settings.langfuse_environment,
                sample_rate=settings.langfuse_sample_rate,
                tracing_enabled=True,
                debug=False,
                timeout=3,
                tracer_provider=self.provider,
            )
        except Exception:
            self.warn()

    def warn(self):
        if not self._warned:
            logger.warning("Langfuse instrumentation unavailable; assistant execution continues")
            self._warned = True

    def content(self, value):
        """Bound opt-in content and strip configured credentials and binary media."""

        def clean(item, depth=0):
            if depth > 8:
                return "[depth limit]"
            if isinstance(item, dict):
                return {
                    str(key): "[redacted]"
                    if SENSITIVE_KEY.search(str(key))
                    else clean(val, depth + 1)
                    for key, val in list(item.items())[:64]
                }
            if isinstance(item, (list, tuple)):
                return [clean(val, depth + 1) for val in item[:64]]
            if isinstance(item, str):
                text = item
                for secret in self._secrets:
                    text = text.replace(secret, "[redacted]")
                text = re.sub(r"data:[^\s\"']+", "[media omitted]", text)
                text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", text)
                return text.encode()[:4000].decode(errors="ignore")
            if item is None or type(item) in {int, float, bool}:
                return item
            return "[unsupported value omitted]"

        result = clean(value)
        encoded = json.dumps(result, ensure_ascii=False).encode()
        if len(encoded) > CONTENT_LIMIT:
            return {"truncated_excerpt": encoded[: CONTENT_LIMIT - 100].decode(errors="ignore")}
        return result

    @contextmanager
    def observation(self, name, *, kind="span", input=None, session_id=None, **attributes):
        cm = None
        observation = Observation(self)
        try:
            if self.client is not None:
                if self.capture_content and input is not None:
                    attributes["input"] = self.content(input)
                cm = self.client.start_as_current_observation(name=name, as_type=kind, **attributes)
                observation = Observation(self, cm.__enter__())
        except Exception:
            cm = None
            self.warn()
        propagation = nullcontext()
        if session_id is not None and observation.span is not None:
            try:
                from langfuse import propagate_attributes

                propagation = propagate_attributes(
                    session_id=hashlib.sha256(session_id.encode()).hexdigest(),
                    user_id="owner",
                    trace_name="assistant.chat",
                )
                propagation.__enter__()
            except Exception:
                propagation = nullcontext()
                self.warn()
        try:
            yield observation
        except BaseException as error:
            # Do not let SDK context managers capture exception text/tracebacks,
            # which may contain URLs, user text, or provider credentials.
            observation.update(level="ERROR", status_message=type(error).__name__)
            raise
        finally:
            try:
                propagation.__exit__(None, None, None)
            except Exception:
                self.warn()
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    self.warn()

    def node(self, name, function):
        @wraps(function)
        async def traced(state):
            with self.observation(
                "graph." + name,
                metadata={
                    "review_round": state.get("revisions", 0),
                    "agent_index": state.get("assigned_task", {}).get("index", -1),
                },
            ) as span:
                result = await function(state)
                summary = {
                    key: result[key]
                    for key in ("review_decision", "review_status", "revisions", "repair_limit")
                    if key in result
                }
                if "plan" in result:
                    agents = result["plan"].get("agents", [])
                    summary.update(
                        specialist=result["plan"]["specialist"],
                        agent_count=len(agents) or 1,
                        action_count=sum(len(agent["actions"]) for agent in agents)
                        if agents
                        else len(result["plan"]["actions"]),
                    )
                span.update(metadata=summary)
                return result

        return traced

    @property
    def status(self):
        return {"active": self.client is not None, "capture_content": self.capture_content}

    async def close(self):
        if self.client is not None:
            try:
                # SDK batches in the background. Flush/shut down once per runtime,
                # outside the event loop, rather than delaying each request.
                await asyncio.to_thread(self.client.shutdown)
            except Exception:
                self.warn()
        if self.provider is not None:
            try:
                await asyncio.to_thread(self.provider.shutdown)
            except Exception:
                self.warn()


async def check_tracing(settings: Settings):
    if not settings.langfuse_enabled:
        raise RuntimeError("Tracing is disabled; set ASSISTANT_LANGFUSE_ENABLED=true first")
    tracer = Tracing(settings)
    try:
        if tracer.client is None or not await asyncio.to_thread(tracer.client.auth_check):
            raise RuntimeError("Langfuse connection failed; check the URL and project API keys")
        with tracer.observation("assistant.tracing_check", metadata={"synthetic": True}):
            pass
        await asyncio.to_thread(tracer.client.flush)
    except Exception:
        raise RuntimeError(
            "Langfuse connection failed; check the URL and project API keys"
        ) from None
    finally:
        await tracer.close()
