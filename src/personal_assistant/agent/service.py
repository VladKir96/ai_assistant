import asyncio
from pathlib import Path

from sqlalchemy import select

from personal_assistant.agent.tool_execution import effective_results
from personal_assistant.core.schemas import ChatRequest, ChatResponse
from personal_assistant.core.tracing import Tracing
from personal_assistant.inference.llm import prompt_json
from personal_assistant.media.images import VisionUnavailable, prepare_image
from personal_assistant.media.pdf_uploads import PDFInputError, PDFUnavailable, PDFUploads
from personal_assistant.media.vision import VisionAnalyzer
from personal_assistant.storage.artifacts import ArtifactStore
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import Message, Run


class AssistantService:
    def __init__(
        self,
        db: Database,
        graph,
        input_budget: int,
        vision: VisionAnalyzer | None = None,
        pdfs: PDFUploads | None = None,
        tracing: Tracing | None = None,
        artifacts: ArtifactStore | None = None,
    ):
        self.db, self.graph, self.input_budget = db, graph, input_budget
        self.vision = vision
        self.pdfs = pdfs
        self.tracing = tracing or Tracing()
        self.artifacts = artifacts or ArtifactStore(Path("data/artifacts"))
        # One owner and API process: serialize Telegram and direct API requests.
        self.lock = asyncio.Lock()

    async def chat(
        self,
        request: ChatRequest,
        image: bytes | None = None,
        *,
        pdf: bytes | None = None,
        filename: str = "document.pdf",
    ) -> ChatResponse:
        with self.tracing.observation(
            "assistant.chat",
            session_id=request.thread_id,
            input={"text": request.text},
            metadata={
                "request_id": str(request.request_id),
                "attachment": "pdf"
                if pdf is not None
                else "image"
                if image is not None
                else "none",
            },
        ) as span:
            response = await self._chat(request, image, pdf=pdf, filename=filename)
            span.update(
                output={"text": response.text},
                metadata={
                    "specialist": response.specialist,
                    "specialists": response.specialists,
                    "review_status": response.review_status,
                    "review_replans": response.review_replans,
                    "max_review_replans": response.max_review_replans,
                    "failed_operations": sum(not item["ok"] for item in response.operations),
                },
                level="WARNING" if response.review_status == "needs_attention" else "DEFAULT",
                status_message=(
                    "Review needs attention"
                    if response.review_status == "needs_attention"
                    else None
                ),
            )
            return response

    async def _chat(
        self,
        request: ChatRequest,
        image: bytes | None = None,
        *,
        pdf: bytes | None = None,
        filename: str = "document.pdf",
    ) -> ChatResponse:
        async with self.lock:
            if image is not None and pdf is not None:
                raise PDFInputError("Send one attachment per request")
            attachment = None
            if pdf is not None:
                if not self.pdfs:
                    raise PDFUnavailable("PDF uploads are disabled; enable the PDF integration")
                attachment = await self.pdfs.prepare(pdf, filename)
            prepared = None
            if image is not None:
                if not self.vision or not self.vision.enabled:
                    raise VisionUnavailable("Image input is disabled for the selected model")
                prepared = await asyncio.to_thread(prepare_image, image)
            digest = prepared.digest if prepared else None
            run_id = str(request.request_id)
            observation = None
            pdf_evidence = None
            async with self.db.sessions.begin() as session:
                run = await session.get(Run, run_id)
                if run:
                    if (
                        run.thread_id != request.thread_id
                        or run.input_text != request.text
                        or run.image_digest != digest
                        or run.pdf_digest != (attachment["sha256"] if attachment else None)
                    ):
                        raise ValueError("Request ID was already used for different input")
                    if run.status == "completed":
                        with self.tracing.observation("request.cache_hit"):
                            pass
                        return ChatResponse.model_validate(run.response)
                    observation = run.image_observation
                    pdf_evidence = run.pdf_evidence
                else:
                    run = Run(
                        id=run_id,
                        thread_id=request.thread_id,
                        input_text=request.text,
                        image_digest=digest,
                        pdf_digest=attachment["sha256"] if attachment else None,
                    )
                    session.add(run)
                    await session.flush()
                    session.add(
                        Message(
                            thread_id=request.thread_id,
                            run_id=run_id,
                            role="user",
                            content=request.text
                            + (f"\n[Attached image sha256:{digest}]" if digest else "")
                            + (
                                "\n[Attached PDF: " + prompt_json(attachment) + "]"
                                if attachment
                                else ""
                            ),
                        )
                    )
            # Separate graph thread per request makes failed executions resumable even after
            # later turns. Conversation history lives in the archive, keyed by user thread.
            # Five repairs with team preparation/fan-out/synthesis need up to 37
            # supersteps. Parallel branches share a superstep rather than adding depth.
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": 48}
            snapshot = await self.graph.aget_state(config)
            initial = {
                "request_id": run_id,
                "thread_id": request.thread_id,
                "text": request.text,
            }
            try:
                if attachment:
                    if pdf_evidence is None:
                        with self.tracing.observation("pdf.extract", kind="tool"):
                            pdf_evidence = await self.pdfs.extract(attachment)
                        async with self.db.sessions.begin() as session:
                            run = await session.get(Run, run_id)
                            run.pdf_evidence = pdf_evidence
                    initial["pdf_evidence"] = pdf_evidence
                if prepared:
                    if observation is None:
                        observation = await self.vision.analyze(request.text, prepared)
                        async with self.db.sessions.begin() as session:
                            run = await session.get(Run, run_id)
                            run.image_observation = observation
                    initial["image_evidence"] = observation
                if snapshot.values and snapshot.next:
                    result = await self.graph.ainvoke(None, config)
                elif snapshot.values and snapshot.values.get("final"):
                    result = snapshot.values
                else:
                    result = await self.graph.ainvoke(initial, config)
                all_artifacts = self.response_artifacts(result["results"], limit=None)
                chart_notice = (
                    "\n\nOnly the first three charts are attached."
                    if len(all_artifacts) > 3
                    else ""
                )
                response = ChatResponse(
                    request_id=request.request_id,
                    thread_id=request.thread_id,
                    text=result["final"] + chart_notice,
                    specialist=result["plan"]["specialist"],
                    specialists=[agent["specialist"] for agent in result["plan"].get("agents", [])]
                    or [result["plan"]["specialist"]],
                    reviewed=result["reviewed"],
                    review_status=result["review_status"],
                    review_replans=result.get("revisions", 0),
                    max_review_replans=result.get("repair_limit", 2),
                    operations=[
                        {
                            "action": item["action"],
                            "ok": item["ok"],
                            **({"error": item["error"]} if not item["ok"] else {}),
                        }
                        for item in result["results"]
                    ],
                    artifacts=all_artifacts[:3],
                )
                async with self.db.sessions.begin() as session:
                    run = await session.get(Run, run_id)
                    run.status, run.response = "completed", response.model_dump(mode="json")
                    if observation:
                        # Archive after graph preparation to avoid duplicating this turn's
                        # observation in both compact history and dedicated image evidence.
                        message = await session.scalar(
                            select(Message).where(Message.run_id == run_id, Message.role == "user")
                        )
                        message.content = (
                            request.text
                            + "\n[Attached image; untrusted model observation: "
                            + prompt_json(observation)
                            + "]"
                        )
                    if pdf_evidence:
                        message = await session.scalar(
                            select(Message).where(Message.run_id == run_id, Message.role == "user")
                        )
                        message.content = (
                            request.text
                            + "\n[Attached PDF; untrusted bounded excerpt: "
                            + prompt_json(pdf_evidence)
                            + "]"
                        )
                    session.add(
                        Message(
                            thread_id=request.thread_id,
                            run_id=run_id,
                            role="assistant",
                            content=response.text,
                        )
                    )
                return response
            except Exception:
                async with self.db.sessions.begin() as session:
                    run = await session.get(Run, run_id)
                    run.status = "failed"
                raise

    @staticmethod
    def response_artifacts(results: list[dict], limit: int | None = 3) -> list[dict]:
        artifacts = {}
        # Supersede only the repaired task, including failed/no-chart replacements.
        # Independently completed tasks survive repairs elsewhere in the request.
        for item in effective_results(results):
            if item["action"] == "run_python" and item["ok"]:
                for artifact in item["result"].get("artifacts", []):
                    artifacts.setdefault(artifact["id"], artifact)
        return list(artifacts.values())[:limit]

    async def history(self, thread_id: str, limit: int = 30) -> list[dict]:
        async with self.db.sessions() as session:
            messages = (
                await session.scalars(
                    select(Message)
                    .where(Message.thread_id == thread_id)
                    .order_by(Message.id.desc())
                    .limit(limit)
                )
            ).all()
            return [{"role": item.role, "content": item.content} for item in reversed(messages)]
