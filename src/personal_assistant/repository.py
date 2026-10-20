import json
import math
import re
from collections import defaultdict
from datetime import UTC, date
from typing import Literal
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

from sqlalchemy import case, delete, func, or_, select, update

from personal_assistant.calendar_store import CalendarStore
from personal_assistant.database import Database
from personal_assistant.llm import Embeddings, ModelError, clip
from personal_assistant.models import (
    CalendarEvent,
    Document,
    Message,
    Operation,
    Revision,
    Run,
    Task,
    now,
)
from personal_assistant.schemas import Action
from personal_assistant.time_utils import date_bounds, utc_timestamp


def document_dict(document: Document, full: bool = False) -> dict:
    return {
        "id": document.id,
        "kind": document.kind,
        "title": document.title,
        "created_at": utc_timestamp(document.created_at),
        "updated_at": utc_timestamp(document.updated_at),
        "tags": document.tags,
        "source": document.source,
        "content": document.content if full else clip(document.content, 1400),
    }


def task_dict(task: Task) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "created_at": utc_timestamp(task.created_at),
        "due_at": utc_timestamp(task.due_at) if task.due_at else None,
        "notification_status": task.notification_status,
        "content": task.content,
    }


class Repository:
    def __init__(self, db: Database, embeddings: Embeddings):
        self.db = db
        self.embeddings = embeddings
        self.timezone = ZoneInfo(embeddings.settings.timezone)
        self.calendar = CalendarStore(db, self.timezone)
        self.retrieval_warning: str | None = None

    def date_filters(self, column, date_from: date | None, date_to: date | None) -> list:
        start, end = date_bounds(date_from, date_to, self.timezone)
        clauses = []
        if start:
            clauses.append(column >= start)
        if end:
            clauses.append(column < end)
        return clauses

    def document_date_filters(self, date_from, date_to, date_field):
        columns = {"created_at": Document.created_at, "updated_at": Document.updated_at}
        if date_field not in columns:
            raise ValueError("Choose created_at or updated_at for date filtering")
        return self.date_filters(columns[date_field], date_from, date_to)

    async def get_document(self, resource_id: str) -> dict:
        async with self.db.sessions() as session:
            document = await session.get(Document, resource_id)
            if not document:
                raise ValueError("Document not found")
            return document_dict(document, full=True)

    async def list_documents(
        self,
        kind: str | None = None,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        date_field: Literal["created_at", "updated_at"] = "created_at",
    ) -> list[dict]:
        filters = self.document_date_filters(date_from, date_to, date_field)
        async with self.db.sessions() as session:
            query = select(Document).where(*filters).order_by(Document.updated_at.desc()).limit(200)
            if kind:
                query = query.where(Document.kind == kind)
            return [document_dict(item) for item in (await session.scalars(query)).all()]

    async def search(
        self,
        query: str,
        limit: int = 6,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        date_field: Literal["created_at", "updated_at"] = "created_at",
    ) -> list[dict]:
        filters = self.document_date_filters(date_from, date_to, date_field)
        self.retrieval_warning = None
        try:
            vector = await self.embeddings.embed(query) if query.strip() else None
        except ModelError:
            vector = None
            self.retrieval_warning = "Semantic retrieval unavailable; keyword search was used."
        terms = re.findall(r"\w+", query.casefold())[:12]
        async with self.db.sessions() as session:
            base = select(Document).where(*filters)
            score = 0
            if terms:
                clauses = []
                for term in terms:
                    term = term.replace("_", r"\_")
                    clauses.extend(
                        [
                            Document.title.ilike(f"%{term}%", escape="\\"),
                            Document.content.ilike(f"%{term}%", escape="\\"),
                        ]
                    )
                    score += case((Document.title.ilike(f"%{term}%", escape="\\"), 4), else_=0)
                    score += case((Document.content.ilike(f"%{term}%", escape="\\"), 1), else_=0)
                base = base.where(or_(*clauses))
                base = base.order_by(score.desc())
            lexical = list(
                (
                    await session.scalars(
                        base.order_by(Document.updated_at.desc(), Document.id).limit(40)
                    )
                ).all()
            )
            semantic = []
            if vector is not None:
                candidates = select(Document).where(
                    *filters,
                    Document.embedding.is_not(None),
                    Document.embedding_model == self.embeddings.settings.embedding_model,
                )
                if self.db.url.startswith("postgresql"):
                    candidates = candidates.order_by(
                        Document.embedding.cosine_distance(vector)
                    ).limit(20)
                    semantic = list((await session.scalars(candidates)).all())
                else:
                    semantic = list((await session.scalars(candidates)).all())

                    def distance(document):
                        norm = math.sqrt(sum(x * x for x in vector)) * math.sqrt(
                            sum(x * x for x in document.embedding)
                        )
                        return -sum(
                            a * b for a, b in zip(vector, document.embedding, strict=True)
                        ) / (norm or 1)

                    semantic.sort(key=distance)
                    semantic = semantic[:20]
            scores = defaultdict(float)
            documents = {}
            for ranking in (lexical, semantic):
                for position, item in enumerate(ranking):
                    scores[item.id] += 1 / (60 + position)
                    documents[item.id] = item
            selected = sorted(scores, key=scores.get, reverse=True)[:limit]
            return [document_dict(documents[key]) for key in selected]

    async def tasks(self) -> list[dict]:
        async with self.db.sessions() as session:
            return [
                task_dict(item)
                for item in (
                    await session.scalars(
                        select(Task)
                        .where(Task.status == "open")
                        .order_by(Task.created_at)
                        .limit(100)
                    )
                ).all()
            ]

    async def search_history(
        self, query: str, *, date_from: date | None = None, date_to: date | None = None
    ) -> list[dict]:
        terms = re.findall(r"\w+", query)[:8]
        if not terms and not (date_from or date_to):
            return []
        filters = self.date_filters(Message.created_at, date_from, date_to)
        if terms:
            filters.append(or_(*[Message.content.ilike(f"%{term}%") for term in terms]))
        async with self.db.sessions() as session:
            messages = (
                await session.scalars(
                    select(Message)
                    .join(Run, Message.run_id == Run.id)
                    .where(
                        Run.status == "completed",
                        *filters,
                    )
                    .order_by(Message.id.desc())
                    .limit(5)
                )
            ).all()
            return [
                {
                    "message_id": item.id,
                    "thread_id": item.thread_id,
                    "role": item.role,
                    "content": clip(item.content, 800),
                    "created_at": utc_timestamp(item.created_at),
                }
                for item in messages
            ]

    async def revisions(self, resource_id: str) -> list[dict]:
        async with self.db.sessions() as session:
            return [
                {"created_at": utc_timestamp(item.created_at), "snapshot": item.snapshot}
                for item in (
                    await session.scalars(
                        select(Revision)
                        .where(Revision.document_id == resource_id)
                        .order_by(Revision.id.desc())
                        .limit(50)
                    )
                ).all()
            ]

    async def mutation_target(self, action: Action) -> dict:
        """Resolve write selectors from storage, never from model-supplied titles."""
        model = {
            "update_note": Document,
            "complete_task": Task,
            "update_event": CalendarEvent,
        }[action.name]
        filters = [Document.kind == "note"] if model is Document else []
        async with self.db.sessions() as session:
            title = await session.scalar(
                select(model.title).where(model.id == action.resource_id, *filters)
            )
            if title is None:
                raise ValueError("Write target not found; retrieve its exact ID first")
            count = await session.scalar(
                select(func.count()).select_from(model).where(model.title == title, *filters)
            )
            return {"title": title, "unique_title": count == 1}

    async def mutate(self, action: Action, operation_id: str, source: str) -> dict:
        # Ledger and mutation commit together. Replaying a graph cannot duplicate a write.
        async with self.db.sessions.begin() as session:
            existing = await session.get(Operation, operation_id)
            if existing:
                return existing.result
            resource_id = str(uuid5(NAMESPACE_URL, operation_id))
            if action.name in {"create_event", "update_event", "cancel_event"}:
                result = await self.calendar.mutate(session, action, resource_id, source)
            elif action.name in {"save_note", "remember", "update_note"}:
                if not action.content.strip():
                    raise ValueError("A note or memory must have content")
                document = None
                if action.name == "update_note":
                    document = await session.get(Document, action.resource_id)
                    if not document or document.kind != "note":
                        raise ValueError("Note not found; retrieve its exact ID before updating")
                elif action.name == "remember":
                    if not action.title.strip():
                        raise ValueError("Memory requires a stable fact key in title")
                    document = await session.scalar(
                        select(Document).where(
                            Document.kind == "fact", Document.title == action.title
                        )
                    )
                if document:
                    session.add(
                        Revision(
                            document_id=document.id, snapshot=document_dict(document, full=True)
                        )
                    )
                    document.content = action.content
                    document.title = action.title or document.title
                    document.tags = action.tags or document.tags
                    document.updated_at = now()
                    document.source = source
                else:
                    document = Document(
                        id=resource_id,
                        kind="fact" if action.name == "remember" else "note",
                        title=action.title or action.content[:80],
                        content=action.content,
                        tags=action.tags,
                        source=source,
                    )
                    session.add(document)
                warning = None
                try:
                    document.embedding = await self.embeddings.embed(
                        f"{document.title}\n{document.content}"
                    )
                except ModelError:
                    document.embedding = None
                    warning = "Note saved without a vector; run reindex after embeddings recover."
                document.embedding_model = (
                    self.embeddings.settings.embedding_model if document.embedding else None
                )
                await session.flush()
                result = {"action": action.name, "status": "saved", **document_dict(document)}
                if warning:
                    result["integration_warning"] = warning
            elif action.name in {"delete_note", "forget"}:
                document = await session.get(Document, action.resource_id)
                expected_kind = "note" if action.name == "delete_note" else "fact"
                if not document or document.kind != expected_kind:
                    raise ValueError("Resource not found; retrieve its exact ID before deleting")
                await session.execute(delete(Revision).where(Revision.document_id == document.id))
                await session.delete(document)
                result = {"action": action.name, "status": "deleted", "id": action.resource_id}
            elif action.name == "create_task":
                if not action.title.strip():
                    raise ValueError("Task title is required")
                if action.due_at and action.due_at.tzinfo is None:
                    raise ValueError("Reminder time must include a timezone offset")
                if action.due_at and action.due_at.astimezone(UTC) <= now():
                    raise ValueError("Reminder time must be in the future")
                task = Task(
                    id=resource_id,
                    title=action.title,
                    content=action.content,
                    due_at=action.due_at,
                )
                session.add(task)
                await session.flush()
                result = {"action": action.name, "status": "saved", "task": task_dict(task)}
            elif action.name == "complete_task":
                task = await session.get(Task, action.resource_id)
                if not task:
                    raise ValueError("Task not found")
                task.status = "completed"
                result = {"action": action.name, "status": "completed", "task": task_dict(task)}
            else:
                raise ValueError("Unsupported mutation")
            session.add(Operation(id=operation_id, result=result))
            return result

    async def reindex(self) -> int:
        if self.embeddings.settings.embedding_backend == "disabled":
            raise ValueError("Enable an embedding service before reindexing")
        count, cursor = 0, ""
        while True:
            async with self.db.sessions() as session:
                identifiers = list(
                    (
                        await session.scalars(
                            select(Document.id)
                            .where(Document.id > cursor)
                            .order_by(Document.id)
                            .limit(32)
                        )
                    ).all()
                )
            if not identifiers:
                return count
            for identifier in identifiers:
                for attempt in range(3):
                    async with self.db.sessions() as session:
                        snapshot = (
                            await session.execute(
                                select(Document.title, Document.content, Document.updated_at).where(
                                    Document.id == identifier
                                )
                            )
                        ).first()
                    if snapshot is None:
                        break  # Deleted while maintenance was running.
                    vector = await self.embeddings.embed(f"{snapshot.title}\n{snapshot.content}")
                    async with self.db.sessions.begin() as session:
                        saved = await session.execute(
                            update(Document)
                            .where(
                                Document.id == identifier,
                                Document.title == snapshot.title,
                                Document.content == snapshot.content,
                                Document.updated_at == snapshot.updated_at,
                            )
                            .values(
                                embedding=vector,
                                embedding_model=self.embeddings.settings.embedding_model,
                            )
                        )
                    if saved.rowcount:
                        count += 1
                        break
                    if attempt == 2:
                        raise ModelError(
                            "Reindex partially completed; a note kept changing. Retry reindex."
                        )
            cursor = identifiers[-1]

    async def export_markdown(self) -> str:
        async with self.db.sessions() as session:
            documents = (
                await session.scalars(select(Document).order_by(Document.created_at))
            ).all()
            parts = []
            for document in documents:
                parts.append(
                    f"# {document.title}\n\n"
                    f"ID: {document.id}\n\nKind: {document.kind}\n\n"
                    f"Created (UTC): {utc_timestamp(document.created_at)}\n\n"
                    f"Updated (UTC): {utc_timestamp(document.updated_at)}\n\n"
                    f"Tags: {json.dumps(document.tags, ensure_ascii=False)}\n\n"
                    f"{document.content}\n"
                )
            return "\n---\n\n".join(parts)
