import asyncio
import logging
from datetime import timedelta

import httpx
from celery import Celery
from sqlalchemy import and_, or_, select, update

from personal_assistant.core.config import Settings
from personal_assistant.storage.database import Database
from personal_assistant.storage.models import Task, now

settings = Settings()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
celery_app = Celery("assistant", broker=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    worker_prefetch_multiplier=1,
    beat_schedule={"due-reminders": {"task": "assistant.notify_due", "schedule": 30.0}},
    timezone="UTC",
    broker_connection_retry_on_startup=True,
)


async def claim_due(db: Database) -> list[dict]:
    current = now()
    async with db.sessions.begin() as session:
        # A crash after claiming the last attempt must reach a visible terminal
        # state once its lease expires, rather than remaining "sending" forever.
        await session.execute(
            update(Task)
            .where(
                Task.status == "open",
                Task.notification_attempts >= 5,
                or_(
                    Task.notification_status == "pending",
                    and_(
                        Task.notification_status == "sending",
                        Task.notification_claimed_at < current - timedelta(minutes=3),
                    ),
                ),
            )
            .values(notification_status="failed", notification_claimed_at=None)
        )
        tasks = (
            await session.scalars(
                select(Task)
                .where(
                    Task.status == "open",
                    Task.due_at <= current,
                    Task.notification_attempts < 5,
                    or_(
                        Task.notification_status == "pending",
                        and_(
                            Task.notification_status == "sending",
                            Task.notification_claimed_at < current - timedelta(minutes=3),
                        ),
                    ),
                )
                .order_by(Task.due_at)
                .limit(20)
                .with_for_update(skip_locked=True)
            )
        ).all()
        claimed = []
        for task in tasks:
            task.notification_status = "sending"
            task.notification_claimed_at = current
            task.notification_attempts += 1
            claimed.append({"id": task.id, "title": task.title})
        return claimed


async def deliver_due(active_settings: Settings):
    if (
        not active_settings.telegram_token.get_secret_value()
        or not active_settings.telegram_owner_id
    ):
        return 0
    db = Database(active_settings.database_url)
    delivered = 0
    try:
        claimed = await claim_due(db)
        async with httpx.AsyncClient(timeout=20) as client:
            for task in claimed:
                status = "pending"
                try:
                    # Recheck completion immediately before delivery.
                    async with db.sessions() as session:
                        current = await session.get(Task, task["id"])
                        if current.status != "open":
                            continue
                    response = await client.post(
                        "https://api.telegram.org/bot"
                        + active_settings.telegram_token.get_secret_value()
                        + "/sendMessage",
                        json={
                            "chat_id": active_settings.telegram_owner_id,
                            "text": f"Reminder: {task['title']}",
                        },
                    )
                    response.raise_for_status()
                    if not response.json().get("ok"):
                        raise ValueError("Telegram rejected reminder")
                    status = "sent"
                    delivered += 1
                except (httpx.HTTPError, ValueError):
                    pass
                async with db.sessions.begin() as session:
                    current = await session.get(Task, task["id"])
                    current.notification_status = (
                        "failed"
                        if status != "sent" and current.notification_attempts >= 5
                        else status
                    )
        return delivered
    finally:
        await db.close()


@celery_app.task(name="assistant.notify_due")
def notify_due():
    return asyncio.run(deliver_due(settings))
