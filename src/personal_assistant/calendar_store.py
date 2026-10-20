from datetime import date, timedelta
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event
from sqlalchemy import or_, select

from personal_assistant.models import CalendarEvent, now
from personal_assistant.schemas import Action
from personal_assistant.time_utils import as_utc, date_bounds, event_time, utc_timestamp


def event_dict(event: CalendarEvent) -> dict:
    zone = ZoneInfo(event.timezone)
    return {
        "id": event.id,
        "title": event.title,
        "starts_at": utc_timestamp(event.starts_at),
        "ends_at": utc_timestamp(event.ends_at),
        "local_start": as_utc(event.starts_at).astimezone(zone).isoformat(),
        "local_end": as_utc(event.ends_at).astimezone(zone).isoformat(),
        "timezone": event.timezone,
        "all_day": event.all_day,
        "status": event.status,
        "created_at": utc_timestamp(event.created_at),
        "updated_at": utc_timestamp(event.updated_at),
        "content": event.content,
    }


class CalendarStore:
    def __init__(self, db, timezone: ZoneInfo):
        self.db, self.timezone = db, timezone

    async def get(self, resource_id: str) -> dict:
        async with self.db.sessions() as session:
            event = await session.get(CalendarEvent, resource_id)
            if not event:
                raise ValueError("Calendar event not found")
            return event_dict(event)

    async def events(
        self,
        query: str = "",
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        include_cancelled: bool = False,
    ) -> list[dict]:
        start, end = date_bounds(date_from, date_to, self.timezone)
        if not (date_from or date_to) and not query.strip():
            start = now()
            end = (start.astimezone(self.timezone) + timedelta(days=7)).astimezone(start.tzinfo)
        clauses = []
        if not include_cancelled:
            clauses.append(CalendarEvent.status == "active")
        if start:
            clauses.append(CalendarEvent.ends_at > start)
        if end:
            clauses.append(CalendarEvent.starts_at < end)
        if query.strip():
            term = query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            clauses.append(
                or_(
                    CalendarEvent.title.ilike(f"%{term}%", escape="\\"),
                    CalendarEvent.content.ilike(f"%{term}%", escape="\\"),
                )
            )
        async with self.db.sessions() as session:
            events = await session.scalars(
                select(CalendarEvent).where(*clauses).order_by(CalendarEvent.starts_at).limit(100)
            )
            return [event_dict(event) for event in events]

    async def mutate(self, session, action: Action, resource_id: str, source: str) -> dict:
        if action.name == "create_event":
            if not action.title.strip() or not action.starts_at or not action.ends_at:
                raise ValueError("An event needs a title, start, and end; ask for missing details")
            event = CalendarEvent(id=resource_id, title=action.title, content=action.content)
            session.add(event)
            zone = action.event_timezone or self.timezone.key
            all_day = bool(action.all_day)
        else:
            event = await session.get(CalendarEvent, action.resource_id)
            if not event:
                raise ValueError("Calendar event not found; retrieve its exact ID first")
            if action.name == "cancel_event":
                event.status, event.updated_at, event.source = "cancelled", now(), source
                await session.flush()
                return {"action": action.name, "status": "cancelled", "event": event_dict(event)}
            if event.status == "cancelled":
                raise ValueError("This event is cancelled; create a new event to schedule it again")
            zone = action.event_timezone or event.timezone
            if zone != event.timezone and not (action.starts_at and action.ends_at):
                raise ValueError("Changing the event timezone requires both local start and end")
            all_day = event.all_day if action.all_day is None else action.all_day
            if action.title:
                event.title = action.title
            if action.content:
                event.content = action.content
        starts_at = (
            event_time(action.starts_at, zone, all_day)
            if action.starts_at
            else as_utc(event.starts_at)
        )
        ends_at = (
            event_time(action.ends_at, zone, all_day) if action.ends_at else as_utc(event.ends_at)
        )
        if ends_at <= starts_at:
            raise ValueError("The event end must be after its start")
        if all_day:
            # Also validate retained times when an update changes the all-day flag.
            for value in (starts_at, ends_at):
                event_time(value.astimezone(ZoneInfo(zone)).isoformat(), zone, all_day=True)
        event.starts_at, event.ends_at = starts_at, ends_at
        event.timezone, event.all_day = zone, all_day
        event.status, event.source, event.updated_at = "active", source, now()
        await session.flush()
        conflicts = await session.scalars(
            select(CalendarEvent)
            .where(
                CalendarEvent.id != event.id,
                CalendarEvent.status == "active",
                CalendarEvent.starts_at < ends_at,
                CalendarEvent.ends_at > starts_at,
            )
            .order_by(CalendarEvent.starts_at)
            .limit(5)
        )
        return {
            "action": action.name,
            "status": "saved",
            "event": event_dict(event),
            "conflicts": [event_dict(other) for other in conflicts],
        }

    async def export_ics(self) -> bytes:
        calendar = Calendar()
        calendar.add("prodid", "-//Personal Assistant//Private Calendar//EN")
        calendar.add("version", "2.0")
        calendar.add("calscale", "GREGORIAN")
        async with self.db.sessions() as session:
            events = await session.scalars(
                select(CalendarEvent)
                .where(CalendarEvent.status == "active")
                .order_by(CalendarEvent.starts_at)
            )
            for stored in events:
                event = Event()
                event.add("uid", f"{stored.id}@personal-assistant.local")
                event.add("summary", stored.title)
                event.add("description", stored.content)
                event.add("dtstamp", as_utc(stored.updated_at))
                event.add("created", as_utc(stored.created_at))
                event.add("last-modified", as_utc(stored.updated_at))
                start, end = as_utc(stored.starts_at), as_utc(stored.ends_at)
                if stored.all_day:
                    zone = ZoneInfo(stored.timezone)
                    start, end = start.astimezone(zone).date(), end.astimezone(zone).date()
                event.add("dtstart", start)
                event.add("dtend", end)
                calendar.add_component(event)
        return calendar.to_ical()
