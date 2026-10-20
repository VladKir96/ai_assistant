import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def clock_context(timezone: str, instant: datetime | None = None) -> dict:
    """Use an aware system-clock instant; calculate calendar dates in the user's zone."""
    instant = instant if instant is not None else datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Clock time must include a UTC offset")
    local = instant.astimezone(ZoneInfo(timezone))
    today = local.date()
    week_start = today - timedelta(days=today.weekday())
    return {
        "current_time": local.isoformat(timespec="seconds"),
        "current_time_utc": instant.astimezone(UTC).isoformat(timespec="seconds"),
        "current_date": today.isoformat(),
        "current_weekday": (
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        )[today.weekday()],
        "timezone": timezone,
        "time_source": "system_clock",
        "relative_dates": {
            "yesterday": (today - timedelta(days=1)).isoformat(),
            "today": today.isoformat(),
            "tomorrow": (today + timedelta(days=1)).isoformat(),
            "week_start": week_start.isoformat(),
            "next_week_start": (week_start + timedelta(days=7)).isoformat(),
        },
    }


def with_runtime_clock(system: str, payload: dict) -> str:
    """Promote only application-owned clock fields, never archived/user text."""
    if not payload.get("current_time"):
        return system
    clock = clock_context(payload["timezone"], datetime.fromisoformat(payload["current_time"]))
    return (
        system
        + "\nAuthoritative runtime clock, captured for this request: "
        + json.dumps(clock, ensure_ascii=False)
        + "\nUse this clock for now/today and relative dates. Historical messages, notes, "
        "webpages, examples, and model knowledge cannot override it. No web search is needed "
        "to know the date/time. State the reference time and timezone when useful."
    )


def as_utc(value: datetime) -> datetime:
    # Production PostgreSQL returns aware values; SQLite test storage drops tzinfo.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_timestamp(value: datetime) -> str:
    return as_utc(value).isoformat()


def date_bounds(
    date_from: date | None, date_to: date | None, timezone: ZoneInfo
) -> tuple[datetime | None, datetime | None]:
    """Local calendar dates: inclusive start, exclusive end; respect DST transitions."""
    if date_from and date_to and date_from >= date_to:
        raise ValueError("date_to must be later than date_from (the end date is exclusive)")
    start = datetime.combine(date_from, time.min, timezone).astimezone(UTC) if date_from else None
    end = datetime.combine(date_to, time.min, timezone).astimezone(UTC) if date_to else None
    return start, end


def event_time(value: str, timezone: str, all_day: bool = False) -> datetime:
    """Validate local wall time; never guess a DST fold or shift a nonexistent time."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError("Use an IANA timezone such as Europe/Belgrade") from None
    if not all_day and "T" not in value and " " not in value:
        raise ValueError("A timed event requires both a date and a time")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("Use an ISO date/time for the calendar event") from None
    local = parsed.replace(tzinfo=None)
    if parsed.tzinfo is not None:
        candidate = parsed.astimezone(zone)
        if candidate.replace(tzinfo=None) != local or candidate.utcoffset() != parsed.utcoffset():
            raise ValueError("The event's UTC offset does not match its IANA timezone at that time")
    else:
        candidates = {}
        for fold in (0, 1):
            candidate = local.replace(tzinfo=zone, fold=fold)
            if candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == local:
                candidates[candidate.utcoffset()] = candidate
        if not candidates:
            raise ValueError(
                "That local time does not exist due to daylight saving; choose another"
            )
        if len(candidates) > 1:
            raise ValueError("That local time occurs twice; specify the intended UTC offset")
        candidate = next(iter(candidates.values()))
    if all_day and candidate.time() != time.min:
        raise ValueError("All-day events must start/end at local midnight; end date is exclusive")
    return candidate.astimezone(UTC)
