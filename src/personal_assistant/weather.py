"""Structured Open-Meteo forecasts. No search excerpts, API keys, or implicit geolocation."""

import asyncio
import copy
import json
import math
import re
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from time import monotonic
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from personal_assistant.config import Settings
from personal_assistant.schemas import Action

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
SOURCE = "https://open-meteo.com/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
CURRENT = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)
HOURLY = (
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)
DAILY = (
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "weather_code",
    "wind_speed_10m_max",
)
CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    97: "Heavy thunderstorm",
    99: "Thunderstorm with heavy hail",
}


def weather_capabilities(settings: Settings) -> dict:
    return {
        "enabled": settings.weather_enabled,
        "provider": "Open-Meteo",
        "default_location": settings.weather_default_name
        if settings.weather_default_latitude is not None
        else None,
        "default_coordinates_configured": settings.weather_default_latitude is not None,
    }


def coordinates(latitude, longitude) -> tuple[float, float]:
    try:
        lat, lon = float(latitude), float(longitude)
    except (ValueError, TypeError):
        raise ValueError("Weather coordinates must be valid latitude/longitude") from None
    if (
        not math.isfinite(lat)
        or not math.isfinite(lon)
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
    ):
        raise ValueError("Weather coordinates must be valid latitude/longitude")
    return lat, lon


def rows(data: dict, group: str, variables: tuple[str, ...]) -> list[dict]:
    values = data[group]
    times = values["time"]
    if not isinstance(times, list) or not all(
        isinstance(values[key], list) and len(values[key]) == len(times) for key in variables
    ):
        raise ValueError("Invalid weather series")
    result = []
    for index, timestamp in enumerate(times):
        row = {"time": timestamp, **{key: values[key][index] for key in variables}}
        if type(timestamp) not in {int, float} or not math.isfinite(timestamp):
            raise ValueError("Invalid forecast timestamp")
        for key in variables:
            value = row[key]
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError("Invalid weather value")
        if row.get("weather_code") is not None:
            row["condition"] = CONDITIONS.get(row["weather_code"], "Unknown condition")
        result.append(row)
    return result


def units(data: dict, group: str, variables: tuple[str, ...], temperature: str) -> dict:
    result = {key: data[f"{group}_units"][key] for key in variables}
    expected = "°F" if temperature == "fahrenheit" else "°C"
    if not all(isinstance(value, str) and value for value in result.values()) or any(
        value != expected
        for key, value in result.items()
        if key.startswith("temperature_") or key == "apparent_temperature"
    ):
        raise ValueError("Open-Meteo returned missing or inconsistent units")
    return result


def bounded_weather(result: dict, budget: int) -> str:
    result = copy.deepcopy(result)
    while len(json.dumps(result, ensure_ascii=False).encode()) > budget:
        hourly = result.get("hourly")
        if isinstance(hourly, dict) and hourly.get("time"):
            for values in hourly.values():
                values.pop()
            result["hourly_omitted"] = result.get("hourly_omitted", 0) + 1
            continue
        group = next(
            (
                key
                for key in ("hourly", "daily", "candidates")
                if isinstance(result.get(key), list) and result[key]
            ),
            None,
        )
        if not group:
            return json.dumps(
                {
                    "status": "evidence_omitted",
                    "provider": "Open-Meteo",
                    "message": "Weather evidence exceeds the context budget; do not infer values.",
                }
            )
        result[group].pop()
        result[f"{group}_omitted"] = result.get(f"{group}_omitted", 0) + 1
    return json.dumps(result, ensure_ascii=False)


class Weather:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings, self.transport = settings, transport
        self.cache = OrderedDict()
        self.lock = asyncio.Lock()

    async def _json(self, url: str, params: dict, ttl: int) -> tuple[dict, datetime, bool, int]:
        key = (url, tuple(sorted(params.items())))
        async with self.lock:
            entry = self.cache.get(key)
            if entry and monotonic() - entry[0] < ttl:
                self.cache.move_to_end(key)
                return copy.deepcopy(entry[2]), entry[1], True, int(monotonic() - entry[0])
            async with httpx.AsyncClient(
                transport=self.transport, timeout=10, trust_env=False
            ) as client:
                for attempt in range(3):
                    try:
                        response = await client.get(url, params=params)
                        response.raise_for_status()
                        data = response.json()
                        if not isinstance(data, dict) or data.get("error"):
                            raise ValueError("Invalid weather response")
                        fetched = datetime.now(UTC)
                        self.cache[key] = (monotonic(), fetched, data)
                        self.cache.move_to_end(key)
                        while len(self.cache) > 128:
                            self.cache.popitem(last=False)
                        return copy.deepcopy(data), fetched, False, 0
                    except httpx.HTTPError as error:
                        retry = not isinstance(
                            error, httpx.HTTPStatusError
                        ) or error.response.status_code in {
                            429,
                            500,
                            502,
                            503,
                            504,
                        }
                        if not retry or attempt == 2:
                            raise ValueError(
                                "Open-Meteo is unavailable or rate-limiting requests. "
                                "Try later; no search snippets were substituted."
                            ) from None
                        await asyncio.sleep(0.2 * 2**attempt)
                    except ValueError:
                        raise ValueError("Open-Meteo returned an invalid response") from None
        raise ValueError("Open-Meteo request failed")

    async def _location(self, query: str) -> dict:
        if not query.strip():
            if self.settings.weather_default_latitude is None:
                return {
                    "status": "needs_location",
                    "message": "Ask which city or coordinates to use",
                }
            return {
                "name": self.settings.weather_default_name or "Configured weather location",
                "latitude": self.settings.weather_default_latitude,
                "longitude": self.settings.weather_default_longitude,
            }
        match = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\s*", query)
        if match:
            lat, lon = coordinates(*match.groups())
            return {"name": f"{lat}, {lon}", "latitude": lat, "longitude": lon}
        if len(query) > 200:
            raise ValueError("Weather query must contain only a city/place name or coordinates")
        language = "ru" if re.search(r"[А-Яа-яЁё]", query) else "en"
        data, _, _, _ = await self._json(
            GEOCODING_URL, {"name": query, "count": 5, "language": language}, 86400
        )
        candidates = []
        for item in data.get("results", []):
            lat, lon = coordinates(item["latitude"], item["longitude"])
            candidates.append(
                {
                    "name": item["name"],
                    "country": item.get("country", ""),
                    "region": item.get("admin1", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "population": item.get("population") or 0,
                }
            )
        if not candidates:
            return {
                "status": "needs_location",
                "message": "Location not found; ask for a country/region or coordinates",
            }
        # Use an overwhelmingly dominant first result; otherwise ask rather than guess.
        if len(candidates) > 1 and candidates[0]["population"] < max(
            1, 10 * max(item["population"] for item in candidates[1:])
        ):
            return {
                "status": "ambiguous_location",
                "message": "Ask the user to choose a country/region",
                "candidates": candidates,
                "geocoding_source": "https://www.geonames.org/",
            }
        return {**candidates[0], "geocoding_source": "https://www.geonames.org/"}

    async def forecast(self, action: Action, reference: datetime | None = None) -> dict:
        if not self.settings.weather_enabled:
            return {
                "status": "disabled",
                "message": "Structured weather is disabled in configuration",
            }
        if reference is not None and reference.utcoffset() is None:
            raise ValueError("Weather reference time must include a UTC offset")
        if action.weather_period == "dates" and (
            not action.date_from or not action.date_to or action.date_from >= action.date_to
        ):
            raise ValueError("Weather dates need date_from and an exclusive later date_to")
        if action.weather_period != "dates" and (action.date_from or action.date_to):
            raise ValueError("Set weather_period=dates when supplying a weather date range")
        try:
            location = await self._location(action.query)
            if "status" in location:
                return location
            data, fetched, cached, age = await self._json(
                FORECAST_URL,
                {
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "timezone": "auto",
                    "timeformat": "unixtime",
                    "forecast_days": 16,
                    "temperature_unit": action.temperature_unit,
                    "wind_speed_unit": "kmh",
                    "precipitation_unit": "mm",
                    "current": ",".join(CURRENT),
                    "hourly": ",".join(HOURLY),
                    "daily": ",".join(DAILY),
                },
                600,
            )
            zone = ZoneInfo(data["timezone"])
            today = (reference or datetime.now(UTC)).astimezone(zone).date()
            start = (
                action.date_from
                if action.weather_period == "dates"
                else today + timedelta(days=1 if action.weather_period == "tomorrow" else 0)
            )
            end = (
                action.date_to
                if action.weather_period == "dates"
                else start + timedelta(days=7 if action.weather_period == "week" else 1)
            )
            daily = rows(data, "daily", DAILY)
            for row in daily:
                # Open-Meteo daily boundaries use its fixed response offset, not a DST-aware zone.
                row["date"] = (
                    datetime.fromtimestamp(row.pop("time") + data["utc_offset_seconds"], UTC)
                    .date()
                    .isoformat()
                )
            selected = [row for row in daily if start.isoformat() <= row["date"] < end.isoformat()]
            if len(selected) != (end - start).days:
                return {
                    "status": "outside_forecast_range",
                    "message": "Requested dates are not covered; do not substitute other dates.",
                    "available_dates": [row["date"] for row in daily],
                    "location": location,
                }
            result = {
                "status": "ok",
                "provider": "Open-Meteo",
                "source": SOURCE,
                "license": LICENSE,
                "data_type": "weather model output; not an observation or guaranteed outcome",
                "retrieved_at": fetched.isoformat(),
                "cached": cached,
                "cache_age_seconds": age,
                "location": {**location, "timezone": zone.key},
                "period": {"start_date": start.isoformat(), "end_date_exclusive": end.isoformat()},
            }
            if action.weather_period == "current":
                current = data["current"]
                if any(
                    value is not None
                    and (type(value) not in {int, float} or not math.isfinite(value))
                    for value in (current.get(key) for key in CURRENT)
                ):
                    raise ValueError("Open-Meteo returned invalid current weather values")
                if not any(current.get(key) is not None for key in CURRENT):
                    raise ValueError("Current weather values missing")
                valid_time = datetime.fromtimestamp(current["time"], UTC)
                valid_age = (reference or fetched) - valid_time
                if valid_age > timedelta(hours=2) or valid_age < -timedelta(minutes=15):
                    raise ValueError(
                        "Open-Meteo current data does not match the request time; "
                        "check the system clock or try the provider later"
                    )
                result["current"] = {
                    **{key: current.get(key) for key in CURRENT},
                    "time": valid_time.astimezone(zone).isoformat(),
                    "interval_seconds": current["interval"],
                    "condition": CONDITIONS.get(current.get("weather_code"), "Unknown condition"),
                }
                result["current_units"] = units(data, "current", CURRENT, action.temperature_unit)
            else:
                result["daily"] = selected
                result["daily_units"] = units(data, "daily", DAILY, action.temperature_unit)
                result["daily_aggregation_utc_offset_seconds"] = data["utc_offset_seconds"]
            if action.hourly:
                hourly = []
                anchor = reference or fetched
                for row in rows(data, "hourly", HOURLY):
                    instant = datetime.fromtimestamp(row["time"], UTC).astimezone(zone)
                    selected_hour = (
                        anchor <= instant < anchor + timedelta(hours=24)
                        if action.weather_period == "current"
                        else start <= instant.date() < end
                    )
                    if selected_hour:
                        row["time"] = instant.isoformat()
                        hourly.append(row)
                columns = ("time", *HOURLY, "condition")
                result["hourly"] = {key: [row.get(key) for row in hourly[:48]] for key in columns}
                result["hourly_omitted"] = max(0, len(hourly) - 48)
                result["hourly_units"] = units(data, "hourly", HOURLY, action.temperature_unit)
            if any(
                datetime.combine(start + timedelta(days=offset), datetime.min.time(), zone)
                .utcoffset()
                .total_seconds()
                != data["utc_offset_seconds"]
                for offset in range((end - start).days + 1)
            ):
                result["time_note"] = (
                    "Daily aggregates use the provider's fixed UTC offset across DST; "
                    "hourly timestamps use the IANA timezone and actual DST offset."
                )
            return result
        except (KeyError, TypeError, OverflowError, ZoneInfoNotFoundError):
            raise ValueError("Open-Meteo returned incomplete or invalid forecast data") from None
