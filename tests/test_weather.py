import copy
import json
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import ValidationError

from personal_assistant.agent.graph import PLANNER, AssistantGraph, Draft
from personal_assistant.core.config import Settings
from personal_assistant.core.schemas import Action, ChatRequest, Plan, Review
from personal_assistant.core.time_utils import with_runtime_clock
from personal_assistant.inference.llm import fit_payload
from personal_assistant.integrations import weather as module
from personal_assistant.integrations.weather import CURRENT, DAILY, HOURLY, LICENSE, Weather
from tests.conftest import ScriptedModel

REFERENCE = datetime(2026, 10, 17, 10, tzinfo=UTC)


def settings(**updates):
    return Settings(
        _env_file=None,
        weather_default_name="Belgrade, Serbia",
        weather_default_latitude=44.804,
        weather_default_longitude=20.4651,
        search_provider="disabled",
        **updates,
    )


def forecast_data(day=date(2026, 10, 17), zone="Europe/Belgrade", temperature="°C"):
    start = datetime.combine(day, datetime.min.time(), ZoneInfo(zone))
    unit = {
        key: (
            temperature
            if key.startswith("temperature") or key == "apparent_temperature"
            else "WMO code"
            if key == "weather_code"
            else "%"
            if "probability" in key or "humidity" in key
            else "km/h"
            if "wind" in key
            else "mm"
        )
        for key in set(CURRENT + DAILY + HOURLY)
    }
    value = {
        key: 3
        if key == "weather_code"
        else 12
        if key == "temperature_2m_min"
        else 18
        if key == "temperature_2m_max"
        else 0
        if "precipitation" in key
        else 15
        for key in unit
    }
    return {
        "timezone": zone,
        "utc_offset_seconds": int(start.utcoffset().total_seconds()),
        "current": {
            "time": int(start.timestamp()) + 36000,
            "interval": 900,
            **{key: value[key] for key in CURRENT},
        },
        "current_units": {key: unit[key] for key in CURRENT},
        "daily": {
            "time": [int(start.timestamp()) + 86400 * i for i in range(16)],
            **{key: [value[key]] * 16 for key in DAILY},
        },
        "daily_units": {key: unit[key] for key in DAILY},
        "hourly": {
            "time": [int(start.timestamp()) + 3600 * i for i in range(16 * 24)],
            **{key: [value[key]] * (16 * 24) for key in HOURLY},
        },
        "hourly_units": {key: unit[key] for key in HOURLY},
    }


async def test_default_location_forecast_units_and_cache_without_search():
    calls = []

    async def handle(request):
        calls.append(request)
        assert request.url.host == "api.open-meteo.com"
        assert request.url.params["latitude"] == "44.804"
        assert request.url.params["longitude"] == "20.4651"
        assert request.url.params["timeformat"] == "unixtime"
        return httpx.Response(200, json=forecast_data())

    weather = Weather(settings(), httpx.MockTransport(handle))
    current = await weather.forecast(Action(name="get_weather"), REFERENCE)
    tomorrow = await weather.forecast(
        Action(name="get_weather", weather_period="tomorrow", hourly=True), REFERENCE
    )
    assert len(calls) == 1
    assert current["current"]["temperature_2m"] == 15
    assert current["current"]["precipitation"] == 0
    assert current["current_units"]["temperature_2m"] == "°C"
    assert current["current"]["condition"] == "Overcast"
    assert "current" not in tomorrow
    assert tomorrow["daily"][0]["date"] == "2026-10-18"
    assert tomorrow["daily"][0]["temperature_2m_min"] == 12
    assert tomorrow["daily"][0]["temperature_2m_max"] == 18
    assert len(tomorrow["hourly"]["time"]) == 24
    assert tomorrow["cached"] and not current["cached"]
    assert tomorrow["retrieved_at"] == current["retrieved_at"]


async def test_expired_cache_retries_without_serving_stale_forecast(monkeypatch):
    tick, failed, calls = [0], [False], []
    monkeypatch.setattr(module, "monotonic", lambda: tick[0])

    async def handle(request):
        calls.append(request)
        return httpx.Response(503 if failed[0] else 200, json=forecast_data())

    weather = Weather(settings(), httpx.MockTransport(handle))
    await weather.forecast(Action(name="get_weather"), REFERENCE)
    tick[0], failed[0] = 601, True
    with pytest.raises(ValueError, match="unavailable"):
        await weather.forecast(Action(name="get_weather"), REFERENCE)
    assert len(calls) == 4


async def test_explicit_city_overrides_default_and_uses_location_timezone():
    calls = []

    async def handle(request):
        calls.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "Лондон, Великобритания"
            assert request.url.params["language"] == "ru"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Лондон",
                            "latitude": 51.5,
                            "longitude": -0.1,
                            "country": "Великобритания",
                            "population": 9000000,
                        }
                    ]
                },
            )
        assert request.url.params["latitude"] == "51.5"
        return httpx.Response(200, json=forecast_data(zone="Europe/London"))

    result = await Weather(settings(), httpx.MockTransport(handle)).forecast(
        Action(name="get_weather", query="Лондон, Великобритания", weather_period="tomorrow"),
        REFERENCE,
    )
    assert len(calls) == 2 and result["location"]["name"] == "Лондон"
    assert result["location"]["timezone"] == "Europe/London"


async def test_ambiguous_place_returns_candidates_without_fetching_forecast():
    calls = []

    async def handle(request):
        calls.append(request)
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Springfield",
                        "latitude": 39.8,
                        "longitude": -89.6,
                        "population": 114000,
                    },
                    {
                        "name": "Springfield",
                        "latitude": 42.1,
                        "longitude": -72.6,
                        "population": 155000,
                    },
                ]
            },
        )

    result = await Weather(settings(), httpx.MockTransport(handle)).forecast(
        Action(name="get_weather", query="Springfield"), REFERENCE
    )
    assert result["status"] == "ambiguous_location"
    assert len(result["candidates"]) == 2 and len(calls) == 1


async def test_no_location_or_disabled_weather_never_contacts_provider():
    async def forbidden(request):
        pytest.fail("No provider request expected")

    for config, status in [
        (Settings(_env_file=None), "needs_location"),
        (settings(weather_enabled=False), "disabled"),
    ]:
        result = await Weather(config, httpx.MockTransport(forbidden)).forecast(
            Action(name="get_weather")
        )
        assert result["status"] == status


async def test_explicit_fahrenheit_and_next_24_hour_window():
    async def handle(request):
        assert request.url.params["temperature_unit"] == "fahrenheit"
        return httpx.Response(200, json=forecast_data(temperature="°F"))

    result = await Weather(settings(), httpx.MockTransport(handle)).forecast(
        Action(name="get_weather", temperature_unit="fahrenheit", hourly=True), REFERENCE
    )
    assert result["current_units"]["temperature_2m"] == "°F"
    times = [datetime.fromisoformat(value) for value in result["hourly"]["time"]]
    assert len(times) == 24 and all(
        REFERENCE <= value < REFERENCE + timedelta(hours=24) for value in times
    )


async def test_invalid_coordinates_and_dates_fail_before_provider_access():
    async def forbidden(request):
        pytest.fail("Invalid weather inputs must not contact a provider")

    weather = Weather(settings(), httpx.MockTransport(forbidden))
    for action in [
        Action(name="get_weather", query="91,0"),
        Action(name="get_weather", weather_period="dates"),
        Action(name="get_weather", date_from=date(2026, 10, 17)),
    ]:
        with pytest.raises(ValueError):
            await weather.forecast(action, REFERENCE)


@pytest.mark.parametrize("code,expected_calls", [(400, 1), (429, 3)])
async def test_http_errors_are_sanitized_and_only_transient_errors_retry(code, expected_calls):
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(code, json={"error": True, "reason": "do-not-expose-upstream-body"})

    with pytest.raises(ValueError) as caught:
        await Weather(settings(), httpx.MockTransport(handle)).forecast(Action(name="get_weather"))
    assert "do-not-expose" not in str(caught.value)
    assert len(calls) == expected_calls


@pytest.mark.parametrize("damage", ["length", "units", "null", "malformed"])
async def test_missing_values_and_malformed_provider_data(damage):
    data = forecast_data()
    if damage == "length":
        data["daily"]["temperature_2m_max"].pop()
    elif damage == "units":
        data["current_units"]["temperature_2m"] = "°F"
    elif damage == "null":
        data["current"]["temperature_2m"] = None
    else:
        data["current"]["temperature_2m"] = "not-a-number"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    service = Weather(settings(), transport)
    if damage == "null":
        assert (await service.forecast(Action(name="get_weather"), REFERENCE))["current"][
            "temperature_2m"
        ] is None
    else:
        with pytest.raises(ValueError):
            await service.forecast(Action(name="get_weather"), REFERENCE)


async def test_fresh_fetch_does_not_make_stale_current_conditions_valid():
    data = forecast_data()
    data["current"]["time"] = int((REFERENCE - timedelta(days=1)).timestamp())
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    with pytest.raises(ValueError, match="does not match the request time"):
        await Weather(settings(), transport).forecast(Action(name="get_weather"), REFERENCE)


async def test_hourly_dst_offsets_and_daily_provider_offset_are_explicit():
    data = forecast_data(day=date(2026, 10, 25))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    result = await Weather(settings(), transport).forecast(
        Action(name="get_weather", weather_period="today", hourly=True),
        datetime(2026, 10, 25, 0, 15, tzinfo=UTC),
    )
    timestamps = result["hourly"]["time"]
    assert "2026-10-25T02:00:00+02:00" in timestamps
    assert "2026-10-25T02:00:00+01:00" in timestamps
    assert result["hourly_omitted"] == 0 and len(timestamps) == 25
    assert "fixed UTC offset" in result["time_note"]


async def test_relative_dates_use_forecast_location_not_user_timezone():
    data = forecast_data(day=date(2026, 10, 16), zone="America/New_York")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    result = await Weather(settings(), transport).forecast(
        Action(name="get_weather", weather_period="tomorrow"),
        datetime(2026, 10, 17, 0, 15, tzinfo=UTC),
    )
    assert result["daily"][0]["date"] == "2026-10-17"


async def test_outside_forecast_range_does_not_return_other_dates():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=forecast_data()))
    result = await Weather(settings(), transport).forecast(
        Action(
            name="get_weather",
            weather_period="dates",
            date_from=date(2026, 12, 1),
            date_to=date(2026, 12, 2),
        ),
        REFERENCE,
    )
    assert result["status"] == "outside_forecast_range" and "daily" not in result


def test_default_coordinates_are_optional_paired_and_validated():
    assert (
        Settings(
            _env_file=None, weather_default_latitude="", weather_default_longitude=""
        ).weather_default_latitude
        is None
    )
    for values in [
        {"weather_default_latitude": 45},
        {"weather_default_latitude": 91, "weather_default_longitude": 20},
    ]:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **values)


async def test_weather_enters_review_preserves_evidence_and_appends_attribution(harness):
    weather = Weather(
        settings(), httpx.MockTransport(lambda request: httpx.Response(200, json=forecast_data()))
    )
    model = ScriptedModel(
        Plan(
            specialist="research",
            actions=[Action(name="get_weather", weather_period="today", hourly=True)],
        ),
        Draft(answer="The forecast maximum is 18 °C."),
        Review(decision="accept"),
    )
    # Match the fixture forecast dates while using the graph's normal captured clock.
    from unittest.mock import patch

    import personal_assistant.agent.graph as graph_module

    with patch.object(graph_module, "clock_context", wraps=graph_module.clock_context) as clock:
        reference_clock = graph_module.clock_context("Europe/Belgrade", REFERENCE)
        clock.side_effect = lambda *args, **kwargs: copy.deepcopy(reference_clock)
        response = await harness.runtime(model, weather=weather).service.chat(
            ChatRequest(text="Show today's hourly weather in Belgrade.")
        )
    assert response.review_status == "accepted"
    assert "open-meteo.com" in response.text and LICENSE in response.text
    for system, payload, schema in model.calls:
        fitted = fit_payload(
            with_runtime_clock(system, payload),
            payload,
            schema.model_json_schema(),
            harness.settings.input_budget - 150,
        )
        if system == PLANNER:
            assert fitted["web_capabilities"]["weather"]["enabled"]
            continue
        result = json.loads(fitted.get("operation_results", fitted.get("evidence"))[0]["data"])
        assert result["daily"][0]["temperature_2m_max"] == 18
        assert result["daily_units"]["temperature_2m_max"] == "°C"
        assert result["location"]["timezone"] == "Europe/Belgrade"


def test_weather_context_trimming_preserves_valid_json_and_marks_omissions():
    result = {
        "status": "ok",
        "provider": "Open-Meteo",
        "source": module.SOURCE,
        "hourly": {
            "time": ["2026-10-17T10:00:00+02:00"] * 200,
            "temperature_2m": [15] * 200,
            "condition": ["Overcast"] * 200,
        },
        "daily": [{"date": "2026-10-17", "temperature_2m_max": 18}],
    }
    bounded = AssistantGraph.bounded_results(
        [{"action": "get_weather", "ok": True, "result": result}]
    )
    parsed = json.loads(bounded[0]["data"])
    assert len(bounded[0]["data"].encode()) <= 6000
    assert parsed["daily"] == result["daily"]
    assert parsed["hourly_omitted"] > 0
    assert len({len(values) for values in parsed["hourly"].values()}) == 1
    assert len(result["hourly"]["time"]) == 200
