from __future__ import annotations

import asyncio
import copy
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import main
from app.api import dependencies, errors
from app.api.routers import chat, form, geocode
from app.services import budget_assessment as budget_service
from app.services.budget_assessment import (
    BudgetAssessment,
    BudgetAssessmentResult,
    BudgetAssessmentUnavailable,
)
from app.schemas.responses import validated_accepted_plan
from app.tools.currency import CurrencyRateUnavailableError


client = TestClient(main.create_app())
AUTH = {"X-User-ID": "test-user"}

_APPROVED_PLANNING_LOG_FIELDS = {
    "request_id",
    "session_id",
    "attempt",
    "stage",
    "destination_country_code",
    "issue_codes",
    "elapsed_ms",
    "outcome",
}
_STANDARD_LOG_RECORD_FIELDS = set(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


def custom_log_record_fields(record: logging.LogRecord) -> set[str]:
    return set(record.__dict__) - _STANDARD_LOG_RECORD_FIELDS


class _RaisingHandler(logging.Handler):
    def emit(self, _record):
        raise RuntimeError("raising log handler")


@contextmanager
def raising_log_handler(target_logger: logging.Logger):
    handler = _RaisingHandler()
    original_level = target_logger.level
    original_propagate = target_logger.propagate
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.INFO)
    target_logger.propagate = False
    try:
        yield
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(original_level)
        target_logger.propagate = original_propagate


def valid_form():
    return {
        "country": "Japan",
        "city": ["Tokyo"],
        "num_people": 2,
        "total_budget": 5000,
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
    }


def qualified_plan_state(
    *,
    revision: int = 1,
    total_base_budget: float = 5000,
    total_convert_budget: float = 160000,
    country: str = "Japan",
    country_code: str = "JP",
    destination_currency: str = "JPY",
    cities: list[str] | None = None,
    num_people: int = 2,
    start_date: str = "2026-08-01",
    end_date: str = "2026-08-05",
    reply: str = "Your accepted itinerary is ready.",
):
    """Build a complete accepted snapshot that passes the real validator."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    configured_cities = cities or ["Tokyo"]
    total_days = (end - start).days + 1
    hotel = {
        "hotel_name": "Verified Destination Hotel",
        "hotel_class": 0.0,
        "overall_rating": None,
        "reviews": None,
        "description": "",
        "price_per_night": 120.0,
        "amenities": [],
        "check_in_time": "",
        "check_out_time": "",
        "location": {
            "lat": 35.68,
            "lng": 139.76,
            "country_code": country_code,
        },
        "image": "",
        "booking_url": "",
        "over_budget": False,
    }

    def flight(
        departure_id: str,
        departure_name: str,
        arrival_id: str,
        arrival_name: str,
        travel_date: str,
    ):
        return {
            "airline": "Grounded Air",
            "flight_number": f"GA-{departure_id}-{arrival_id}",
            "airline_logo": "",
            "travel_class": "",
            "airplane": "",
            "departure_airport": {
                "id": departure_id,
                "name": departure_name,
                "time": f"{travel_date} 08:00",
            },
            "arrival_airport": {
                "id": arrival_id,
                "name": arrival_name,
                "time": f"{travel_date} 14:00",
            },
            "departure_time": f"{travel_date} 08:00",
            "arrival_time": f"{travel_date} 14:00",
            "duration": 360,
            "stops": 0,
            "layovers": [],
            "price": 500.0,
            "booking_url": None,
            "over_budget": False,
        }

    outbound = flight(
        "KUL",
        "Kuala Lumpur International Airport",
        "DST",
        "Verified Destination Airport",
        start_date,
    )
    returning = flight(
        "DST",
        "Verified Destination Airport",
        "KUL",
        "Kuala Lumpur International Airport",
        end_date,
    )
    itinerary = []
    maps = {}
    current = start
    day = 1
    while current <= end:
        activity_name = f"Verified Museum {day}"
        requested_city = configured_cities[(day - 1) % len(configured_cities)]
        activity = {
            "name": activity_name,
            "type": "attraction",
            "description": "A verified destination attraction.",
            "category": "museum",
            "rating": 4.5,
            "address": f"{day} Museum Road",
            "thumbnail": "https://images.test/museum.jpg",
            "suggested_time": "10:00",
            "estimated_cost": 20,
            "is_estimated": True,
            "order": 1,
            "location": {
                "place_name": activity_name,
                "country_code": country_code,
                "latitude": 35.0 + day / 100,
                "longitude": 139.0 + day / 100,
                "requested_city": requested_city,
                "verified_locality": requested_city,
            },
        }
        is_first = day == 1
        is_last = day == total_days
        flights = (
            [outbound, returning]
            if is_first and is_last
            else [outbound]
            if is_first
            else [returning]
            if is_last
            else None
        )
        day_hotel = hotel if not is_last else None
        day_total = 20.0
        if is_first:
            day_total += 500.0
        if is_last:
            day_total += 500.0
        if not is_last:
            day_total += 120.0
        itinerary.append(
            {
                "day": day,
                "date": current.strftime("%Y-%m-%d"),
                "flight": flights,
                "hotel": day_hotel,
                "activities": [activity],
                "route": None,
                "day_total_cost": day_total,
            }
        )
        maps[day] = {
            "type": "FeatureCollection",
            "features": [
                *(
                    [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    hotel["location"]["lng"],
                                    hotel["location"]["lat"],
                                ],
                            },
                            "properties": {
                                "name": hotel["hotel_name"],
                                "type": "hotel",
                                "order": 0,
                            },
                        }
                    ]
                    if day_hotel is not None
                    else []
                ),
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            activity["location"]["longitude"],
                            activity["location"]["latitude"],
                        ],
                    },
                    "properties": {
                        "name": activity_name,
                        "type": "attraction",
                        "order": 1,
                    },
                }
            ],
        }
        current += timedelta(days=1)
        day += 1

    allocation = {
        "transportation": total_convert_budget * 0.25,
        "accommodation": total_convert_budget * 0.35,
        "food": total_convert_budget * 0.15,
        "activity": total_convert_budget * 0.15,
        "shopping": total_convert_budget * 0.05,
        "emergency_fund": total_convert_budget * 0.05,
    }
    snapshot = {
        "plan_revision": revision,
        "total_base_budget": total_base_budget,
        "base_currency_code": "MYR",
        "dest_currency_code": destination_currency,
        "total_convert_budget": total_convert_budget,
        "budget_allocation": copy.deepcopy(allocation),
        "draft_itinerary": copy.deepcopy(itinerary),
        "daily_map_info": copy.deepcopy(maps),
    }
    return {
        "origin_country": "Malaysia",
        "country": country,
        "city": configured_cities,
        "start_date": start_date,
        "end_date": end_date,
        "num_people": num_people,
        "base_currency_code": "MYR",
        "dest_currency_code": destination_currency,
        "total_base_budget": total_base_budget,
        "total_convert_budget": total_convert_budget,
        "budget_allocation": allocation,
        "draft_itinerary": itinerary,
        "daily_map_info": maps,
        "plan_revision": revision,
        "accepted_plan_snapshot": snapshot,
        "messages": [
            HumanMessage(content="Plan my trip"),
            AIMessage(
                content=reply,
                additional_kwargs={"plan_revision": revision},
            ),
        ],
    }


def test_accepted_projection_rejects_unexpected_map_day_before_public_serialization():
    """Catch a canonical but out-of-trip map key surviving API revalidation."""
    state = qualified_plan_state()
    snapshot = state["accepted_plan_snapshot"]
    snapshot["daily_map_info"][99] = copy.deepcopy(snapshot["daily_map_info"][1])

    assert validated_accepted_plan(state) is None


def test_accepted_projection_rejects_route_metric_disagreement_with_map():
    """Catch backend acceptance of a snapshot the frontend route guard rejects."""
    state = qualified_plan_state()
    snapshot = state["accepted_plan_snapshot"]
    day = snapshot["draft_itinerary"][0]
    daily_map = snapshot["daily_map_info"][1]
    point_coordinates = [
        feature["geometry"]["coordinates"]
        for feature in daily_map["features"]
        if feature["geometry"]["type"] == "Point"
    ]
    day["route"] = {
        "ordered_stops": [
            feature["properties"]["name"]
            for feature in daily_map["features"]
            if feature["geometry"]["type"] == "Point"
        ],
        "profiles": {
            "driving": {"distance_km": 10.0, "duration_mins": 15.0},
        },
    }
    daily_map["features"].append(
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [point_coordinates[0], point_coordinates[-1]],
            },
            "properties": {
                "type": "route",
                "profile": "driving",
                "distance_km": 999.0,
                "duration_mins": 15.0,
            },
        }
    )

    assert validated_accepted_plan(state) is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda feature: feature["properties"].update({"type": "hotel"}),
        lambda feature: feature["properties"].update({"order": 99}),
        lambda feature: feature["properties"].update({"profile": "walking"}),
    ],
    ids=["point-type", "point-order", "point-route-property"],
)
def test_accepted_projection_rejects_geojson_point_semantic_mismatch(mutation):
    state = qualified_plan_state()
    feature = state["accepted_plan_snapshot"]["daily_map_info"][1]["features"][-1]
    mutation(feature)

    assert validated_accepted_plan(state) is None


def test_accepted_projection_rejects_unsupported_polygon_feature():
    state = qualified_plan_state()
    state["accepted_plan_snapshot"]["daily_map_info"][1]["features"].append(
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {"name": "Injected", "type": "attraction", "order": 2},
        }
    )

    assert validated_accepted_plan(state) is None


def pending_chat_state(expires_at="2999-08-16T13:00:00+00:00"):
    """Return a valid pending assessment state without exposing private evidence."""
    assessment = {
        "assessment_id": "assessment-1",
        "calculation_version": "allocation-v1",
        "origin": "Malaysia",
        "destination": "China",
        "destination_city": "Shanghai",
        "start_date": "2026-08-17",
        "end_date": "2026-08-20",
        "num_people": 1,
        "base_currency": "MYR",
        "destination_currency": "CNY",
        "exchange_rate": 2.0,
        "minimum_destination_budget": 7000.0,
        "recommended_minimum_budget": 3500.0,
        "evidence": {
            "outbound_flight": {"price": 900},
            "return_flight": {"price": 850},
            "hotel": {"price_per_night": 400},
            "outbound_flight_price": 900,
            "return_flight_price": 850,
            "hotel_price_per_night": 400,
            "hotel_nights": 3,
        },
        "created_at": "2026-08-16T12:00:00+00:00",
        "expires_at": expires_at,
    }
    return {
        **{
            key: assessment[key]
            for key in ("origin", "destination", "start_date", "end_date", "num_people")
        },
        "country": assessment["destination"],
        "city": [assessment["destination_city"]],
        "origin_country": assessment["origin"],
        "messages": [
            HumanMessage(content="Use RM 500"),
            AIMessage(content="Confirm RM 3,500 first."),
        ],
        "draft_itinerary": [{"day": 1, "hotel": {"hotel_name": "Accepted"}}],
        "pending_budget_confirmation": {
            "budget_assessment_id": "assessment-1",
            "reason": "insufficient_budget",
            "stated_budget": 500,
            "recommended_minimum_budget": 3500,
            "base_currency": "MYR",
            "destination_currency": "CNY",
            "expires_at": expires_at,
            "evidence": {
                "outbound_flight_price": 900,
                "return_flight_price": 850,
                "hotel_price_per_night": 400,
                "hotel_nights": 3,
            },
            "assessment": assessment,
        },
        "budget_gate_outcome": "budget_confirmation_required",
        "budget_gate_message": "Confirm RM 3,500 first.",
    }


@pytest.fixture(autouse=True)
def _authoritative_public_confirmation_cache():
    """Ordinary projection fixtures represent the exact persisted assessment."""
    canonical = BudgetAssessment.model_validate(pending_chat_state()["pending_budget_confirmation"]["assessment"])

    def load_cached(*, assessment_id: str, **trip):
        if assessment_id != canonical.assessment_id:
            return None
        expected = {
            "origin": canonical.origin,
            "destination": canonical.destination,
            "destination_city": canonical.destination_city,
            "start_date": canonical.start_date,
            "end_date": canonical.end_date,
            "num_people": canonical.num_people,
        }
        return canonical if trip == expected else None

    with patch.object(chat, "load_confirmed_budget_assessment", side_effect=load_cached):
        yield


@pytest.mark.api
class TestApplicationSurface:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/", {"message": "Travel Assistant AI is running"}),
            ("/health", {"status": "ok"}),
            ("/healthz", {"status": "ok"}),
        ],
    )
    def test_public_health_endpoints(self, path, expected):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == expected
        assert response.headers["X-Process-Time"].endswith("ms")

    def test_openapi_contains_all_business_routes(self):
        schema = client.get("/openapi.json").json()
        assert {
            "/api/profile/",
            "/api/form/submit",
            "/api/chat/message",
            "/api/chat/history",
            "/api/geocode/",
        }.issubset(schema["paths"])

    @pytest.mark.parametrize(
        "path,method",
        [
            ("/api/profile/", "get"),
            ("/api/profile/", "post"),
            ("/api/form/submit", "post"),
            ("/api/chat/message", "post"),
            ("/api/chat/history", "get"),
            ("/api/geocode/?q=Tokyo", "get"),
        ],
    )
    def test_protected_routes_reject_missing_user_header(self, path, method):
        response = client.post(path, json={}) if method == "post" else client.get(path)
        assert response.status_code == 401
        assert "Missing X-User-ID" in response.json()["detail"]

    @pytest.mark.parametrize("value", [" ", "x" * 129])
    def test_invalid_user_identifier_is_rejected(self, value):
        response = client.get("/api/profile/", headers={"X-User-ID": value})
        assert response.status_code == 401

    @pytest.mark.parametrize("header", ["X-User-ID", "X-UID", "X-Auth-User-ID"])
    def test_supported_user_header_aliases(self, header):
        with patch("app.core.supabase_db.fetch_user_profile", return_value={}):
            response = client.get("/api/profile/", headers={header: " user-1 "})
        assert response.status_code == 200

    def test_cors_preflight_accepts_dynamic_local_vite_port(self):
        response = client.options(
            "/api/profile/",
            headers={
                "Origin": "http://localhost:5179",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-user-id",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:5179"
        )

    def test_unknown_route_is_404(self):
        assert client.get("/does-not-exist").status_code == 404


@pytest.mark.api
class TestProfileRoutes:
    def test_create_profile_maps_public_fields_to_storage_fields(self):
        with patch(
            "app.core.supabase_db.create_user_profile", return_value=True
        ) as create:
            response = client.post(
                "/api/profile/",
                headers=AUTH,
                json={
                    "name": "Alex",
                    "origin_country": "Malaysia",
                    "origin_state": "Selangor",
                },
            )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "user_id": "test-user"}
        create.assert_called_once_with(
            "test-user",
            {"name": "Alex", "home_country": "Malaysia", "home_state": "Selangor"},
        )

    def test_create_profile_reports_storage_failure(self):
        with patch("app.core.supabase_db.create_user_profile", return_value=False):
            response = client.post(
                "/api/profile/",
                headers=AUTH,
                json={
                    "name": "Alex",
                    "origin_country": "Malaysia",
                    "origin_state": "Selangor",
                },
            )
        assert response.status_code == 503

    @pytest.mark.parametrize(
        "profile,onboarded",
        [
            ({}, False),
            (
                {"name": "Alex", "home_country": "Malaysia", "home_state": "Selangor"},
                True,
            ),
            ({"name": "Alex", "home_country": "Malaysia"}, False),
        ],
    )
    def test_get_profile_onboarding_gate(self, profile, onboarded):
        with patch("app.core.supabase_db.fetch_user_profile", return_value=profile):
            response = client.get("/api/profile/", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["onboarded"] is onboarded
        assert response.json()["profile"] == profile
        assert response.headers["cache-control"] == "no-store, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert "X-User-ID" in response.headers["vary"]

    @pytest.mark.parametrize("missing", ["name", "origin_country", "origin_state"])
    def test_profile_validation_reports_missing_fields(self, missing):
        payload = {
            "name": "Alex",
            "origin_country": "Malaysia",
            "origin_state": "Selangor",
        }
        payload.pop(missing)
        assert (
            client.post("/api/profile/", headers=AUTH, json=payload).status_code == 422
        )


@pytest.mark.api
class TestFormRoute:
    def assessment(self, *, minimum=160000.0, recommended=5000.0):
        return BudgetAssessment.model_validate(
            {
                "assessment_id": "assessment-1",
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Japan",
                "destination_city": "Tokyo",
                "start_date": "2026-08-01",
                "end_date": "2026-08-05",
                "num_people": 2,
                "base_currency": "MYR",
                "destination_currency": "JPY",
                "exchange_rate": 32.0,
                "minimum_destination_budget": minimum,
                "recommended_minimum_budget": recommended,
                "evidence": {
                    "outbound_flight": {"price": 32000.0},
                    "return_flight": {"price": 8000.0},
                    "hotel": {"price_per_night": 10000.0},
                    "outbound_flight_price": 32000.0,
                    "return_flight_price": 8000.0,
                    "hotel_price_per_night": 10000.0,
                    "hotel_nights": 4,
                },
                "created_at": "2026-08-16T12:00:00+00:00",
                "expires_at": "2999-08-16T13:00:00+00:00",
            }
        )

    def final_state(self, *, total_base_budget=5000):
        return qualified_plan_state(total_base_budget=total_base_budget)

    def test_success_builds_state_from_profile_and_returns_session(self, caplog):
        final_state = self.final_state()
        final_state["destination_country_code"] = "ZZ"
        invoke = AsyncMock(return_value=final_state)
        assessment = self.assessment()
        caplog.set_level(logging.INFO, logger=form.logger.name)
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={
                    "name": "Alex",
                    "home_country": "Malaysia",
                    "home_state": "Selangor",
                },
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post(
                "/api/form/submit",
                headers={**AUTH, "X-Request-ID": "form-request-1"},
                json=valid_form(),
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "success"
        assert body["currency"] == "JPY"
        assert body["total_budget"] == 160000
        assert body["session_id"]
        assert body["destination_country_code"] == "JP"
        assert body["chat_reply"] == "Your accepted itinerary is ready."
        assert "flight" not in body["budget_allocation"]
        call = invoke.call_args.kwargs
        assert call["user_id"] == "test-user"
        assert call["thread_id"] == body["session_id"]
        assert call["request_id"] == "form-request-1"
        initial = call["initial_state"]
        assert initial["origin_country"] == "Malaysia"
        assert initial["total_base_budget"] == 5000.0
        assert initial["city"] == ["Tokyo"]
        assert initial["budget_assessment"]["assessment_id"] == "assessment-1"
        records = [
            record
            for record in caplog.records
            if record.name == form.logger.name
            and record.getMessage() in {"form.submit.start", "form.submit.done"}
        ]
        assert [record.getMessage() for record in records] == [
            "form.submit.start",
            "form.submit.done",
        ]
        assert all(
            custom_log_record_fields(record) == _APPROVED_PLANNING_LOG_FIELDS
            for record in records
        )

    @pytest.mark.parametrize(
        "stated_budget,expected_status",
        [(200, "success"), (199, "budget_confirmation_required")],
    )
    def test_same_day_form_uses_flight_only_budget_gate(
        self, stated_budget, expected_status
    ):
        payload = {
            **valid_form(),
            "total_budget": stated_budget,
            "end_date": "2026-08-01",
        }
        final_state = qualified_plan_state(
            total_base_budget=200,
            total_convert_budget=200,
            end_date="2026-08-01",
        )
        invoke = AsyncMock(return_value=final_state)
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(budget_service, "get_cached_data", return_value=None),
            patch.object(
                budget_service,
                "claim_cached_data",
                return_value=("claimed", None, "day-trip-claim"),
            ),
            patch.object(budget_service, "complete_cached_data_claim", return_value=True),
            patch.object(budget_service, "get_currency_code", side_effect=["MYR", "JPY"]),
            patch.object(budget_service, "get_currency_rate", return_value=1),
            patch.object(
                budget_service,
                "fetch_flights_api",
                return_value=[{"price": 25, "flight_number": "BOUNDARY"}],
            ),
            patch.object(
                budget_service,
                "fetch_hotels_api",
                side_effect=AssertionError("same-day form must not query hotels"),
            ) as hotel_provider,
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == expected_status
        hotel_provider.assert_not_called()
        if expected_status == "success":
            invoke.assert_awaited_once()
        else:
            invoke.assert_not_awaited()
            assert body["recommended_minimum_budget"] == 200
            assert body["evidence"]["hotel_nights"] == 0
            assert body["evidence"]["hotel_price_per_night"] == 0

    @pytest.mark.parametrize(
        "replacement_message",
        [
            AIMessage(content=""),
            AIMessage(content="Unmarked planning text."),
            AIMessage(
                content="Reply for a different revision.",
                additional_kwargs={"plan_revision": 999},
            ),
        ],
    )
    def test_form_success_requires_nonempty_promoted_reply_for_same_revision(
        self,
        replacement_message,
    ):
        """Catch an unreviewed or revision-mismatched initial success reply."""
        final_state = self.final_state()
        final_state["messages"][-1] = replacement_message
        assessment = self.assessment()
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
            ),
            patch.object(
                form,
                "invoke_new_trip_authenticated",
                AsyncMock(return_value=final_state),
            ),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())

        assert response.status_code == 200
        assert response.json()["status"] == "planning_unavailable"
        assert "Unmarked planning text" not in response.text
        assert "different revision" not in response.text

    def test_successful_form_outcome_survives_raising_log_handler(self):
        final_state = self.final_state()
        assessment = self.assessment()
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={
                    "name": "Alex",
                    "home_country": "Malaysia",
                    "home_state": "Selangor",
                },
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
                create=True,
            ),
            patch.object(
                form,
                "invoke_new_trip_authenticated",
                AsyncMock(return_value=final_state),
            ),
            raising_log_handler(form.logger),
        ):
            response = client.post(
                "/api/form/submit",
                headers={**AUTH, "X-Request-ID": "raising-form-request"},
                json=valid_form(),
            )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_success_is_refused_without_qualified_accepted_snapshot(self):
        """Catch the initial API falling back to a draft or private candidate."""
        rejected = qualified_plan_state()
        rejected.pop("accepted_plan_snapshot")
        rejected["candidate_plan"] = {
            "itinerary": copy.deepcopy(rejected["draft_itinerary"]),
            "maps": copy.deepcopy(rejected["daily_map_info"]),
            "review": {"approved": False},
        }
        assessment = self.assessment()
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
            ),
            patch.object(
                form,
                "invoke_new_trip_authenticated",
                AsyncMock(return_value=rejected),
            ),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())

        body = response.json()
        assert response.status_code == 200
        assert body == {
            "status": "planning_unavailable",
            "reason": "validation_failed",
            "chat_reply": "Planning is temporarily unavailable. Please retry.",
            "retryable": True,
            "itinerary": None,
            "daily_geojson_maps": None,
        }
        assert "candidate_plan" not in response.text

    def test_insufficient_budget_returns_confirmation_without_invoking_graph(self):
        invoke = AsyncMock()
        assessment = self.assessment(minimum=192000, recommended=6000)
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "budget_confirmation_required"
        assert body["reason"] == "insufficient_budget"
        assert body["stated_budget"] == 5000
        assert body["recommended_minimum_budget"] == 6000
        assert body["itinerary"] is None
        assert body["daily_geojson_maps"] is None
        assert "session_id" not in body
        invoke.assert_not_awaited()

    def test_unknown_budget_returns_recommendation_without_invoking_graph(self):
        invoke = AsyncMock()
        assessment = self.assessment(minimum=192000, recommended=6000)
        payload = {
            **valid_form(),
            "total_budget": None,
            "request_budget_recommendation": True,
        }
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        body = response.json()
        assert body["status"] == "budget_confirmation_required"
        assert body["reason"] == "recommendation_requested"
        assert body["stated_budget"] is None
        assert body["recommended_minimum_budget"] == 6000
        assert body["itinerary"] is None
        invoke.assert_not_awaited()

    def test_valid_cached_confirmation_plans_without_fresh_assessment(self):
        invoke = AsyncMock(return_value=self.final_state(total_base_budget=6000))
        assessment = self.assessment(minimum=192000, recommended=6000)
        payload = {
            **valid_form(),
            "total_budget": 6000,
            "budget_assessment_id": "assessment-1",
        }
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "load_confirmed_budget_assessment",
                return_value=assessment,
                create=True,
            ) as load_assessment,
            patch.object(
                form,
                "get_or_create_budget_assessment",
                side_effect=AssertionError("confirmation must not refresh providers"),
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        assert response.json()["status"] == "success"
        load_assessment.assert_called_once_with(
            assessment_id="assessment-1",
            origin="Malaysia",
            destination="Japan",
            destination_city="Tokyo",
            start_date="2026-08-01",
            end_date="2026-08-05",
            num_people=2,
        )
        invoke.assert_awaited_once()

    def test_confirmed_amount_below_cached_minimum_is_rejected_as_tampered(self):
        invoke = AsyncMock()
        assessment = self.assessment(minimum=192000, recommended=6000)
        payload = {
            **valid_form(),
            "total_budget": 5999,
            "budget_assessment_id": "assessment-1",
        }
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "load_confirmed_budget_assessment",
                return_value=assessment,
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["reason"] == "assessment_expired_or_invalid"
        invoke.assert_not_awaited()

    def test_confirmed_amount_below_displayed_rounded_minimum_is_rejected(self):
        invoke = AsyncMock()
        assessment = self.assessment(minimum=191999.0, recommended=6000.0)
        payload = {
            **valid_form(),
            # Mathematically clears the raw destination minimum, but is less
            # than the exact displayed amount the user is confirming.
            "total_budget": 5999.99,
            "budget_assessment_id": "assessment-1",
        }
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "load_confirmed_budget_assessment",
                return_value=assessment,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        assert response.json()["status"] == "budget_check_unavailable"
        invoke.assert_not_awaited()

    def test_invalid_or_expired_confirmation_fails_closed(self):
        invoke = AsyncMock()
        payload = {
            **valid_form(),
            "total_budget": 6000,
            "budget_assessment_id": "expired",
        }
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "load_confirmed_budget_assessment",
                return_value=None,
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=payload)

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["reason"] == "assessment_expired_or_invalid"
        assert body["itinerary"] is None
        assert "session_id" not in body
        invoke.assert_not_awaited()

    @pytest.mark.parametrize(
        "persisted,side_effect,expected_reason",
        [
            (False, None, "assessment_cache_unavailable"),
            (True, BudgetAssessmentUnavailable("offline"), "provider_data_unavailable"),
        ],
    )
    def test_unusable_assessment_never_invokes_graph(
        self,
        persisted,
        side_effect,
        expected_reason,
    ):
        invoke = AsyncMock()
        assessment = self.assessment(minimum=192000, recommended=6000)
        assessment_result = BudgetAssessmentResult(
            assessment=assessment,
            persisted=persisted,
        )
        with (
            patch.object(
                form,
                "fetch_user_profile",
                return_value={"home_country": "Malaysia", "home_state": "Selangor"},
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=assessment_result,
                side_effect=side_effect,
                create=True,
            ),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["reason"] == expected_reason
        assert body["itinerary"] is None
        invoke.assert_not_awaited()

    def test_incomplete_onboarding_is_conflict_and_does_not_invoke_graph(self):
        invoke = AsyncMock()
        with (
            patch.object(form, "fetch_user_profile", return_value={}),
            patch.object(form, "invoke_new_trip_authenticated", invoke),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())
        assert response.status_code == 409
        invoke.assert_not_awaited()

    @pytest.mark.parametrize(
        "changes",
        [
            {"num_people": 0},
            {"total_budget": 0},
            {"end_date": "2026-07-01"},
            {"start_date": "bad"},
            {"country": ""},
        ],
    )
    def test_invalid_forms_are_422_before_storage_or_graph(self, changes):
        response = client.post(
            "/api/form/submit", headers=AUTH, json={**valid_form(), **changes}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "exception,status_code",
        [
            (asyncio.TimeoutError(), 504),
            (httpx.ConnectError("offline"), 502),
            (RuntimeError("secret internal detail"), 500),
        ],
    )
    def test_workflow_failures_are_safely_mapped(self, exception, status_code):
        assessment = self.assessment()
        with (
            patch.object(
                form, "fetch_user_profile", return_value={"home_country": "Malaysia"}
            ),
            patch.object(
                form,
                "get_or_create_budget_assessment",
                return_value=BudgetAssessmentResult(
                    assessment=assessment,
                    persisted=True,
                ),
                create=True,
            ),
            patch.object(
                form, "invoke_new_trip_authenticated", AsyncMock(side_effect=exception)
            ),
            raising_log_handler(form.logger),
        ):
            response = client.post("/api/form/submit", headers=AUTH, json=valid_form())
        assert response.status_code == status_code
        assert "secret internal detail" not in response.text


@pytest.mark.api
class TestChatRoute:
    def test_planning_unavailable_never_contains_candidate_payload(self):
        """Catch a rejected chat candidate crossing the wire or replacing state."""
        state = qualified_plan_state(reply="Planning could not be completed safely.")
        accepted_before = copy.deepcopy(state["accepted_plan_snapshot"])
        itinerary_before = copy.deepcopy(state["draft_itinerary"])
        state.update(
            {
                "planning_outcome": "unavailable",
                "planning_issue_codes": ["activity.country.mismatch"],
                "candidate_plan": {
                    "itinerary": [{"day": 1, "country": "Rejected"}],
                    "maps": {1: {"private": True}},
                    "review": {"approved": False},
                    "tool_payload": {"secret": True},
                },
            }
        )
        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "session-1", "user_message": "Replan everything"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "planning_unavailable"
        assert body["reason"] == "validation_failed"
        assert body["retryable"] is True
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert "candidate_plan" not in response.text
        assert "tool_payload" not in response.text
        assert state["accepted_plan_snapshot"] == accepted_before
        assert state["draft_itinerary"] == itinerary_before

    @pytest.mark.parametrize("internal_key", ["candidate_plan", "review", "debug", "tool_payload"])
    def test_nested_internal_map_data_fails_closed(self, internal_key):
        """Catch private/debug data hidden inside an otherwise valid accepted map."""
        state = qualified_plan_state()
        state["accepted_plan_snapshot"]["daily_map_info"][1][internal_key] = {
            "secret": True
        }
        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "nested-private", "user_message": "hello"},
            )

        assert response.json()["status"] == "planning_unavailable"
        assert internal_key not in response.text

    def test_arbitrary_nested_tool_field_name_fails_closed_at_chat_response(self):
        """Catch non-denylisted provider payloads hitchhiking in GeoJSON."""
        state = qualified_plan_state()
        properties = state["accepted_plan_snapshot"]["daily_map_info"][1][
            "features"
        ][0]["properties"]
        properties["provider_blob"] = {"secret": "must-not-serialize"}

        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "nested-arbitrary", "user_message": "hello"},
            )

        assert response.json()["status"] == "planning_unavailable"
        assert "must-not-serialize" not in response.text

    def test_colliding_normalized_map_keys_fail_closed_at_chat_response(self):
        """Validate and project the same canonical map object without key overwrite."""
        state = qualified_plan_state()
        valid_day = state["accepted_plan_snapshot"]["daily_map_info"][1]
        state["accepted_plan_snapshot"]["daily_map_info"] = {
            1: valid_day,
            "1": {"type": "FeatureCollection", "features": []},
        }

        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "map-key-collision", "user_message": "hello"},
            )

        assert response.json()["status"] == "planning_unavailable"
        assert response.json()["daily_map_info"] == {}

    def test_trailing_tool_message_cannot_become_live_chat_reply(self):
        """Only a completed AI message from the current turn may cross the wire."""
        state = qualified_plan_state()
        state["messages"] = [
            HumanMessage(content="Change the hotel"),
            AIMessage(
                content="",
                tool_calls=[{"name": "edit_itinerary", "args": {}, "id": "1"}],
            ),
            ToolMessage(
                name="edit_itinerary",
                content='{"tool_result":{"secret":"must-not-serialize"}}',
                tool_call_id="1",
            ),
        ]

        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "trailing-tool", "user_message": "Change hotel"},
            )

        body = response.json()
        assert body["status"] == "planning_unavailable"
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert "must-not-serialize" not in response.text

    def test_history_preserves_safe_text_but_skips_invalid_plan_attachment(self):
        """Catch history restoring raw plan fields when its snapshot is invalid."""
        state = qualified_plan_state()
        state["messages"] = [
            HumanMessage(content="Plan a safe trip"),
            AIMessage(content="Here is the saved response."),
        ]
        state["accepted_plan_snapshot"]["plan_revision"] = 999

        restored = chat._serialize_history_session(
            {
                "session_id": "history-invalid-snapshot",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        assert [message["content"] for message in restored.messages] == [
            "Plan a safe trip",
            "Here is the saved response.",
        ]
        assert all("itinerary" not in message for message in restored.messages)
        assert all("maps" not in message for message in restored.messages)

    @pytest.mark.parametrize(
        "unsafe",
        [
            '{"draft_itinerary":[{"day":1}]}',
            'Summary before {"candidate_plan":{"country":"Malaysia"}} after.',
        ],
        ids=["whole-json", "embedded-json"],
    )
    def test_history_replaces_unsafe_legacy_ai_output_but_retains_user_text(
        self, unsafe
    ):
        state = qualified_plan_state()
        state["messages"] = [
            HumanMessage(content="Keep this user request"),
            AIMessage(
                content=unsafe,
                additional_kwargs={"plan_revision": state["plan_revision"]},
            ),
            HumanMessage(content="And keep this follow-up"),
            AIMessage(content="Safe saved answer."),
        ]

        restored = chat._serialize_history_session(
            {
                "session_id": "unsafe-legacy-history",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        contents = [message["content"] for message in restored.messages]
        assert "Keep this user request" in contents
        assert "And keep this follow-up" in contents
        assert "Safe saved answer." in contents
        assert unsafe not in contents
        assert chat._UNSAFE_HISTORY_MESSAGE in contents
        assert "draft_itinerary" not in repr(restored.messages)
        assert "candidate_plan" not in repr(restored.messages)

    def test_failed_modifying_turn_keeps_prior_history_plan_on_accepted_reply(self):
        """A failed edit response must not inherit the prior accepted snapshot."""
        state = qualified_plan_state()
        state["planning_outcome"] = "unavailable"
        state["messages"] = [
            HumanMessage(content="Plan my trip"),
            AIMessage(
                content="Initial accepted plan",
                additional_kwargs={"plan_revision": 1},
            ),
            HumanMessage(content="Change the hotel"),
            AIMessage(
                content="",
                tool_calls=[{"name": "edit_itinerary", "args": {}, "id": "1"}],
            ),
            ToolMessage(name="edit_itinerary", content="{}", tool_call_id="1"),
            AIMessage(content=errors.PLANNING_UNAVAILABLE_MESSAGE),
        ]

        restored = chat._serialize_history_session(
            {
                "session_id": "failed-edit-history",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        assert [
            (message["role"], message["content"])
            for message in restored.messages
        ] == [
            ("user", "Plan my trip"),
            ("ai", "Initial accepted plan"),
            ("user", "Change the hotel"),
            ("ai", errors.PLANNING_UNAVAILABLE_MESSAGE),
        ]
        assert restored.messages[1]["itinerary"] == state["draft_itinerary"]
        assert restored.messages[1]["maps"] == state["daily_map_info"]
        assert "itinerary" not in restored.messages[3]
        assert "maps" not in restored.messages[3]

    @pytest.mark.asyncio
    async def test_same_owner_session_chat_invocations_serialize_revision_updates(self):
        """Catch two overlapping requests both promoting from revision N."""
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release_first = asyncio.Event()
        revision_store = {"value": 1}
        observed_bases: list[int] = []

        async def invoke(**kwargs):
            base_revision = revision_store["value"]
            observed_bases.append(base_revision)
            if len(observed_bases) == 1:
                first_entered.set()
                await release_first.wait()
            else:
                second_entered.set()
            revision_store["value"] = base_revision + 1
            return qualified_plan_state(revision=base_revision + 1)

        with patch.object(chat, "invoke_chat_authenticated", side_effect=invoke):
            first = asyncio.create_task(
                chat.chat_with_ai(
                    user_id="lock-owner",
                    chat=chat.ChatRequest(session_id="shared", user_message="first"),
                    request=Mock(),
                    request_id="request-1",
                )
            )
            await first_entered.wait()
            second = asyncio.create_task(
                chat.chat_with_ai(
                    user_id="lock-owner",
                    chat=chat.ChatRequest(session_id="shared", user_message="second"),
                    request=Mock(),
                    request_id="request-2",
                )
            )
            try:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not second_entered.is_set()
            finally:
                release_first.set()
            first_response, second_response = await asyncio.gather(first, second)

        assert observed_bases == [1, 2]
        assert revision_store["value"] == 3
        assert first_response.status == "success"
        assert second_response.status == "success"

    @pytest.mark.asyncio
    async def test_different_chat_sessions_are_not_serialized_together(self):
        """Catch a global lock unnecessarily blocking unrelated sessions."""
        entered = {"one": asyncio.Event(), "two": asyncio.Event()}
        release = asyncio.Event()

        async def invoke(**kwargs):
            entered[kwargs["thread_id"]].set()
            await release.wait()
            return qualified_plan_state()

        with patch.object(chat, "invoke_chat_authenticated", side_effect=invoke):
            tasks = [
                asyncio.create_task(
                    chat.chat_with_ai(
                        user_id="lock-owner",
                        chat=chat.ChatRequest(session_id=session, user_message="replan"),
                        request=Mock(),
                        request_id=f"request-{session}",
                    )
                )
                for session in ("one", "two")
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in entered.values())),
                    timeout=1,
                )
            finally:
                release.set()
            responses = await asyncio.gather(*tasks)

        assert [response.status for response in responses] == ["success", "success"]

    @pytest.mark.asyncio
    async def test_cancelled_chat_releases_session_and_registry_entry(self):
        """Catch cancellation stranding a lock or retaining its session key."""
        first_entered = asyncio.Event()
        never_release = asyncio.Event()
        call_count = 0

        async def invoke(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_entered.set()
                await never_release.wait()
            return qualified_plan_state(revision=call_count)

        with patch.object(chat, "invoke_chat_authenticated", side_effect=invoke):
            first = asyncio.create_task(
                chat.chat_with_ai(
                    user_id="cancel-owner",
                    chat=chat.ChatRequest(session_id="cancel-session", user_message="one"),
                    request=Mock(),
                    request_id="cancel-1",
                )
            )
            await first_entered.wait()
            second = asyncio.create_task(
                chat.chat_with_ai(
                    user_id="cancel-owner",
                    chat=chat.ChatRequest(session_id="cancel-session", user_message="two"),
                    request=Mock(),
                    request_id="cancel-2",
                )
            )
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            response = await asyncio.wait_for(second, timeout=1)

        assert response.status == "success"
        assert chat._SESSION_MUTATION_LOCKS._entries == {}

    @pytest.mark.asyncio
    async def test_failed_chat_releases_session_lock_for_next_invocation(self):
        """Catch mapped graph errors retaining a session lock or registry key."""
        calls = 0

        async def invoke(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private graph failure")
            return qualified_plan_state()

        with patch.object(chat, "invoke_chat_authenticated", side_effect=invoke):
            with pytest.raises(HTTPException):
                await chat.chat_with_ai(
                    user_id="error-owner",
                    chat=chat.ChatRequest(session_id="error-session", user_message="one"),
                    request=Mock(),
                    request_id="error-1",
                )
            response = await asyncio.wait_for(
                chat.chat_with_ai(
                    user_id="error-owner",
                    chat=chat.ChatRequest(session_id="error-session", user_message="two"),
                    request=Mock(),
                    request_id="error-2",
                ),
                timeout=1,
            )

        assert response.status == "success"
        assert chat._SESSION_MUTATION_LOCKS._entries == {}

    @pytest.mark.parametrize("surface", ["live", "history"])
    def test_budget_gate_ignores_private_persisted_copy(self, surface):
        """Only deterministic public copy may replace a persisted gate string."""
        private = "PRIVATE provider payload https://evil.invalid/?token=secret-123"
        state = pending_chat_state()
        state["budget_gate_message"] = private
        state["messages"] = [
            HumanMessage(content="Use RM 500"),
            AIMessage(content=private),
        ]

        if surface == "live":
            with patch.object(
                chat,
                "invoke_chat_authenticated",
                AsyncMock(return_value=state),
            ):
                response = client.post(
                    "/api/chat/message",
                    headers=AUTH,
                    json={"session_id": "s", "user_message": "Try RM 500"},
                )
            body = response.json()
            public_copy = body["chat_reply"]
            assert body["budget_confirmation"]["chat_reply"] == public_copy
        else:
            session = chat._serialize_history_session(
                {
                    "session_id": "s",
                    "updated_at": "2026-08-16T12:01:00+00:00",
                    "state": state,
                }
            )
            public_copy = session.messages[-1]["content"]

        assert public_copy == (
            "MYR 500.00 is below the provider-grounded minimum of MYR "
            "3,500.00. Use the recommended budget to continue planning."
        )
        assert private not in public_copy
        assert "secret-123" not in public_copy

    @pytest.mark.parametrize("surface", ["live", "history"])
    def test_uncached_coherent_pending_assessment_is_not_projected_as_actionable(
        self, surface
    ):
        state = pending_chat_state()
        with patch.object(chat, "load_confirmed_budget_assessment", return_value=None):
            if surface == "live":
                with patch.object(
                    chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
                ):
                    response = client.post(
                        "/api/chat/message",
                        headers=AUTH,
                        json={"session_id": "s", "user_message": "Use RM 500"},
                    )
                body = response.json()
                assert body["status"] == "budget_check_unavailable"
                assert body["budget_confirmation"] is None
            else:
                restored = chat._serialize_history_session(
                    {
                        "session_id": "s",
                        "updated_at": "2026-08-16T12:01:00+00:00",
                        "state": state,
                    }
                )
                assert "budget_confirmation" not in restored.messages[-1]

    @pytest.mark.parametrize("surface", ["live", "history"])
    @pytest.mark.parametrize(
        "reason,stated_budget",
        [
            ("recommendation_requested", 500),
            ("insufficient_budget", None),
            ("insufficient_budget", 3500),
        ],
    )
    def test_budget_confirmation_reason_amount_and_sufficiency_are_coherent(
        self,
        surface,
        reason,
        stated_budget,
    ):
        """Contradictory checkpoint claims must never create an action card."""
        state = pending_chat_state()
        state["pending_budget_confirmation"] = copy.deepcopy(
            state["pending_budget_confirmation"]
        )
        state["pending_budget_confirmation"]["reason"] = reason
        state["pending_budget_confirmation"]["stated_budget"] = stated_budget

        if surface == "live":
            with patch.object(
                chat,
                "invoke_chat_authenticated",
                AsyncMock(return_value=state),
            ):
                response = client.post(
                    "/api/chat/message",
                    headers=AUTH,
                    json={"session_id": "s", "user_message": "budget"},
                )
            body = response.json()
            assert body["status"] == "budget_check_unavailable"
            assert body["budget_confirmation"] is None
        else:
            session = chat._serialize_history_session(
                {
                    "session_id": "s",
                    "updated_at": "2026-08-16T12:01:00+00:00",
                    "state": state,
                }
            )
            assert "budget_confirmation" not in session.messages[-1]

    @pytest.mark.parametrize(
        "gate_message",
        [
            None,
            "   ",
            {"assessment": {"evidence": {"hotel_price_per_night": 900}}},
            ["assessment", "evidence", 900],
            pending_chat_state()["pending_budget_confirmation"]["assessment"],
        ],
        ids=["missing", "blank", "dict", "list", "nested_assessment"],
    )
    def test_chat_unavailable_rejects_malformed_gate_message_without_leaking_state(
        self, gate_message
    ):
        """Catch unavailable replies that stringify checkpoint-owned assessment data."""
        state = {
            "messages": [AIMessage(content="assessment evidence costs 900 and 3500")],
            "budget_gate_outcome": "budget_check_unavailable",
            "budget_gate_message": gate_message,
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["chat_reply"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["budget_confirmation"] is None
        assert "assessment" not in body["chat_reply"]
        assert "evidence" not in body["chat_reply"]
        assert "900" not in body["chat_reply"]
        assert "3500" not in body["chat_reply"]

    @pytest.mark.parametrize(
        "gate_message",
        [
            None,
            "   ",
            {"assessment": {"evidence": {"hotel_price_per_night": 900}}},
            ["assessment", "evidence", 900],
            pending_chat_state()["pending_budget_confirmation"]["assessment"],
        ],
        ids=["missing", "blank", "dict", "list", "nested_assessment"],
    )
    def test_history_unavailable_rejects_malformed_gate_message_without_plan(
        self, gate_message
    ):
        """Catch history restoring a stale plan from an incoherent gate state."""
        state = {
            "messages": [
                HumanMessage(content="Use RM 500"),
                AIMessage(content="assessment evidence costs 900 and 3500"),
            ],
            "budget_gate_outcome": "budget_check_unavailable",
            "budget_gate_message": gate_message,
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        session = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        newest = session.messages[-1]
        assert newest["content"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert "itinerary" not in newest
        assert "maps" not in newest
        assert "budget_confirmation" not in newest
        assert "assessment" not in newest["content"]
        assert "evidence" not in newest["content"]
        assert "900" not in newest["content"]
        assert "3500" not in newest["content"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("budget_assessment_id", "other-assessment"),
            ("recommended_minimum_budget", 1),
            ("base_currency", "USD"),
            ("destination_currency", "JPY"),
            ("expires_at", "2999-08-17T13:00:00+00:00"),
            (
                "evidence",
                {
                    "outbound_flight_price": 900,
                    "return_flight_price": 850,
                    "hotel_price_per_night": 999,
                    "hotel_nights": 3,
                },
            ),
            ("reason", "untrusted_reason"),
            ("stated_budget", True),
            ("stated_budget", 0),
        ],
    )
    def test_chat_confirmation_rejects_mismatched_outer_proposal_data(
        self, field, value
    ):
        """Catch a checkpoint proposal whose public action differs from its assessment."""
        state = pending_chat_state()
        state["pending_budget_confirmation"] = copy.deepcopy(
            state["pending_budget_confirmation"]
        )
        state["pending_budget_confirmation"][field] = value
        state["messages"] = [AIMessage(content="Confirm RM 3,500 first.")]
        state["draft_itinerary"] = [{"day": 1}]
        state["daily_map_info"] = {1: {"type": "FeatureCollection", "features": []}}
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["budget_confirmation"] is None
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert "assessment" not in response.text
        assert "outbound_flight" not in response.text

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("budget_assessment_id", "other-assessment"),
            ("recommended_minimum_budget", 1),
            ("base_currency", "USD"),
            ("destination_currency", "JPY"),
            ("expires_at", "2999-08-17T13:00:00+00:00"),
            (
                "evidence",
                {
                    "outbound_flight_price": 900,
                    "return_flight_price": 850,
                    "hotel_price_per_night": 999,
                    "hotel_nights": 3,
                },
            ),
            ("reason", "untrusted_reason"),
            ("stated_budget", True),
            ("stated_budget", 0),
        ],
    )
    def test_history_confirmation_rejects_mismatched_outer_proposal_data(
        self, field, value
    ):
        """Catch history exposing a button derived from corrupt outer proposal data."""
        state = pending_chat_state()
        state["pending_budget_confirmation"] = copy.deepcopy(
            state["pending_budget_confirmation"]
        )
        state["pending_budget_confirmation"][field] = value
        session = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        newest = session.messages[-1]
        assert newest["content"] == (
            "This budget check expired or no longer matches the trip. Request a refreshed "
            "recommendation before planning."
        )
        assert "budget_confirmation" not in newest
        assert "itinerary" not in newest
        assert "maps" not in newest
        assert "assessment" not in str(newest)
        assert "outbound_flight" not in str(newest)

    def test_chat_confirmation_rejects_oversized_stated_budget_without_leakage(
        self,
    ):
        """Catch arbitrary-precision checkpoint amounts crashing the live route."""
        state = pending_chat_state()
        state["pending_budget_confirmation"] = copy.deepcopy(
            state["pending_budget_confirmation"]
        )
        state["pending_budget_confirmation"]["stated_budget"] = 10**10000
        state["messages"] = [AIMessage(content="Confirm RM 3,500 first.")]
        state["draft_itinerary"] = [{"day": 1}]
        state["daily_map_info"] = {1: {"type": "FeatureCollection", "features": []}}
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["chat_reply"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert body["budget_confirmation"] is None
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert "assessment" not in response.text
        assert "evidence" not in response.text
        assert "stated_budget" not in response.text
        assert len(response.content) < 1024

    def test_history_confirmation_rejects_oversized_stated_budget_without_leakage(
        self,
    ):
        """Catch arbitrary-precision checkpoint amounts crashing history restoration."""
        state = pending_chat_state()
        state["pending_budget_confirmation"] = copy.deepcopy(
            state["pending_budget_confirmation"]
        )
        state["pending_budget_confirmation"]["stated_budget"] = 10**10000
        session = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        newest = session.messages[-1]
        assert newest["content"] == (
            "This budget check expired or no longer matches the trip. Request a refreshed "
            "recommendation before planning."
        )
        assert "budget_confirmation" not in newest
        assert "itinerary" not in newest
        assert "maps" not in newest
        assert "stated_budget" not in newest
        assert "assessment" not in newest
        assert "evidence" not in newest

    @pytest.mark.parametrize(
        "outcome,pending",
        [
            ("unknown_outcome", None),
            ("accepted", pending_chat_state()["pending_budget_confirmation"]),
            ("budget_confirmation_required", None),
            (
                "budget_check_unavailable",
                pending_chat_state()["pending_budget_confirmation"],
            ),
            (None, pending_chat_state()["pending_budget_confirmation"]),
        ],
        ids=[
            "unknown",
            "accepted_with_pending",
            "confirmation_without_pending",
            "unavailable_with_pending",
            "none_with_pending",
        ],
    )
    def test_chat_incoherent_budget_gate_state_fails_closed(self, outcome, pending):
        """Catch non-ordinary gate states falling through to planning success."""
        state = {
            "messages": [AIMessage(content="A plan reply")],
            "budget_gate_outcome": outcome,
            "budget_gate_message": "A gate reply",
            "pending_budget_confirmation": copy.deepcopy(pending),
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["chat_reply"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["budget_confirmation"] is None

    @pytest.mark.parametrize(
        "outcome,pending",
        [
            ("unknown_outcome", None),
            ("accepted", pending_chat_state()["pending_budget_confirmation"]),
            ("budget_confirmation_required", None),
            (
                "budget_check_unavailable",
                pending_chat_state()["pending_budget_confirmation"],
            ),
            (None, pending_chat_state()["pending_budget_confirmation"]),
        ],
        ids=[
            "unknown",
            "accepted_with_pending",
            "confirmation_without_pending",
            "unavailable_with_pending",
            "none_with_pending",
        ],
    )
    def test_history_incoherent_budget_gate_state_hides_stale_plan(
        self, outcome, pending
    ):
        """Catch history attaching stale checkpoint plans to invalid gate states."""
        state = {
            "messages": [
                HumanMessage(content="Use RM 500"),
                AIMessage(content="A plan reply"),
            ],
            "budget_gate_outcome": outcome,
            "budget_gate_message": "A gate reply",
            "pending_budget_confirmation": copy.deepcopy(pending),
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        session = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        newest = session.messages[-1]
        assert newest["content"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert "budget_confirmation" not in newest
        assert "itinerary" not in newest
        assert "maps" not in newest

    def test_history_restores_only_valid_pending_budget_action(self):
        """Catch stale checkpoints leaking a confirmation or accepted itinerary."""
        valid = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": pending_chat_state(),
            }
        )
        newest = valid.messages[-1]
        assert newest["budget_confirmation"]["budget_assessment_id"] == "assessment-1"
        assert "assessment" not in newest["budget_confirmation"]
        assert "itinerary" not in newest

        expired = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": pending_chat_state("2000-01-01T00:00:00+00:00"),
            }
        )
        expired_message = expired.messages[-1]
        assert "expired or no longer matches" in expired_message["content"]
        assert "budget_confirmation" not in expired_message
        assert "itinerary" not in expired_message

    def test_chat_confirmation_required_hides_existing_planning_payload(self):
        """Catch a rejected budget gate returning a checkpoint's old itinerary."""
        state = {
            **pending_chat_state(),
            "messages": [AIMessage(content="Confirm RM 3,500 first.")],
            "budget_gate_outcome": "budget_confirmation_required",
            "budget_gate_message": "Confirm RM 3,500 first.",
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "budget_confirmation_required"
        assert body["chat_reply"] == (
            "MYR 500.00 is below the provider-grounded minimum of MYR 3,500.00. "
            "Use the recommended budget to continue planning."
        )
        assert body["itinerary_modified"] is False
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["budget_confirmation"]["budget_assessment_id"] == "assessment-1"
        assert "assessment" not in body["budget_confirmation"]

    def test_chat_budget_unavailable_returns_no_planning_payload(self):
        """Catch an unavailable budget check exposing a stale itinerary."""
        state = {
            "messages": [AIMessage(content="Refresh the grounded budget check.")],
            "budget_gate_outcome": "budget_check_unavailable",
            "budget_gate_reason": "provider_data_unavailable",
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "budget_check_unavailable"
        assert body["chat_reply"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["itinerary_modified"] is False
        assert body["budget_confirmation"] is None

    def test_chat_mismatched_confirmation_outcome_fails_closed(self):
        """Catch a confirmation outcome without its verified proposal exposing a plan."""
        state = {
            "messages": [AIMessage(content="Confirm first.")],
            "budget_gate_outcome": "budget_confirmation_required",
            "budget_gate_message": "Confirm first.",
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_check_unavailable"
        assert body["chat_reply"] == (
            "Unable to verify the budget right now. Your existing trip plan has not "
            "been changed."
        )
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["budget_confirmation"] is None

    def test_chat_confirmation_reconstructs_copy_without_gate_message(self):
        """The public confirmation projection, not checkpoint text, authorizes copy."""
        state = {
            **pending_chat_state(),
            "messages": [],
            "budget_gate_outcome": "budget_confirmation_required",
            "budget_gate_message": None,
            "draft_itinerary": [{"day": 1}],
            "daily_map_info": {1: {"type": "FeatureCollection", "features": []}},
        }
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "Try RM 500"},
            )

        body = response.json()
        assert body["status"] == "budget_confirmation_required"
        assert body["chat_reply"] == (
            "MYR 500.00 is below the provider-grounded minimum of MYR 3,500.00. "
            "Use the recommended budget to continue planning."
        )
        assert body["draft_itinerary"] == []
        assert body["daily_map_info"] == {}
        assert body["budget_confirmation"]["budget_assessment_id"] == "assessment-1"

    def test_structured_chat_confirmation_is_forwarded_exactly(self):
        """Catch the route dropping or changing the accepted opaque action."""
        invoke = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Accepted and re-planned.")],
                "draft_itinerary": [],
                "daily_map_info": {},
            }
        )
        with patch.object(chat, "invoke_chat_authenticated", invoke):
            response = client.post(
                "/api/chat/message",
                headers={**AUTH, "X-Request-ID": "chat-request-1"},
                json={
                    "session_id": "s",
                    "user_message": "Use recommended budget",
                    "budget_action": "accept_recommended",
                    "budget_assessment_id": "assessment-1",
                    "recommended_minimum_budget": 1,
                    "assessment": {"evidence": "client-controlled"},
                },
            )

        assert response.status_code == 200
        assert invoke.await_args.kwargs == {
            "user_message": "Use recommended budget",
            "thread_id": "s",
            "user_id": "test-user",
            "budget_action": "accept_recommended",
            "budget_assessment_id": "assessment-1",
            "request_id": "chat-request-1",
        }

    @pytest.mark.parametrize(
        "name", ["propose_budget_change", "confirm_recommended_budget"]
    )
    def test_budget_intent_tools_do_not_mark_itinerary_modified(self, name):
        """Catch confirmation-only tool calls being rendered as a new plan."""
        messages = [
            HumanMessage(content="budget"),
            AIMessage(content=""),
            ToolMessage(name=name, content="{}", tool_call_id="1"),
            AIMessage(content="Confirm first."),
        ]
        assert chat._plan_modified_this_turn(messages) is False

    @pytest.mark.parametrize(
        "messages",
        [
            [
                HumanMessage(content="Use RM 4,000"),
                ToolMessage(
                    name="propose_budget_change",
                    content="{}",
                    tool_call_id="proposal",
                ),
                AIMessage(content="Budget accepted and itinerary updated."),
            ],
            [
                HumanMessage(content="Yes, use it"),
                ToolMessage(
                    name="confirm_recommended_budget",
                    content="{}",
                    tool_call_id="confirm",
                ),
                AIMessage(content="Budget accepted and itinerary updated."),
            ],
            [
                HumanMessage(content="Use recommended budget"),
                AIMessage(content="Budget accepted and itinerary updated."),
            ],
        ],
        ids=["sufficient-proposal", "natural-confirmation", "structured-confirmation"],
    )
    def test_completed_budget_replans_use_server_revision_marker(self, messages):
        """Catch complete budget replans being hidden by legacy tool-name inference."""
        state = qualified_plan_state(revision=2)
        state.update({"messages": messages, "_itinerary_modified": True})
        with patch.object(
            chat,
            "invoke_chat_authenticated",
            AsyncMock(return_value=state),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "budget decision"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["itinerary_modified"] is True
        assert body["draft_itinerary"] == state["draft_itinerary"]
        assert body["daily_map_info"] == {
            str(day): daily_map for day, daily_map in state["daily_map_info"].items()
        }

    @pytest.mark.parametrize("gate_kind", ["pending", "unavailable", "expired"])
    def test_history_keeps_accepted_snapshot_on_prior_plan_message(self, gate_kind):
        """Catch a gate card/error removing the previously accepted canvas."""
        gate = pending_chat_state()
        state = qualified_plan_state(
            country="China",
            country_code="CN",
            destination_currency="CNY",
            cities=["Beijing"],
            num_people=1,
            start_date="2026-08-17",
            end_date="2026-08-20",
            total_convert_budget=10000,
        )
        state.update(
            {
                "pending_budget_confirmation": gate["pending_budget_confirmation"],
                "budget_gate_outcome": gate["budget_gate_outcome"],
                "budget_gate_message": gate["budget_gate_message"],
            }
        )
        state["messages"] = [
            HumanMessage(content="Plan my trip"),
            AIMessage(content="Initial accepted plan"),
            HumanMessage(content="Use RM 500"),
            AIMessage(content="Untrusted persisted gate copy"),
        ]
        if gate_kind == "unavailable":
            state["budget_gate_outcome"] = "budget_check_unavailable"
            state["pending_budget_confirmation"] = None
        elif gate_kind == "expired":
            state["pending_budget_confirmation"]["expires_at"] = (
                "2000-01-01T00:00:00+00:00"
            )
            state["pending_budget_confirmation"]["assessment"]["expires_at"] = (
                "2000-01-01T00:00:00+00:00"
            )

        session = chat._serialize_history_session(
            {
                "session_id": "s",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        prior_plan = next(
            message
            for message in session.messages
            if message["content"] == "Initial accepted plan"
        )
        gate_message = session.messages[-1]
        assert prior_plan["itinerary"] == state["draft_itinerary"]
        assert prior_plan["maps"] == state["daily_map_info"]
        assert "itinerary" not in gate_message
        assert "maps" not in gate_message

    @pytest.mark.parametrize("gate_kind", ["pending", "unavailable", "expired"])
    def test_legacy_gated_history_does_not_attach_unvalidated_draft_state(
        self, gate_kind
    ):
        state = pending_chat_state()
        state.pop("accepted_plan_snapshot", None)
        state.update(
            {
                "plan_revision": 1,
                "total_base_budget": 5000,
                "total_convert_budget": 10000,
                "dest_currency_code": "CNY",
                "budget_allocation": {"food": 1500},
                "daily_map_info": {1: {"name": "accepted-map"}},
                "messages": [
                    HumanMessage(content="Plan my trip"),
                    AIMessage(content="Initial accepted plan"),
                    HumanMessage(content="Use RM 500"),
                    AIMessage(content="checkpoint gate text"),
                ],
            }
        )
        if gate_kind == "unavailable":
            state["budget_gate_outcome"] = "budget_check_unavailable"
            state["pending_budget_confirmation"] = None
        elif gate_kind == "expired":
            state["pending_budget_confirmation"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            state["pending_budget_confirmation"]["assessment"]["expires_at"] = "2000-01-01T00:00:00+00:00"

        restored = chat._serialize_history_session(
            {
                "session_id": "legacy",
                "updated_at": "2026-08-16T12:01:00+00:00",
                "state": state,
            }
        )

        assert "itinerary" not in restored.messages[1]
        assert "maps" not in restored.messages[1]
        assert "itinerary" not in restored.messages[-1]

    def test_history_restores_only_display_messages_and_latest_plan(self):
        state = qualified_plan_state(cities=["Tokyo", "Kyoto"])
        state["messages"] = [
                HumanMessage(content="Plan my trip"),
                AIMessage(content="Initial plan"),
                HumanMessage(content="Change the hotel"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "edit_itinerary", "args": {}, "id": "1"}],
                ),
                ToolMessage(name="edit_itinerary", content="{}", tool_call_id="1"),
                AIMessage(
                    content="Hotel updated",
                    additional_kwargs={"plan_revision": 1},
                ),
            ]
        records = [
            {
                "session_id": "session-1",
                "updated_at": "2026-07-15T08:00:00+00:00",
                "state": state,
            }
        ]
        with patch.object(
            chat, "fetch_chat_history_checkpoints", return_value=records
        ) as fetch:
            response = client.get("/api/chat/history?limit=5", headers=AUTH)

        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert "X-User-ID" in response.headers["vary"]
        fetch.assert_called_once_with("test-user", 5)
        session = response.json()["sessions"][0]
        assert session["id"] == "session-1"
        assert session["title"] == "Tokyo, Kyoto, Japan"
        assert [message["content"] for message in session["messages"]] == [
            "Plan my trip",
            "Initial plan",
            "Change the hotel",
            "Hotel updated",
        ]
        restored_plan = session["messages"][-1]
        assert restored_plan["itinerary"] == state["draft_itinerary"]
        assert restored_plan["budget"]["allocation"] == state["budget_allocation"]

    def test_plain_chat_reply_does_not_claim_itinerary_modified(self, caplog):
        state = qualified_plan_state()
        state["messages"] = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there"),
        ]
        caplog.set_level(logging.INFO, logger=chat.logger.name)
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ) as invoke:
            response = client.post(
                "/api/chat/message",
                headers={**AUTH, "X-Request-ID": "chat-request-2"},
                json={"session_id": "session-1", "user_message": "hello"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["chat_reply"] == "Hi there"
        assert body["itinerary_modified"] is False
        assert "flight" not in body["budget_allocation"]
        assert invoke.await_args.kwargs == {
            "user_message": "hello",
            "thread_id": "session-1",
            "user_id": "test-user",
            "request_id": "chat-request-2",
        }
        records = [
            record
            for record in caplog.records
            if record.name == chat.logger.name
            and record.getMessage() in {"chat.message.start", "chat.message.done"}
        ]
        assert [record.getMessage() for record in records] == [
            "chat.message.start",
            "chat.message.done",
        ]
        assert all(
            custom_log_record_fields(record) == _APPROVED_PLANNING_LOG_FIELDS
            for record in records
        )

    def test_successful_chat_outcome_survives_raising_log_handler(self):
        state = qualified_plan_state()
        state["messages"] = [
            HumanMessage(content="hello"),
            AIMessage(content="Hi there"),
        ]
        with (
            patch.object(
                chat,
                "invoke_chat_authenticated",
                AsyncMock(return_value=state),
            ),
            raising_log_handler(chat.logger),
        ):
            response = client.post(
                "/api/chat/message",
                headers={**AUTH, "X-Request-ID": "raising-chat-request"},
                json={"session_id": "session-1", "user_message": "hello"},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_tool_turn_marks_itinerary_modified_and_extracts_list_content(self):
        messages = [
            HumanMessage(content="change hotel"),
            AIMessage(
                content="",
                tool_calls=[{"name": "edit_itinerary", "args": {}, "id": "1"}],
            ),
            ToolMessage(name="edit_itinerary", content="{}", tool_call_id="1"),
            AIMessage(content=[{"type": "text", "text": "Hotel updated"}]),
        ]
        state = qualified_plan_state()
        state["messages"] = messages
        with patch.object(
            chat, "invoke_chat_authenticated", AsyncMock(return_value=state)
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "session-1", "user_message": "change hotel"},
            )
        assert response.status_code == 200
        assert response.json()["chat_reply"] == "Hotel updated"
        assert response.json()["itinerary_modified"] is True

    @pytest.mark.parametrize(
        "payload",
        [
            {"session_id": "", "user_message": "hello"},
            {"session_id": "s", "user_message": ""},
            {"session_id": "s", "user_message": "x" * 8193},
        ],
    )
    def test_chat_validation(self, payload):
        assert (
            client.post("/api/chat/message", headers=AUTH, json=payload).status_code
            == 422
        )

    def test_chat_timeout_maps_to_504(self):
        with (
            patch.object(
                chat,
                "invoke_chat_authenticated",
                AsyncMock(side_effect=asyncio.TimeoutError()),
            ),
            raising_log_handler(chat.logger),
        ):
            response = client.post(
                "/api/chat/message",
                headers=AUTH,
                json={"session_id": "s", "user_message": "hello"},
            )
        assert response.status_code == 504


@pytest.mark.api
class TestGeocodeRoute:
    def test_geocode_success_rounds_bias_for_cache_key(self):
        hit = {"name": "Museum", "address": "KL", "lat": 3.1, "lng": 101.7}
        with patch.object(geocode, "_geocode", return_value=hit) as lookup:
            response = client.get(
                "/api/geocode/?q=Museum&lat=3.1234&lng=101.6789", headers=AUTH
            )
        assert response.status_code == 200
        assert response.json()["found"] is True
        lookup.assert_called_once_with("Museum", "@3.12,101.68,13z")

    def test_geocode_miss_is_success_false(self):
        with patch.object(geocode, "_geocode", return_value=None):
            response = client.get("/api/geocode/?q=Unknown", headers=AUTH)
        assert response.json() == {"status": "success", "found": False}

    @pytest.mark.parametrize(
        "exception", [requests.ConnectionError(), ValueError("bad json")]
    )
    def test_geocode_provider_failure_is_error_false(self, exception):
        with patch.object(geocode, "_geocode", side_effect=exception):
            response = client.get("/api/geocode/?q=Unknown", headers=AUTH)
        assert response.status_code == 200
        assert response.json() == {"status": "error", "found": False}

    @pytest.mark.parametrize(
        "query",
        ["ab", "x" * 301],
    )
    def test_geocode_query_length_validation(self, query):
        assert (
            client.get("/api/geocode/", headers=AUTH, params={"q": query}).status_code
            == 422
        )

    @pytest.mark.parametrize(
        "params",
        [
            {"q": "Tokyo", "lat": 91},
            {"q": "Tokyo", "lat": -91},
            {"q": "Tokyo", "lng": 181},
            {"q": "Tokyo", "lng": -181},
        ],
    )
    def test_geocode_coordinate_bounds(self, params):
        assert (
            client.get("/api/geocode/", headers=AUTH, params=params).status_code == 422
        )


@pytest.mark.unit
class TestDependenciesAndErrorMapping:
    @pytest.mark.parametrize(
        "state,expected",
        [
            ({"planning_issue_codes": ["activity.country.mismatch"]}, "validation_failed"),
            ({"output_review_issue_codes": ["review.unavailable"]}, "review_unavailable"),
            ({"planning_issue_codes": ["provider.data.unavailable"]}, "provider_data_unavailable"),
            ({"planning_issue_codes": ["deadline.exhausted"]}, "deadline_exhausted"),
        ],
    )
    def test_planning_unavailable_reason_is_safe_and_bounded(self, state, expected):
        assert errors.planning_unavailable_reason(state) == expected

    @pytest.mark.asyncio
    async def test_request_id_reuses_header_or_cached_generated_value(self):
        request = Mock()
        request.state = Mock(spec=[])
        request.state.__dict__ = {}
        assert await dependencies.get_request_id(request, " trace-1 ") == "trace-1"

        request2 = Mock()
        request2.state = type("State", (), {})()
        first = await dependencies.get_request_id(request2, None)
        second = await dependencies.get_request_id(request2, None)
        assert first == second
        assert len(first) == 36

    def test_get_postgres_pool_rejects_closed_pool(self):
        with patch.object(dependencies, "connection_pool", Mock(closed=True)):
            with pytest.raises(HTTPException) as exc:
                dependencies.get_postgres_pool()
        assert exc.value.status_code == 503

    @pytest.mark.parametrize(
        "exception,status_code,detail",
        [
            (
                CurrencyRateUnavailableError("no provider"),
                503,
                "Exchange-rate service temporarily unavailable",
            ),
            (asyncio.TimeoutError(), 504, "Workflow execution timed out"),
            (httpx.ReadTimeout("slow"), 504, "Upstream HTTP call timed out"),
            (httpx.ConnectError("offline"), 502, "Upstream HTTP service error"),
            (RuntimeError("database password secret"), 500, "unexpected error"),
        ],
    )
    def test_exception_mapping_is_specific_and_does_not_leak_raw_messages(
        self, exception, status_code, detail
    ):
        mapped = errors.map_exception_to_http(exception)
        assert mapped.status_code == status_code
        assert detail.lower() in mapped.detail.lower()
        assert "password" not in mapped.detail.lower()
