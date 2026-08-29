from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from app.agents.state import AgentState
from app.tools import currency
from app.tools.currency import BudgetAdjustment
from app.tools.trip_details import _valid_date, update_trip_details


@pytest.fixture(autouse=True)
def clear_currency_caches():
    currency._country_cache.clear()
    currency._rate_cache.clear()
    yield
    currency._country_cache.clear()
    currency._rate_cache.clear()


@pytest.mark.unit
class TestCountryAndCurrencyResolution:
    @pytest.mark.parametrize(
        "country,expected",
        [
            ("Malaysia", "MYR"),
            ("Japan", "JPY"),
            ("United States", "USD"),
            ("Taiwan", "TWD"),
            (" malaysia ", "MYR"),
        ],
    )
    def test_known_country_currency_codes(self, country, expected):
        assert currency.get_currency_code(country) == expected

    @pytest.mark.parametrize("country", ["", "   ", None])
    def test_empty_country_falls_back_to_usd(self, country):
        assert currency.get_currency_code(country) == "USD"

    def test_unknown_country_falls_back_and_is_cached(self):
        with patch.object(
            currency.pycountry.countries, "search_fuzzy", side_effect=LookupError
        ):
            assert currency.get_currency_code("Atlantis") == "USD"
            assert currency.get_currency_code("ATLANTIS") == "USD"

    def test_same_currency_never_calls_provider(self):
        with patch.object(currency._session, "get") as get:
            assert currency.get_currency_rate("myr", " MYR ") == 1.0
        get.assert_not_called()

    def test_primary_provider_success_and_reciprocal_cache(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": {"JPY": 32.5}}
        with patch.object(currency._session, "get", return_value=response) as get:
            assert currency.get_currency_rate("MYR", "JPY") == 32.5
            assert currency.get_currency_rate("JPY", "MYR") == pytest.approx(1 / 32.5)
        assert get.call_count == 1
        assert get.call_args.kwargs["params"]["apikey"] == "test-currency-key"

    def test_primary_client_error_switches_to_fallback_without_retry(self):
        response = Mock()
        err = requests.HTTPError("unsupported")
        err.response = Mock(status_code=422)
        response.raise_for_status.side_effect = err
        with (
            patch.object(currency._session, "get", return_value=response) as get,
            patch.object(
                currency, "_fetch_rate_open_erapi", return_value=6.75
            ) as fallback,
            patch.object(currency.time, "sleep") as sleep,
        ):
            assert currency.get_currency_rate("MYR", "TWD") == 6.75
        assert get.call_count == 1
        fallback.assert_called_once_with("MYR", "TWD")
        sleep.assert_not_called()

    def test_all_providers_failure_returns_none(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.ConnectionError("offline")
        with (
            patch.object(currency._session, "get", return_value=response),
            patch.object(currency, "_fetch_rate_open_erapi", return_value=None),
            patch.object(currency.time, "sleep"),
        ):
            assert currency.get_currency_rate("MYR", "JPY", max_retries=2) is None

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"result": "success", "rates": {"TWD": 6.8}}, 6.8),
            ({"result": "error", "rates": {"TWD": 6.8}}, None),
            ({"result": "success", "rates": {}}, None),
        ],
    )
    def test_fallback_provider_payloads(self, payload, expected):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        with patch.object(currency._session, "get", return_value=response):
            assert currency._fetch_rate_open_erapi("MYR", "TWD") == expected


@pytest.mark.unit
class TestBudgetCalculations:
    @pytest.mark.parametrize(
        "amount,rate,expected",
        [(100, 2, 200), (10.555, 1.2, 12.67), (0, 5, 0), (-5, 2, -10)],
    )
    def test_budget_calculation(self, amount, rate, expected):
        assert currency.budget_calculation(amount, rate) == expected

    def test_budget_calculation_propagates_missing_rate(self):
        assert currency.budget_calculation(100, None) is None

    def test_currency_pipeline_preserves_base_and_converts_destination(self):
        state = AgentState(
            origin_country="Malaysia", country="Japan", total_base_budget=1000
        )
        with patch.object(currency, "get_currency_rate", return_value=32.5):
            result = currency.currency_pipeline(state)
        assert result["base_currency_code"] == "MYR"
        assert result["dest_currency_code"] == "JPY"
        assert result["total_convert_budget"] == 32500
        assert result["exchange_rate"] == {"MYR": 1.0, "JPY": 32.5}
        assert result["currency_fetched_at"].endswith("Z")

    def test_currency_pipeline_rejects_provider_failure_instead_of_false_one_to_one(
        self,
    ):
        state = AgentState(
            origin_country="Malaysia", country="Japan", total_base_budget=1000
        )
        with patch.object(currency, "get_currency_rate", return_value=None):
            with pytest.raises(currency.CurrencyRateUnavailableError, match="MYR->JPY"):
                currency.currency_pipeline(state)

    def test_chat_currency_conversion_success(self):
        with patch.object(currency, "get_currency_rate", return_value=3.25):
            result = currency.chat_currency_conversion.func("Malaysia", "Japan", 100)
        assert result["from"] == "MYR"
        assert result["to"] == "JPY"
        assert result["converted_amount"] == 325

    def test_chat_currency_conversion_failure_is_explicit(self):
        with patch.object(currency, "get_currency_rate", return_value=None):
            result = currency.chat_currency_conversion.func("Malaysia", "Japan", 100)
        assert "error" in result

    def test_update_total_budget_scales_every_category(self):
        state = AgentState(
            total_convert_budget=1000,
            budget_allocation={"food": 200, "activity": 300, "accommodation": 500},
        )
        result = currency.update_total_budget.func(2000, state)
        assert result["total_convert_budget"] == 2000
        assert result["budget_allocation"] == {
            "food": 400,
            "activity": 600,
            "accommodation": 1000,
        }

    def test_update_total_budget_handles_uninitialised_allocation(self):
        state = AgentState(total_convert_budget=0, budget_allocation={})
        result = currency.update_total_budget.func(500, state)
        assert result == {"total_convert_budget": 500, "budget_allocation": {}}

    @pytest.mark.regression
    @pytest.mark.parametrize("new_budget", [0, -1, -999])
    def test_update_total_budget_rejects_non_positive_values(self, new_budget):
        state = AgentState(
            total_convert_budget=100,
            budget_allocation={"food": 100},
        )
        result = currency.update_total_budget.func(new_budget, state)
        assert result == {"error": "new_total_budget must be greater than 0."}

    def test_category_increase_rebalances_other_categories(self):
        state = AgentState(
            total_convert_budget=1000,
            budget_allocation={
                "transportation": 250,
                "accommodation": 350,
                "food": 150,
                "activity": 150,
                "shopping": 50,
                "emergency_fund": 50,
            },
        )
        adjustment = [BudgetAdjustment(category="food", add_delete_amount=100)]
        result = currency.update_budget_category.func(adjustment, state)
        assert result["budget_allocation"]["food"] == 250
        assert sum(result["budget_allocation"].values()) == pytest.approx(
            1000, abs=0.05
        )

    def test_vague_modifier_uses_ten_percent_of_existing_category(self):
        state = AgentState(total_convert_budget=1000, budget_allocation={"food": 200})
        adjustment = [BudgetAdjustment(category="food", modifier="small_increase")]
        result = currency.update_budget_category.func(adjustment, state)
        assert result["budget_allocation"]["food"] == 220

    def test_legacy_flight_alias_is_removed(self):
        state = AgentState(
            total_convert_budget=1000,
            budget_allocation={"flight": 250, "transportation": 250, "food": 750},
        )
        result = currency.update_budget_category.func([], state)
        assert "flight" not in result["budget_allocation"]

    @pytest.mark.regression
    def test_large_decrease_rebalances_only_the_amount_actually_removed(self):
        state = AgentState(
            total_convert_budget=100,
            budget_allocation={
                "food": 10,
                "transportation": 20,
                "accommodation": 20,
                "activity": 20,
                "shopping": 15,
                "emergency_fund": 15,
            },
        )
        result = currency.update_budget_category.func(
            [BudgetAdjustment(category="food", add_delete_amount=-100)], state
        )
        allocation = result["budget_allocation"]
        assert allocation["food"] == 0
        assert sum(allocation.values()) == pytest.approx(100, abs=0.05)
        assert "warnings" not in result

    def test_excessive_increase_is_trimmed_to_total_budget(self):
        state = AgentState(total_convert_budget=100, budget_allocation={"food": 100})
        result = currency.update_budget_category.func(
            [BudgetAdjustment(category="food", add_delete_amount=1000)], state
        )
        assert sum(result["budget_allocation"].values()) == pytest.approx(100, abs=0.05)
        assert result["warnings"]


@pytest.mark.unit
class TestTripDetailUpdates:
    @pytest.mark.parametrize("value", ["2026-01-01", "2024-02-29", "1999-12-31"])
    def test_valid_date(self, value):
        assert _valid_date(value)

    @pytest.mark.parametrize("value", ["", None, "2026-02-30", "01/01/2026"])
    def test_invalid_date(self, value):
        assert not _valid_date(value)

    def test_multiple_updates_are_cleaned_and_returned(self):
        result = update_trip_details.func(
            state=AgentState(),
            num_people=4,
            start_date="2026-10-01",
            end_date="2026-10-05",
            country="  Japan ",
            city=[" Tokyo ", "", " Kyoto"],
        )
        assert result["status"] == "success"
        assert result["updates"] == {
            "num_people": 4,
            "start_date": "2026-10-01",
            "end_date": "2026-10-05",
            "country": "Japan",
            "city": ["Tokyo", "Kyoto"],
        }

    @pytest.mark.parametrize(
        "kwargs,error_fragment",
        [
            ({"num_people": 0}, "at least 1"),
            ({"start_date": "bad"}, "not YYYY-MM-DD"),
            ({"end_date": "bad"}, "not YYYY-MM-DD"),
            ({"start_date": "2026-10-05", "end_date": "2026-10-01"}, "after"),
            ({}, "No trip details"),
        ],
    )
    def test_invalid_updates_return_actionable_errors(self, kwargs, error_fragment):
        result = update_trip_details.func(state=AgentState(), **kwargs)
        assert error_fragment in result["error"]

    def test_empty_city_list_is_a_valid_clear_operation(self):
        result = update_trip_details.func(state=AgentState(), city=[])
        assert result["updates"] == {"city": []}

    @pytest.mark.parametrize(
        "updates",
        [
            {"start_date": "2026-10-06"},
            {"end_date": "2026-09-30"},
        ],
    )
    def test_single_boundary_update_validates_against_current_trip(self, updates):
        state = AgentState(start_date="2026-10-01", end_date="2026-10-05")

        result = update_trip_details.func(state=state, **updates)

        assert result == {"error": "end_date must be after start_date."}
