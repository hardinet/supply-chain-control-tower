"""Tests for what-if scenario simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sctower.domain.forecasting import build_model
from sctower.services import scenarios
from sctower.services.scenarios import Scenario, apply_scenario, projected_demand


@pytest.fixture
def base_forecast(daily_series: pd.DataFrame) -> object:
    model = build_model("seasonal_naive").fit(daily_series)
    return model.predict(21)


def test_scenario_rejects_non_positive_multiplier() -> None:
    with pytest.raises(ValueError, match="demand_multiplier"):
        Scenario("bad", demand_multiplier=0.0)


def test_apply_scenario_scales_forecast(base_forecast: object) -> None:
    outcome = apply_scenario(
        Scenario("up", demand_multiplier=1.2),
        base_forecast=base_forecast,  # type: ignore[arg-type]
        avg_daily_demand=100.0,
        demand_std=20.0,
        base_lead_time_days=7.0,
        service_level=0.95,
    )
    base_total = float(np.sum(base_forecast.yhat))  # type: ignore[attr-defined]
    assert projected_demand(outcome) == pytest.approx(base_total * 1.2)


def test_apply_scenario_recomputes_policy(base_forecast: object) -> None:
    base = apply_scenario(
        Scenario("base"),
        base_forecast=base_forecast,  # type: ignore[arg-type]
        avg_daily_demand=100.0,
        demand_std=20.0,
        base_lead_time_days=7.0,
        service_level=0.95,
    )
    longer = apply_scenario(
        Scenario("slow", lead_time_delta_days=7.0),
        base_forecast=base_forecast,  # type: ignore[arg-type]
        avg_daily_demand=100.0,
        demand_std=20.0,
        base_lead_time_days=7.0,
        service_level=0.95,
    )
    # A longer lead time increases both safety stock and reorder point.
    assert longer.policy.safety_stock > base.policy.safety_stock
    assert longer.policy.reorder_point > base.policy.reorder_point


def test_lead_time_floored_at_one(base_forecast: object) -> None:
    outcome = apply_scenario(
        Scenario("fast", lead_time_delta_days=-100.0),
        base_forecast=base_forecast,  # type: ignore[arg-type]
        avg_daily_demand=100.0,
        demand_std=20.0,
        base_lead_time_days=7.0,
        service_level=0.95,
    )
    assert outcome.policy.lead_time_days == pytest.approx(1.0)


def test_projected_demand_is_sum(base_forecast: object) -> None:
    outcome = scenarios.apply_scenario(
        Scenario("flat"),
        base_forecast=base_forecast,  # type: ignore[arg-type]
        avg_daily_demand=50.0,
        demand_std=10.0,
        base_lead_time_days=5.0,
        service_level=0.9,
    )
    assert projected_demand(outcome) == pytest.approx(float(np.sum(outcome.forecast.yhat)))
