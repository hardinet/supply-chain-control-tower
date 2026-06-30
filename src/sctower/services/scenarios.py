"""What-if scenario simulation on top of a forecast and an inventory policy.

Two operational levers are exposed:

- a **demand multiplier** (e.g. +20% => 1.2) that scales the forecast and the
  demand statistics feeding the policy;
- a **lead-time delta** (in days) that shifts replenishment responsiveness.

Each scenario recomputes the inventory policy from first principles so the impact
on safety stock and reorder point is explicit and traceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sctower.domain.forecasting import ForecastResult
from sctower.domain.inventory import InventoryPolicy


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named what-if configuration."""

    name: str
    demand_multiplier: float = 1.0
    lead_time_delta_days: float = 0.0

    def __post_init__(self) -> None:
        if self.demand_multiplier <= 0:
            raise ValueError("demand_multiplier must be > 0")


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """Result of applying a scenario: scaled forecast and recomputed policy."""

    scenario: Scenario
    forecast: ForecastResult
    policy: InventoryPolicy


def _scale_forecast(result: ForecastResult, factor: float) -> ForecastResult:
    """Return a copy of ``result`` with the point forecast and bounds scaled."""
    return ForecastResult(
        model_name=result.model_name,
        ds=result.ds,
        yhat=result.yhat * factor,
        yhat_lower=None if result.yhat_lower is None else result.yhat_lower * factor,
        yhat_upper=None if result.yhat_upper is None else result.yhat_upper * factor,
    )


def apply_scenario(
    scenario: Scenario,
    *,
    base_forecast: ForecastResult,
    avg_daily_demand: float,
    demand_std: float,
    base_lead_time_days: float,
    service_level: float,
    ordering_cost: float = 50.0,
    holding_cost_per_unit: float = 1.0,
) -> ScenarioOutcome:
    """Apply a scenario, returning the scaled forecast and recomputed policy.

    The demand multiplier scales both the mean and the standard deviation of
    demand (a multiplicative shock preserves the coefficient of variation), and
    the lead time is shifted by the scenario delta (floored at 1 day).
    """
    factor = scenario.demand_multiplier
    lead_time = max(1.0, base_lead_time_days + scenario.lead_time_delta_days)
    policy = InventoryPolicy.from_demand(
        avg_daily_demand=avg_daily_demand * factor,
        demand_std=demand_std * factor,
        lead_time_days=lead_time,
        service_level=service_level,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=holding_cost_per_unit,
    )
    return ScenarioOutcome(
        scenario=scenario,
        forecast=_scale_forecast(base_forecast, factor),
        policy=policy,
    )


def projected_demand(outcome: ScenarioOutcome) -> float:
    """Total demand projected over the forecast horizon for a scenario."""
    return float(np.sum(outcome.forecast.yhat))
