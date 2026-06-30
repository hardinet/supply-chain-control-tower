"""Tests for the stock alerting service."""

from __future__ import annotations

import pandas as pd

from sctower.domain.inventory import StockStatus
from sctower.services import alerts
from sctower.services.alerts import POLICY_COLUMNS


def test_top_stores_orders_by_volume(curated: pd.DataFrame) -> None:
    # Store 3 has the highest base demand, then 2, then 1.
    assert alerts.top_stores(curated, n=2) == [3, 2]


def test_compute_store_policies_schema(curated: pd.DataFrame) -> None:
    policies = alerts.compute_store_policies(curated, [1, 2, 3])
    assert list(policies.columns) == POLICY_COLUMNS
    assert len(policies) == 3
    assert (policies["reorder_point"] >= policies["safety_stock"]).all()


def test_simulate_positions_is_deterministic(curated: pd.DataFrame) -> None:
    policies = alerts.compute_store_policies(curated, [1, 2, 3])
    a = alerts.simulate_positions(policies, seed=42)
    b = alerts.simulate_positions(policies, seed=42)
    pd.testing.assert_series_equal(a["on_hand"], b["on_hand"])
    valid = {s.value for s in StockStatus}
    assert set(a["status"]).issubset(valid)


def test_simulate_positions_empty() -> None:
    empty = pd.DataFrame(columns=POLICY_COLUMNS)
    out = alerts.simulate_positions(empty)
    assert out.empty


def test_build_alerts_end_to_end(curated: pd.DataFrame) -> None:
    result = alerts.build_alerts(curated, n_stores=3)
    assert "status" in result.columns
    assert "days_of_cover" in result.columns
    summary = alerts.alerts_summary(result)
    assert sum(summary.values()) == len(result)


def test_alerts_summary_empty() -> None:
    assert alerts.alerts_summary(pd.DataFrame()) == {}
