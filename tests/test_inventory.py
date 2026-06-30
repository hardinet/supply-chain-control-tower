"""Tests for inventory policy mathematics."""

from __future__ import annotations

import math

import pytest

from sctower.domain import inventory
from sctower.domain.inventory import InventoryPolicy, StockStatus


def test_z_from_service_level() -> None:
    assert inventory.z_from_service_level(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert inventory.z_from_service_level(0.5) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.2])
def test_z_from_service_level_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        inventory.z_from_service_level(bad)


def test_safety_stock_demand_only() -> None:
    # z(0.95)=1.6449, sigma=10, L=4 -> 1.6449 * 10 * 2 = 32.9
    ss = inventory.safety_stock(10.0, 4.0, 0.95)
    assert ss == pytest.approx(1.6449 * 10 * math.sqrt(4), abs=1e-2)


def test_safety_stock_with_lead_time_variability_is_larger() -> None:
    base = inventory.safety_stock(10.0, 7.0, 0.95)
    with_lt = inventory.safety_stock(10.0, 7.0, 0.95, lead_time_std=2.0, avg_daily_demand=50.0)
    assert with_lt > base


def test_safety_stock_validation() -> None:
    with pytest.raises(ValueError):
        inventory.safety_stock(-1.0, 5.0, 0.95)
    with pytest.raises(ValueError):
        inventory.safety_stock(10.0, 0.0, 0.95)


def test_reorder_point() -> None:
    assert inventory.reorder_point(50.0, 7.0, 30.0) == pytest.approx(380.0)
    with pytest.raises(ValueError):
        inventory.reorder_point(-1.0, 7.0, 30.0)


def test_eoq_known_value() -> None:
    # sqrt(2*D*S/H) with D=3650, S=50, H=2 -> sqrt(182500) ~ 427.2
    assert inventory.economic_order_quantity(3650.0, 50.0, 2.0) == pytest.approx(
        math.sqrt(2 * 3650 * 50 / 2)
    )
    with pytest.raises(ValueError):
        inventory.economic_order_quantity(100.0, 10.0, 0.0)


def test_days_of_cover() -> None:
    assert inventory.days_of_cover(100.0, 20.0) == pytest.approx(5.0)
    assert inventory.days_of_cover(100.0, 0.0) == math.inf
    with pytest.raises(ValueError):
        inventory.days_of_cover(-1.0, 20.0)


def test_policy_from_demand_and_classification() -> None:
    policy = InventoryPolicy.from_demand(
        avg_daily_demand=100.0,
        demand_std=20.0,
        lead_time_days=7.0,
        service_level=0.95,
    )
    assert policy.safety_stock > 0
    assert policy.reorder_point > policy.safety_stock
    assert policy.order_quantity > 0
    assert policy.overstock_threshold > policy.reorder_point

    assert policy.classify(0.0) is StockStatus.STOCKOUT
    assert policy.classify(policy.safety_stock * 0.5) is StockStatus.AT_RISK
    assert policy.classify(policy.reorder_point - 1) is StockStatus.AT_RISK
    mid = (policy.reorder_point + policy.overstock_threshold) / 2
    assert policy.classify(mid) is StockStatus.HEALTHY
    assert policy.classify(policy.overstock_threshold + 1) is StockStatus.OVERSTOCK


def test_classify_rejects_negative() -> None:
    policy = InventoryPolicy.from_demand(
        avg_daily_demand=10.0, demand_std=2.0, lead_time_days=3.0, service_level=0.9
    )
    with pytest.raises(ValueError):
        policy.classify(-5.0)
