"""Inventory policy mathematics.

Implements the classic continuous-review (s, Q) policy used to translate a demand
forecast into operational levers:

- **safety stock**   buffer that absorbs demand variability over the lead time,
  sized to a target cycle service level.
- **reorder point**  the on-hand level that triggers a replenishment order.
- **economic order quantity (EOQ)**  the order size that minimizes the sum of
  ordering and holding costs.

The inverse normal quantile (the ``z`` factor) comes from the standard library
(:class:`statistics.NormalDist`), so this module stays dependency-free and pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist


def z_from_service_level(service_level: float) -> float:
    """Return the standard-normal quantile (z) for a cycle service level.

    A 95% service level maps to z ~= 1.645.
    """
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be in the open interval (0, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(
    demand_std: float,
    lead_time_days: float,
    service_level: float,
    *,
    lead_time_std: float = 0.0,
    avg_daily_demand: float = 0.0,
) -> float:
    """Compute safety stock for a target cycle service level.

    Uses the standard formula that combines demand and (optionally) lead-time
    variability::

        SS = z * sqrt(L * sigma_d^2 + mu_d^2 * sigma_L^2)

    When ``lead_time_std`` is 0 this reduces to ``z * sigma_d * sqrt(L)``.

    Args:
        demand_std: standard deviation of daily demand (sigma_d).
        lead_time_days: average lead time in days (L).
        service_level: target cycle service level in (0, 1).
        lead_time_std: standard deviation of the lead time (sigma_L), optional.
        avg_daily_demand: mean daily demand (mu_d), required only when
            ``lead_time_std`` > 0.

    Returns:
        Safety stock in demand units (non-negative).
    """
    if demand_std < 0 or lead_time_days <= 0 or lead_time_std < 0:
        raise ValueError("demand_std, lead_time_std must be >= 0 and lead_time_days > 0")
    z = z_from_service_level(service_level)
    variance = lead_time_days * demand_std**2 + (avg_daily_demand**2) * (lead_time_std**2)
    return float(z * math.sqrt(variance))


def reorder_point(
    avg_daily_demand: float,
    lead_time_days: float,
    safety_stock_units: float,
) -> float:
    """Reorder point = expected demand over the lead time + safety stock."""
    if avg_daily_demand < 0 or lead_time_days <= 0 or safety_stock_units < 0:
        raise ValueError("inputs must be non-negative and lead_time_days > 0")
    return float(avg_daily_demand * lead_time_days + safety_stock_units)


def economic_order_quantity(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
) -> float:
    """Economic Order Quantity: sqrt(2 * D * S / H)."""
    if annual_demand < 0 or ordering_cost < 0 or holding_cost_per_unit <= 0:
        raise ValueError("annual_demand, ordering_cost >= 0 and holding_cost > 0 required")
    return float(math.sqrt(2.0 * annual_demand * ordering_cost / holding_cost_per_unit))


def days_of_cover(on_hand: float, avg_daily_demand: float) -> float:
    """How many days of demand the current on-hand stock covers.

    Returns ``inf`` when there is no demand (cannot run out).
    """
    if on_hand < 0:
        raise ValueError("on_hand must be >= 0")
    if avg_daily_demand <= 0:
        return float("inf")
    return float(on_hand / avg_daily_demand)


class StockStatus(StrEnum):
    """Operational status of a stock position relative to its policy."""

    STOCKOUT = "stockout"
    AT_RISK = "at_risk"
    HEALTHY = "healthy"
    OVERSTOCK = "overstock"


@dataclass(frozen=True, slots=True)
class InventoryPolicy:
    """A continuous-review (s, Q) policy derived from a demand forecast."""

    avg_daily_demand: float
    demand_std: float
    lead_time_days: float
    service_level: float
    safety_stock: float
    reorder_point: float
    order_quantity: float

    @classmethod
    def from_demand(
        cls,
        *,
        avg_daily_demand: float,
        demand_std: float,
        lead_time_days: float,
        service_level: float,
        ordering_cost: float = 50.0,
        holding_cost_per_unit: float = 1.0,
        lead_time_std: float = 0.0,
    ) -> InventoryPolicy:
        """Build a policy from demand statistics and cost parameters."""
        ss = safety_stock(
            demand_std,
            lead_time_days,
            service_level,
            lead_time_std=lead_time_std,
            avg_daily_demand=avg_daily_demand,
        )
        rop = reorder_point(avg_daily_demand, lead_time_days, ss)
        eoq = economic_order_quantity(
            annual_demand=avg_daily_demand * 365.0,
            ordering_cost=ordering_cost,
            holding_cost_per_unit=holding_cost_per_unit,
        )
        return cls(
            avg_daily_demand=avg_daily_demand,
            demand_std=demand_std,
            lead_time_days=lead_time_days,
            service_level=service_level,
            safety_stock=ss,
            reorder_point=rop,
            order_quantity=eoq,
        )

    @property
    def overstock_threshold(self) -> float:
        """Upper bound above which stock is considered excess.

        Defined as the order-up-to level plus one extra order cycle of cover,
        i.e. ``reorder_point + 2 * order_quantity``.
        """
        return self.reorder_point + 2.0 * self.order_quantity

    def classify(self, on_hand: float) -> StockStatus:
        """Classify an on-hand position against this policy."""
        if on_hand < 0:
            raise ValueError("on_hand must be >= 0")
        if on_hand <= self.safety_stock:
            return StockStatus.STOCKOUT if on_hand <= 0 else StockStatus.AT_RISK
        if on_hand <= self.reorder_point:
            return StockStatus.AT_RISK
        if on_hand > self.overstock_threshold:
            return StockStatus.OVERSTOCK
        return StockStatus.HEALTHY
