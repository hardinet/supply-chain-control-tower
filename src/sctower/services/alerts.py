"""Stockout / overstock alerting per store.

The public Rossmann dataset contains sales but no inventory levels, so on-hand
positions are *simulated* deterministically from each store's demand-based policy
purely to demonstrate the alerting logic end to end. This is clearly flagged in
the UI and the README; the policy math itself is driven entirely by real demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sctower.config import Settings, get_settings
from sctower.domain.inventory import InventoryPolicy, days_of_cover
from sctower.logging import get_logger
from sctower.services.pipeline import build_series, recent_demand_stats

logger = get_logger(__name__)

POLICY_COLUMNS = [
    "store",
    "avg_daily_demand",
    "demand_std",
    "safety_stock",
    "reorder_point",
    "order_quantity",
    "overstock_threshold",
]


def top_stores(curated: pd.DataFrame, n: int = 50) -> list[int]:
    """Return the ``n`` highest-volume stores (stable, by total sales)."""
    totals = curated.groupby("store")["sales"].sum().sort_values(ascending=False)
    return [int(s) for s in totals.head(n).index]


def compute_store_policies(
    curated: pd.DataFrame,
    stores: list[int],
    *,
    settings: Settings | None = None,
    window: int = 90,
) -> pd.DataFrame:
    """Build an inventory policy per store from its recent demand statistics."""
    settings = settings or get_settings()
    rows: list[dict[str, float]] = []
    for store in stores:
        series = build_series(curated, store)
        mean, std = recent_demand_stats(series, window=window)
        if mean <= 0:
            continue
        policy = InventoryPolicy.from_demand(
            avg_daily_demand=mean,
            demand_std=std,
            lead_time_days=settings.lead_time_days,
            service_level=settings.service_level,
        )
        rows.append(
            {
                "store": float(store),
                "avg_daily_demand": policy.avg_daily_demand,
                "demand_std": policy.demand_std,
                "safety_stock": policy.safety_stock,
                "reorder_point": policy.reorder_point,
                "order_quantity": policy.order_quantity,
                "overstock_threshold": policy.overstock_threshold,
            }
        )
    return pd.DataFrame(rows, columns=POLICY_COLUMNS)


def _policy_from_row(row: pd.Series) -> InventoryPolicy:
    return InventoryPolicy(
        avg_daily_demand=float(row["avg_daily_demand"]),
        demand_std=float(row["demand_std"]),
        lead_time_days=0.0,  # not needed for classification
        service_level=0.95,
        safety_stock=float(row["safety_stock"]),
        reorder_point=float(row["reorder_point"]),
        order_quantity=float(row["order_quantity"]),
    )


def simulate_positions(policies: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Attach a simulated on-hand position, status and days of cover per store."""
    if policies.empty:
        return policies.assign(on_hand=[], days_of_cover=[], status=[])
    rng = np.random.default_rng(seed)
    rop = policies["reorder_point"].to_numpy()
    # Most positions hover around the reorder point; a minority is pushed high to
    # surface overstock cases. Deterministic given the seed.
    base = rng.normal(loc=rop, scale=0.6 * np.maximum(rop, 1.0))
    overstock_mask = rng.random(len(policies)) < 0.15
    base = np.where(overstock_mask, base + 3.0 * policies["order_quantity"].to_numpy(), base)
    on_hand = np.clip(base, 0.0, None)

    out = policies.copy()
    out["on_hand"] = on_hand
    out["days_of_cover"] = [
        days_of_cover(float(oh), float(mu))
        for oh, mu in zip(on_hand, policies["avg_daily_demand"], strict=True)
    ]
    out["status"] = [
        _policy_from_row(row).classify(float(row["on_hand"])).value for _, row in out.iterrows()
    ]
    return out


def build_alerts(
    curated: pd.DataFrame,
    *,
    settings: Settings | None = None,
    stores: list[int] | None = None,
    n_stores: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """End-to-end: select stores, size policies, simulate positions and classify."""
    settings = settings or get_settings()
    selected = stores if stores is not None else top_stores(curated, n_stores)
    policies = compute_store_policies(curated, selected, settings=settings)
    alerts = simulate_positions(policies, seed=seed)
    logger.info("alerts_built", stores=len(alerts))
    return alerts


def alerts_summary(alerts: pd.DataFrame) -> dict[str, int]:
    """Count stores by status (stockout / at_risk / healthy / overstock)."""
    if alerts.empty or "status" not in alerts:
        return {}
    counts = alerts["status"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}
