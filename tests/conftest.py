"""Shared pytest fixtures.

Synthetic data here is *test scaffolding* (small, seeded, used to exercise the
code paths), not analytical input. The real analysis always runs on the curated
Rossmann dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_series() -> pd.DataFrame:
    """A 220-day daily demand series with weekly seasonality and a mild trend."""
    rng = np.random.default_rng(0)
    n = 220
    ds = pd.date_range("2014-01-01", periods=n, freq="D")
    weekly = 12.0 * np.sin(2 * np.pi * ds.dayofweek.to_numpy() / 7.0)
    trend = 0.08 * np.arange(n)
    y = np.clip(120.0 + weekly + trend + rng.normal(0, 4, n), 0, None)
    promo = (rng.random(n) < 0.3).astype(int)
    return pd.DataFrame(
        {
            "ds": ds,
            "y": y,
            "promo": promo,
            "school_holiday": (rng.random(n) < 0.15).astype(int),
            "state_holiday": (rng.random(n) < 0.05).astype(int),
        }
    )


@pytest.fixture
def curated() -> pd.DataFrame:
    """A small curated table: 3 stores x 150 days, matching the real schema."""
    rng = np.random.default_rng(7)
    days = 150
    ds = pd.date_range("2014-01-01", periods=days, freq="D")
    dow = ds.dayofweek.to_numpy()
    weekday_factor = np.array([1.0, 0.95, 0.95, 1.0, 1.1, 1.2, 0.0])
    frames = []
    for store in (1, 2, 3):
        base = 1000.0 * store
        open_flag = (weekday_factor[dow] > 0).astype(int)
        sales = np.where(
            open_flag == 1,
            base * weekday_factor[dow] * rng.normal(1.0, 0.05, days),
            0.0,
        )
        frames.append(
            pd.DataFrame(
                {
                    "store": store,
                    "ds": ds,
                    "day_of_week": dow + 1,
                    "sales": np.clip(sales, 0, None),
                    "customers": np.clip(sales / 10.0, 0, None),
                    "open": open_flag,
                    "promo": (rng.random(days) < 0.3).astype(int),
                    "state_holiday": (rng.random(days) < 0.04).astype(int),
                    "school_holiday": (rng.random(days) < 0.15).astype(int),
                    "store_type": "a",
                    "assortment": "c",
                    "competition_distance": 500.0 * store,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
