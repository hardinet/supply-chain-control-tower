"""Tests for the series-building pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from sctower.domain.forecasting.base import ForecastResult
from sctower.services import pipeline


def test_list_stores(curated: pd.DataFrame) -> None:
    assert pipeline.list_stores(curated) == [1, 2, 3]


def test_build_total_series_sums_stores(curated: pd.DataFrame) -> None:
    series = pipeline.build_series(curated, store=None)
    assert set(series.columns) >= {"ds", "y", "promo", "school_holiday", "state_holiday"}
    assert series["ds"].is_unique
    # Total on a given day equals the sum across stores.
    day = curated["ds"].iloc[0]
    expected = curated.loc[curated["ds"] == day, "sales"].sum()
    assert series.loc[series["ds"] == day, "y"].iloc[0] == pytest.approx(expected)


def test_build_store_series(curated: pd.DataFrame) -> None:
    series = pipeline.build_series(curated, store=2)
    assert (series["y"] >= 0).all()
    assert len(series) == curated["ds"].nunique()


def test_build_series_unknown_store(curated: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown store"):
        pipeline.build_series(curated, store=999)


def test_build_series_fills_gaps(curated: pd.DataFrame) -> None:
    # Drop a day for store 1 and check it is reinstated with zero demand.
    gap_day = curated["ds"].unique()[10]
    pruned = curated.drop(curated[(curated["store"] == 1) & (curated["ds"] == gap_day)].index)
    series = pipeline.build_series(pruned, store=1, fill_gaps=True)
    assert pd.Timestamp(gap_day) in set(series["ds"])


def test_recent_demand_stats_open_only(curated: pd.DataFrame) -> None:
    series = pipeline.build_series(curated, store=1)
    mean, std = pipeline.recent_demand_stats(series, window=90, open_only=True)
    assert mean > 0
    assert std >= 0


def test_fit_and_forecast_returns_result(curated: pd.DataFrame) -> None:
    series = pipeline.build_series(curated, store=1)
    result = pipeline.fit_and_forecast(series, "seasonal_naive", horizon=10)
    assert isinstance(result, ForecastResult)
    assert result.horizon == 10
