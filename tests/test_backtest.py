"""Tests for the rolling-origin backtester."""

from __future__ import annotations

import pandas as pd
import pytest

from sctower.services import backtest


def test_rolling_origin_splits_shape() -> None:
    splits = backtest.rolling_origin_splits(100, folds=3, horizon=10)
    assert splits == [(70, 80), (80, 90), (90, 100)]


def test_rolling_origin_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        backtest.rolling_origin_splits(15, folds=3, horizon=10)


def test_rolling_origin_invalid_args() -> None:
    with pytest.raises(ValueError):
        backtest.rolling_origin_splits(100, folds=0, horizon=10)


def test_backtest_model_pools_folds(daily_series: pd.DataFrame) -> None:
    result = backtest.backtest_model(daily_series, "seasonal_naive", horizon=14, folds=2)
    assert result.model_name == "seasonal_naive"
    assert result.n_folds == 2
    assert result.metrics.n == 28  # 2 folds x 14
    row = result.to_row()
    assert row["model"] == "seasonal_naive"
    assert "wape" in row


def test_compare_models_sorted_by_wape(daily_series: pd.DataFrame) -> None:
    table = backtest.compare_models(daily_series, ["seasonal_naive", "gbm"], horizon=14, folds=2)
    assert list(table["model"])  # non-empty
    assert table["wape"].is_monotonic_increasing


def test_compare_models_all_failing_raises(daily_series: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="no model"):
        backtest.compare_models(daily_series, ["unknown_model"], horizon=14, folds=2)
