"""Tests for forecast accuracy metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sctower.domain import metrics


def test_perfect_forecast_is_zero_error() -> None:
    y = [10.0, 20.0, 30.0]
    assert metrics.mae(y, y) == 0.0
    assert metrics.rmse(y, y) == 0.0
    assert metrics.mape(y, y) == 0.0
    assert metrics.wape(y, y) == 0.0
    assert metrics.bias(y, y) == 0.0


def test_mae_and_rmse_known_values() -> None:
    y_true = [0.0, 0.0, 0.0]
    y_pred = [1.0, 2.0, 2.0]
    assert metrics.mae(y_true, y_pred) == pytest.approx(5 / 3)
    assert metrics.rmse(y_true, y_pred) == pytest.approx(math.sqrt(9 / 3))


def test_mape_excludes_zero_actuals() -> None:
    # Only the non-zero actual contributes; 10% error there.
    assert metrics.mape([0.0, 100.0], [5.0, 110.0]) == pytest.approx(10.0)


def test_mape_all_zero_actuals_is_nan() -> None:
    assert math.isnan(metrics.mape([0.0, 0.0], [1.0, 2.0]))


def test_wape_is_volume_weighted() -> None:
    assert metrics.wape([100.0, 100.0], [110.0, 90.0]) == pytest.approx(10.0)


def test_wape_zero_volume_is_nan() -> None:
    assert math.isnan(metrics.wape([0.0, 0.0], [1.0, 1.0]))


def test_smape_bounds_and_zero() -> None:
    assert metrics.smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    value = metrics.smape([100.0], [0.0])
    assert 0.0 <= value <= 200.0


def test_bias_sign_and_pct() -> None:
    assert metrics.bias([10.0, 10.0], [12.0, 14.0]) == pytest.approx(3.0)
    assert metrics.bias_pct([10.0, 10.0], [12.0, 14.0]) == pytest.approx(30.0)
    assert math.isnan(metrics.bias_pct([0.0, 0.0], [1.0, 1.0]))


def test_compute_metrics_bundle() -> None:
    bundle = metrics.compute_metrics([10.0, 20.0, 30.0], [11.0, 19.0, 33.0])
    as_dict = bundle.to_dict()
    assert set(as_dict) == {"mae", "rmse", "mape", "smape", "wape", "bias", "bias_pct", "n"}
    assert bundle.n == 3
    assert as_dict["mae"] > 0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.mae([1.0, 2.0], [1.0])


def test_empty_arrays_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        metrics.mae(np.array([]), np.array([]))
