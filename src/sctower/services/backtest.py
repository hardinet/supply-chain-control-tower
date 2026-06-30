"""Rolling-origin backtesting to compare forecasting models honestly.

We evaluate each model on several consecutive hold-out windows of length
``horizon`` that all end at the series' tail (an expanding-window / rolling-origin
scheme). Predictions and actuals are pooled across folds before scoring, so the
reported metrics reflect genuine out-of-sample performance rather than an
in-sample fit. There is no leakage: a fold only ever trains on data strictly
before its test window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sctower.domain.forecasting import build_model
from sctower.domain.metrics import ForecastMetrics, compute_metrics
from sctower.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Pooled backtest outcome for a single model."""

    model_name: str
    metrics: ForecastMetrics
    n_folds: int

    def to_row(self) -> dict[str, float | str]:
        """Flatten to a single record for a comparison table."""
        return {"model": self.model_name, "folds": self.n_folds, **self.metrics.to_dict()}


def rolling_origin_splits(n: int, folds: int, horizon: int) -> list[tuple[int, int]]:
    """Return ``folds`` (test_start, test_end) index pairs ending at ``n``.

    Each test window has length ``horizon``; training uses everything before it.
    Raises if the series is too short to host the requested scheme.
    """
    if folds < 1 or horizon < 1:
        raise ValueError("folds and horizon must be >= 1")
    first_test_start = n - folds * horizon
    if first_test_start < horizon:
        raise ValueError(f"series too short: need >= {(folds + 1) * horizon} points, got {n}")
    return [(n - (folds - k) * horizon, n - (folds - k - 1) * horizon) for k in range(folds)]


def backtest_model(
    series: pd.DataFrame,
    model_name: str,
    *,
    horizon: int,
    folds: int,
    exog: tuple[str, ...] = (),
) -> BacktestResult:
    """Backtest one model and return pooled out-of-sample metrics."""
    n = len(series)
    splits = rolling_origin_splits(n, folds, horizon)
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for test_start, test_end in splits:
        train = series.iloc[:test_start]
        test = series.iloc[test_start:test_end]
        model = build_model(model_name, exog=exog)
        model.fit(train)
        future = test if exog else None
        result = model.predict(horizon, future=future)
        y_true_all.append(test["y"].to_numpy(dtype=np.float64))
        y_pred_all.append(result.yhat)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    metrics = compute_metrics(y_true, y_pred)
    logger.info("backtest", model=model_name, folds=folds, wape=round(metrics.wape, 2))
    return BacktestResult(model_name=model_name, metrics=metrics, n_folds=folds)


def compare_models(
    series: pd.DataFrame,
    model_names: list[str],
    *,
    horizon: int,
    folds: int,
    exog: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Backtest several models and return a comparison table sorted by WAPE."""
    rows: list[dict[str, float | str]] = []
    for name in model_names:
        try:
            result = backtest_model(series, name, horizon=horizon, folds=folds, exog=exog)
            rows.append(result.to_row())
        except Exception as exc:  # one bad model must not sink the whole panel
            logger.warning("backtest_model_failed", model=name, error=str(exc))
    if not rows:
        raise RuntimeError("no model produced a backtest result")
    table = pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
    return table
