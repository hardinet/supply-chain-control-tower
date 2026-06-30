"""Cached data-access layer for the dashboard.

The app is driven exclusively by the real curated Rossmann dataset. When it is
missing, accessors raise :class:`DataUnavailable` and the application renders a
clear "run `make data`" message rather than fabricating any numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from sctower.config import get_settings
from sctower.domain.forecasting import ForecastResult
from sctower.io.loaders import load_curated
from sctower.logging import get_logger
from sctower.services.alerts import build_alerts
from sctower.services.backtest import compare_models
from sctower.services.pipeline import build_series, fit_and_forecast, recent_demand_stats

logger = get_logger(__name__)

# Modest backtest configuration so the dashboard stays responsive on free tiers.
APP_BACKTEST_HORIZON = 28
APP_BACKTEST_FOLDS = 3


class DataUnavailable(RuntimeError):
    """Raised when the curated dataset has not been built yet."""


@dataclass(frozen=True, slots=True)
class AppData:
    """The real curated dataset backing the app."""

    curated: pd.DataFrame


@lru_cache(maxsize=1)
def get_app_data() -> AppData:
    """Load the curated Rossmann dataset (cached)."""
    try:
        curated = load_curated()
    except FileNotFoundError as exc:
        raise DataUnavailable(str(exc)) from exc
    logger.info("app_data_loaded", rows=len(curated))
    return AppData(curated=curated)


def is_data_available() -> bool:
    """Return True when the curated dataset can be loaded."""
    try:
        get_app_data()
        return True
    except DataUnavailable:
        return False


@lru_cache(maxsize=1)
def get_total_series() -> pd.DataFrame:
    """Aggregate daily demand series across all stores (cached)."""
    return build_series(get_app_data().curated, store=None)


@lru_cache(maxsize=32)
def get_store_series(store: int) -> pd.DataFrame:
    """Daily demand series for a single store (cached)."""
    return build_series(get_app_data().curated, store=store)


@lru_cache(maxsize=1)
def get_store_ids() -> tuple[int, ...]:
    """Sorted tuple of available store ids."""
    return tuple(sorted(int(s) for s in get_app_data().curated["store"].unique()))


@lru_cache(maxsize=1)
def get_backtest_table() -> pd.DataFrame:
    """Model comparison on the aggregate series (cached on disk and in memory)."""
    cache_path = get_settings().data_curated_dir / "backtest_results.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    from sctower.domain.forecasting import available_models

    series = get_total_series()
    table = compare_models(
        series,
        list(available_models()),
        horizon=APP_BACKTEST_HORIZON,
        folds=APP_BACKTEST_FOLDS,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    except OSError as exc:  # pragma: no cover - best effort cache
        logger.warning("backtest_cache_failed", error=str(exc))
    return table


@lru_cache(maxsize=1)
def get_alerts() -> pd.DataFrame:
    """Stock alerts for the top stores (cached)."""
    return build_alerts(get_app_data().curated, n_stores=40)


def _series_for(store_value: str) -> pd.DataFrame:
    """Resolve 'total' or a store id string to the matching series."""
    if store_value == "total":
        return get_total_series()
    return get_store_series(int(store_value))


@lru_cache(maxsize=128)
def get_base_forecast(store_value: str, model_name: str, horizon: int) -> ForecastResult:
    """Fit and cache a base forecast, keyed by (scope, model, horizon)."""
    return fit_and_forecast(_series_for(store_value), model_name, horizon)


@lru_cache(maxsize=64)
def get_demand_stats(store_value: str, window: int = 90) -> tuple[float, float]:
    """Cached (mean, std) of recent demand for a scope."""
    return recent_demand_stats(_series_for(store_value), window=window)
