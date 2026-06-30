"""Turn the curated table into model-ready demand series and forecasts.

A *series* is a daily, gap-free DataFrame with columns ``ds`` (timestamp), ``y``
(demand) and the exogenous flags. The default forecasting path is univariate:
exogenous promotion/holiday regressors are supported by the model layer but kept
out of the production forecast because future promotion calendars are not part of
the public dataset (see the case study in the README).
"""

from __future__ import annotations

import pandas as pd

from sctower.domain.forecasting import ForecastResult, build_model
from sctower.logging import get_logger

logger = get_logger(__name__)

EXOG_COLS: tuple[str, ...] = ("promo", "school_holiday", "state_holiday")
TOTAL_KEY = "total"


def list_stores(curated: pd.DataFrame) -> list[int]:
    """Return the sorted list of store identifiers in the curated table."""
    return sorted(int(s) for s in curated["store"].unique())


def build_series(
    curated: pd.DataFrame,
    store: int | None = None,
    *,
    fill_gaps: bool = True,
) -> pd.DataFrame:
    """Build a daily demand series for one store, or the aggregate of all stores.

    Args:
        curated: the curated sales table.
        store: store id, or ``None`` for the all-stores aggregate.
        fill_gaps: reindex to a continuous daily range, filling missing demand
            and flags with zeros (keeps seasonal models well-behaved).

    Returns:
        DataFrame with columns ``ds``, ``y`` and the exogenous flags.
    """
    if store is None:
        grouped = curated.groupby("ds", as_index=False).agg(
            y=("sales", "sum"),
            promo=("promo", "mean"),
            school_holiday=("school_holiday", "mean"),
            state_holiday=("state_holiday", "max"),
        )
        series = grouped
    else:
        subset = curated.loc[curated["store"] == store]
        if subset.empty:
            raise ValueError(f"unknown store: {store}")
        series = subset.rename(columns={"sales": "y"})[["ds", "y", *EXOG_COLS]].copy()

    series = series.sort_values("ds").reset_index(drop=True)

    if fill_gaps and not series.empty:
        full = pd.date_range(series["ds"].min(), series["ds"].max(), freq="D")
        series = series.set_index("ds").reindex(full).rename_axis("ds").reset_index()
        series["y"] = series["y"].fillna(0.0)
        for col in EXOG_COLS:
            series[col] = series[col].fillna(0.0)

    series["y"] = series["y"].astype("float64")
    return series


def fit_and_forecast(
    series: pd.DataFrame,
    model_name: str,
    horizon: int,
    *,
    exog: tuple[str, ...] = (),
) -> ForecastResult:
    """Fit ``model_name`` on the series and forecast ``horizon`` days ahead."""
    model = build_model(model_name, exog=exog)
    model.fit(series)
    future = series.tail(horizon) if exog else None
    logger.info("forecast", model=model_name, horizon=horizon, n_train=len(series))
    return model.predict(horizon, future=future)


def recent_demand_stats(
    series: pd.DataFrame,
    window: int = 90,
    *,
    open_only: bool = True,
) -> tuple[float, float]:
    """Return (mean, std) of recent daily demand.

    When ``open_only`` is set, zero-demand days (typically closures) are excluded
    so the statistics describe genuine selling days, which is what an inventory
    policy should be sized against.
    """
    tail = series.tail(window)["y"]
    if open_only:
        tail = tail[tail > 0]
    if tail.empty:
        return 0.0, 0.0
    return float(tail.mean()), float(tail.std(ddof=1) if len(tail) > 1 else 0.0)
