"""Seasonal-naive baseline forecaster.

The forecast for day *t* is the demand observed ``season`` days earlier (default
weekly seasonality of 7 days). Trivial, dependency-free, and the honest baseline
every more sophisticated model must beat -- reporting it keeps the evaluation
grounded.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from sctower.domain.forecasting.base import (
    ForecastModel,
    ForecastResult,
    Y,
    future_index,
    register_model,
    validate_history,
)


@register_model
class SeasonalNaiveModel(ForecastModel):
    """Repeat the value from ``season`` periods ago, with a residual interval."""

    name: ClassVar[str] = "seasonal_naive"

    def __init__(self, season: int = 7, z: float = 1.96, exog: tuple[str, ...] = ()) -> None:
        super().__init__(exog=())  # baseline ignores exogenous regressors
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season
        self.z = z
        self._y: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._resid_std: float = 0.0

    def fit(self, history: pd.DataFrame) -> Self:
        hist = validate_history(history)
        self._history = hist
        self._y = hist[Y].to_numpy(dtype=np.float64)
        if len(self._y) > self.season:
            in_sample = self._y[self.season :]
            shifted = self._y[: -self.season]
            if len(in_sample) > 1:
                self._resid_std = float(np.std(in_sample - shifted, ddof=1))
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> ForecastResult:
        if self._history is None:
            raise RuntimeError("model must be fit before predict")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        # Recursively roll the last season forward.
        rolled = list(self._y[-self.season :]) if len(self._y) >= self.season else list(self._y)
        preds: list[float] = []
        for _ in range(horizon):
            value = rolled[-self.season] if len(rolled) >= self.season else rolled[-1]
            preds.append(value)
            rolled.append(value)
        yhat = np.asarray(preds, dtype=np.float64)
        margin = self.z * self._resid_std
        return ForecastResult(
            model_name=self.name,
            ds=future_index(self._history, horizon),
            yhat=yhat,
            yhat_lower=np.clip(yhat - margin, 0.0, None),
            yhat_upper=yhat + margin,
        )
