"""SARIMA(X) forecaster built on statsmodels.

A seasonal ARIMA with weekly seasonality (m=7) captures the strong day-of-week
pattern of retail sales. Exogenous regressors (e.g. promotions, holidays) are
supported through the SARIMAX formulation when supplied at fit and forecast time.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Self

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sctower.domain.forecasting.base import (
    ForecastModel,
    ForecastResult,
    Y,
    future_index,
    register_model,
    validate_history,
)

Order = tuple[int, int, int]
SeasonalOrder = tuple[int, int, int, int]


@register_model
class SarimaModel(ForecastModel):
    """Seasonal ARIMA with optional exogenous regressors."""

    name: ClassVar[str] = "sarima"

    def __init__(
        self,
        order: Order = (1, 1, 1),
        seasonal_order: SeasonalOrder = (1, 0, 1, 7),
        alpha: float = 0.05,
        exog: tuple[str, ...] = (),
    ) -> None:
        super().__init__(exog=exog)
        self.order = order
        self.seasonal_order = seasonal_order
        self.alpha = alpha
        self._result: object | None = None

    def fit(self, history: pd.DataFrame) -> Self:
        hist = validate_history(history, exog=self.exog)
        self._history = hist
        endog = hist[Y].to_numpy(dtype=np.float64)
        exog = hist[list(self.exog)].to_numpy(dtype=np.float64) if self.exog else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter is expected here
            model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = model.fit(disp=False)
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> ForecastResult:
        if self._result is None or self._history is None:
            raise RuntimeError("model must be fit before predict")
        future_exog = self._require_future_exog(horizon, future)
        exog_arr = (
            future_exog[list(self.exog)].to_numpy(dtype=np.float64)
            if future_exog is not None
            else None
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._result.get_forecast(steps=horizon, exog=exog_arr)  # type: ignore[attr-defined]
            mean = np.asarray(forecast.predicted_mean, dtype=np.float64)
            conf = np.asarray(forecast.conf_int(alpha=self.alpha), dtype=np.float64)
        return ForecastResult(
            model_name=self.name,
            ds=future_index(self._history, horizon),
            yhat=np.clip(mean, 0.0, None),
            yhat_lower=np.clip(conf[:, 0], 0.0, None),
            yhat_upper=np.clip(conf[:, 1], 0.0, None),
        )
