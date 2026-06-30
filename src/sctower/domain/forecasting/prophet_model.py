"""Prophet forecaster (optional dependency).

This module imports ``prophet`` at top level. When the package is not installed,
the import fails and :mod:`sctower.domain.forecasting` skips registration, so the
rest of the system keeps working. Prophet is exercised in Docker and CI, where it
installs reliably (see docs/adr/0002-prophet-isolation.md).
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
import pandas as pd
from prophet import Prophet

from sctower.domain.forecasting.base import (
    DS,
    ForecastModel,
    ForecastResult,
    Y,
    future_index,
    register_model,
    validate_history,
)


@register_model
class ProphetModel(ForecastModel):
    """Additive model with weekly/yearly seasonality and optional regressors."""

    name: ClassVar[str] = "prophet"

    def __init__(self, interval_width: float = 0.95, exog: tuple[str, ...] = ()) -> None:
        super().__init__(exog=exog)
        self.interval_width = interval_width
        self._model: Prophet | None = None

    def fit(self, history: pd.DataFrame) -> Self:
        hist = validate_history(history, exog=self.exog)
        self._history = hist
        model = Prophet(
            interval_width=self.interval_width,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        for col in self.exog:
            model.add_regressor(col)
        train = hist.rename(columns={DS: "ds", Y: "y"})
        self._model = model.fit(train)
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> ForecastResult:
        if self._model is None or self._history is None:
            raise RuntimeError("model must be fit before predict")
        future_exog = self._require_future_exog(horizon, future)
        ds_future = future_index(self._history, horizon)
        future_df = pd.DataFrame({"ds": ds_future})
        for col in self.exog:
            assert future_exog is not None
            future_df[col] = future_exog[col].to_numpy()
        forecast = self._model.predict(future_df)
        return ForecastResult(
            model_name=self.name,
            ds=ds_future,
            yhat=np.clip(forecast["yhat"].to_numpy(dtype=np.float64), 0.0, None),
            yhat_lower=np.clip(forecast["yhat_lower"].to_numpy(dtype=np.float64), 0.0, None),
            yhat_upper=np.clip(forecast["yhat_upper"].to_numpy(dtype=np.float64), 0.0, None),
        )
