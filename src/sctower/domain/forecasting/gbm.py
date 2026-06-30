"""Gradient-boosting forecaster (LightGBM) with recursive multi-step prediction.

The series is turned into a supervised problem with calendar features, lags and
rolling means; exogenous regressors (known for the future, e.g. promotions) are
appended. Multi-step forecasts are produced recursively, feeding each prediction
back in to compute the next step's lag features. The prediction interval is
derived from the empirical distribution of in-sample residuals.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMRegressor

from sctower.domain.forecasting.base import (
    DS,
    ForecastModel,
    ForecastResult,
    Y,
    future_index,
    register_model,
    validate_history,
)

LAGS: tuple[int, ...] = (1, 7, 14, 28)
ROLLS: tuple[int, ...] = (7, 14, 28)
_MIN_HISTORY = max(max(LAGS), max(ROLLS)) + 1


def _calendar_features(ds: pd.Series) -> pd.DataFrame:
    """Deterministic calendar features derived from timestamps."""
    dt = pd.DatetimeIndex(pd.to_datetime(ds))
    return pd.DataFrame(
        {
            "dow": dt.dayofweek,
            "month": dt.month,
            "day": dt.day,
            "doy": dt.dayofyear,
            "week": dt.isocalendar().week.to_numpy(dtype=np.int64),
            "is_weekend": (dt.dayofweek >= 5).astype(np.int64),
        }
    )


@register_model
class GradientBoostingModel(ForecastModel):
    """LightGBM regressor on engineered time-series features."""

    name: ClassVar[str] = "gbm"

    def __init__(self, z: float = 1.96, exog: tuple[str, ...] = (), **lgbm_params: Any) -> None:
        super().__init__(exog=exog)
        params: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        params.update(lgbm_params)
        self._model = LGBMRegressor(**params)
        self.z = z
        self._feature_cols: list[str] = []
        self._resid_std: float = 0.0

    def _design_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build the full feature matrix (calendar + lags + rolls + exog)."""
        feats = _calendar_features(frame[DS])
        y = frame[Y].reset_index(drop=True)
        for lag in LAGS:
            feats[f"lag_{lag}"] = y.shift(lag)
        for win in ROLLS:
            feats[f"roll_{win}"] = y.shift(1).rolling(win).mean()
        for col in self.exog:
            feats[col] = frame[col].reset_index(drop=True)
        return feats

    def fit(self, history: pd.DataFrame) -> Self:
        hist = validate_history(history, exog=self.exog)
        if len(hist) <= _MIN_HISTORY:
            raise ValueError(f"need more than {_MIN_HISTORY} observations to fit {self.name}")
        self._history = hist
        feats = self._design_matrix(hist)
        target = hist[Y].reset_index(drop=True)
        mask = feats.notna().all(axis=1)
        x_train = feats.loc[mask]
        y_train = target.loc[mask]
        self._feature_cols = list(x_train.columns)
        self._model.fit(x_train, y_train)
        resid = y_train.to_numpy() - self._model.predict(x_train)
        self._resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        return self

    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> ForecastResult:
        if self._history is None:
            raise RuntimeError("model must be fit before predict")
        future_exog = self._require_future_exog(horizon, future)
        ds_future = future_index(self._history, horizon)

        # Rolling buffer of known + predicted demand for recursive lag features.
        work_y: list[float] = list(self._history[Y].to_numpy(dtype=np.float64))
        preds: list[float] = []
        for step in range(horizon):
            ts = pd.Timestamp(ds_future[step])
            row: dict[str, float] = {
                "dow": float(ts.dayofweek),
                "month": float(ts.month),
                "day": float(ts.day),
                "doy": float(ts.dayofyear),
                "week": float(ts.isocalendar()[1]),
                "is_weekend": float(ts.dayofweek >= 5),
            }
            for lag in LAGS:
                row[f"lag_{lag}"] = work_y[-lag]
            for win in ROLLS:
                row[f"roll_{win}"] = float(np.mean(work_y[-win:]))
            for col in self.exog:
                assert future_exog is not None  # guaranteed by _require_future_exog
                row[col] = float(future_exog.iloc[step][col])
            x_step = pd.DataFrame([row])[self._feature_cols]
            yhat = float(self._model.predict(x_step)[0])
            yhat = max(yhat, 0.0)
            preds.append(yhat)
            work_y.append(yhat)

        yhat_arr: npt.NDArray[np.float64] = np.asarray(preds, dtype=np.float64)
        margin = self.z * self._resid_std
        return ForecastResult(
            model_name=self.name,
            ds=ds_future,
            yhat=yhat_arr,
            yhat_lower=np.clip(yhat_arr - margin, 0.0, None),
            yhat_upper=yhat_arr + margin,
        )
