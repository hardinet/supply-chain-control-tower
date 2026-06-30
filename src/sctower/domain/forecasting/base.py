"""Common forecasting interface, result container and model registry.

The contract is intentionally small so models stay interchangeable:

- ``fit(history)`` where ``history`` has columns ``ds`` (daily timestamps) and
  ``y`` (demand), plus any exogenous columns the model was configured with.
- ``predict(horizon, future)`` returns a :class:`ForecastResult` with a point
  forecast and, when the model supports it, a prediction interval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

DS = "ds"
Y = "y"


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """A horizon of forecasts with an optional prediction interval."""

    model_name: str
    ds: pd.DatetimeIndex
    yhat: npt.NDArray[np.float64]
    yhat_lower: npt.NDArray[np.float64] | None = None
    yhat_upper: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if len(self.ds) != len(self.yhat):
            raise ValueError("ds and yhat must have the same length")

    @property
    def horizon(self) -> int:
        """Number of forecasted steps."""
        return len(self.yhat)

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy frame with ds, yhat and (if present) interval bounds."""
        data: dict[str, object] = {DS: self.ds, "yhat": self.yhat}
        if self.yhat_lower is not None:
            data["yhat_lower"] = self.yhat_lower
        if self.yhat_upper is not None:
            data["yhat_upper"] = self.yhat_upper
        return pd.DataFrame(data)


def validate_history(history: pd.DataFrame, exog: Sequence[str] = ()) -> pd.DataFrame:
    """Validate and normalize a training frame.

    Ensures ``ds``/``y`` are present, ``ds`` is datetime and strictly increasing,
    and required exogenous columns exist. Returns a sorted copy.
    """
    missing = {DS, Y, *exog} - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    out = history.loc[:, [DS, Y, *exog]].copy()
    out[DS] = pd.to_datetime(out[DS])
    out = out.sort_values(DS).reset_index(drop=True)
    if out[DS].duplicated().any():
        raise ValueError("history contains duplicate timestamps")
    if len(out) < 2:
        raise ValueError("history must contain at least two observations")
    return out


def future_index(history: pd.DataFrame, horizon: int, freq: str = "D") -> pd.DatetimeIndex:
    """Build the future timestamp index that follows the end of ``history``."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    last = pd.to_datetime(history[DS].iloc[-1])
    return pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:]


class ForecastModel(ABC):
    """Abstract base class shared by every forecasting model."""

    name: ClassVar[str] = "abstract"

    def __init__(self, exog: Sequence[str] = ()) -> None:
        self.exog: tuple[str, ...] = tuple(exog)
        self._history: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> Self:
        """Fit the model on ``history`` (columns ds, y [, exog...])."""

    @abstractmethod
    def predict(self, horizon: int, future: pd.DataFrame | None = None) -> ForecastResult:
        """Forecast ``horizon`` steps ahead.

        ``future`` must provide the exogenous columns for the horizon when the
        model was configured with any.
        """

    def _require_future_exog(
        self, horizon: int, future: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """Validate exogenous inputs for the forecast horizon."""
        if not self.exog:
            return None
        if future is None or len(future) < horizon:
            raise ValueError(f"future must provide {horizon} rows of exog {list(self.exog)}")
        missing = set(self.exog) - set(future.columns)
        if missing:
            raise ValueError(f"future is missing exog columns: {sorted(missing)}")
        return future.iloc[:horizon].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[ForecastModel]] = {}


def register_model(cls: type[ForecastModel]) -> type[ForecastModel]:
    """Class decorator that registers a model under its ``name``."""
    if cls.name in _REGISTRY:
        raise ValueError(f"model already registered: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def available_models() -> tuple[str, ...]:
    """Names of all registered, runnable models, in stable order."""
    return tuple(sorted(_REGISTRY))


def build_model(name: str, exog: Sequence[str] = ()) -> ForecastModel:
    """Instantiate a registered model by name."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}") from None
    return cls(exog=exog)
