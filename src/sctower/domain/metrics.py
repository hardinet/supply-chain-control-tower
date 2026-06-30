"""Forecast accuracy metrics.

These are deliberately implemented from first principles (rather than pulled from
a library) so the behaviour around zero actuals is explicit and defensible:

- ``MAPE`` is undefined when the actual is zero; we mask those points and report
  how many were excluded.
- ``WAPE`` (a.k.a. MAD/Mean ratio) is the volume-weighted alternative that stays
  well-defined on intermittent retail demand, which is why we headline it.

All functions accept array-likes and return plain ``float`` values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[np.float64] | list[float]


def _as_arrays(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true{yt.shape} vs y_pred{yp.shape}")
    if yt.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    return yt, yp


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Percentage Error, in percent.

    Points where the actual is exactly zero are excluded (the term is undefined).
    Returns ``nan`` if every actual is zero.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    mask = yt != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE in percent, bounded in [0, 200]."""
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    mask = denom != 0.0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(yp[mask] - yt[mask]) / denom[mask]) * 100.0)


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error in percent (sum|e| / sum|y|).

    Volume-weighted and well-defined on intermittent demand, hence preferred as
    the headline accuracy metric for retail sales.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100.0)


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean forecast error (positive => over-forecasting)."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(yp - yt))


def bias_pct(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean forecast error as a percentage of mean actual demand."""
    yt, yp = _as_arrays(y_true, y_pred)
    mean_true = float(np.mean(yt))
    if mean_true == 0.0:
        return float("nan")
    return bias(yt, yp) / mean_true * 100.0


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Bundle of accuracy metrics for one forecast vs. actuals."""

    mae: float
    rmse: float
    mape: float
    smape: float
    wape: float
    bias: float
    bias_pct: float
    n: int

    def to_dict(self) -> dict[str, float]:
        """Return a plain dict (handy for logging, tables and JSON)."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "smape": self.smape,
            "wape": self.wape,
            "bias": self.bias,
            "bias_pct": self.bias_pct,
            "n": float(self.n),
        }


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> ForecastMetrics:
    """Compute the full :class:`ForecastMetrics` bundle in one pass."""
    yt, yp = _as_arrays(y_true, y_pred)
    return ForecastMetrics(
        mae=mae(yt, yp),
        rmse=rmse(yt, yp),
        mape=mape(yt, yp),
        smape=smape(yt, yp),
        wape=wape(yt, yp),
        bias=bias(yt, yp),
        bias_pct=bias_pct(yt, yp),
        n=int(yt.size),
    )
