"""Forecasting models behind a single, swappable interface.

Every model implements :class:`~sctower.domain.forecasting.base.ForecastModel`,
so the backtester, the pipeline and the UI treat them interchangeably and can
compare them on equal footing.

Models that depend on optional packages (Prophet) register themselves only when
the dependency is importable; :func:`available_models` reflects what can actually
run in the current environment.
"""

from __future__ import annotations

from sctower.domain.forecasting.base import (
    ForecastModel,
    ForecastResult,
    available_models,
    build_model,
    register_model,
)

__all__ = [
    "ForecastModel",
    "ForecastResult",
    "available_models",
    "build_model",
    "register_model",
]

# Importing the modules triggers their registration side effects.
import contextlib

from sctower.domain.forecasting import gbm, naive, sarima  # noqa: F401

with contextlib.suppress(ImportError):  # pragma: no cover - only when Prophet is installed
    from sctower.domain.forecasting import prophet_model  # noqa: F401
