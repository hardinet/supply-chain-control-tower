"""Tests for the forecasting models and their shared infrastructure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sctower.domain.forecasting import available_models, base, build_model
from sctower.domain.forecasting.base import ForecastResult, future_index, validate_history
from sctower.domain.forecasting.naive import SeasonalNaiveModel


def test_validate_history_requires_columns(daily_series: pd.DataFrame) -> None:
    validated = validate_history(daily_series)
    assert list(validated.columns) == ["ds", "y"]
    with pytest.raises(ValueError, match="missing columns"):
        validate_history(daily_series.drop(columns=["y"]))


def test_validate_history_rejects_duplicates() -> None:
    df = pd.DataFrame({"ds": ["2020-01-01", "2020-01-01"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_history(df)


def test_future_index_follows_history(daily_series: pd.DataFrame) -> None:
    idx = future_index(daily_series, 5)
    assert len(idx) == 5
    assert idx[0] > daily_series["ds"].iloc[-1]
    with pytest.raises(ValueError):
        future_index(daily_series, 0)


def test_registry_lists_and_builds() -> None:
    names = available_models()
    assert "seasonal_naive" in names
    assert "sarima" in names
    assert "gbm" in names
    model = build_model("seasonal_naive")
    assert isinstance(model, SeasonalNaiveModel)


def test_build_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        build_model("does_not_exist")


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @base.register_model
        class Dup(SeasonalNaiveModel):
            name = "seasonal_naive"


def test_forecast_result_length_guard() -> None:
    with pytest.raises(ValueError, match="same length"):
        ForecastResult(
            model_name="x",
            ds=pd.date_range("2020-01-01", periods=2),
            yhat=np.array([1.0]),
        )


def test_seasonal_naive_repeats_last_season() -> None:
    ds = pd.date_range("2020-01-01", periods=14, freq="D")
    y = np.array([float(i % 7) for i in range(14)])
    model = SeasonalNaiveModel(season=7).fit(pd.DataFrame({"ds": ds, "y": y}))
    result = model.predict(7)
    assert result.yhat == pytest.approx([0, 1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("name", ["seasonal_naive", "sarima", "gbm"])
def test_models_fit_and_predict(name: str, daily_series: pd.DataFrame) -> None:
    model = build_model(name)
    model.fit(daily_series)
    result = model.predict(14)
    assert result.horizon == 14
    assert result.yhat_lower is not None
    assert result.yhat_upper is not None
    assert np.all(result.yhat_upper >= result.yhat_lower - 1e-6)
    assert np.all(result.yhat >= 0)
    frame = result.to_frame()
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"} <= set(frame.columns)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit before predict"):
        SeasonalNaiveModel().predict(5)


def test_gbm_requires_minimum_history() -> None:
    short = pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=10, freq="D"), "y": np.arange(10.0)}
    )
    with pytest.raises(ValueError, match="more than"):
        build_model("gbm").fit(short)


def test_exog_model_requires_future(daily_series: pd.DataFrame) -> None:
    model = build_model("sarima", exog=("promo",))
    model.fit(daily_series)
    with pytest.raises(ValueError, match="future must provide"):
        model.predict(10, future=None)
