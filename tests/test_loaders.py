"""Tests for the Rossmann loaders and curation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sctower.config import Settings
from sctower.io.loaders import (
    CURATED_COLUMNS,
    curate,
    curate_rossmann,
    load_curated,
    load_raw_rossmann,
)


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "DayOfWeek": [1, 2, 1, 2],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-01", "2015-01-02"],
            "Sales": [5000, 5200, 0, 4800],
            "Customers": [500, 520, 0, 480],
            "Open": [1, 1, 0, 1],
            "Promo": [1, 0, 0, 1],
            "StateHoliday": ["a", "0", "0", "0"],
            "SchoolHoliday": [1, 0, 0, 1],
        }
    )
    store = pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["c", "a"],
            "CompetitionDistance": [500.0, 1500.0],
        }
    )
    return train, store


def test_curate_rossmann_schema_and_state_holiday() -> None:
    train, store = _raw_frames()
    curated = curate_rossmann(train, store)
    assert list(curated.columns) == CURATED_COLUMNS
    assert curated["state_holiday"].isin([0, 1]).all()
    # The 'a' state holiday becomes 1, the '0' values become 0.
    assert curated.loc[curated["store"] == 1, "state_holiday"].iloc[0] == 1


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        data_curated_dir=tmp_path / "curated",
    )


def test_curate_end_to_end(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    settings.data_raw_dir.mkdir(parents=True)
    train, store = _raw_frames()
    train.to_csv(settings.rossmann_train_csv, index=False)
    store.to_csv(settings.rossmann_store_csv, index=False)

    written = curate(settings)
    assert settings.curated_sales_parquet.exists()
    loaded = load_curated(settings)
    assert len(loaded) == len(written) == 4


def test_load_curated_missing(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="curated dataset not found"):
        load_curated(settings)


def test_load_raw_missing(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing raw data"):
        load_raw_rossmann(settings=settings)
