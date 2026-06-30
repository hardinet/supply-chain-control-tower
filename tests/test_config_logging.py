"""Tests for configuration and structured logging."""

from __future__ import annotations

from sctower.config import REPO_ROOT, Settings, get_settings
from sctower.logging import configure_logging, get_logger


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_derived_paths() -> None:
    settings = get_settings()
    assert settings.data_raw_dir.is_absolute()
    assert settings.rossmann_train_csv.name == "train.csv"
    assert settings.rossmann_store_csv.name == "store.csv"
    assert settings.curated_sales_parquet.name == "sales.parquet"


def test_relative_paths_resolve_under_root() -> None:
    settings = Settings(_env_file=None, data_raw_dir="data/raw")
    assert str(settings.data_raw_dir).startswith(str(REPO_ROOT))


def test_logging_does_not_raise() -> None:
    configure_logging(force=True)
    logger = get_logger("test")
    logger.info("event", key="value")  # should not raise
