"""Tests for the database access layer (SQLite stand-in for Postgres)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from sctower.io import db


def test_roundtrip(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    assert db.ping(engine) is True
    frame = pd.DataFrame({"store": [1, 2], "status": ["healthy", "stockout"]})
    written = db.save_dataframe(frame, "alerts", engine=engine)
    assert written == 2
    loaded = db.load_table("alerts", engine=engine)
    assert len(loaded) == 2
    assert set(loaded["status"]) == {"healthy", "stockout"}


def test_ping_handles_failure() -> None:
    class _BadEngine:
        def connect(self) -> object:
            raise RuntimeError("no database")

    assert db.ping(_BadEngine()) is False  # type: ignore[arg-type]


def test_get_engine_is_cached() -> None:
    e1 = db.get_engine("sqlite:///:memory:")
    e2 = db.get_engine("sqlite:///:memory:")
    assert e1 is e2
