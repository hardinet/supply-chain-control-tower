"""Thin PostgreSQL access layer (SQLAlchemy 2.0).

The serving database stores the published alert and policy tables consumed by the
dashboard. The application degrades gracefully to file-based parquet when no
database is reachable, so this layer is intentionally minimal: an engine factory,
a health probe and round-trip helpers for DataFrames.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from sctower.config import get_settings
from sctower.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine(db_url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy engine for the configured database."""
    url = db_url or get_settings().db_url
    return create_engine(url, pool_pre_ping=True, future=True)


def ping(engine: Engine | None = None) -> bool:
    """Return True when the database answers ``SELECT 1``, False otherwise."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # health probe must never raise
        logger.warning("db_ping_failed", error=str(exc))
        return False


def save_dataframe(
    df: pd.DataFrame,
    table: str,
    *,
    engine: Engine | None = None,
    if_exists: Literal["fail", "replace", "append"] = "replace",
) -> int:
    """Write a DataFrame to ``table`` and return the number of rows written."""
    engine = engine or get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False)
    logger.info("table_written", table=table, rows=len(df))
    return len(df)


def load_table(table: str, *, engine: Engine | None = None) -> pd.DataFrame:
    """Read an entire table into a DataFrame."""
    engine = engine or get_engine()
    return pd.read_sql_table(table, engine)
