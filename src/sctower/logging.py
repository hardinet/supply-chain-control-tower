"""Structured logging setup built on ``structlog``.

A single :func:`configure_logging` call wires both the standard library and
structlog so that every log line is either human-readable (local) or JSON
(production / containers), with consistent timestamps and a bound logger API.
"""

from __future__ import annotations

import logging
import sys

import structlog

from sctower.config import get_settings

_CONFIGURED = False


def configure_logging(*, force: bool = False) -> None:
    """Configure process-wide structured logging.

    Idempotent: safe to call from multiple entrypoints. Set ``force=True`` to
    reconfigure (used in tests).
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level)
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger, initializing logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
