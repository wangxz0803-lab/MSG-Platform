from __future__ import annotations

import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_format: str = "console",
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_format: "console" for colored dev output, "json" for production.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.stdlib.NAME_TO_LEVEL.get(level.upper(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str, **initial_context: Any) -> Any:
    """Return a logger bound with the given module name.

    Usage::

        logger = get_logger(__name__)
        logger.info("event_happened", key="value")
    """
    if not _CONFIGURED:
        setup_logging(
            level=os.environ.get("MSG_LOG_LEVEL", "INFO"),
            log_format=os.environ.get("MSG_LOG_FORMAT", "console"),
        )
    return structlog.get_logger(name, **initial_context)
