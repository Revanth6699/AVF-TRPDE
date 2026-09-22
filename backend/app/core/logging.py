from __future__ import annotations

import logging
import sys
from typing import Final


DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = DEFAULT_LOG_LEVEL,
) -> None:
    """
    Configure application-wide logging.

    Logging is written to stdout so that API and research-process
    logs are visible in the application runtime environment.
    """

    normalized_level = level.strip().upper()

    numeric_level = getattr(
        logging,
        normalized_level,
        None,
    )

    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level: {level!r}. "
            "Expected DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    root_logger = logging.getLogger()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    handler.setFormatter(formatter)

    root_logger.setLevel(numeric_level)

    for existing_handler in root_logger.handlers:
        if getattr(
            existing_handler,
            "_avf_trpde_handler",
            False,
        ):
            existing_handler.setLevel(numeric_level)
            existing_handler.setFormatter(formatter)
            return

    handler._avf_trpde_handler = True  # type: ignore[attr-defined]

    root_logger.addHandler(handler)


def get_logger(
    name: str,
) -> logging.Logger:
    """
    Return a logger for an AVF-TRPDE module.
    """

    if not name or not name.strip():
        raise ValueError("Logger name must not be empty.")

    return logging.getLogger(name.strip())