"""Logging configuration using Loguru.

This module provides structured logging setup with file rotation, retention,
and customizable log levels. Loguru is used for its simplicity and powerful features.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

from vector_sentiment.config.constants import LOG_FORMAT


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Add file handler if specified
    if log_file is not None:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=LOG_FORMAT,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Logging initialized at level: {level}")


def get_logger(name: str) -> "Logger":
    return logger.bind(name=name)
