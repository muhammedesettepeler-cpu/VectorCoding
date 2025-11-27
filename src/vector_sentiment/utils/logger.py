"""Logging configuration using Loguru.

This module provides structured logging setup with file rotation, retention,
and customizable log levels. Loguru is used for its simplicity and powerful features.
"""

import sys
from pathlib import Path

from loguru import Logger, logger

from vector_sentiment.config.constants import LOG_FORMAT


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Setup application logging with Loguru.

    Configures both console and file logging with rotation and retention policies.

    Args:
        level: Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is enabled
        rotation: When to rotate log file (e.g., "500 MB", "1 week", "1 day")
        retention: How long to keep old log files (e.g., "10 days", "1 month")

    Example:
        >>> setup_logging(level="DEBUG", log_file=Path("logs/app.log"))
    """
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
    """Get a logger instance with the specified name.

    This is a convenience function that returns the Loguru logger.
    The name is used for context in log messages.

    Args:
        name: Name for the logger (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Processing data")
    """
    return logger.bind(name=name)
