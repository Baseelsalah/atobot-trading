"""Structured logging setup using loguru for AtoBot Trading."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "logs/atobot.log") -> None:
    """Configure loguru for console and file logging.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the main log file.
    """
    # Remove default handler
    logger.remove()

    # Console handler with colours
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Main log file with rotation
    logger.add(
        log_file,
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{module}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention=5,
        compression="zip",
        enqueue=True,  # Thread-safe
    )

    # Separate error log file
    error_log = log_file.replace(".log", "_error.log")
    logger.add(
        error_log,
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{module}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention=5,
        compression="zip",
        enqueue=True,
    )

    logger.info("Logger initialised — level={}, file={}", log_level, log_file)


def get_logger():
    """Return the loguru logger instance."""
    return logger
