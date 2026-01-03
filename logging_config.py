"""
Logging configuration for nano-eval.

Best practices:
- Call setup_logging() once at application entry point
- Use logging.getLogger(__name__) in each module
- NullHandler at package level prevents warnings when used as library
"""

from __future__ import annotations

import logging
import sys

PACKAGE_NAME = "nano_eval"

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbosity: int = 0) -> None:
    """
    Configure logging for the application.

    Args:
        verbosity: 0=WARNING (default), 1=INFO, 2+=DEBUG
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name."""
    return logging.getLogger(name)
