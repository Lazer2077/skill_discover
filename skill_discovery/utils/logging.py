"""Lightweight console logging with consistent formatting."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "skill_discovery", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that prints to stdout.

    Repeated calls with the same name return the same logger without
    duplicating handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
