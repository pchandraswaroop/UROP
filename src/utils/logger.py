"""
logger.py — Logging setup
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str = None, filename: str = None) -> None:
    """Configure root logger.

    Args:
        level:    Logging level string ("DEBUG", "INFO", "WARNING", etc.).
        log_dir:  Optional directory to also write logs to file.
        filename: Log file name (default: run.log).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fname = filename or "run.log"
        handlers.append(logging.FileHandler(Path(log_dir) / fname))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
