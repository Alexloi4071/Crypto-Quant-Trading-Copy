# -*- coding: utf-8 -*-
"""
Lightweight logging setup used by optuna_system.

Provides a single setup_logging(name) function returning a configured logger.
This avoids importing external project loggers and keeps optuna_system isolated.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging(name: str = __name__) -> logging.Logger:
    level = os.getenv("OPTUNA_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(getattr(logging, level, logging.INFO))
        return logger

    logger.setLevel(getattr(logging, level, logging.INFO))
    fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # Optional file logging
    log_dir = Path("logs/system")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "optuna_system.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        logger.addHandler(file_handler)
    except Exception:
        # If filesystem not writable, ignore file logging
        pass

    return logger


