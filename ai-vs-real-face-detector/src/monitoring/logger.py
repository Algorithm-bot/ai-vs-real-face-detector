"""Structured logging and prediction metrics."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from src.config import get_config


def setup_logger(name: str = "face_detector") -> logging.Logger:
    cfg = get_config().logging
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


@contextmanager
def log_inference_timing(operation: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s completed in %.2f ms", operation, elapsed_ms)


def log_prediction(
    label: str,
    confidence: float,
    filename: Optional[str] = None,
    model_version: Optional[str] = None,
) -> None:
    cfg = get_config().logging
    if not cfg.log_predictions:
        return
    logger.info(
        "prediction label=%s confidence=%.2f file=%s model=%s",
        label,
        confidence,
        filename or "unknown",
        model_version or "unknown",
    )
