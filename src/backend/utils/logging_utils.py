from __future__ import annotations

import logging
from pathlib import Path

from src.backend.utils.io import ensure_dir


def setup_logging(log_file: str | Path, level: str = "INFO") -> None:
    target = Path(log_file)
    ensure_dir(target.parent)

    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger.setLevel(level.upper())

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
