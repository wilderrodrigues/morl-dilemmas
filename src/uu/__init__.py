# MIT License - 2026

import logging


def _build_default_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
    )
    return handler


logger = logging.getLogger("uu")

if not logger.handlers:
    logger.addHandler(_build_default_handler())

logger.setLevel(logging.INFO)
logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logger
    return logger.getChild(name)


__all__ = ["get_logger", "logger"]
