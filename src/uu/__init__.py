# MIT License - 2026
"""Shared logging utilities for the ``uu`` package."""

import logging


def _build_default_handler() -> logging.Handler:
    """Create the default stream handler for package logging.

    Returns
    -------
    logging.Handler
        Configured stream handler using the package's standard log format.
    """
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
    """Return the package logger or a named child logger.

    Parameters
    ----------
    name : str | None, optional
        Optional child-logger name. When omitted, the root package logger is
        returned.

    Returns
    -------
    logging.Logger
        Logger instance associated with the package or the requested child
        namespace.
    """
    if not name:
        return logger
    return logger.getChild(name)


__all__ = ["get_logger", "logger"]
