# Imports.
from __future__ import annotations

import logging
import sys


# Get logger.
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a minimal stderr handler for local runs."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
