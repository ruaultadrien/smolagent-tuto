"""Logger for the app."""

import logging


def get_logger() -> logging.Logger:
    """Get logger of the app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = get_logger()
