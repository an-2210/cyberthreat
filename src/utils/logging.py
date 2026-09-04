"""
Structured logging module for Cyber Threat Detection project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "cyberthreat",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger module.
        log_file: Path to a file where logs should be written.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger instance configured with standard formatters.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
