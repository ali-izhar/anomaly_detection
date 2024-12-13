# src/model/utils/logger.py

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up logging configuration with different levels for console and file.

    Args:
        config: Dictionary containing logging configuration with:
            - dir: Log directory
            - level: Root logger level
            - console_level: Console output level
            - file_level: File output level
            - console_format: Format for console output
            - file_format: Format for file output
    """
    log_dir = Path(config["dir"])
    log_dir.mkdir(exist_ok=True)

    # Create logger with root level (lowest of console and file levels)
    logger = logging.getLogger("link_predictor")
    root_level = min(
        getattr(logging, config["console_level"].upper()),
        getattr(logging, config["file_level"].upper()),
    )
    logger.setLevel(root_level)

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(config["file_format"])
    console_formatter = logging.Formatter(config["console_format"])

    # File handler with detailed formatting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
    file_handler.setLevel(getattr(logging, config["file_level"].upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler with simpler formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config["console_level"].upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial configuration (to file only)
    file_handler.handle(
        logger.makeRecord(
            "link_predictor",
            logging.DEBUG,
            "",
            0,
            "Logging initialized:\n"
            + f"Root level: {logging.getLevelName(root_level)}\n"
            + f"Console level: {config['console_level']}\n"
            + f"File level: {config['file_level']}",
            None,
            None,
        )
    )

    return logger
