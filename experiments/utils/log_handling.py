# experiments/utils/log_handling.py

"""Centralized logging configuration for the entire application."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import colorlog


class ApplicationFilter(logging.Filter):
    """Filter to only allow logs from our application packages."""

    def __init__(self, app_packages: list[str]):
        super().__init__()
        self.app_packages = app_packages

    def filter(self, record: logging.LogRecord) -> bool:
        """Check if log record is from our application.

        Returns True if log is from one of our packages, False otherwise.
        """
        return any(record.name.startswith(pkg) for pkg in self.app_packages)


def setup_root_logger(config: Dict[str, Any], experiment_name: str) -> None:
    """Configure root logger for the entire application with colored output."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create colored console formatter
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    # File formatter (no colors needed)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create application filter
    app_filter = ApplicationFilter(["src", "tests"])

    # Console handler (handles all levels)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(config.logging.level)
    console_handler.addFilter(app_filter)  # Add filter
    root_logger.addHandler(console_handler)

    # File handler
    if config.logging.file:
        log_path = Path(config.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(config.logging.level)
        file_handler.addFilter(app_filter)  # Add filter
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to file: {log_path}")

    # Set up exception handling to log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module."""
    return logging.getLogger(name)
