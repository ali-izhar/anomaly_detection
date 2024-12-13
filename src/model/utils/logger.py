# src/model/utils/logger.py

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """Set up logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
    )

    return logging.getLogger(__name__)
