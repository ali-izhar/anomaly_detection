# src/configs/__init__.py

"""Configuration module for the application."""

from .loader import load_model_config, get_config, get_full_model_name
from .plotting import (
    FIGURE_DIMENSIONS,
    TYPOGRAPHY,
    LINE_STYLE,
    COLORS,
    DEFAULT_NETWORK_STYLE,
    get_network_style,
)

__all__ = [
    "load_model_config",
    "get_config",
    "get_full_model_name",
    "FIGURE_DIMENSIONS",
    "TYPOGRAPHY",
    "LINE_STYLE",
    "COLORS",
    "DEFAULT_NETWORK_STYLE",
    "get_network_style",
]
