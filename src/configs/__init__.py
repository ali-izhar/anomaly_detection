# src/configs/__init__.py

"""Configuration module for the application."""

from .loader import load_model_config, get_config, get_full_model_name

__all__ = [
    "load_model_config",
    "get_config",
    "get_full_model_name",
]
