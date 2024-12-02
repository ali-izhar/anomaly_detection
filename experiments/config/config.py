# config/config.py

import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Union


class Config(SimpleNamespace):
    """Provides attribute-style access to configuration parameters loaded from YAML."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super().__init__()
        # Recursively convert dictionary items to Config attributes
        for key, value in config_dict.items():
            setattr(self, key, self._convert(value))

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [self._convert(item) for item in value]
        return value


def load_config(yaml_path: Union[str, Path]) -> Config:
    """Load a YAML configuration file into a Config object."""
    path = Path(yaml_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' does not exist.")

    with path.open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
