# src/configs/loader.py

"""Loads model configurations from YAML files."""

from pathlib import Path
from typing import Dict, Any, Union

import yaml
from dacite import from_dict

from .params import BAParams, WSParams, ERParams, SBMParams

# Type mapping for model names to their parameter classes
MODEL_PARAMS = {
    "barabasi_albert": BAParams,
    "watts_strogatz": WSParams,
    "erdos_renyi": ERParams,
    "stochastic_block_model": SBMParams,
    # Aliases
    "ba": BAParams,
    "ws": WSParams,
    "er": ERParams,
    "sbm": SBMParams,
}


def load_model_config(
    model_name: str,
) -> Union[BAParams, WSParams, ERParams, SBMParams]:
    """Load model configuration from YAML and validate parameters.

    Args:
        model_name: Target model identifier
    Returns:
        Validated parameter instance
    Raises:
        ValueError: If model not found or config invalid
    """
    param_class = MODEL_PARAMS.get(model_name.lower())
    if not param_class:
        raise ValueError(f"Unknown model: {model_name}")

    config_path = Path(__file__).parent / "models.yaml"
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    model_config = configs.get(model_name)
    if not model_config:
        raise ValueError(f"No configuration found for model: {model_name}")

    return from_dict(data_class=param_class, data=model_config)


def get_config(model_name: str, **overrides) -> Dict[str, Any]:
    """Get model configuration with optional parameter overrides.

    Args:
        model_name: Target model identifier
        **overrides: Parameter values to override defaults
    Returns:
        Dict containing model name and validated parameters
    """
    params = load_model_config(model_name)

    if overrides:
        param_class = MODEL_PARAMS[model_name.lower()]
        param_dict = {**params.__dict__, **overrides}
        params = from_dict(data_class=param_class, data=param_dict)

    return {"model": model_name, "params": params}
