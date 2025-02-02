# Configs Module

Configuration management for graph models and visualization.

## Files

- `params.py`: Parameter dataclasses for each graph model (BA, WS, ER, SBM) with type validation
- `models.yaml`: Default parameter values and ranges for all graph models
- `loader.py`: YAML configuration loader with parameter validation and override support
- `plotting.py`: Visual style configurations for consistent plot appearance

## Key Classes

- `BAParams`: Barabási-Albert model parameters
- `WSParams`: Watts-Strogatz model parameters
- `ERParams`: Erdős-Rényi model parameters
- `SBMParams`: Stochastic Block Model parameters
- `MetricConfig`: Metric computation settings
