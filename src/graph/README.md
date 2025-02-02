# Graph Module

Core functionality for graph generation, analysis, and visualization.

## Files

- `generator.py`: Generates dynamic graph sequences using various models (BA, WS, ER, SBM) with controlled parameter evolution
- `features.py`: Extracts graph features (basic metrics, centrality measures, spectral properties) using modular extractors
- `metrics.py`: Computes comparison metrics between actual and predicted graph features, including distribution-based metrics
- `evolution.py`: Controls parameter evolution for dynamic graph generation with model-specific rules
- `utils.py`: Common utilities for graph manipulation and conversion
- `visualizer.py`: Visualization tools for graphs, features, and their evolution over time

## Key Classes

- `GraphGenerator`: Creates graph sequences with change points
- `NetworkFeatureExtractor`: Extracts and processes graph features
- `NetworkVisualizer`: Renders graphs and feature plots
- `EvolutionManager`: Controls graph parameter evolution
