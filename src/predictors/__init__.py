"""Network prediction models.

This package provides various predictors for forecasting network evolution:

1. Base Predictor: Abstract base class for all predictors
2. Weighted Predictor: Uses weighted averaging of recent states
3. Spectral Predictor: Uses spectral properties of networks
4. Dynamical Predictor: Models networks as dynamical systems
5. Embedding Predictor: Uses graph embeddings
6. Hybrid Predictor: Combines multiple prediction strategies
7. Ensemble Predictor: Combines multiple predictors
8. Adaptive Predictor: Dynamically adjusts ensemble weights
"""

from .base import BasePredictor
from .weighted import WeightedPredictor
from .spectral import SpectralPredictor
from .dynamical import DynamicalPredictor
from .embedding import EmbeddingPredictor
from .hybrid import HybridPredictor
from .ensemble import EnsemblePredictor
from .adaptive import AdaptivePredictor
from .visualize import plot_metric_evolution

__all__ = [
    "BasePredictor",
    "WeightedPredictor",
    "SpectralPredictor",
    "DynamicalPredictor",
    "EmbeddingPredictor",
    "HybridPredictor",
    "EnsemblePredictor",
    "AdaptivePredictor",
    "plot_metric_evolution",
]
