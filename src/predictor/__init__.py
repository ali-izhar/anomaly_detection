"""Network prediction module.

This module provides various predictors for network evolution forecasting:

1. WeightedPredictor: Uses weighted averaging of recent states
2. SpectralPredictor: Uses spectral decomposition patterns
3. EmbeddingPredictor: Uses node embedding evolution
4. DynamicalPredictor: Models network as dynamical system
5. EnsemblePredictor: Combines multiple base predictors
6. AdaptivePredictor: Dynamically adjusts ensemble weights
7. HybridPredictor: Integrates multiple prediction strategies
"""

from .base import BasePredictor
from .weighted import WeightedPredictor
from .spectral import SpectralPredictor
from .embedding import EmbeddingPredictor
from .dynamical import DynamicalPredictor
from .ensemble import EnsemblePredictor
from .adaptive import AdaptivePredictor
from .hybrid import HybridPredictor
from .visualize import Visualizer

__all__ = [
    "BasePredictor",
    "WeightedPredictor",
    "SpectralPredictor",
    "EmbeddingPredictor",
    "DynamicalPredictor",
    "EnsemblePredictor",
    "AdaptivePredictor",
    "HybridPredictor",
    "Visualizer",
]
