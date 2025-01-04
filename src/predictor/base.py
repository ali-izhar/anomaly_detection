# src/predictor/base.py

"""Abstract base class for network predictors."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BasePredictor(ABC):
    """Abstract base class for network prediction models."""

    @abstractmethod
    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            Historical network data
        horizon : int, optional
            Number of steps to predict ahead, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices
        """
        pass
