"""Base class for network predictors.

This module provides the abstract base class that all network predictors must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


def pad_or_truncate_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    """Pad or truncate a matrix to match the target size.

    If matrix is smaller, pad with zeros.
    If matrix is larger, truncate to target size.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to resize
    target_size : int
        Desired size of output matrix

    Returns
    -------
    np.ndarray
        Resized matrix of shape (target_size, target_size)
    """
    current_size = matrix.shape[0]
    if current_size == target_size:
        return matrix

    if current_size < target_size:
        # Pad with zeros
        padded = np.zeros((target_size, target_size))
        padded[:current_size, :current_size] = matrix
        return padded
    else:
        # Truncate
        return matrix[:target_size, :target_size]


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
