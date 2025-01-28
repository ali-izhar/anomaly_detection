# src/predictor/base.py

"""Base class for network prediction algorithms."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BasePredictor(ABC):
    """Abstract base class for network prediction algorithms."""

    @property
    def history_size(self) -> int:
        """Get the required history size for this predictor.

        Returns
        -------
        int
            Number of historical states needed for prediction
        """
        # Default to 5 if not specified by child class
        return getattr(self, "_history_size", 5)

    @abstractmethod
    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """Predict future network states.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            List of historical network states. Each state is a dictionary containing
            at minimum an 'adjacency' key with the adjacency matrix.
            Additional keys may include network features, timestamps, etc.
        horizon : int, optional
            Number of steps ahead to predict, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices [hat{A}_{t+1}, ..., hat{A}_{t+h}]
        """
        pass

    @abstractmethod
    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """Update predictor's internal state with new observation.

        This method allows the predictor to update its internal parameters,
        detect changes, or adapt to new patterns based on observed data.

        Parameters
        ----------
        actual_state : Dict[str, Any]
            The actual observed network state, with same format as history items
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the predictor.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current predictor state, which may include:
            - Model parameters
            - Performance metrics
            - Detected network properties
            - Change points
            - etc.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the predictor to its initial state.

        This includes resetting all internal parameters, buffers,
        and historical data to their initial values.
        """
        pass
