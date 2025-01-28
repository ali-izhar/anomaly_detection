# src/predictor/adaptive.py

"""
Implements the predictive (preemptive) forecasting logic described in:
Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting.

 - Input: current graph adjacency, history of adjacency matrices, horizon h, etc.
 - Output: list of predicted adjacency matrices [hat{A}_{t+1}, ..., hat{A}_{t+h}].
 
Then feed these predicted adjacencies into the martingale pipeline to
compute horizon martingales, etc.
"""

from collections import deque
from typing import List, Dict, Any

import numpy as np
import logging
from .base import BasePredictor

logger = logging.getLogger(__name__)


class AdaptivePredictor(BasePredictor):
    """A predictor that replicates the paper's approach:
    1) Weighted average (temporal memory) with adaptive decay.
    2) Structural role preservation with adaptive gamma.
    3) Adaptive mixing parameter.
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 0.8,
        initial_gamma: float = 0.1,
        initial_beta: float = 0.5,
        error_window: int = 5,
    ):
        """
        Parameters
        ----------
        k : int
            Size of the historical window to look back for temporal predictions.
        alpha : float
            Base decay factor for weighted averaging of adjacency matrices, in (0,1).
        initial_gamma : float
            Initial coefficient for structural constraints, will adapt over time.
        initial_beta : float
            Starting value for the adaptive mixing parameter beta_t.
        error_window : int
            Number of recent prediction errors to track for adaptation.
        """
        self.k = k
        self._history_size = k  # Set history size property
        self.alpha = alpha
        self.gamma = initial_gamma
        self.beta = initial_beta
        self._error_window = error_window
        self._prev_mses = deque(maxlen=error_window)
        self._metrics = {"mae": [], "rmse": []}
        self._density_history = deque(maxlen=k)
        self._prediction_history = deque(maxlen=k)  # Track recent predictions
        self._last_prediction = None

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """
        Predict future adjacency matrices with consistent smoothing.

        Parameters
        ----------
        history : List[Dict[str, Any]]
            List of historical network states, each containing an 'adjacency' key
        horizon : int, optional
            Number of steps to predict ahead, by default 1

        Returns
        -------
        List[np.ndarray]
            List of predicted adjacency matrices
        """
        # Extract adjacency matrices from history
        history_adjs = [state["adjacency"] for state in history]
        if len(history_adjs) < 1:
            raise ValueError("Need at least one historical state")

        current_adj = history_adjs[-1]
        extended_history = history_adjs[:-1]

        predictions = []
        n = current_adj.shape[0]

        # Compute smoothed density statistics using full history
        densities = [np.mean(adj) for adj in extended_history + [current_adj]]
        self._density_history.extend(densities)

        # Use exponential moving average for smoother trend detection
        if len(self._density_history) > 2:
            weights = np.exp(np.linspace(-2, 0, len(self._density_history)))
            weights /= np.sum(weights)
            smooth_densities = np.array(list(self._density_history))
            weighted_densities = np.convolve(weights, smooth_densities, mode="valid")[
                -3:
            ]
            trend = (
                np.mean(np.diff(weighted_densities))
                if len(weighted_densities) > 1
                else 0
            )
            volatility = (
                np.std(weighted_densities) if len(weighted_densities) > 1 else 0
            )
        else:
            trend = 0
            volatility = 0

        # Unified stability measure (combines trend and volatility)
        stability_score = 1.0 - min(1.0, (abs(trend) / 0.02 + volatility / 0.03))

        # Base prediction with temporal patterns
        base_pred = self._temporal_prediction(extended_history + [current_adj])

        # Structural preservation with adaptive gamma
        struct_pred = self._structural_preservation(
            base_pred, current_adj, stability_score
        )

        # Initial prediction
        pred = self._adaptive_integration(base_pred, struct_pred, stability_score)

        # Apply consistent thresholding
        threshold = 0.5  # Fixed base threshold
        if self._last_prediction is not None:
            # Adjust threshold based on previous prediction's density
            target_density = np.mean(self._last_prediction) + trend
            current_density = np.mean(pred)
            if abs(current_density - target_density) > 0.05:
                # Use quantile-based threshold
                sorted_vals = np.sort(pred.flatten())
                k = int((1 - target_density) * n * n)
                k = max(0, min(k, len(sorted_vals) - 1))  # Ensure k is valid
                threshold = sorted_vals[k] if k > 0 else 0.5

        # Single thresholding step with stability-based smoothing
        binary_pred = (pred > threshold).astype(float)
        smooth_factor = stability_score * 0.2  # Max 20% smoothing
        pred = (1 - smooth_factor) * binary_pred + smooth_factor * pred

        predictions.append(pred)
        self._prediction_history.append(pred)
        prev_pred = pred

        # Make dependent predictions for horizon > 1
        for step in range(1, horizon):
            # Use stability score to determine mixing weights
            temporal_weight = 0.6 + 0.3 * (1 - stability_score)  # 0.6-0.9 range

            # Temporal prediction using both history and recent predictions
            temp_history = extended_history + [current_adj] + predictions
            temp_pred = self._temporal_prediction(temp_history)

            # Mix with previous prediction
            mixed_pred = temporal_weight * temp_pred + (1 - temporal_weight) * prev_pred

            # Structural preservation
            struct_pred = self._structural_preservation(
                mixed_pred, prev_pred, stability_score
            )

            # Final integration
            next_pred = self._adaptive_integration(
                mixed_pred, struct_pred, stability_score
            )

            # Consistent thresholding
            binary_next = (next_pred > threshold).astype(float)
            next_pred = (1 - smooth_factor) * binary_next + smooth_factor * next_pred

            predictions.append(next_pred)
            prev_pred = next_pred

        return predictions

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """
        Update predictor's internal state with new observation.

        Parameters
        ----------
        actual_state : Dict[str, Any]
            The actual observed network state containing 'adjacency' matrix
        """
        actual_adj = actual_state["adjacency"]
        if self._last_prediction is not None:
            self._update_beta(actual_adj, self._last_prediction)
        self._last_prediction = actual_adj.copy()

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the predictor.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current predictor state and metrics
        """
        return {
            "metrics": {
                "mae": np.array(self._metrics["mae"]),
                "rmse": np.array(self._metrics["rmse"]),
            },
            "parameters": {
                "current_beta": self.beta,
                "current_gamma": self.gamma,
            },
            "statistics": {
                "mean_mae": (
                    np.mean(self._metrics["mae"]) if self._metrics["mae"] else None
                ),
                "mean_rmse": (
                    np.mean(self._metrics["rmse"]) if self._metrics["rmse"] else None
                ),
                "density_history": list(self._density_history),
            },
        }

    def reset(self) -> None:
        """Reset the predictor to its initial state."""
        self._prev_mses.clear()
        self._metrics = {"mae": [], "rmse": []}
        self._density_history.clear()
        self._prediction_history.clear()
        self._last_prediction = None
        self.beta = 0.5  # Reset to initial value
        self.gamma = 0.1  # Reset to initial value

    # --------------------------------------------------------------------------
    # PRIVATE HELPER METHODS

    def _temporal_prediction(self, extended_history: list) -> np.ndarray:
        """Temporal prediction with consistent smoothing."""
        use_history = (
            extended_history[-self.k :]
            if len(extended_history) >= self.k
            else extended_history
        )
        m = len(use_history)

        if m == 0:
            return np.zeros_like(extended_history[-1])

        # Compute exponential weights
        weights = np.exp(np.linspace(-2, 0, m))  # Consistent exp decay
        weights = weights / np.sum(weights)

        # Weighted combination
        N = use_history[-1].shape[0]
        A_T = np.zeros((N, N))
        for idx, mat in enumerate(use_history):
            A_T += weights[idx] * mat

        return A_T

    def _structural_preservation(
        self, A_t_T: np.ndarray, current_adj: np.ndarray, stability_score: float
    ) -> np.ndarray:
        """Structural preservation with stability-based adaptation."""
        # Adapt gamma based on stability
        if len(self._prev_mses) > 0:
            mean_error = np.mean(list(self._prev_mses))
            # More stable gamma adaptation
            self.gamma = self.gamma * np.exp(-mean_error * 2)
            self.gamma = max(0.01, min(0.3, self.gamma))

            # Reduce gamma more during instability
            self.gamma *= stability_score

        return self._optimize_structure(A_t_T, current_adj)

    def _optimize_structure(self, A_t_T: np.ndarray, A_init: np.ndarray) -> np.ndarray:
        """
        Optimize the adjacency matrix with stability controls.
        """
        N = A_t_T.shape[0]

        # Start with more aggressive threshold for high confidence edges
        A_opt = (A_t_T > 0.55).astype(float)  # Lower threshold from 0.6

        # Count how many more edges we need
        target_density = np.mean(A_t_T)
        target_edges = int(target_density * N * (N - 1))
        current_edges = int(np.sum(A_opt))
        remaining_edges = max(0, target_edges - current_edges)

        if remaining_edges > 0:
            mask = ~np.eye(N, dtype=bool) & (A_opt == 0)
            probs = A_t_T[mask]

            if len(probs) > remaining_edges:
                # Enhanced community structure bonus
                community_bonus = self._get_community_bonus(A_t_T)
                community_weight = 0.3  # Increased from 0.2
                probs = probs + community_weight * community_bonus[mask]

                # Use soft thresholding for remaining edges
                sorted_probs = np.sort(probs)
                k = min(remaining_edges, len(sorted_probs) - 1)  # Ensure k is valid
                threshold = sorted_probs[-k] if k > 0 else 0.5
                confidence_factor = np.minimum(1.0, (probs - threshold + 0.1) / 0.1)
                A_opt[mask] = confidence_factor * (probs >= threshold)

        # Ensure symmetry and zero diagonal
        A_opt = np.maximum(A_opt, A_opt.T)
        np.fill_diagonal(A_opt, 0)

        return A_opt

    def _get_community_bonus(self, A: np.ndarray) -> np.ndarray:
        """
        Compute bonus scores for edges that maintain community structure.
        Uses simple block structure detection based on density patterns.
        """
        N = A.shape[0]
        bonus = np.zeros((N, N))

        # Detect potential communities using density patterns
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Check if nodes i and j share many neighbors
                    common_neighbors = np.sum(A[i, :] * A[j, :])
                    total_neighbors = np.sum(A[i, :] + A[j, :])
                    if total_neighbors > 0:
                        similarity = common_neighbors / total_neighbors
                        bonus[i, j] = similarity

        return bonus

    def _adaptive_integration(
        self, A_t_T: np.ndarray, A_t_S: np.ndarray, stability_score: float
    ) -> np.ndarray:
        """Integration with stability-based mixing."""
        # Adjust beta based on stability
        target_beta = 0.5 + 0.3 * (1 - stability_score)  # 0.5-0.8 range
        self.beta = 0.7 * self.beta + 0.3 * target_beta  # Smooth beta changes

        return self.beta * A_t_T + (1.0 - self.beta) * A_t_S

    def _update_beta(self, actual_adj: np.ndarray, predicted_adj: np.ndarray) -> None:
        """
        Enhanced beta adaptation based on prediction errors and density changes.
        """
        # Compute current error
        diff = actual_adj - predicted_adj
        mse = np.mean(diff**2)
        self._prev_mses.append(mse)

        # Track density change
        actual_density = np.mean(actual_adj)
        pred_density = np.mean(predicted_adj)
        density_error = abs(actual_density - pred_density)

        # Compute weighted error combining MSE and density error
        if len(self._prev_mses) > 1:
            # Weight recent errors more heavily
            weights = np.exp(np.arange(len(self._prev_mses)))
            weights = weights / np.sum(weights)
            weighted_mse = np.sum(weights * np.array(self._prev_mses))

            # Equal weight to MSE and density error
            delta = 0.5 * weighted_mse + 0.5 * density_error

            # More aggressive beta adaptation
            self.beta = 1.0 / (1.0 + np.exp(delta * 8))
            logger.debug(
                f"Updated beta to {self.beta:.3f} (MSE: {mse:.3f}, density error: {density_error:.3f})"
            )

        # Store metrics
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(mse)
        self._metrics["mae"].append(mae)
        self._metrics["rmse"].append(rmse)
