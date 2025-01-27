# src/predictor/predictor.py

"""
Implements the predictive (preemptive) forecasting logic described in:
Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting.

 - Input: current graph adjacency, history of adjacency matrices, horizon h, etc.
 - Output: list of predicted adjacency matrices [hat{A}_{t+1}, ..., hat{A}_{t+h}].
 
Then feed these predicted adjacencies into the martingale pipeline to
compute horizon martingales, etc.
"""

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class GraphPredictor:
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
        self.alpha = alpha
        self.gamma = initial_gamma
        self.beta = initial_beta
        self._error_window = error_window
        self._prev_mses = deque(maxlen=error_window)
        self._metrics = {"mae": [], "rmse": []}
        self._density_history = deque(maxlen=k)  # Track recent densities
        self._last_prediction = None  # Store last prediction for adaptation

    def forecast(
        self,
        history_adjs: np.ndarray,
        current_adj: np.ndarray,
        h: int = 5,
    ) -> np.ndarray:
        """Forecast the next h adjacency matrices using the method in Section 5 of the paper."""
        logger.info(f"Starting forecast for horizon h={h}")

        # Initialize storage for final predictions
        N = current_adj.shape[0]
        predicted_adjs = np.zeros((h, N, N))

        # Start with the "history" plus the current adjacency as the new window
        extended_history = []
        if history_adjs is not None and len(history_adjs) > 0:
            for mat in history_adjs:
                extended_history.append(mat)
        extended_history.append(current_adj)
        logger.debug(f"Extended history size: {len(extended_history)}")

        for i in range(h):
            logger.debug(f"Predicting step {i+1}/{h}")

            # -- (A) TEMPORAL PREDICTION
            A_t_T = self._temporal_prediction(extended_history)
            logger.debug(f"Temporal prediction density: {np.mean(A_t_T):.3f}")

            # -- (B) STRUCTURAL PRESERVATION
            A_t_S = self._structural_preservation(A_t_T)
            logger.debug(f"Structural preservation density: {np.mean(A_t_S):.3f}")

            # -- (C) ADAPTIVE INTEGRATION
            A_final = self._adaptive_integration(A_t_T, A_t_S)
            logger.debug(f"Final prediction density: {np.mean(A_final):.3f}")

            predicted_adjs[i] = A_final
            extended_history.append(A_final)

        logger.info(f"Completed forecast for {h} steps")
        return predicted_adjs

    def update_beta(self, actual_adj: np.ndarray, predicted_adj: np.ndarray) -> None:
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

            # Equal weight to MSE and density error (changed from 0.7/0.3)
            delta = 0.5 * weighted_mse + 0.5 * density_error

            # More aggressive beta adaptation
            self.beta = 1.0 / (
                1.0 + np.exp(delta * 8)
            )  # Increased sensitivity from 5 to 8
            logger.debug(
                f"Updated beta to {self.beta:.3f} (MSE: {mse:.3f}, density error: {density_error:.3f})"
            )

        # Store metrics
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(mse)
        self._metrics["mae"].append(mae)
        self._metrics["rmse"].append(rmse)

        # Store prediction for next update
        self._last_prediction = predicted_adj.copy()

    def get_metrics(self) -> dict:
        """
        Get current performance metrics.

        Returns
        -------
        dict
            Dictionary containing MAE and RMSE histories
        """
        return {
            "mae": np.array(self._metrics["mae"]),
            "rmse": np.array(self._metrics["rmse"]),
            "current_beta": self.beta,
            "mean_mae": np.mean(self._metrics["mae"]) if self._metrics["mae"] else None,
            "mean_rmse": (
                np.mean(self._metrics["rmse"]) if self._metrics["rmse"] else None
            ),
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics tracking."""
        self._metrics = {"mae": [], "rmse": []}
        self._prev_mses.clear()

    # --------------------------------------------------------------------------
    # PRIVATE HELPER METHODS

    def _temporal_prediction(self, extended_history: list) -> np.ndarray:
        """
        Enhanced weighted average with adaptive decay and momentum-based stabilization.
        More aggressive adaptation to downward trends.
        """
        use_history = (
            extended_history[-self.k :]
            if len(extended_history) >= self.k
            else extended_history
        )
        m = len(use_history)

        # Compute base weights with newer matrices getting higher weights
        weights_unnorm = [self.alpha ** (m - j - 1) for j in range(m)]

        # Adapt decay rate based on recent density changes
        densities = [np.mean(mat) for mat in use_history]
        self._density_history.extend(densities)

        if len(self._density_history) > 1:
            # Measure both immediate and trend volatility
            recent_changes = np.diff(list(self._density_history)[-4:])
            immediate_volatility = np.std(recent_changes)
            trend_direction = np.mean(recent_changes)

            # Detect if we're in a transition period
            in_transition = immediate_volatility > 0.03 or abs(trend_direction) > 0.02

            if in_transition:
                # During transition, use very recent history with more aggressive weights
                weights_unnorm = [0.0] * len(weights_unnorm)
                weights_unnorm[-3:] = [0.1, 0.3, 0.6]  # More weight on most recent

                # Adjust based on trend direction
                if abs(trend_direction) > 0.02:
                    # Base trend factor
                    trend_factor = 1.0 + np.sign(trend_direction) * min(
                        abs(trend_direction), 0.1
                    )

                    # More aggressive for downward trends
                    if trend_direction < 0:
                        trend_factor *= 1.2  # 20% more aggressive on downward trends
                        # Extra weight to recent matrices for downward trends
                        weights_unnorm[-1] *= 1.2
                        weights_unnorm[-2] *= 1.1

                    weights_unnorm[-1] *= trend_factor
            else:
                # Normal operation with decay
                decay_factor = np.exp(-immediate_volatility * 8)
                weights_unnorm = [
                    w * (decay_factor**i) for i, w in enumerate(weights_unnorm)
                ]

        # Normalize weights
        sum_w = sum(weights_unnorm)
        weights = [w / sum_w for w in weights_unnorm]

        logger.debug(f"Temporal weights: {weights}")

        # Weighted sum with momentum stabilization
        N = use_history[-1].shape[0]
        A_T = np.zeros((N, N))

        for idx, mat in enumerate(use_history):
            A_T += weights[idx] * mat

        # Apply momentum stabilization if we have previous prediction
        if self._last_prediction is not None:
            current_density = np.mean(A_T)
            last_density = np.mean(self._last_prediction)
            density_diff = abs(current_density - last_density)

            if density_diff > 0.1:  # Large change
                # More aggressive momentum for downward changes
                if current_density < last_density:
                    A_T = (
                        0.8 * A_T + 0.2 * self._last_prediction
                    )  # Less smoothing for downward
                else:
                    A_T = (
                        0.7 * A_T + 0.3 * self._last_prediction
                    )  # Normal smoothing for upward

        return A_T

    def _structural_preservation(self, A_t_T: np.ndarray) -> np.ndarray:
        """
        Project the matrix A_t_T onto a structurally correct adjacency A_t_S,
        with adaptive gamma and stability controls.
        """
        # Adapt gamma based on recent prediction errors and density changes
        if len(self._prev_mses) > 0:
            mean_error = np.mean(list(self._prev_mses))

            # Check for transition period
            if len(self._density_history) > 3:
                recent_densities = list(self._density_history)[-4:]
                density_changes = np.diff(recent_densities)

                # Detect if we're in transition
                volatility = np.std(density_changes)
                trend = np.mean(density_changes)
                in_transition = volatility > 0.03 or abs(trend) > 0.02

                if in_transition:
                    # Reduce structural preservation during transition
                    self.gamma = max(0.01, self.gamma * 0.5)
                else:
                    # Normal gamma adaptation
                    self.gamma = self.gamma * np.exp(-mean_error * 4)
                    self.gamma = max(0.01, min(0.3, self.gamma))

            logger.debug(f"Adapted gamma: {self.gamma:.3f}")

        A_bin = (A_t_T > 0.5).astype(float)
        np.fill_diagonal(A_bin, 0.0)
        A_bin = np.maximum(A_bin, A_bin.T)

        if self.gamma > 0:
            A_opt = self._optimize_structure(A_t_T, A_bin)
            return A_opt

        return A_bin

    def _optimize_structure(self, A_t_T: np.ndarray, A_init: np.ndarray) -> np.ndarray:
        """
        Optimize the adjacency matrix with stability controls.
        """
        N = A_t_T.shape[0]

        # Start with high confidence edges
        A_opt = (A_t_T > 0.6).astype(float)

        # Count how many more edges we need
        target_density = np.mean(A_t_T)
        target_edges = int(target_density * N * (N - 1))
        current_edges = int(np.sum(A_opt))
        remaining_edges = max(0, target_edges - current_edges)

        if remaining_edges > 0:
            mask = ~np.eye(N, dtype=bool) & (A_opt == 0)
            probs = A_t_T[mask]

            if len(probs) > remaining_edges:
                # Add stability bonus for edges that existed in previous prediction
                if self._last_prediction is not None:
                    stability_bonus = 0.1 * self._last_prediction[mask]
                    probs = probs + stability_bonus

                # Add community structure bonus
                community_bonus = self._get_community_bonus(A_t_T)
                probs = probs + 0.2 * community_bonus[mask]

                threshold = np.sort(probs)[-remaining_edges]
                A_opt[mask] = (probs >= threshold).astype(float)

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

    def _adaptive_integration(self, A_t_T: np.ndarray, A_t_S: np.ndarray) -> np.ndarray:
        """
        Combine the temporal adjacency and structural adjacency with adaptive mixing.
        During transitions, especially favor temporal predictions.
        """
        # Check if we're in a transition period
        if len(self._density_history) > 3:
            recent_changes = np.diff(list(self._density_history)[-4:])
            volatility = np.std(recent_changes)
            trend = np.mean(recent_changes)

            if volatility > 0.03 or abs(trend) > 0.02:  # In transition
                # Increase temporal weight more aggressively
                if trend < 0:  # Downward trend
                    self.beta = min(
                        0.9, self.beta + 0.15
                    )  # More aggressive for downward
                else:
                    self.beta = min(0.8, self.beta + 0.1)  # Normal transition
                logger.debug(f"Transition detected, increased beta to {self.beta:.3f}")

        return self.beta * A_t_T + (1.0 - self.beta) * A_t_S
