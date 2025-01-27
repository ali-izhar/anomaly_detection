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
from scipy import optimize
from collections import deque
import logging

# Setup logging
logger = logging.getLogger(__name__)


class GraphPredictor:
    """A predictor that replicates the paper's approach:
    1) Weighted average (temporal memory).
    2) Structural role preservation.
    3) Adaptive mixing parameter.
    """

    def __init__(
        self,
        k: int = 10,
        alpha: float = 0.8,
        gamma: float = 0.1,
        initial_beta: float = 0.5,
        error_window: int = 5,  # Window size for error tracking
    ):
        """
        Parameters
        ----------
        k : int
            Size of the historical window to look back for temporal predictions.
        alpha : float
            Decay factor for weighted averaging of adjacency matrices, in (0,1).
        gamma : float
            Coefficient controlling how strongly we enforce structural constraints.
        initial_beta : float
            Starting value for the adaptive mixing parameter beta_t.
        error_window : int
            Number of recent prediction errors to track for beta adaptation.
        """
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.beta = initial_beta  # will adapt over time
        self._prev_mse = None  # track recent prediction error for adaptivity
        self._error_window = error_window
        self._prev_mses = deque(maxlen=error_window)  # Track recent MSEs
        self._metrics = {"mae": [], "rmse": []}  # Track performance metrics

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
        Adapt the mixing parameter beta based on the MSE between
        predicted adjacency and the actual adjacency observed.
        """
        # Compute error
        diff = actual_adj - predicted_adj
        mse = np.mean(diff**2)

        # Update error history
        self._prev_mses.append(mse)

        # Compute weighted delta using error window
        if len(self._prev_mses) > 1:
            # Exponential weighting of recent errors
            weights = np.exp(np.arange(len(self._prev_mses)))
            weights = weights / np.sum(weights)
            delta = np.sum(weights * np.array(self._prev_mses))
        else:
            delta = mse

        # Update beta using sigmoid function
        self.beta = 1.0 / (1.0 + np.exp(delta))

        # Store current MSE
        self._prev_mse = mse

        # Update metrics
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(mse)
        self._metrics["mae"].append(mae)
        self._metrics["rmse"].append(rmse)

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
        Weighted average of up to k adjacency matrices in 'extended_history',
        using a geometric decay factor alpha^j.
        """
        # We'll pick the last k from extended_history if it is longer than k
        use_history = (
            extended_history[-self.k :]
            if len(extended_history) >= self.k
            else extended_history
        )
        m = len(use_history)
        # compute weights alpha^j
        # e.g. w_j = alpha^j / sum_{l=1..m} alpha^l
        # note we might do j from 0..m-1 or 1..m
        # We'll do j=1..m with the newest adjacency having the largest alpha^m
        # but you can invert if you want older -> smaller alpha^1
        weights_unnorm = []
        for j in range(1, m + 1):
            weights_unnorm.append(self.alpha**j)
        sum_w = sum(weights_unnorm)
        weights = [w / sum_w for w in weights_unnorm]

        # Weighted sum
        N = use_history[-1].shape[0]
        A_T = np.zeros((N, N))
        for idx, mat in enumerate(reversed(use_history)):
            # reversed so that idx=0 => newest adjacency has alpha^m
            A_T += weights[idx] * mat

        return A_T

    def _structural_preservation(self, A_t_T: np.ndarray) -> np.ndarray:
        """
        Project the matrix A_t_T onto a "structurally correct" adjacency A_t_S,
        by solving:
           A_t_S = argmin_{A in A_set} ||A_t_T - A||_F + gamma * R(A)
        where R(A) is a penalty term for structural constraints.

        For simplicity, we do a naive approach:
         1) Round A_t_T > 0.5 => 1, else 0.
         2) Possibly tweak degrees or edge density if needed.

        More elaborate methods can be done by actual optimization.
        """
        # Keep existing implementation for basic constraints
        A_bin = (A_t_T > 0.5).astype(float)
        np.fill_diagonal(A_bin, 0.0)
        A_bin = np.maximum(A_bin, A_bin.T)

        # Add structural optimization if gamma > 0
        if self.gamma > 0:
            A_opt = self._optimize_structure(A_t_T, A_bin)
            return A_opt

        return A_bin

    def _optimize_structure(self, A_t_T: np.ndarray, A_init: np.ndarray) -> np.ndarray:
        """Optimize the adjacency matrix to preserve structural properties."""
        N = A_t_T.shape[0]
        x0 = A_init.flatten()
        logger.debug(f"Starting structural optimization for {N}x{N} matrix")

        def objective(x):
            A = x.reshape(N, N)
            diff = A - A_t_T
            frobenius = np.sum(diff * diff)
            penalty = np.sum(np.abs(A - A.T))
            obj_value = frobenius + self.gamma * penalty
            logger.debug(f"Objective value: {obj_value:.3f}")
            return obj_value

        # Gradient of objective
        def gradient(x):
            A = x.reshape(N, N)
            grad = 2 * (A - A_t_T)
            grad += self.gamma * (
                np.ones_like(A) - np.ones_like(A).T
            )  # Symmetry gradient
            return grad.flatten()

        # Constraints
        constraints = []

        # 1. Symmetry constraint: A[i,j] = A[j,i]
        for i in range(N):
            for j in range(i + 1, N):
                idx1 = i * N + j
                idx2 = j * N + i
                # Create constraint vector
                c = np.zeros(N * N)
                c[idx1] = 1
                c[idx2] = -1

                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x, c=c: np.array(
                            [c.dot(x)]
                        ),  # Wrap in array for consistent dimensions
                        "jac": lambda x, c=c: c.reshape(1, -1),  # Ensure Jacobian is 2D
                    }
                )

        # 2. Diagonal constraint: A[i,i] = 0
        for i in range(N):
            idx = i * N + i
            c = np.zeros(N * N)
            c[idx] = 1
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, c=c: np.array([c.dot(x)]),  # Wrap in array
                    "jac": lambda x, c=c: c.reshape(1, -1),  # 2D Jacobian
                }
            )

        # Bounds: 0 ≤ A[i,j] ≤ 1
        bounds = [(0, 1) for _ in range(N * N)]

        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        # Reshape result back to matrix
        A_opt = result.x.reshape(N, N)

        # Ensure binary values
        A_opt = (A_opt > 0.5).astype(float)

        # Final cleanup
        np.fill_diagonal(A_opt, 0.0)  # Ensure zero diagonal
        A_opt = np.maximum(A_opt, A_opt.T)  # Ensure symmetry

        logger.debug(f"Optimization completed with status: {result.status}")
        return A_opt

    def _adaptive_integration(self, A_t_T: np.ndarray, A_t_S: np.ndarray) -> np.ndarray:
        """
        Combine the temporal adjacency and structural adjacency with the
        current mixing parameter self.beta:
            A_final = beta * A_t_T + (1 - beta) * A_t_S
        """
        return self.beta * A_t_T + (1.0 - self.beta) * A_t_S
