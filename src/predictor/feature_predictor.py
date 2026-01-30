"""Feature-based predictor using Holt's linear exponential smoothing with trend.

This predictor works directly in feature space rather than predicting graphs,
which is more accurate for the martingale framework that operates on features.

Key improvements over graph-based prediction:
1. Trend detection and extrapolation (not just weighted averaging)
2. Adaptive smoothing that responds to variance changes
3. Direct feature prediction avoids graph->feature extraction errors
"""

import numpy as np
from typing import List, Dict, Any, Optional


class FeaturePredictor:
    """Predicts future feature values using Holt's double exponential smoothing.

    Uses level + trend components for extrapolation:
    - Level: l_t = α * x_t + (1 - α) * (l_{t-1} + b_{t-1})
    - Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
    - Forecast: x_{t+h} = l_t + h * b_t

    Adaptive parameters respond to variance changes for faster adaptation
    after regime changes.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        n_history: int = 10,
        adaptive: bool = True,
        min_alpha: float = 0.2,
        max_alpha: float = 0.8,
        variance_window: int = 5,
    ):
        """Initialize feature predictor.

        Args:
            alpha: Smoothing parameter for level (0 < α < 1)
            beta: Smoothing parameter for trend (0 < β < 1)
            n_history: Number of historical observations to use
            adaptive: Whether to adapt parameters based on prediction error
            min_alpha: Minimum alpha value for adaptive smoothing
            max_alpha: Maximum alpha value for adaptive smoothing
            variance_window: Window size for variance-based adaptation
        """
        self.alpha = alpha
        self.beta = beta
        self.n_history = n_history
        self.adaptive = adaptive
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.variance_window = variance_window

        # State
        self.level = None  # Current level estimate per feature
        self.trend = None  # Current trend estimate per feature
        self.feature_history = []
        self.prediction_errors = []

    def fit(self, features: np.ndarray) -> None:
        """Fit the predictor on historical feature data.

        Args:
            features: Array of shape (n_timesteps, n_features)
        """
        features = np.asarray(features)
        n_timesteps, n_features = features.shape

        if n_timesteps < 2:
            raise ValueError("Need at least 2 timesteps to fit")

        # Initialize level and trend
        self.level = features[-1].copy()
        self.trend = np.zeros(n_features)

        # Compute initial trend from recent data
        if n_timesteps >= 3:
            # Use linear regression on last few points for initial trend
            recent = features[-min(5, n_timesteps):]
            for k in range(n_features):
                t = np.arange(len(recent))
                if np.std(recent[:, k]) > 1e-10:
                    slope = np.polyfit(t, recent[:, k], 1)[0]
                    self.trend[k] = slope

        # Store history
        self.feature_history = list(features[-self.n_history:])

        # Compute adaptive alpha based on recent variance
        if self.adaptive and n_timesteps >= self.variance_window:
            self._adapt_parameters(features)

    def predict(self, horizon: int = 5) -> np.ndarray:
        """Predict future feature values.

        Args:
            horizon: Number of steps ahead to predict

        Returns:
            Array of shape (horizon, n_features) with predictions
        """
        if self.level is None:
            raise ValueError("Predictor not fitted. Call fit() first.")

        predictions = []
        for h in range(1, horizon + 1):
            # Holt's forecast: x_{t+h} = l_t + h * b_t
            pred = self.level + h * self.trend
            predictions.append(pred)

        return np.array(predictions)

    def update(self, observation: np.ndarray) -> None:
        """Update predictor with new observation.

        Args:
            observation: New feature vector of shape (n_features,)
        """
        observation = np.asarray(observation)

        if self.level is None:
            # First observation - initialize
            self.level = observation.copy()
            self.trend = np.zeros_like(observation)
            self.feature_history = [observation]
            return

        # Compute prediction error for adaptive parameters
        predicted = self.level + self.trend
        error = np.abs(observation - predicted)
        self.prediction_errors.append(np.mean(error))

        # Adapt alpha if error is high (regime change)
        if self.adaptive:
            self._adapt_alpha_on_error(error)

        # Holt's update equations
        prev_level = self.level.copy()

        # Level: l_t = α * x_t + (1 - α) * (l_{t-1} + b_{t-1})
        self.level = self.alpha * observation + (1 - self.alpha) * (self.level + self.trend)

        # Trend: b_t = β * (l_t - l_{t-1}) + (1 - β) * b_{t-1}
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

        # Update history
        self.feature_history.append(observation)
        if len(self.feature_history) > self.n_history:
            self.feature_history.pop(0)

    def _adapt_parameters(self, features: np.ndarray) -> None:
        """Adapt smoothing parameters based on data characteristics."""
        # Compute variance in recent window
        recent = features[-self.variance_window:]
        variance = np.mean(np.var(recent, axis=0))

        # Higher variance -> higher alpha (more responsive)
        # Scale variance to [0, 1] range approximately
        normalized_var = min(1.0, variance / 2.0)
        self.alpha = self.min_alpha + normalized_var * (self.max_alpha - self.min_alpha)

    def _adapt_alpha_on_error(self, error: np.ndarray) -> None:
        """Increase alpha when prediction error is high (regime change)."""
        mean_error = np.mean(error)

        # If error is high, increase alpha for faster adaptation
        if mean_error > 1.0:  # Threshold for "high" error on normalized data
            # Boost alpha temporarily
            self.alpha = min(self.max_alpha, self.alpha + 0.1)
        elif mean_error < 0.3 and len(self.prediction_errors) > 5:
            # Low error - can reduce alpha for stability
            self.alpha = max(self.min_alpha, self.alpha - 0.02)

    def reset(self) -> None:
        """Reset predictor state."""
        self.level = None
        self.trend = None
        self.feature_history = []
        self.prediction_errors = []
        self.alpha = 0.3  # Reset to default


class GraphFeaturePredictor:
    """Predicts graph features by extracting features and using FeaturePredictor.

    Uses Holt's double exponential smoothing for true trend-based forecasting.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        n_history: int = 10,
        feature_names: Optional[List[str]] = None,
        **kwargs  # Accept extra kwargs for compatibility
    ):
        """Initialize graph feature predictor.

        Args:
            alpha: Smoothing parameter for level
            beta: Smoothing parameter for trend
            n_history: Number of historical observations
            feature_names: List of feature names to extract and predict
        """
        self.feature_predictor = FeaturePredictor(
            alpha=alpha, beta=beta, n_history=n_history, adaptive=True
        )
        self.feature_names = feature_names or [
            'mean_degree', 'density', 'mean_clustering', 'mean_betweenness'
        ]
        self.n_history = n_history
        self._extractor = None
        self._fitted = False

    def _get_extractor(self):
        """Lazy load feature extractor."""
        if self._extractor is None:
            from src.graph import NetworkFeatureExtractor
            self._extractor = NetworkFeatureExtractor()
        return self._extractor

    def _extract_features(self, adj_matrix: np.ndarray) -> np.ndarray:
        """Extract features from adjacency matrix."""
        from src.graph.utils import adjacency_to_graph
        extractor = self._get_extractor()
        graph = adjacency_to_graph(adj_matrix)
        numeric = extractor.get_numeric_features(graph)
        return np.array([numeric[name] for name in self.feature_names])

    def predict(self, history: List[Dict], horizon: int = 5) -> List[np.ndarray]:
        """Predict future graphs (returns adjacency matrices for compatibility).

        Note: This returns dummy adjacency matrices. Use predict_features() for
        actual feature predictions which is what the martingale needs.

        Args:
            history: List of dicts with 'adjacency' key
            horizon: Number of steps to predict

        Returns:
            List of predicted adjacency matrices (dummy - use predict_features instead)
        """
        if len(history) < self.n_history:
            raise ValueError(f"Need at least {self.n_history} historical graphs")

        # Extract features from history
        features = []
        for g in history:
            adj = np.array(g['adjacency']) if not isinstance(g['adjacency'], np.ndarray) else g['adjacency']
            features.append(self._extract_features(adj))
        features = np.array(features)

        # Fit predictor on history
        self.feature_predictor.fit(features)
        self._fitted = True

        # Store predicted features for get_predicted_features()
        self._last_predicted_features = self.feature_predictor.predict(horizon)

        # Return dummy adjacency matrices (same as last observed)
        # The martingale should use predict_features() instead
        last_adj = np.array(history[-1]['adjacency'])
        return [last_adj.copy() for _ in range(horizon)]

    def predict_features(self, history: List[Dict], horizon: int = 5) -> np.ndarray:
        """Predict future feature values directly.

        This is the preferred method for use with the martingale framework.

        Args:
            history: List of dicts with 'adjacency' key
            horizon: Number of steps to predict

        Returns:
            Array of shape (horizon, n_features) with predicted features
        """
        if len(history) < self.n_history:
            raise ValueError(f"Need at least {self.n_history} historical graphs")

        # Extract features from history
        features = []
        for g in history:
            adj = np.array(g['adjacency']) if not isinstance(g['adjacency'], np.ndarray) else g['adjacency']
            features.append(self._extract_features(adj))
        features = np.array(features)

        # Fit and predict
        self.feature_predictor.fit(features)
        self._fitted = True

        return self.feature_predictor.predict(horizon)

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """Update predictor with new observation."""
        adj = actual_state['adjacency']
        if not isinstance(adj, np.ndarray):
            adj = np.array(adj)
        features = self._extract_features(adj)
        self.feature_predictor.update(features)

    def get_state(self) -> Dict:
        """Get predictor state."""
        return {
            'alpha': self.feature_predictor.alpha,
            'n_history': self.n_history,
            'fitted': self._fitted
        }

    def reset(self) -> None:
        """Reset predictor state."""
        self.feature_predictor.reset()
        self._fitted = False
