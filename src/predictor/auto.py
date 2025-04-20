# src/predictor/auto.py

#####################################
#### EXPERIMENTAL AUTO PREDICTOR ####
#####################################

"""Predictor with automatic changepoint detection using feature distribution monitoring."""

from typing import List, Dict, Any
from collections import deque

import logging
import numpy as np
import networkx as nx

from sklearn.cluster import SpectralClustering
from scipy.stats import wasserstein_distance

from .base import BasePredictor


logger = logging.getLogger(__name__)


class AutoChangepointPredictor(BasePredictor):
    """Predictor with automatic changepoint detection using feature distribution monitoring."""

    def __init__(
        self, n_history: int = 5, alpha: float = 0.85, min_phase_length: int = 40
    ):
        """
        Parameters
        ----------
        n_history : int
            Number of historical states to keep
        alpha : float

            Exponential decay factor for temporal weighting
        min_phase_length : int
            Minimum timesteps between changepoints (40 as specified)
        """
        self.n_history = n_history
        self.alpha = alpha
        self.min_phase_length = min_phase_length
        self.current_phase_start = 0
        self.structural_params = {}
        self.history = []
        self.feature_history = deque(maxlen=100)

        self.network_type = None
        self._initialize_feature_tracking()

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        """
        Predict future network states with changepoint awareness.

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
        if not history:
            raise ValueError("Need at least one historical state")

        if len(history) < self.n_history:
            raise ValueError(f"Need at least {self.n_history} historical states")

        # Extract current adjacency and update feature tracking
        current_adj = history[-1]["adjacency"]
        features = self._compute_features(current_adj)
        for k in self.feature_buffers:
            self.feature_buffers[k].append(features[k])

        # Detect changepoint
        t = len(history) - 1  # Current timestep
        if self._detect_changepoint(features):
            logger.info(f"Changepoint detected at t={t}")
            self.current_phase_start = t
            self.history = [current_adj.copy()]
            self._update_structural_params(current_adj)
        else:
            self.history.append(current_adj.copy())

        # Maintain phase history
        self.history = self.history[-self.min_phase_length :]

        predictions = []
        for _ in range(horizon):
            temp_pred = self._temporal_prediction()
            struct_pred = self._apply_structural_correction(temp_pred)
            binary_pred = self._threshold_matrix(struct_pred)
            predictions.append(binary_pred)
            self.history.append(binary_pred.copy())

        # Keep history length in check after predictions
        self.history = self.history[-self.min_phase_length :]

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
        features = self._compute_features(actual_adj)

        # Update feature tracking
        for k in self.feature_buffers:
            self.feature_buffers[k].append(features[k])

        # Update network type if needed
        new_type = self._detect_network_type(features)
        if new_type != self.network_type:
            self.network_type = new_type
            self._update_structural_params(actual_adj)

        # Update history
        self.history.append(actual_adj.copy())
        self.history = self.history[-self.min_phase_length :]  # Maintain length

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the predictor.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing current predictor state and metrics
        """
        return {
            "parameters": {
                "alpha": self.alpha,
                "network_type": self.network_type,
                "current_phase_start": self.current_phase_start,
            },
            "features": {k: list(v) for k, v in self.feature_buffers.items()},
            "structural_params": self.structural_params,
            "history_length": len(self.history),
        }

    def reset(self) -> None:
        """Reset the predictor to its initial state."""
        self.current_phase_start = 0
        self.structural_params = {}
        self.history = []
        self.feature_history.clear()
        self.network_type = None
        self._initialize_feature_tracking()

    def _initialize_feature_tracking(self):
        """Initialize feature tracking buffers"""
        self.feature_buffers = {
            "density": deque(maxlen=50),
            "clustering": deque(maxlen=50),
            "degree_mean": deque(maxlen=50),
            "degree_std": deque(maxlen=50),
            "spectral_gap": deque(maxlen=50),
        }

    def _compute_features(self, adj: np.ndarray) -> dict:
        """Compute network features for change detection"""
        G = nx.from_numpy_array(adj)
        features = {
            "density": nx.density(G),
            "clustering": nx.average_clustering(G),
            "degree_mean": np.mean([d for _, d in G.degree()]),
            "degree_std": np.std([d for _, d in G.degree()]),
        }

        # Spectral features
        try:
            vals, _ = np.linalg.eigh(adj)
            features["spectral_gap"] = np.sort(np.abs(vals))[-2]
        except:
            features["spectral_gap"] = 0

        return features

    def _detect_changepoint(self, current_features: dict) -> bool:
        """Detect distribution shifts using Wasserstein distance with phase constraints"""
        if len(self.feature_buffers["density"]) < 20:
            return False  # Not enough history

        # Check minimum phase duration
        if len(self.history) - self.current_phase_start < self.min_phase_length:
            return False

        # Compute distribution distances for key features
        distances = []
        for feature in ["degree_mean", "degree_std", "clustering"]:
            hist_vals = np.array(self.feature_buffers[feature])
            current_val = current_features[feature]

            # Use Wasserstein distance between hist and current
            distances.append(wasserstein_distance(hist_vals, [current_val]))

        combined_score = np.mean(distances)
        threshold = 0.15  # Empirical threshold, could be parameterized

        # Detect network type change
        old_type = self.network_type
        new_type = self._detect_network_type(current_features)
        type_changed = (new_type != old_type) if old_type is not None else False

        return combined_score > threshold or type_changed

    def _detect_network_type(self, features: dict) -> str:
        """Classify network type using feature thresholds"""
        if features["clustering"] > 0.3 and features["degree_std"] < 1.5:
            return "ws"
        if features["degree_std"] > 2.0 and features["spectral_gap"] > 0.5:
            return "ba"
        if features["clustering"] > 0.2 and 1.0 < features["degree_std"] < 2.0:
            return "sbm"
        return "er"

    def _update_structural_params(self, adj: np.ndarray):
        """Update structural parameters based on current network type"""
        self.network_type = self._detect_network_type(self._compute_features(adj))

        if self.network_type == "sbm":
            n = adj.shape[0]
            labels = SpectralClustering(
                n_clusters=2, affinity="precomputed", random_state=42
            ).fit_predict(adj)
            self.structural_params = {"communities": labels}
        elif self.network_type == "ba":
            degrees = adj.sum(axis=0)
            hubs = np.argsort(degrees)[-int(adj.shape[0] * 0.1) :]
            self.structural_params = {"hubs": hubs}
        elif self.network_type == "ws":
            G = nx.from_numpy_array(adj)
            self.structural_params = {"clustering": nx.average_clustering(G)}
        else:
            self.structural_params = {}

    def _apply_structural_correction(self, pred: np.ndarray) -> np.ndarray:
        """Apply network-type specific structural corrections"""
        if self.network_type == "sbm":
            return self._correct_sbm(pred)
        elif self.network_type == "ba":
            return self._correct_ba(pred)
        elif self.network_type == "ws":
            return self._correct_ws(pred)
        return pred

    def _correct_sbm(self, pred: np.ndarray) -> np.ndarray:
        communities = self.structural_params.get("communities", np.zeros(pred.shape[0]))
        for i in range(pred.shape[0]):
            for j in range(i + 1, pred.shape[0]):
                if communities[i] == communities[j]:
                    pred[i, j] = min(pred[i, j] * 1.2, 1.0)
                else:
                    pred[i, j] *= 0.8
        return pred

    def _correct_ba(self, pred: np.ndarray) -> np.ndarray:
        hubs = self.structural_params.get("hubs", [])
        for hub in hubs:
            pred[hub] = np.clip(pred[hub] * 1.3, 0, 1)
            pred[:, hub] = pred[hub]
        return pred

    def _correct_ws(self, pred: np.ndarray) -> np.ndarray:
        G = nx.from_numpy_array(pred)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    pred[neighbors[i], neighbors[j]] = max(
                        pred[neighbors[i], neighbors[j]], 0.7
                    )
        return pred

    def _temporal_prediction(self) -> np.ndarray:
        """Exponential weighted average of current phase's history"""
        m = len(self.history)
        weights = np.array([self.alpha ** (m - i - 1) for i in range(m)])
        weights /= weights.sum()
        return np.sum([w * adj for w, adj in zip(weights, self.history)], axis=0)

    def _threshold_matrix(self, pred: np.ndarray) -> np.ndarray:
        """Binarize while preserving phase density"""
        if not self.history:
            return (pred > 0.5).astype(float)

        target_density = np.mean(self.history[-1])
        n = pred.shape[0]
        triu = np.triu_indices(n, k=1)
        probs = pred[triu]
        threshold = np.quantile(probs, 1 - target_density)
        binary = (pred >= threshold).astype(float)
        np.fill_diagonal(binary, 0)
        return np.maximum(binary, binary.T)
