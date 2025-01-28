# src/predictor/aware.py

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from typing import List, Dict, Any
import networkx as nx
import logging

from .base import BasePredictor

logger = logging.getLogger(__name__)


class ChangepointAwarePredictor(BasePredictor):
    """Robust predictor using multivariate change detection and model-aware corrections."""

    def __init__(self, n_nodes: int = 50, alpha: float = 0.99, min_phase: int = 40):
        super().__init__()
        self.n = n_nodes
        self.alpha = alpha  # Forgetting factor for covariance estimation
        self.min_phase = min_phase
        self._history_size = min_phase

        # State tracking
        self.current_model = None
        self.cov_estimator = LedoitWolf(store_precision=False)
        self.feature_buffer = []
        self.model_probs = None

        # Model parameters
        self.models = {
            "ba": {"m": 1, "hub_degree": None},
            "ws": {"k": 6, "p": 0.1},
            "sbm": {"blocks": None, "intra_prob": 0.95},
            "er": {"p": 0.05},
        }

        # Pre-trained GMM for model classification
        self.gmm = GaussianMixture(n_components=4, covariance_type="tied")

    def predict(
        self, history: List[Dict[str, Any]], horizon: int = 1
    ) -> List[np.ndarray]:
        # Multivariate change detection
        if self._detect_change(history):
            self._identify_model(history[-1]["adjacency"])

        # Model-aware prediction
        preds = []
        for _ in range(horizon):
            if self.current_model == "ba":
                pred = self._ba_predictor()
            elif self.current_model == "ws":
                pred = self._ws_predictor()
            elif self.current_model == "sbm":
                pred = self._sbm_predictor()
            else:
                pred = self._er_predictor()

            preds.append(self._threshold(pred))

        return preds

    def update_state(self, actual_state: Dict[str, Any]) -> None:
        """Update predictor state with new observation."""
        adj = actual_state["adjacency"]
        features = self._extract_features(adj)
        feature_vector = np.array([v for v in features.values()])

        # Maintain rolling window of features
        self.feature_buffer.append(feature_vector)
        if len(self.feature_buffer) > 100:
            self.feature_buffer.pop(0)

        # Update covariance estimator with all available features
        if len(self.feature_buffer) > 10:
            feature_matrix = np.array(self.feature_buffer)
            self.cov_estimator.fit(feature_matrix)

    def _detect_change(self, history: List) -> bool:
        """Multivariate change detection using regularized covariance."""
        if len(history) < 2 * self.min_phase:
            return False

        # Split into reference and test windows
        feature_matrix = np.array(self.feature_buffer)
        ref = feature_matrix[-self.min_phase : -self.min_phase // 2]
        test = feature_matrix[-self.min_phase // 2 :]

        # Compute Hotelling's T² with regularized covariance
        ref_mean = np.mean(ref, axis=0)
        test_mean = np.mean(test, axis=0)

        try:
            # Fit covariance on reference window
            self.cov_estimator.fit(ref)
            cov = self.cov_estimator.covariance_
            inv_cov = np.linalg.pinv(cov)

            # Compute test statistic
            T2 = (test_mean - ref_mean).T @ inv_cov @ (test_mean - ref_mean)
            F_stat = (len(ref) * len(test)) / (len(ref) + len(test)) * T2
            return F_stat > 7.8  # 99% CI for 5 features
        except:
            logger.warning("Covariance estimation failed, skipping change detection")
            return False

    def _identify_model(self, adj: np.ndarray) -> None:
        """Bayesian model selection using topological signatures."""
        # Precomputed model signatures from training data
        model_params = {
            "ba": {"degree_skew": 2.1, "clustering": 0.12},
            "ws": {"degree_skew": 0.8, "clustering": 0.42},
            "sbm": {"degree_skew": 1.3, "clustering": 0.25},
            "er": {"degree_skew": 0.2, "clustering": 0.05},
        }

        features = self._extract_features(adj)
        log_probs = {}

        # Compute likelihood under each model
        for model, params in model_params.items():
            residuals = np.array([features[k] - v for k, v in params.items()])
            log_probs[model] = -0.5 * residuals.T @ np.diag([1, 0.1]) @ residuals

        # Softmax selection
        self.current_model = max(log_probs, key=log_probs.get)
        self._update_model_params(adj)

    def _update_model_params(self, adj: np.ndarray) -> None:
        """Model-specific parameter estimation."""
        if self.current_model == "ba":
            degrees = adj.sum(axis=0)
            self.models["ba"]["m"] = int(np.median(degrees) // 2)
            self.models["ba"]["hub_degree"] = np.max(degrees)
        elif self.current_model == "sbm":
            self.models["sbm"]["blocks"] = SpectralClustering(
                n_clusters=2, affinity="precomputed"
            ).fit_predict(adj)
            intra_edges = adj[self.models["sbm"]["blocks"] == 0][
                :, self.models["sbm"]["blocks"] == 0
            ]
            self.models["sbm"]["intra_prob"] = intra_edges.mean()

    # Model-specific predictors
    def _ba_predictor(self) -> np.ndarray:
        """BA growth model with preferential attachment."""
        m = self.models["ba"]["m"]
        hub_boost = np.log(self.models["ba"]["hub_degree"] + 1)
        prob = np.outer(np.arange(self.n) ** hub_boost, np.arange(self.n) ** hub_boost)
        return prob / prob.max()

    def _sbm_predictor(self) -> np.ndarray:
        """SBM predictor with block structure preservation."""
        prob = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.models["sbm"]["blocks"][i] == self.models["sbm"]["blocks"][j]:
                    prob[i, j] = self.models["sbm"]["intra_prob"]
                else:
                    prob[i, j] = 0.01
        return prob

    def _threshold(self, pred: np.ndarray) -> np.ndarray:
        """Optimal thresholding preserving degree distribution."""
        if len(self.feature_buffer) == 0:
            target_density = 0.05  # Default density if no history
        else:
            # Access density as the last feature in the feature vector
            target_density = self.feature_buffer[-1][4]  # Density is the 5th feature

        sorted_edges = np.sort(pred[np.triu_indices(self.n, k=1)])
        threshold = sorted_edges[-int(target_density * (self.n * (self.n - 1) / 2))]
        adj = (pred >= threshold).astype(float)
        return np.maximum(adj, adj.T)

    def _extract_features(self, adj: np.ndarray) -> dict:
        """Efficient feature extraction with topological invariants."""
        degrees = adj.sum(axis=0)
        n = adj.shape[0]
        density = np.sum(adj) / (n * (n - 1))  # Add density calculation

        return {
            "degree_skew": (degrees.max() - degrees.mean()) / degrees.std(),
            "clustering": nx.average_clustering(nx.from_numpy_array(adj)),
            "assortativity": nx.degree_assortativity_coefficient(
                nx.from_numpy_array(adj)
            ),
            "spectral_gap": np.linalg.eigvalsh(adj)[-2] - np.linalg.eigvalsh(adj)[-3],
            "density": density,  # Add density to features
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current predictor state."""
        return {
            "current_model": self.current_model,
            "model_parameters": (
                self.models[self.current_model] if self.current_model else None
            ),
            "feature_buffer_size": len(self.feature_buffer),
            "model_probabilities": self.model_probs,
        }

    def reset(self) -> None:
        """Reset predictor to initial state."""
        self.current_model = None
        self.feature_buffer = []
        self.model_probs = None
        self.cov_estimator = LedoitWolf(store_precision=False)

        # Reset model parameters to defaults
        self.models = {
            "ba": {"m": 1, "hub_degree": None},
            "ws": {"k": 6, "p": 0.1},
            "sbm": {"blocks": None, "intra_prob": 0.95},
            "er": {"p": 0.05},
        }

    def _ws_predictor(self) -> np.ndarray:
        """WS predictor preserving local clustering."""
        k = self.models["ws"]["k"]
        p = self.models["ws"]["p"]

        # Create base lattice structure
        adj = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(1, k // 2 + 1):
                adj[i, (i + j) % self.n] = 1
                adj[i, (i - j) % self.n] = 1

        # Add random rewiring probability
        rewire_mask = np.random.random(adj.shape) < p
        adj = np.where(rewire_mask, 1 - adj, adj)

        return np.maximum(adj, adj.T)

    def _er_predictor(self) -> np.ndarray:
        """ER predictor with density preservation."""
        if len(self.feature_buffer) > 0:
            # Access density as the last feature in the feature vector
            p = self.feature_buffer[-1][4]  # Density is the 5th feature
        else:
            p = self.models["er"]["p"]
        return np.random.random((self.n, self.n)) < p
