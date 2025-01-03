# src/graph/features.py

"""Feature extraction module for dynamic graphs.

This module provides functions to extract various graph features and embeddings:

Categories:
1. Basic Metrics:
   - Average degree, density, clustering
   - Maximum degree, spectral properties
2. Centrality Features:
   - Degree, betweenness, eigenvector, closeness
3. Structural Features:
   - Embeddings (SVD, LSVD)
   - Community structure
4. Temporal Features:
   - Link prediction
   - Evolution patterns
5. Advanced Features:
   - Strangeness scores
   - Link prediction metrics
"""

import networkx as nx
import numpy as np
import logging
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


@dataclass
class NetworkMetrics:
    """Container for network metrics."""

    avg_degree: float
    max_degree: float
    density: float
    clustering: float
    avg_betweenness: float
    max_betweenness: float
    avg_eigenvector: float
    max_eigenvector: float
    avg_closeness: float
    spectral_gap: float
    algebraic_connectivity: float


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract features from a graph."""
        pass


class BasicMetricsExtractor(BaseFeatureExtractor):
    """Extracts basic network metrics."""

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract basic network metrics."""
        return {
            "avg_degree": np.mean([d for _, d in graph.degree()]),
            "density": nx.density(graph),
            "clustering": nx.average_clustering(graph),
            "max_degree": max(dict(graph.degree()).values()),
        }


class CentralityMetricsExtractor(BaseFeatureExtractor):
    """Extracts centrality-based metrics."""

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract centrality metrics."""
        betweenness = nx.betweenness_centrality(graph)
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
        closeness = nx.closeness_centrality(graph)

        return {
            "avg_betweenness": np.mean(list(betweenness.values())),
            "max_betweenness": max(betweenness.values()),
            "avg_eigenvector": np.mean(list(eigenvector.values())),
            "max_eigenvector": max(eigenvector.values()),
            "avg_closeness": np.mean(list(closeness.values())),
        }


class SpectralMetricsExtractor(BaseFeatureExtractor):
    """Extracts spectral metrics."""

    def extract(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract spectral metrics."""
        adj_matrix = nx.to_numpy_array(graph)
        _, S, _ = np.linalg.svd(adj_matrix)
        laplacian = nx.laplacian_matrix(graph).toarray()
        _, L_S, _ = np.linalg.svd(laplacian)

        return {
            "spectral_gap": S[0] - S[1] if len(S) > 1 else S[0],
            "algebraic_connectivity": L_S[1],
        }


class NetworkFeatureExtractor:
    """Main class for extracting network features."""

    def __init__(self):
        """Initialize feature extractors."""
        self.extractors = {
            "basic": BasicMetricsExtractor(),
            "centrality": CentralityMetricsExtractor(),
            "spectral": SpectralMetricsExtractor(),
        }

    def get_all_metrics(self, graph: nx.Graph) -> NetworkMetrics:
        """Get all network metrics in a structured format."""
        metrics = {}
        for extractor in self.extractors.values():
            metrics.update(extractor.extract(graph))

        return NetworkMetrics(**metrics)

    def get_metrics(
        self, graph: nx.Graph, metric_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get specific network metrics."""
        if metric_types is None:
            metric_types = list(self.extractors.keys())

        metrics = {}
        for metric_type in metric_types:
            if metric_type in self.extractors:
                metrics.update(self.extractors[metric_type].extract(graph))
            else:
                logger.warning(f"Unknown metric type: {metric_type}")

        return metrics


class StrangenessDetector:
    """Detects strangeness in network features."""

    def __init__(
        self,
        n_clusters: int = 1,
        random_state: Optional[int] = 42,
        batch_size: Optional[int] = None,
    ):
        """Initialize detector."""
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size

    def compute_strangeness(self, data: Union[List[Any], np.ndarray]) -> np.ndarray:
        """Compute strangeness scores.

        Parameters
        ----------
        data : Union[List[Any], np.ndarray]
            Input vectors to compute strangeness for

        Returns
        -------
        np.ndarray
            Array of strangeness scores
        """
        if not data:
            raise ValueError("Empty data sequence")

        try:
            data_array = np.array(data)
            if data_array.ndim == 3:
                data_array = data_array.reshape(-1, data_array.shape[-1])

                if self.batch_size and data_array.shape[0] > self.batch_size:
                    from sklearn.cluster import MiniBatchKMeans

                    kmeans = MiniBatchKMeans(
                        n_clusters=self.n_clusters,
                        batch_size=self.batch_size,
                        random_state=self.random_state,
                    )
            else:
                kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    n_init="auto",
                    random_state=self.random_state,
                )

                return kmeans.fit_transform(data_array)

        except Exception as e:
            logger.error(f"Strangeness computation failed: {str(e)}")
            raise RuntimeError(f"Strangeness computation failed: {str(e)}")


def calculate_error_metrics(
    actual_metrics: Dict[str, float], predicted_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Calculate error metrics between actual and predicted network properties.

    Parameters
    ----------
    actual_metrics : Dict[str, float]
        Dictionary of actual network metrics
    predicted_metrics : Dict[str, float]
        Dictionary of predicted network metrics

    Returns
    -------
    Dict[str, float]
        Dictionary containing absolute errors for each metric
    """
    return {
        key: abs(actual_metrics[key] - predicted_metrics[key])
        for key in actual_metrics.keys()
        if key in predicted_metrics
    }
