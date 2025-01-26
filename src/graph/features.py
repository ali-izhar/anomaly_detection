# src/graph/features.py

"""Extracts features from networkx graphs."""

from abc import ABC, abstractmethod
from typing import Dict, List

import logging
import warnings
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""

    @abstractmethod
    def extract(self, graph: nx.Graph) -> Dict[str, List[float]]:
        """Extract features from graph.

        Args:
            graph: Input graph
        Returns:
            Dict mapping feature names to lists of raw values
        """
        pass

    @property
    def feature_names(self) -> List[str]:
        """Get list of features this extractor provides."""
        return []

    def _handle_empty_graph(self) -> Dict[str, List[float]]:
        """Return empty feature values for an empty graph."""
        return {name: [] for name in self.feature_names}


class BasicMetricsExtractor(BaseFeatureExtractor):
    """Extracts basic network metrics (degrees, density, clustering)."""

    @property
    def feature_names(self) -> List[str]:
        return ["degrees", "density", "clustering"]

    def extract(self, graph: nx.Graph) -> Dict[str, List[float]]:
        """Extract basic network metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with raw metrics:
            - degrees: List of node degrees
            - density: float
            - clustering: List of node clustering coefficients
        """
        if graph.number_of_nodes() == 0:
            return {"degrees": [], "density": 0.0, "clustering": []}

        degrees = [float(deg) for _, deg in graph.degree()]
        clustering = list(nx.clustering(graph).values())

        return {
            "degrees": degrees,
            "density": float(nx.density(graph)),
            "clustering": clustering,
        }


class CentralityMetricsExtractor(BaseFeatureExtractor):
    """Extracts centrality metrics (betweenness, eigenvector, closeness)."""

    @property
    def feature_names(self) -> List[str]:
        return ["betweenness", "eigenvector", "closeness"]

    def extract(self, graph: nx.Graph) -> Dict[str, List[float]]:
        """Extract centrality metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with raw metrics:
            - betweenness: List of node betweenness values
            - eigenvector: List of node eigenvector values
            - closeness: List of node closeness values
        """
        if graph.number_of_nodes() == 0:
            return self._handle_empty_graph()

        # Compute betweenness centrality
        betweenness = list(nx.betweenness_centrality(graph).values())

        # Compute eigenvector centrality using numpy (more robust)
        adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        largest_eigenvalue_idx = np.argmax(eigenvalues)
        eigenvector_values = np.abs(eigenvectors[:, largest_eigenvalue_idx])
        eigenvector_values = eigenvector_values / eigenvector_values.sum()

        # Compute closeness centrality
        closeness = list(nx.closeness_centrality(graph).values())

        return {
            "betweenness": betweenness,
            "eigenvector": list(eigenvector_values),
            "closeness": closeness,
        }


class SpectralMetricsExtractor(BaseFeatureExtractor):
    """Extracts spectral metrics from adjacency and Laplacian matrices."""

    @property
    def feature_names(self) -> List[str]:
        return ["singular_values", "laplacian_eigenvalues"]

    def extract(self, graph: nx.Graph) -> Dict[str, List[float]]:
        """Extract spectral metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with raw metrics:
            - singular_values: List of adjacency matrix singular values
            - laplacian_eigenvalues: List of Laplacian eigenvalues
        """
        if graph.number_of_nodes() == 0:
            return self._handle_empty_graph()

        # Get singular values of adjacency matrix
        A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        singular_values = list(np.linalg.svd(A, compute_uv=False))

        # Get Laplacian eigenvalues
        L = nx.laplacian_matrix(graph).todense()
        laplacian_eigenvalues = list(np.linalg.eigvalsh(L))

        return {
            "singular_values": [float(s) for s in singular_values],
            "laplacian_eigenvalues": [float(e) for e in laplacian_eigenvalues],
        }


class NetworkFeatureExtractor:
    """Main class for extracting network features."""

    def __init__(self, extractors: Dict[str, BaseFeatureExtractor] = None):
        """Initialize with feature extractors.

        Args:
            extractors: Dict mapping extractor names to instances
        """
        if extractors is None:
            extractors = {
                "basic": BasicMetricsExtractor(),
                "centrality": CentralityMetricsExtractor(),
                "spectral": SpectralMetricsExtractor(),
            }
        self.extractors = extractors

    @property
    def available_features(self) -> List[str]:
        """Get list of all available feature types."""
        return list(self.extractors.keys())

    def get_features(
        self, graph: nx.Graph, feature_types: List[str] = None
    ) -> Dict[str, List[float]]:
        """Extract raw features from graph.

        Args:
            graph: Input graph
            feature_types: List of feature types to extract (default: all)
        Returns:
            Dict mapping feature names to lists of raw values
        """
        if feature_types is None:
            feature_types = self.available_features

        features = {}
        for ftype in feature_types:
            if ftype in self.extractors:
                features.update(self.extractors[ftype].extract(graph))
            else:
                logger.warning(f"Unknown feature type: {ftype}")

        return features
