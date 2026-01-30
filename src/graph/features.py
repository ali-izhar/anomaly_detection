"""Extracts features from networkx graphs."""

from abc import ABC, abstractmethod
from typing import Dict, List

import logging
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


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

    @abstractmethod
    def extract_numeric(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract numeric summary features from graph.

        Args:
            graph: Input graph
        Returns:
            Dict mapping feature names to single numeric values
        """
        pass


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
            logger.debug(
                "Empty graph detected in BasicMetricsExtractor, returning empty features"
            )
            return {"degrees": [], "density": 0.0, "clustering": []}

        degrees = [float(deg) for _, deg in graph.degree()]
        density = float(nx.density(graph))
        clustering = list(nx.clustering(graph).values())

        logger.debug(
            f"Collected BasicMetrics. Returning {len(degrees)} degrees, {density} density, and {len(clustering)} clustering coefficients"
        )
        return {
            "degrees": degrees,
            "density": density,
            "clustering": clustering,
        }

    def extract_numeric(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract numeric summary features for basic metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with numeric summaries:
            - mean_degree: Average node degree
            - density: Graph density
            - mean_clustering: Average clustering coefficient
        """
        raw_features = self.extract(graph)
        return {
            "mean_degree": (
                np.mean(raw_features["degrees"]) if raw_features["degrees"] else 0.0
            ),
            "density": raw_features["density"],
            "mean_clustering": (
                np.mean(raw_features["clustering"])
                if raw_features["clustering"]
                else 0.0
            ),
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
            logger.debug(
                "Empty graph detected in CentralityMetricsExtractor, returning empty features"
            )
            return self._handle_empty_graph()

        # Compute betweenness centrality
        betweenness = list(nx.betweenness_centrality(graph).values())

        # Compute eigenvector centrality using numpy (more robust)
        adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        largest_eigenvalue_idx = np.argmax(eigenvalues)
        eigenvector_values = np.abs(eigenvectors[:, largest_eigenvalue_idx])
        eigenvector_values = eigenvector_values / eigenvector_values.sum()
        eigenvector_values = list(eigenvector_values)

        # Compute closeness centrality
        closeness = list(nx.closeness_centrality(graph).values())

        logger.debug(
            f"Collected CentralityMetrics. Returning {len(betweenness)} betweenness values, {len(eigenvector_values)} eigenvector values, and {len(closeness)} closeness values"
        )
        return {
            "betweenness": betweenness,
            "eigenvector": eigenvector_values,
            "closeness": closeness,
        }

    def extract_numeric(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract numeric summary features for centrality metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with numeric summaries:
            - mean_betweenness: Average betweenness centrality
            - mean_eigenvector: Average eigenvector centrality
            - mean_closeness: Average closeness centrality
        """
        raw_features = self.extract(graph)
        return {
            "mean_betweenness": (
                np.mean(raw_features["betweenness"])
                if raw_features["betweenness"]
                else 0.0
            ),
            "mean_eigenvector": (
                np.mean(raw_features["eigenvector"])
                if raw_features["eigenvector"]
                else 0.0
            ),
            "mean_closeness": (
                np.mean(raw_features["closeness"]) if raw_features["closeness"] else 0.0
            ),
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
            logger.debug(
                "Empty graph detected in SpectralMetricsExtractor, returning empty features"
            )
            return self._handle_empty_graph()

        # Get singular values of adjacency matrix
        A = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        singular_values = list(np.linalg.svd(A, compute_uv=False))

        # Get Laplacian eigenvalues
        L = nx.laplacian_matrix(graph).todense()
        laplacian_eigenvalues = list(np.linalg.eigvalsh(L))

        singular_values = [float(s) for s in singular_values]
        laplacian_eigenvalues = [float(e) for e in laplacian_eigenvalues]

        logger.debug(
            f"Collected SpectralMetrics. Returning {len(singular_values)} singular values and {len(laplacian_eigenvalues)} Laplacian eigenvalues"
        )
        return {
            "singular_values": singular_values,
            "laplacian_eigenvalues": laplacian_eigenvalues,
        }

    def extract_numeric(self, graph: nx.Graph) -> Dict[str, float]:
        """Extract numeric summary features for spectral metrics.

        Args:
            graph: Input graph
        Returns:
            Dict with numeric summaries:
            - max_singular_value: Largest singular value
            - min_nonzero_laplacian: Smallest non-zero Laplacian eigenvalue
        """
        raw_features = self.extract(graph)
        return {
            "max_singular_value": (
                max(raw_features["singular_values"])
                if raw_features["singular_values"]
                else 0.0
            ),
            "min_nonzero_laplacian": min(
                (x for x in raw_features["laplacian_eigenvalues"] if x > 1e-10),
                default=0.0,
            ),
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

    def get_numeric_features(
        self, graph: nx.Graph, feature_types: List[str] = None
    ) -> Dict[str, float]:
        """Extract numeric summary features from graph.

        Args:
            graph: Input graph
            feature_types: List of feature types to extract (default: all)
        Returns:
            Dict mapping feature names to single numeric values
        """
        if feature_types is None:
            feature_types = self.available_features

        features = {}
        for ftype in feature_types:
            if ftype in self.extractors:
                features.update(self.extractors[ftype].extract_numeric(graph))
            else:
                logger.warning(f"Unknown feature type: {ftype}")

        return features
