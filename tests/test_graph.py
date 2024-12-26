# tests/test_graph.py

"""Test suite for graph generation and feature extraction.

This module provides tests for the graph generator and feature extraction modules. It verifies:
1. Graph generation for different models
2. Feature computation correctness
3. Performance benchmarks
4. Edge cases and error handling
"""

import unittest
import numpy as np
import networkx as nx
import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.graph_generator import GraphGenerator
from src.graph.features import (
    extract_centralities,
    compute_embeddings,
    strangeness_point,
    compute_laplacian,
    compute_graph_statistics,
)
from src.graph.params import (
    BAParams,
    ERParams,
    SBMParams,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGraphGeneration(unittest.TestCase):
    """Test suite for graph generation functionality."""

    def setUp(self):
        """Initialize graph generator and common parameters."""
        self.generator = GraphGenerator()

        # Register graph models with correct parameter mapping
        self.generator.register_model(
            "BA",
            lambda n, initial_edges, **kwargs: nx.barabasi_albert_graph(
                n=n, m=initial_edges
            ),
            BAParams,
        )
        self.generator.register_model(
            "ER",
            lambda n, initial_prob, **kwargs: nx.erdos_renyi_graph(n=n, p=initial_prob),
            ERParams,
        )
        self.generator.register_model(
            "SBM",
            lambda n, num_blocks, initial_intra_prob, initial_inter_prob, **kwargs: nx.stochastic_block_model(
                sizes=[n // num_blocks] * num_blocks,
                p=[
                    [
                        initial_intra_prob if i == j else initial_inter_prob
                        for j in range(num_blocks)
                    ]
                    for i in range(num_blocks)
                ],
            ),
            SBMParams,
        )

        # Common parameters - adjusted for feasible change points
        self.base_params = {
            "n": 100,
            "seq_len": 100,
            "min_segment": 10,
            "min_changes": 1,
            "max_changes": 3,
        }

    def test_ba_generation(self):
        """Test Barabási-Albert graph generation."""
        params = BAParams(
            **self.base_params, initial_edges=3, min_edges=2, max_edges=5, pref_exp=1.0
        )

        sequence = self.generator.generate_sequence("BA", params)

        self.assertEqual(len(sequence["graphs"]), params.seq_len)
        self.assertTrue(
            all(g.shape == (params.n, params.n) for g in sequence["graphs"])
        )

        G = nx.from_numpy_array(sequence["graphs"][0])

        # Get degree distribution
        degrees = [d for _, d in G.degree()]
        max_degree = max(degrees)
        min_degree = min(degrees)

        # Verify minimum degree is at least half of initial_edges
        # (since initial nodes can have lower degree)
        self.assertGreaterEqual(min_degree, params.initial_edges // 2)

        # Verify most nodes have degree >= initial_edges
        degree_counts = nx.degree_histogram(G)
        nodes_with_target_degree = sum(degree_counts[params.initial_edges :])
        total_nodes = sum(degree_counts)
        self.assertGreater(
            nodes_with_target_degree / total_nodes, 0.8
        )  # At least 80% of nodes

        # Verify we have a reasonable degree range for a scale-free network
        self.assertGreater(max_degree, params.initial_edges * 2)

        # Verify the distribution has a long tail (characteristic of scale-free networks)
        nonzero_degrees = [i for i, count in enumerate(degree_counts) if count > 0]
        self.assertGreater(len(nonzero_degrees), params.initial_edges)

        # Verify the degree distribution is skewed (more low-degree nodes)
        self.assertGreater(degree_counts[params.initial_edges], degree_counts[-1])

    def test_er_generation(self):
        """Test Erdős-Rényi graph generation."""
        params = ERParams(
            **self.base_params, initial_prob=0.1, min_prob=0.05, max_prob=0.15
        )

        sequence = self.generator.generate_sequence("ER", params)

        self.assertEqual(len(sequence["graphs"]), params.seq_len)

        # Verify edge probability
        G = nx.from_numpy_array(sequence["graphs"][0])
        density = nx.density(G)
        self.assertAlmostEqual(density, params.initial_prob, delta=0.05)

    def test_sbm_generation(self):
        """Test Stochastic Block Model generation."""
        params = SBMParams(
            **self.base_params,
            num_blocks=3,
            min_block_size=20,
            max_block_size=40,
            initial_intra_prob=0.3,
            initial_inter_prob=0.05,
            min_intra_prob=0.2,
            max_intra_prob=0.4,
            min_inter_prob=0.02,
            max_inter_prob=0.08,
        )

        sequence = self.generator.generate_sequence("SBM", params)

        self.assertEqual(len(sequence["graphs"]), params.seq_len)

        # Verify community structure
        G = nx.from_numpy_array(sequence["graphs"][0])
        communities = nx.community.greedy_modularity_communities(G)
        self.assertGreaterEqual(len(communities), params.num_blocks - 1)

    def test_change_points(self):
        """Test change point generation and validity."""
        params = BAParams(
            **self.base_params, initial_edges=3, min_edges=2, max_edges=5, pref_exp=1.0
        )

        sequence = self.generator.generate_sequence("BA", params)

        # Verify change points exist and are within bounds
        self.assertGreaterEqual(len(sequence["change_points"]), params.min_changes)
        self.assertLessEqual(len(sequence["change_points"]), params.max_changes)

        # Verify change points are ordered and within sequence length
        change_points = sequence["change_points"]
        self.assertTrue(
            all(0 < cp < params.seq_len for cp in change_points),
            "Change points must be within sequence length",
        )
        self.assertEqual(
            sorted(change_points),
            change_points,
            "Change points must be in ascending order",
        )

        # Verify minimum segment length
        changes = sorted([0] + change_points + [params.seq_len])
        segments = [changes[i + 1] - changes[i] for i in range(len(changes) - 1)]

        # Log segment lengths for debugging
        logger.info(f"Change points: {change_points}")
        logger.info(f"Segment lengths: {segments}")
        logger.info(
            f"Sequence length: {params.seq_len}, Min segment: {params.min_segment}"
        )
        logger.info(
            f"Min changes: {params.min_changes}, Max changes: {params.max_changes}"
        )

        for i, length in enumerate(segments):
            self.assertGreaterEqual(
                length,
                params.min_segment,
                f"Segment {i} (between points {changes[i]}-{changes[i+1]}) "
                f"length {length} is less than minimum {params.min_segment}. "
                f"All segments: {segments}",
            )


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for feature extraction functionality."""

    def setUp(self):
        """Initialize test graphs."""
        self.n_nodes = 50
        self.n_graphs = 10

        # Generate test graphs with higher density to ensure connectivity
        self.graphs = []
        for _ in range(self.n_graphs):
            G = nx.erdos_renyi_graph(self.n_nodes, 0.3)  # Increased probability
            self.graphs.append(nx.to_numpy_array(G))

    def test_centrality_computation(self):
        """Test centrality measure computation."""
        centralities = extract_centralities(
            self.graphs, measures=["degree", "betweenness"]
        )

        self.assertIn("degree", centralities)
        self.assertIn("betweenness", centralities)

        # Verify shapes
        self.assertEqual(len(centralities["degree"]), self.n_graphs)
        self.assertEqual(len(centralities["degree"][0]), self.n_nodes)

        # Verify values
        self.assertTrue(all(0 <= v <= 1 for v in centralities["degree"][0]))
        self.assertTrue(all(0 <= v <= 1 for v in centralities["betweenness"][0]))

    def test_embedding_computation(self):
        """Test graph embedding computation."""
        n_components = 5
        embeddings = compute_embeddings(
            self.graphs, method="svd", n_components=n_components
        )

        # Verify shapes
        self.assertEqual(len(embeddings), self.n_graphs)
        self.assertEqual(embeddings[0].shape, (self.n_nodes, n_components))

        # Verify orthogonality after normalization
        first_embedding = embeddings[0]
        # Normalize the embedding
        norms = np.linalg.norm(first_embedding, axis=0)
        first_embedding_normalized = first_embedding / norms

        product = first_embedding_normalized.T @ first_embedding_normalized
        np.testing.assert_array_almost_equal(product, np.eye(n_components), decimal=5)

    def test_strangeness_computation(self):
        """Test strangeness score computation."""
        n_clusters = 3
        embeddings = compute_embeddings(self.graphs, n_components=5)
        strangeness = strangeness_point(embeddings, n_clusters=n_clusters)

        # Verify shape
        self.assertEqual(strangeness.shape[0], len(self.graphs) * self.n_nodes)
        self.assertEqual(strangeness.shape[1], n_clusters)

        # Verify values are non-negative distances
        self.assertTrue(np.all(strangeness >= 0))

    def test_laplacian_computation(self):
        """Test Laplacian matrix computation."""
        # Test unnormalized Laplacian
        L = compute_laplacian(self.graphs[0])

        # Verify properties
        self.assertTrue(np.allclose(L, L.T))  # Symmetric
        self.assertTrue(np.all(np.diag(L) >= 0))  # Non-negative diagonal
        self.assertTrue(np.all(np.sum(L, axis=1) == 0))  # Row sums are 0

        # Test normalized Laplacian
        L_norm = compute_laplacian(self.graphs[0], normalized=True)
        eigenvals = np.linalg.eigvals(L_norm)
        self.assertTrue(np.all(eigenvals >= -1e-10))  # Positive semi-definite

    def test_graph_statistics(self):
        """Test graph statistics computation."""
        stats = compute_graph_statistics(self.graphs)

        self.assertIn("density", stats)
        self.assertIn("avg_degree", stats)
        self.assertIn("clustering", stats)
        self.assertIn("diameter", stats)

        # Verify values
        self.assertTrue(all(0 <= d <= 1 for d in stats["density"]))
        self.assertTrue(all(c >= 0 for c in stats["clustering"]))
        self.assertTrue(all(isinstance(d, (int, float)) for d in stats["diameter"]))


class TestPerformance(unittest.TestCase):
    """Performance tests for graph operations."""

    def setUp(self):
        """Initialize large test graphs."""
        self.n_nodes = 1000
        self.n_graphs = 5
        self.graphs = [
            nx.to_numpy_array(nx.erdos_renyi_graph(self.n_nodes, 0.01))
            for _ in range(self.n_graphs)
        ]

    def test_centrality_performance(self):
        """Test centrality computation performance."""
        start_time = time.time()

        centralities = extract_centralities(
            self.graphs, measures=["degree"], batch_size=2, n_jobs=-1
        )

        duration = time.time() - start_time
        logger.info(f"Centrality computation took {duration:.2f} seconds")

        # Should complete within reasonable time
        self.assertLess(duration, 30)

    def test_embedding_performance(self):
        """Test embedding computation performance."""
        start_time = time.time()

        embeddings = compute_embeddings(
            self.graphs, method="svd", n_components=10, use_sparse=True
        )

        duration = time.time() - start_time
        logger.info(f"Embedding computation took {duration:.2f} seconds")

        # Should complete within reasonable time
        self.assertLess(duration, 30)


if __name__ == "__main__":
    unittest.main()
