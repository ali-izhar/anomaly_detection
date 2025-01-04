# tests/test_graph.py

"""
Test the `src/graph` module.

1. Graph generation for different models
2. Feature computation correctness (using NetworkFeatureExtractor)
3. Performance benchmarks
4. Edge cases and error handling
"""

import unittest
import networkx as nx
import logging
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor, NetworkMetrics
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

        # Register simplified custom models using the new param fields
        self.generator.register_model(
            "BA",
            # We rename 'm' instead of 'initial_edges'
            lambda n, m, **kwargs: nx.barabasi_albert_graph(n=n, m=m),
            BAParams,
        )
        self.generator.register_model(
            "ER",
            lambda n, prob, **kwargs: nx.erdos_renyi_graph(n=n, p=prob),
            ERParams,
        )
        self.generator.register_model(
            "SBM",
            # We rename 'intra_prob'/'inter_prob' instead of 'initial_intra_prob'/'initial_inter_prob'
            lambda n, num_blocks, intra_prob, inter_prob, **kwargs: nx.stochastic_block_model(
                sizes=[n // num_blocks] * num_blocks,
                p=[
                    [intra_prob if i == j else inter_prob for j in range(num_blocks)]
                    for i in range(num_blocks)
                ],
            ),
            SBMParams,
        )

        # Common base parameters for dynamic sequences
        self.base_params = {
            "n": 100,
            "seq_len": 50,
            "min_segment": 10,
            "min_changes": 1,
            "max_changes": 2,
        }

    def test_ba_generation(self):
        """Test Barabási-Albert graph generation using new BAParams."""
        # 'm' replaces 'initial_edges'; 'min_m'/'max_m' used for anomaly injection
        params = BAParams(
            **self.base_params,
            m=3,
            min_m=2,
            max_m=5,
            n_std=None,
            m_std=None,
        )

        sequence = self.generator.generate_sequence("BA", params)

        self.assertEqual(len(sequence["graphs"]), params.seq_len)
        self.assertTrue(
            all(g.shape == (params.n, params.n) for g in sequence["graphs"])
        )

        G = nx.from_numpy_array(sequence["graphs"][0])

        # Check degree distribution
        degrees = [d for _, d in G.degree()]
        max_degree = max(degrees)
        min_degree = min(degrees)

        # Verify min_degree >= half of m (typical for small BA seeds)
        self.assertGreaterEqual(min_degree, params.m // 2)

        # Count how many nodes have degree >= m
        degree_counts = nx.degree_histogram(G)
        nodes_with_target_degree = sum(degree_counts[params.m :])
        total_nodes = sum(degree_counts)
        # At least 80% of nodes should have degree >= m for typical BA
        self.assertGreater(nodes_with_target_degree / total_nodes, 0.8)

        # Reasonable degree range for scale-free
        self.assertGreater(max_degree, params.m * 2)

        # Ensure a "long tail" by comparing counts in degree histogram
        nonzero_degrees = [i for i, cnt in enumerate(degree_counts) if cnt > 0]
        self.assertGreater(len(nonzero_degrees), params.m)
        # More low-degree nodes than very high-degree
        self.assertGreater(degree_counts[params.m], degree_counts[-1])

    def test_er_generation(self):
        """Test Erdős-Rényi graph generation using new ERParams."""
        # 'prob' replaces 'initial_prob'
        params = ERParams(
            **self.base_params,
            prob=0.1,
            min_prob=0.05,
            max_prob=0.15,
            n_std=None,
            prob_std=None,
        )

        sequence = self.generator.generate_sequence("ER", params)
        self.assertEqual(len(sequence["graphs"]), params.seq_len)

        # Check density vs. prob
        G = nx.from_numpy_array(sequence["graphs"][0])
        density = nx.density(G)
        self.assertAlmostEqual(density, params.prob, delta=0.05)

    def test_sbm_generation(self):
        """Test Stochastic Block Model generation using new SBMParams."""
        # 'intra_prob' replaces 'initial_intra_prob'; 'inter_prob' replaces 'initial_inter_prob'
        params = SBMParams(
            **self.base_params,
            num_blocks=3,
            min_block_size=15,
            max_block_size=40,
            intra_prob=0.3,
            inter_prob=0.05,
            min_intra_prob=0.2,
            max_intra_prob=0.4,
            min_inter_prob=0.02,
            max_inter_prob=0.08,
            n_std=None,
            blocks_std=None,
            intra_prob_std=None,
            inter_prob_std=None,
        )

        sequence = self.generator.generate_sequence("SBM", params)
        self.assertEqual(len(sequence["graphs"]), params.seq_len)

        G = nx.from_numpy_array(sequence["graphs"][0])
        communities = nx.community.greedy_modularity_communities(G)
        # Expect at least 'num_blocks - 1' real communities found
        self.assertGreaterEqual(len(communities), params.num_blocks - 1)

    def test_change_points(self):
        """Test that change points are generated and valid."""
        # We'll just reuse BAParams for a simpler test
        params = BAParams(
            **self.base_params,
            m=3,
            min_m=2,
            max_m=5,
        )

        sequence = self.generator.generate_sequence("BA", params)

        # Check presence and ordering of change points
        self.assertGreaterEqual(len(sequence["change_points"]), params.min_changes)
        self.assertLessEqual(len(sequence["change_points"]), params.max_changes)

        cps = sequence["change_points"]
        self.assertTrue(all(0 < cp < params.seq_len for cp in cps))
        self.assertEqual(sorted(cps), cps)

        # Each segment must be >= min_segment
        segments = [0] + cps + [params.seq_len]
        lengths = [segments[i + 1] - segments[i] for i in range(len(segments) - 1)]
        for seg_len in lengths:
            self.assertGreaterEqual(seg_len, params.min_segment)


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for aggregated feature extraction using NetworkFeatureExtractor."""

    def setUp(self):
        """Initialize test graphs (Erdos-Renyi for variety)."""
        self.n_nodes = 50
        self.n_graphs = 5

        # Create some moderate ER graphs
        self.graphs = []
        for _ in range(self.n_graphs):
            G = nx.erdos_renyi_graph(self.n_nodes, 0.3)
            self.graphs.append(G)

        self.extractor = NetworkFeatureExtractor()

    def test_centrality_metrics(self):
        """Test that centrality metrics (avg_betweenness, etc.) are computed properly."""
        G = self.graphs[0]
        metrics: NetworkMetrics = self.extractor.get_all_metrics(G)

        self.assertIsNotNone(metrics.avg_betweenness)
        self.assertIsNotNone(metrics.max_betweenness)
        self.assertGreaterEqual(metrics.avg_betweenness, 0.0)
        self.assertGreaterEqual(metrics.max_betweenness, 0.0)
        self.assertIsNotNone(metrics.avg_eigenvector)
        self.assertGreaterEqual(metrics.avg_eigenvector, 0.0)
        self.assertIsNotNone(metrics.max_eigenvector)
        self.assertGreaterEqual(metrics.max_eigenvector, 0.0)
        self.assertIsNotNone(metrics.avg_closeness)
        self.assertGreaterEqual(metrics.avg_closeness, 0.0)

    def test_basic_and_spectral_metrics(self):
        """Test basic metrics (avg_degree, density, etc.) and spectral metrics."""
        G = self.graphs[1]
        metrics: NetworkMetrics = self.extractor.get_all_metrics(G)

        # Basic checks
        self.assertGreaterEqual(metrics.avg_degree, 0.0)
        self.assertGreaterEqual(metrics.max_degree, 0.0)
        self.assertGreaterEqual(metrics.density, 0.0)
        self.assertGreaterEqual(metrics.clustering, 0.0)

        # Spectral checks
        self.assertGreaterEqual(metrics.spectral_gap, 0.0)
        self.assertGreaterEqual(metrics.algebraic_connectivity, 0.0)


class TestPerformance(unittest.TestCase):
    """Performance tests for feature extraction on larger graphs."""

    def setUp(self):
        self.n_nodes = 300
        self.n_graphs = 3
        self.graphs = [
            nx.erdos_renyi_graph(self.n_nodes, 0.01) for _ in range(self.n_graphs)
        ]
        self.extractor = NetworkFeatureExtractor()

    def test_extraction_performance(self):
        """Ensure feature extraction completes within a reasonable time for moderate size."""
        start_time = time.time()
        for G in self.graphs:
            _ = self.extractor.get_all_metrics(G)
        duration = time.time() - start_time

        logger.info(
            f"Feature extraction on {self.n_graphs} graphs took {duration:.2f}s"
        )
        self.assertLess(duration, 60, "Feature extraction took too long.")


if __name__ == "__main__":
    unittest.main()
