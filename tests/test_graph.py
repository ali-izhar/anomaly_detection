# tests/test_graph.py

"""
Test the `src/graph` module.

1. Graph generation for different models
2. Feature computation correctness (using NetworkFeatureExtractor)
3. Performance benchmarks
4. Edge cases and error handling
"""

import pytest
import numpy as np
import networkx as nx

from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor


# Load test configurations
@pytest.fixture
def test_configs():
    """Load test configurations for different graph models."""
    return {
        "ba": {
            "n": 50,
            "seq_len": 100,
            "min_segment": 20,
            "min_changes": 1,
            "max_changes": 2,
            "m": 2,
            "min_m": 1,
            "max_m": 3,
            "n_std": None,
            "m_std": 0.1,
        },
        "ws": {
            "n": 50,
            "seq_len": 100,
            "min_segment": 20,
            "min_changes": 1,
            "max_changes": 2,
            "k_nearest": 4,
            "min_k": 2,
            "max_k": 6,
            "rewire_prob": 0.1,
            "min_prob": 0.05,
            "max_prob": 0.15,
            "n_std": None,
            "k_std": 0.2,
            "prob_std": 0.01,
        },
        "er": {
            "n": 50,
            "seq_len": 100,
            "min_segment": 20,
            "min_changes": 1,
            "max_changes": 2,
            "prob": 0.1,
            "min_prob": 0.05,
            "max_prob": 0.15,
            "n_std": None,
            "prob_std": 0.01,
        },
        "sbm": {
            "n": 50,
            "seq_len": 100,
            "min_segment": 20,
            "min_changes": 1,
            "max_changes": 2,
            "num_blocks": 2,
            "min_block_size": 25,
            "max_block_size": 25,
            "intra_prob": 0.3,
            "inter_prob": 0.05,
            "min_intra_prob": 0.2,
            "max_intra_prob": 0.4,
            "min_inter_prob": 0.02,
            "max_inter_prob": 0.08,
            "n_std": None,
            "blocks_std": None,
            "intra_prob_std": 0.01,
            "inter_prob_std": 0.005,
        },
    }


# Test Graph Generation
class TestGraphGeneration:
    """Test suite for graph generation functionality."""

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_generator_initialization(self, model):
        """Test generator initialization for each model."""
        generator = GraphGenerator(model)
        assert generator.model == model

    def test_invalid_model(self):
        """Test initialization with invalid model."""
        with pytest.raises(ValueError):
            GraphGenerator("invalid_model")

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_sequence_generation(self, model, test_configs):
        """Test sequence generation for each model."""
        generator = GraphGenerator(model)
        result = generator.generate_sequence(test_configs[model])

        # Check basic properties
        assert len(result["graphs"]) == test_configs[model]["seq_len"]
        assert all(isinstance(g, np.ndarray) for g in result["graphs"])
        assert all(
            g.shape == (test_configs[model]["n"], test_configs[model]["n"])
            for g in result["graphs"]
        )

        # Check change points
        assert len(result["change_points"]) <= test_configs[model]["max_changes"]
        assert all(
            cp >= test_configs[model]["min_segment"] for cp in result["change_points"]
        )

        # Check parameters evolution
        assert len(result["parameters"]) == len(result["change_points"]) + 1

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_parameter_bounds(self, model, test_configs):
        """Test that generated parameters stay within bounds."""
        generator = GraphGenerator(model)
        result = generator.generate_sequence(test_configs[model])

        for params in result["parameters"]:
            for key, value in params.items():
                min_key = f"min_{key}"
                max_key = f"max_{key}"
                if min_key in params and max_key in params:
                    assert params[min_key] <= value <= params[max_key]


# Test Feature Extraction
class TestFeatureExtraction:
    """Test suite for feature extraction functionality."""

    @pytest.fixture
    def sample_graphs(self):
        """Generate sample graphs for testing."""
        return {
            "empty": nx.Graph(),
            "single_node": nx.Graph([(0, 0)]),
            "path": nx.path_graph(5),
            "complete": nx.complete_graph(5),
            "random": nx.gnp_random_graph(10, 0.3, seed=42),
        }

    def test_basic_metrics(self, sample_graphs):
        """Test basic metrics extraction."""
        extractor = NetworkFeatureExtractor()
        for name, graph in sample_graphs.items():
            features = extractor.get_features(graph, ["basic"])

            # Check degrees
            assert len(features["degrees"]) == graph.number_of_nodes()
            assert all(isinstance(d, float) for d in features["degrees"])

            # Check density
            assert isinstance(features["density"], float)
            assert 0 <= features["density"] <= 1

            # Check clustering
            assert len(features["clustering"]) == graph.number_of_nodes()
            assert all(0 <= c <= 1 for c in features["clustering"])

    def test_centrality_metrics(self, sample_graphs):
        """Test centrality metrics extraction."""
        extractor = NetworkFeatureExtractor()
        for name, graph in sample_graphs.items():
            if graph.number_of_nodes() > 0:  # Skip empty graph
                features = extractor.get_features(graph, ["centrality"])
                n = graph.number_of_nodes()

                # Check all centrality measures
                for measure in ["betweenness", "eigenvector", "closeness"]:
                    assert len(features[measure]) == n
                    assert all(isinstance(v, float) for v in features[measure])

    def test_spectral_metrics(self, sample_graphs):
        """Test spectral metrics extraction."""
        extractor = NetworkFeatureExtractor()
        for name, graph in sample_graphs.items():
            if graph.number_of_nodes() > 0:  # Skip empty graph
                features = extractor.get_features(graph, ["spectral"])

                # Check singular values
                assert len(features["singular_values"]) == graph.number_of_nodes()
                assert all(s >= 0 for s in features["singular_values"])

                # Check Laplacian eigenvalues
                assert len(features["laplacian_eigenvalues"]) == graph.number_of_nodes()
                assert features["laplacian_eigenvalues"][0] >= -1e-10  # Should be ≈ 0

    def test_all_features_combined(self, sample_graphs):
        """Test extraction of all features together."""
        extractor = NetworkFeatureExtractor()
        for name, graph in sample_graphs.items():
            features = extractor.get_features(graph)
            assert all(
                k in features
                for k in [
                    "degrees",
                    "density",
                    "clustering",
                    "betweenness",
                    "eigenvector",
                    "closeness",
                    "singular_values",
                    "laplacian_eigenvalues",
                ]
            )


# Test Edge Cases and Error Handling
class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_minimum_graph_size(self, test_configs):
        """Test generation with minimum allowed graph size."""
        min_configs = test_configs.copy()
        for model in min_configs:
            min_configs[model]["n"] = 3  # Minimum size for most models
            generator = GraphGenerator(model)
            result = generator.generate_sequence(min_configs[model])
            assert all(g.shape == (3, 3) for g in result["graphs"])

    def test_invalid_parameters(self, test_configs):
        """Test handling of invalid parameters."""
        invalid_configs = test_configs.copy()
        for model in invalid_configs:
            # Test negative values
            with pytest.raises(ValueError):
                invalid_configs[model]["n"] = -1
                generator = GraphGenerator(model)
                generator.generate_sequence(invalid_configs[model])

            # Test invalid probability values
            if "prob" in invalid_configs[model]:
                with pytest.raises(ValueError):
                    invalid_configs[model]["prob"] = 1.5
                    generator = GraphGenerator(model)
                    generator.generate_sequence(invalid_configs[model])

    def test_feature_extraction_errors(self):
        """Test error handling in feature extraction."""
        extractor = NetworkFeatureExtractor()

        # Test with invalid feature type
        graph = nx.path_graph(5)
        features = extractor.get_features(graph, ["invalid_type"])
        assert features == {}

        # Test with empty graph
        empty_features = extractor.get_features(nx.Graph())
        assert all(isinstance(v, (list, float)) for v in empty_features.values())

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_reproducibility(self, model, test_configs):
        """Test reproducibility with same random seed."""
        np.random.seed(42)
        generator1 = GraphGenerator(model)
        result1 = generator1.generate_sequence(test_configs[model])

        np.random.seed(42)
        generator2 = GraphGenerator(model)
        result2 = generator2.generate_sequence(test_configs[model])

        assert all(
            np.array_equal(g1, g2)
            for g1, g2 in zip(result1["graphs"], result2["graphs"])
        )
        assert result1["change_points"] == result2["change_points"]
