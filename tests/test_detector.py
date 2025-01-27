# tests/test_detector.py

"""Test script for martingale-based change detection on network sequences.

Tests the ability to detect changes in network evolution using martingale-based
change point detection with different network models and feature combinations.
"""

import sys
import pytest
import numpy as np
import networkx as nx
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.configs.loader import get_config
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def extract_numeric_features(feature_dict: dict) -> np.ndarray:
    """Extract numeric features from feature dictionary in a consistent order."""
    # Extract basic metrics
    degrees = feature_dict.get("degrees", [])
    avg_degree = np.mean(degrees) if degrees else 0.0
    density = feature_dict.get("density", 0.0)
    clustering = feature_dict.get("clustering", [])
    avg_clustering = np.mean(clustering) if clustering else 0.0

    # Extract centrality metrics
    betweenness = feature_dict.get("betweenness", [])
    avg_betweenness = np.mean(betweenness) if betweenness else 0.0
    eigenvector = feature_dict.get("eigenvector", [])
    avg_eigenvector = np.mean(eigenvector) if eigenvector else 0.0
    closeness = feature_dict.get("closeness", [])
    avg_closeness = np.mean(closeness) if closeness else 0.0

    # Extract spectral metrics
    singular_values = feature_dict.get("singular_values", [])
    largest_sv = max(singular_values) if singular_values else 0.0
    laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
    smallest_nonzero_le = (
        min(x for x in laplacian_eigenvalues if x > 1e-10)
        if laplacian_eigenvalues
        else 0.0
    )

    return np.array(
        [
            avg_degree,
            density,
            avg_clustering,
            avg_betweenness,
            avg_eigenvector,
            avg_closeness,
            largest_sv,
            smallest_nonzero_le,
        ]
    )


@pytest.fixture
def feature_names():
    """Get list of feature names used in detection."""
    return [
        "degree",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_value",
        "laplacian",
    ]


class TestChangePointDetection:
    """Test suite for change point detection functionality."""

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_basic_detection(self, model, feature_names):
        """Test basic change point detection for each model."""
        # Setup
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__

        # Generate sequence
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]
        true_change_points = result["change_points"]

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        data = np.array(features_numeric)

        # Run detector
        cpd = ChangePointDetector()
        multiview_result = cpd.detect_changes_multiview(
            data=[data[:, i : i + 1] for i in range(data.shape[1])],
            threshold=60.0,
            epsilon=0.7,
            max_window=None,
            random_state=42,
        )

        # Verify results
        detected_points = multiview_result["change_points"]
        assert len(detected_points) > 0, "Should detect at least one change point"

        # Check detection accuracy
        if true_change_points and detected_points:
            errors = []
            for true_cp in true_change_points:
                closest_detected = min(detected_points, key=lambda x: abs(x - true_cp))
                error = abs(closest_detected - true_cp)
                errors.append(error)
            avg_error = np.mean(errors)

            # Detection should be within reasonable delay
            assert avg_error <= 20, f"Average detection delay ({avg_error}) too high"

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_no_changes(self, model, feature_names):
        """Test detection when there are no actual changes."""
        # Setup with no changes
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__
        params.update(
            {
                "min_changes": 0,
                "max_changes": 0,
                "seq_len": 100,
            }
        )

        # Generate sequence
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]
        true_change_points = result["change_points"]
        assert len(true_change_points) == 0, "Should have no true change points"

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        data = np.array(features_numeric)

        # Run detector with high threshold to avoid false positives
        cpd = ChangePointDetector()
        multiview_result = cpd.detect_changes_multiview(
            data=[data[:, i : i + 1] for i in range(data.shape[1])],
            threshold=100.0,  # High threshold
            epsilon=0.7,
            max_window=None,
            random_state=42,
        )

        # Verify results
        detected_points = multiview_result["change_points"]
        assert len(detected_points) == 0, "Should not detect any change points"

    @pytest.mark.parametrize("model", ["ba", "ws", "er", "sbm"])
    def test_multiple_changes(self, model, feature_names):
        """Test detection of multiple change points."""
        # Setup with multiple changes
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__
        params.update(
            {
                "min_changes": 3,
                "max_changes": 3,
                "seq_len": 200,
                "min_segment": 40,  # Ensure enough separation between changes
            }
        )

        # Generate sequence
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]
        true_change_points = result["change_points"]
        assert len(true_change_points) == 3, "Should have exactly 3 true change points"

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        data = np.array(features_numeric)

        # Run detector
        cpd = ChangePointDetector()
        multiview_result = cpd.detect_changes_multiview(
            data=[data[:, i : i + 1] for i in range(data.shape[1])],
            threshold=60.0,
            epsilon=0.7,
            max_window=None,
            random_state=42,
        )

        # Verify results
        detected_points = multiview_result["change_points"]
        assert len(detected_points) >= 2, "Should detect at least 2 change points"

        # Check detection accuracy
        if true_change_points and detected_points:
            errors = []
            for true_cp in true_change_points:
                closest_detected = min(detected_points, key=lambda x: abs(x - true_cp))
                error = abs(closest_detected - true_cp)
                errors.append(error)
            avg_error = np.mean(errors)

            # Detection should be within reasonable delay
            assert avg_error <= 25, f"Average detection delay ({avg_error}) too high"

    def test_parameter_sensitivity(self):
        """Test sensitivity to detector parameters."""
        # Setup
        model = "ba"  # Use BA model for parameter sensitivity test
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__

        # Generate sequence
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            feature_dict = feature_extractor.get_features(graph)
            numeric_features = extract_numeric_features(feature_dict)
            features_numeric.append(numeric_features)

        data = np.array(features_numeric)

        # Test different parameter combinations
        cpd = ChangePointDetector()
        thresholds = [20.0, 60.0, 100.0]
        epsilons = [0.3, 0.7, 0.9]

        for threshold in thresholds:
            for epsilon in epsilons:
                result = cpd.detect_changes_multiview(
                    data=[data[:, i : i + 1] for i in range(data.shape[1])],
                    threshold=threshold,
                    epsilon=epsilon,
                    max_window=None,
                    random_state=42,
                )

                # Basic checks
                assert "change_points" in result
                assert "martingales_sum" in result
                assert "martingales_avg" in result
                assert "individual_martingales" in result

                # Higher threshold should generally mean fewer detections
                if threshold == 100.0:
                    assert (
                        len(result["change_points"]) <= len(graphs) * 0.1
                    ), "High threshold should result in few detections"
