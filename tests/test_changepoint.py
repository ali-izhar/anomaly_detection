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


@pytest.fixture
def feature_names():
    """Get list of feature names used in detection."""
    return [
        "mean_degree",
        "density",
        "mean_clustering",
        "mean_betweenness",
        "mean_eigenvector",
        "mean_closeness",
        "max_singular_value",
        "min_nonzero_laplacian",
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
            numeric_features = feature_extractor.get_numeric_features(graph)
            # Convert to array in consistent order
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Run detector
        cpd = ChangePointDetector(
            martingale_method="multiview",
            threshold=60.0,
            max_window=None,
            random_state=42,
            betting_func_config={"name": "power", "params": {"epsilon": 0.7}},
        )
        multiview_result = cpd.run(
            data=data,
            predicted_data=None,
        )

        # Verify results
        detected_points = multiview_result["traditional_change_points"]
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
            numeric_features = feature_extractor.get_numeric_features(graph)
            # Convert to array in consistent order
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Run detector with high threshold to avoid false positives
        cpd = ChangePointDetector(
            martingale_method="multiview",
            threshold=100.0,
            max_window=None,
            random_state=42,
            betting_func_config={"name": "power", "params": {"epsilon": 0.7}},
        )
        multiview_result = cpd.run(
            data=data,
            predicted_data=None,
        )

        # Verify results
        detected_points = multiview_result["traditional_change_points"]
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
            numeric_features = feature_extractor.get_numeric_features(graph)
            # Convert to array in consistent order
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Run detector
        cpd = ChangePointDetector(
            martingale_method="multiview",
            threshold=60.0,
            max_window=None,
            random_state=42,
            betting_func_config={"name": "power", "params": {"epsilon": 0.7}},
        )
        multiview_result = cpd.run(
            data=data,
            predicted_data=None,
        )

        # Verify results
        detected_points = multiview_result["traditional_change_points"]
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

    def test_parameter_sensitivity(self, feature_names):
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
            numeric_features = feature_extractor.get_numeric_features(graph)
            # Convert to array in consistent order
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Test different parameter combinations
        thresholds = [20.0, 60.0, 100.0]
        epsilons = [0.3, 0.7, 0.9]

        for threshold in thresholds:
            for epsilon in epsilons:
                # Create detector with current parameters
                cpd = ChangePointDetector(
                    martingale_method="multiview",
                    threshold=threshold,
                    max_window=None,
                    random_state=42,
                    betting_func_config={
                        "name": "power",
                        "params": {"epsilon": epsilon},
                    },
                )
                result = cpd.run(
                    data=data,
                    predicted_data=None,
                )

                # Basic checks
                assert "traditional_change_points" in result
                assert "horizon_change_points" in result
                assert "traditional_sum_martingales" in result
                assert "traditional_avg_martingales" in result
                assert "horizon_sum_martingales" in result
                assert "horizon_avg_martingales" in result
                assert "individual_traditional_martingales" in result
                assert "individual_horizon_martingales" in result

                # Higher threshold should generally mean fewer detections
                if threshold == 100.0:
                    assert (
                        len(result["traditional_change_points"]) <= len(graphs) * 0.1
                    ), "High threshold should result in few detections"

    def test_betting_functions(self, feature_names):
        """Test different betting functions for change detection."""
        # Test different betting functions
        betting_configs = [
            {"name": "power", "params": {"epsilon": 0.7}},
            {"name": "exponential", "params": {"lambd": 1.0}},
            {"name": "mixture", "params": {"epsilons": [0.5, 0.7, 0.9]}},
            {"name": "constant", "params": {}},
            {"name": "beta", "params": {"a": 0.5, "b": 1.5}},
        ]

        # Setup base configuration once
        model = "ba"  # Use BA model for betting function test
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__

        # Use same parameters as successful tests
        params.update(
            {
                "min_changes": 2,
                "max_changes": 2,
                "seq_len": 150,
                "min_segment": 40,
            }
        )

        for betting_config in betting_configs:
            # Generate sequence
            generator = GraphGenerator(model)
            result = generator.generate_sequence(params)
            graphs = result["graphs"]

            # Extract features
            feature_extractor = NetworkFeatureExtractor()
            features_numeric = []
            for adj_matrix in graphs:
                graph = nx.from_numpy_array(adj_matrix)
                numeric_features = feature_extractor.get_numeric_features(graph)
                feature_vector = np.array(
                    [numeric_features[name] for name in feature_names]
                )
                features_numeric.append(feature_vector)

            data = np.array(features_numeric)

            # Create detector with current betting function
            cpd = ChangePointDetector(
                martingale_method="multiview",
                threshold=60.0,
                max_window=None,
                random_state=42,
                betting_func_config=betting_config,
            )
            result = cpd.run(
                data=data,
                predicted_data=None,
            )

            # Basic checks
            assert "traditional_change_points" in result
            assert "horizon_change_points" in result
            assert "traditional_sum_martingales" in result
            assert "traditional_avg_martingales" in result
            assert "horizon_sum_martingales" in result
            assert "horizon_avg_martingales" in result

            # Check that martingales are non-negative
            assert np.all(result["traditional_sum_martingales"] >= 0)
            assert np.all(result["horizon_sum_martingales"] >= 0)

            # Print initial values for debugging
            print(f"\nTesting {betting_config['name']} betting function:")
            print(
                f"Initial traditional sum: {result['traditional_sum_martingales'][0]}"
            )
            print(f"Expected sum: {len(feature_names)}")
            print(
                f"Difference: {abs(result['traditional_sum_martingales'][0] - len(feature_names))}"
            )

            # Check martingale properties based on betting function type
            if betting_config["name"] == "power":
                # Power martingale should initialize close to number of features
                assert (
                    abs(result["traditional_sum_martingales"][0] - len(feature_names))
                    < 1e-6
                )
                assert (
                    abs(result["horizon_sum_martingales"][0] - len(feature_names))
                    < 1e-6
                )
            elif betting_config["name"] == "constant":
                # Constant martingale uses factors 1.5 or 0.5
                assert result["traditional_sum_martingales"][0] > 0
                assert result["horizon_sum_martingales"][0] > 0
            else:
                # For other betting functions, just ensure reasonable initialization
                assert (
                    0
                    < result["traditional_sum_martingales"][0]
                    < len(feature_names) * 2
                )
                assert 0 < result["horizon_sum_martingales"][0] < len(feature_names) * 2

            # Check that martingales evolve over time
            assert not np.allclose(
                result["traditional_sum_martingales"],
                result["traditional_sum_martingales"][0],
                rtol=1e-3,
            ), f"{betting_config['name']} martingale values should change over time"

            # Verify martingale average relationship
            assert np.allclose(
                result["traditional_avg_martingales"],
                result["traditional_sum_martingales"] / len(feature_names),
                rtol=1e-6,
            ), f"{betting_config['name']} average calculation incorrect"

    def test_distance_measures(self, feature_names):
        """Test different distance measures for strangeness computation."""
        # Setup
        model = "ba"  # Use BA model for distance measure test
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__

        # Generate sequence with known change points
        params.update(
            {
                "min_changes": 2,
                "max_changes": 2,
                "seq_len": 150,
                "min_segment": 40,
            }
        )
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]
        true_change_points = result["change_points"]

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            numeric_features = feature_extractor.get_numeric_features(graph)
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Test different distance measures
        distance_measures = ["euclidean", "mahalanobis"]

        for distance_measure in distance_measures:
            # Create detector with current distance measure
            cpd = ChangePointDetector(
                martingale_method="multiview",
                threshold=60.0,
                max_window=None,
                random_state=42,
                betting_func_config={"name": "power", "params": {"epsilon": 0.7}},
                distance_measure=distance_measure,
            )
            result = cpd.run(
                data=data,
                predicted_data=None,
            )

            # Basic checks
            assert "traditional_change_points" in result
            assert "horizon_change_points" in result

            # Verify detection capability
            detected_points = result["traditional_change_points"]
            assert (
                len(detected_points) > 0
            ), f"{distance_measure} distance failed to detect any changes"

            # Check detection accuracy
            if true_change_points and detected_points:
                errors = []
                for true_cp in true_change_points:
                    closest_detected = min(
                        detected_points, key=lambda x: abs(x - true_cp)
                    )
                    error = abs(closest_detected - true_cp)
                    errors.append(error)
                avg_error = np.mean(errors)

                # Detection should be within reasonable delay
                assert (
                    avg_error <= 25
                ), f"{distance_measure} distance: Average detection delay ({avg_error}) too high"

    def test_combined_parameters(self, feature_names):
        """Test combinations of betting functions and distance measures."""
        # Setup
        model = "ba"
        model_name = get_full_model_name(model)
        config = get_config(model_name)
        params = config["params"].__dict__
        params.update(
            {
                "min_changes": 1,
                "max_changes": 1,
                "seq_len": 100,
                "min_segment": 40,
            }
        )

        # Generate sequence
        generator = GraphGenerator(model)
        result = generator.generate_sequence(params)
        graphs = result["graphs"]

        # Extract features
        feature_extractor = NetworkFeatureExtractor()
        features_numeric = []
        for adj_matrix in graphs:
            graph = nx.from_numpy_array(adj_matrix)
            numeric_features = feature_extractor.get_numeric_features(graph)
            feature_vector = np.array(
                [numeric_features[name] for name in feature_names]
            )
            features_numeric.append(feature_vector)

        data = np.array(features_numeric)

        # Test combinations
        betting_configs = [
            {"name": "power", "params": {"epsilon": 0.7}},
            {"name": "exponential", "params": {"lambd": 1.0}},
        ]
        distance_measures = ["euclidean", "mahalanobis"]

        for betting_config in betting_configs:
            for distance_measure in distance_measures:
                # Create detector with current combination
                cpd = ChangePointDetector(
                    martingale_method="multiview",
                    threshold=60.0,
                    max_window=None,
                    random_state=42,
                    betting_func_config=betting_config,
                    distance_measure=distance_measure,
                )
                result = cpd.run(
                    data=data,
                    predicted_data=None,
                )

                # Basic checks
                assert "traditional_change_points" in result
                assert "horizon_change_points" in result
                assert "traditional_sum_martingales" in result
                assert "horizon_sum_martingales" in result

                # Check martingale properties
                assert np.all(result["traditional_sum_martingales"] >= 0)
                assert np.all(result["horizon_sum_martingales"] >= 0)
