"""Tests for change point detection using the martingale framework.

This module contains comprehensive tests for the change point detector,
including configuration validation, input validation, and detection methods.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import (
    DetectorConfig,
    ChangePointDetector,
)
from src.changepoint.martingale import MartingaleConfig


# Default betting function config for tests
DEFAULT_BETTING_CONFIG = {
    "name": "power",
    "params": {"epsilon": 0.7},
}


# Test DetectorConfig validation
def test_detector_config_validation():
    """Test validation of DetectorConfig parameters."""
    # Valid configurations
    DetectorConfig()  # Test default values
    DetectorConfig(
        method="single_view",
        threshold=50.0,
        history_size=20,
        batch_size=500,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )
    DetectorConfig(
        method="multiview",
        threshold=100.0,
        max_window=1000,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )

    # Invalid threshold
    with pytest.raises(ValueError, match="Threshold must be positive"):
        DetectorConfig(threshold=0)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        DetectorConfig(threshold=-1.0)

    # Invalid history size
    with pytest.raises(ValueError, match="History size must be at least 1"):
        DetectorConfig(history_size=0)
    with pytest.raises(ValueError, match="History size must be at least 1"):
        DetectorConfig(history_size=-5)

    # Invalid batch size
    with pytest.raises(ValueError, match="Batch size must be at least 1"):
        DetectorConfig(batch_size=0)
    with pytest.raises(ValueError, match="Batch size must be at least 1"):
        DetectorConfig(batch_size=-100)

    # Invalid max window
    with pytest.raises(ValueError, match="Max window must be at least 1"):
        DetectorConfig(max_window=0)
    with pytest.raises(ValueError, match="Max window must be at least 1"):
        DetectorConfig(max_window=-50)

    # Invalid distance order parameter
    with pytest.raises(ValueError, match="Distance order parameter must be positive"):
        DetectorConfig(distance_p=0)
    with pytest.raises(ValueError, match="Distance order parameter must be positive"):
        DetectorConfig(distance_p=-2.0)


# Test input validation
def test_detector_input_validation():
    """Test input validation for the detector."""
    # Test single-view input validation
    single_view_detector = ChangePointDetector(
        DetectorConfig(
            method="single_view",
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Invalid input dimensions for single-view
    with pytest.raises(ValueError, match="Single-view data must be 1-dimensional"):
        single_view_detector.run(np.random.rand(10, 2))
    with pytest.raises(ValueError, match="Single-view predictions must have shape"):
        single_view_detector.run(
            np.random.rand(10),
            predicted_data=np.random.rand(5, 2, 2),
        )

    # Test multiview input validation
    multiview_detector = ChangePointDetector(
        DetectorConfig(
            method="multiview",
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Invalid input dimensions for multiview
    with pytest.raises(ValueError, match="Multiview data must have shape"):
        multiview_detector.run(np.random.rand(10))
    with pytest.raises(ValueError, match="Multiview predictions must have shape"):
        multiview_detector.run(
            np.random.rand(10, 2),
            predicted_data=np.random.rand(5, 2),
        )

    # Feature dimension mismatch
    with pytest.raises(ValueError, match="Number of features .* does not match"):
        multiview_detector.run(
            np.random.rand(10, 2),
            predicted_data=np.random.rand(5, 3, 3),
        )


# Test single-view detection
def test_single_view_detection():
    """Test single-view change point detection."""
    np.random.seed(42)  # Set seed for reproducibility
    detector = ChangePointDetector(
        DetectorConfig(
            method="single_view",
            threshold=10.0,
            history_size=5,
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Generate test data with a change point
    n_samples = 50
    data = np.concatenate(
        [
            np.random.normal(0, 1, n_samples // 2),
            np.random.normal(3, 1, n_samples // 2),
        ]
    )  # Keep as 1D array

    # Run detection without predictions
    results = detector.run(data)
    assert "traditional_change_points" in results
    assert "traditional_martingales" in results
    assert (
        len(results["traditional_martingales"]) >= n_samples
    )  # May include initialization points
    assert results["horizon_martingales"] is None

    # Run detection with predictions
    predicted_data = np.random.normal(0, 1, (n_samples, 3))  # 3-step ahead predictions
    results_with_pred = detector.run(data, predicted_data)
    assert "horizon_change_points" in results_with_pred
    assert "horizon_martingales" in results_with_pred
    assert len(results_with_pred["horizon_martingales"]) >= n_samples


# Test multiview detection
def test_multiview_detection():
    """Test multiview change point detection."""
    np.random.seed(42)  # Set seed for reproducibility
    detector = ChangePointDetector(
        DetectorConfig(
            method="multiview",
            threshold=20.0,
            history_size=5,
            batch_size=10,
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Generate multivariate test data with a change point
    n_samples, n_features = 50, 3
    data = np.concatenate(
        [
            np.random.normal(0, 1, (n_samples // 2, n_features)),
            np.random.normal(3, 1, (n_samples // 2, n_features)),
        ]
    )

    # Create martingale config for multiview test
    martingale_config = MartingaleConfig(
        threshold=detector.config.threshold,
        history_size=detector.config.history_size,
        window_size=detector.config.max_window,
        betting_func_config=detector.config.betting_func_config,
    )

    # Run detection without predictions
    results = detector.run(data)
    assert "traditional_change_points" in results
    assert "traditional_sum_martingales" in results
    assert "traditional_avg_martingales" in results
    assert "individual_traditional_martingales" in results
    assert (
        len(results["traditional_sum_martingales"]) >= n_samples
    )  # May include initialization points
    assert len(results["individual_traditional_martingales"]) == n_features

    # Run detection with predictions
    predicted_data = np.random.normal(
        0, 1, (n_samples, 3, n_features)
    )  # 3-step ahead predictions
    results_with_pred = detector.run(data, predicted_data)
    assert "horizon_change_points" in results_with_pred
    assert "horizon_sum_martingales" in results_with_pred
    assert "horizon_avg_martingales" in results_with_pred
    assert "individual_horizon_martingales" in results_with_pred


# Test detector reset functionality
def test_detector_reset():
    """Test detector reset functionality."""
    np.random.seed(42)  # Set seed for reproducibility
    detector = ChangePointDetector(
        DetectorConfig(
            method="single_view",
            threshold=10.0,
            reset=True,
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Generate data with multiple change points
    data = np.concatenate(
        [
            np.random.normal(0, 1, 20),
            np.random.normal(3, 1, 20),
            np.random.normal(0, 1, 20),
        ]
    )  # Keep as 1D array

    # Run detection
    results = detector.run(data)
    change_points = results["traditional_change_points"]

    # Verify that martingale values reset after each detection
    martingales = results["traditional_martingales"]
    for cp in change_points:
        if cp + 1 < len(martingales):
            assert np.isclose(martingales[cp + 1], 1.0), (
                f"Martingale not reset to 1.0 after change point at {cp}, "
                f"got {martingales[cp + 1]}"
            )


# Test numerical stability
def test_numerical_stability():
    """Test numerical stability of the detector."""
    np.random.seed(42)  # Set seed for reproducibility
    detector = ChangePointDetector(
        DetectorConfig(
            method="single_view",
            betting_func_config=DEFAULT_BETTING_CONFIG,
        )
    )

    # Test with different scales of data
    scales = [1e-10, 1.0, 1e10]
    for scale in scales:
        data = np.random.rand(30) * scale  # Keep as 1D array
        results = detector.run(data)
        assert np.all(
            np.isfinite(results["traditional_martingales"])
        ), f"Non-finite martingale values found with scale {scale}"

    # Test with mixed scales
    data = np.concatenate(
        [
            np.random.rand(15) * 1e10,
            np.random.rand(15) * 1e-10,
        ]
    )  # Keep as 1D array
    results = detector.run(data)
    assert np.all(
        np.isfinite(results["traditional_martingales"])
    ), "Non-finite martingale values found with mixed scales"


# Test betting function configurations
def test_betting_functions():
    """Test detector with different betting functions."""
    np.random.seed(42)  # Set seed for reproducibility
    betting_configs = [
        {"name": "power", "params": {"epsilon": 0.5}},
        {"name": "mixture", "params": {"epsilons": [0.3, 0.5, 0.7]}},
        {"name": "constant"},
    ]

    data = np.random.rand(30)  # Keep as 1D array
    for betting_config in betting_configs:
        detector = ChangePointDetector(
            DetectorConfig(
                method="single_view",
                betting_func_config=betting_config,
            )
        )
        results = detector.run(data)
        assert np.all(
            np.isfinite(results["traditional_martingales"])
        ), f"Non-finite martingale values found with betting function {betting_config['name']}"
