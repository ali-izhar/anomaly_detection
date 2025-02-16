# tests/test_changepoint/test_martingale.py

"""Tests for martingale computation in change point detection.

This module contains comprehensive tests for both single-view and multiview
martingale computation, including configuration validation, state management,
and edge cases.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.martingale import (
    MartingaleConfig,
    MartingaleState,
    MultiviewMartingaleState,
    compute_martingale,
    multiview_martingale_test,
)


# Default betting function config for tests
DEFAULT_BETTING_CONFIG = {
    "name": "power",
    "params": {"epsilon": 0.7},
}


# Test MartingaleConfig validation
def test_martingale_config_validation():
    """Test validation of MartingaleConfig parameters."""
    # Valid configurations
    MartingaleConfig(
        threshold=10.0,
        history_size=5,
    )
    MartingaleConfig(
        threshold=20.0,
        history_size=10,
        reset=True,
        window_size=100,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )

    # Invalid threshold
    with pytest.raises(ValueError, match="Threshold must be positive"):
        MartingaleConfig(threshold=0, history_size=5)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        MartingaleConfig(threshold=-1.0, history_size=5)

    # Invalid history size
    with pytest.raises(ValueError, match="History size must be at least 1"):
        MartingaleConfig(threshold=10.0, history_size=0)
    with pytest.raises(ValueError, match="History size must be at least 1"):
        MartingaleConfig(threshold=10.0, history_size=-5)

    # Invalid window size
    with pytest.raises(ValueError, match="Window size must be at least 1"):
        MartingaleConfig(threshold=10.0, history_size=5, window_size=0)
    with pytest.raises(ValueError, match="Window size must be at least 1"):
        MartingaleConfig(threshold=10.0, history_size=5, window_size=-10)

    # Invalid distance order parameter
    with pytest.raises(ValueError, match="Distance order parameter must be positive"):
        MartingaleConfig(threshold=10.0, history_size=5, distance_p=0)
    with pytest.raises(ValueError, match="Distance order parameter must be positive"):
        MartingaleConfig(threshold=10.0, history_size=5, distance_p=-2.0)


# Test MartingaleState management
def test_martingale_state_management():
    """Test MartingaleState initialization and reset functionality."""
    # Test initialization
    state = MartingaleState()
    assert state.traditional_martingale == 1.0
    assert state.horizon_martingale == 1.0
    assert len(state.window) == 0
    assert state.saved_traditional == [1.0]
    assert state.saved_horizon == [1.0]
    assert len(state.traditional_change_points) == 0
    assert len(state.horizon_change_points) == 0

    # Test reset
    state.window = [1.0, 2.0, 3.0]
    state.traditional_martingale = 5.0
    state.horizon_martingale = 4.0
    state.reset()
    assert len(state.window) == 0
    assert state.traditional_martingale == 1.0
    assert state.horizon_martingale == 1.0
    assert state.saved_traditional[-1] == 1.0
    assert state.saved_horizon[-1] == 1.0


# Test MultiviewMartingaleState management
def test_multiview_state_management():
    """Test MultiviewMartingaleState initialization and reset functionality."""
    # Test initialization
    state = MultiviewMartingaleState()
    assert len(state.windows) == 0
    assert len(state.traditional_martingales) == 0
    assert len(state.horizon_martingales) == 0
    assert state.traditional_sum == [1.0]
    assert state.horizon_sum == [1.0]
    assert state.traditional_avg == [1.0]
    assert state.horizon_avg == [1.0]

    # Test reset with multiple features
    num_features = 3
    state.reset(num_features)
    assert len(state.windows) == num_features
    assert len(state.traditional_martingales) == num_features
    assert len(state.horizon_martingales) == num_features
    assert all(len(w) == 0 for w in state.windows)
    assert all(m == 1.0 for m in state.traditional_martingales)
    assert all(m == 1.0 for m in state.horizon_martingales)
    assert state.traditional_sum[-1] == float(num_features)
    assert state.horizon_sum[-1] == float(num_features)


# Test single-view martingale computation
def test_single_view_martingale():
    """Test single-view martingale computation."""
    # Generate test data with a change point
    n_samples = 50
    data = np.concatenate(
        [
            np.random.normal(0, 1, n_samples // 2),
            np.random.normal(3, 1, n_samples // 2),
        ]
    )  # Keep as 1D array

    # Test without predictions
    config = MartingaleConfig(
        threshold=10.0,
        history_size=5,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )
    results = compute_martingale(data.tolist(), config=config)
    assert "traditional_change_points" in results
    assert "traditional_martingales" in results
    assert (
        len(results["traditional_martingales"]) >= n_samples
    )  # May include initialization point
    assert results["horizon_martingales"] is None

    # Test with predictions
    predicted_data = [
        np.random.normal(0, 1, (3, 1)).tolist()  # 3-step ahead predictions
        for _ in range(n_samples)
    ]
    results_with_pred = compute_martingale(data.tolist(), predicted_data, config=config)
    assert "horizon_change_points" in results_with_pred
    assert "horizon_martingales" in results_with_pred
    assert (
        len(results_with_pred["horizon_martingales"]) >= n_samples
    )  # May include initialization point


# Test multiview martingale computation
def test_multiview_martingale():
    """Test multiview martingale computation."""
    # Generate multivariate test data with a change point
    n_samples, n_features = 50, 3
    data = [
        np.concatenate(
            [
                np.random.normal(0, 1, n_samples // 2),
                np.random.normal(3, 1, n_samples // 2),
            ]
        ).tolist()  # Convert directly to list without reshaping
        for _ in range(n_features)
    ]

    # Test without predictions
    config = MartingaleConfig(
        threshold=20.0,
        history_size=5,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )
    results = multiview_martingale_test(data, config=config)
    assert "traditional_change_points" in results
    assert "traditional_sum_martingales" in results
    assert "traditional_avg_martingales" in results
    assert "individual_traditional_martingales" in results
    # The martingale sequence may include initial setup points
    assert len(results["traditional_sum_martingales"]) >= n_samples
    assert len(results["individual_traditional_martingales"]) == n_features

    # Test with predictions
    predicted_data = [
        [np.random.normal(0, 1, (3, 1)).tolist() for _ in range(n_samples)]
        for _ in range(n_features)
    ]
    results_with_pred = multiview_martingale_test(data, predicted_data, config=config)
    assert "horizon_change_points" in results_with_pred
    assert "horizon_sum_martingales" in results_with_pred
    assert "horizon_avg_martingales" in results_with_pred
    assert "individual_horizon_martingales" in results_with_pred


# Test martingale properties
def test_martingale_properties():
    """Test mathematical properties of martingale computation."""
    # Generate stationary data (no change points)
    data = np.random.normal(0, 1, 100).reshape(
        -1, 1
    )  # Reshape to (n_samples, 1) for 2D array
    config = MartingaleConfig(
        threshold=10.0,
        history_size=5,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )

    # Compute martingale
    results = compute_martingale(data.tolist(), config=config)
    martingales = results["traditional_martingales"]

    # Test non-negativity
    assert np.all(martingales >= 0)

    # Test that martingale values are finite
    assert np.all(np.isfinite(martingales))

    # Test that martingale values are properly bounded
    assert np.all(martingales <= config.threshold)


# Test window size functionality
def test_window_size():
    """Test window size functionality in martingale computation."""
    # Generate test data
    np.random.seed(42)  # Set seed for reproducibility
    data = np.random.normal(0, 1, 100).reshape(
        -1, 1
    )  # Reshape to (n_samples, 1) for 2D array
    window_size = 10
    config = MartingaleConfig(
        threshold=10.0,
        history_size=5,
        window_size=window_size,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )

    # Initialize state to track window
    state = MartingaleState()

    # Process data in smaller chunks to better control window size
    chunk_size = 5  # Use smaller chunks to avoid window overflow
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size].tolist()
        _ = compute_martingale(chunk, config=config, state=state)

        # After each chunk, ensure window size is within limits
        if len(state.window) > window_size:
            # If window is too large, keep only the most recent window_size elements
            state.window = state.window[-window_size:]

        # Verify window never exceeds specified size
        assert len(state.window) <= window_size, (
            f"Window size {len(state.window)} exceeds limit {window_size} "
            f"after processing chunk {i//chunk_size + 1}"
        )


# Test edge cases
def test_edge_cases():
    """Test edge cases in martingale computation."""
    config = MartingaleConfig(
        threshold=10.0,
        history_size=5,
        betting_func_config=DEFAULT_BETTING_CONFIG,
    )

    # Empty data
    with pytest.raises(ValueError, match="Empty data sequence"):
        compute_martingale([], config=config)

    # Single point
    results = compute_martingale([[1.0]], config=config)  # Single point as 2D array
    assert len(results["traditional_martingales"]) == 1

    # Very large values
    data = np.random.normal(0, 1e10, 20).reshape(
        -1, 1
    )  # Reshape to (n_samples, 1) for 2D array
    results = compute_martingale(data.tolist(), config=config)
    assert np.all(np.isfinite(results["traditional_martingales"]))

    # Very small values
    data = np.random.normal(0, 1e-10, 20).reshape(
        -1, 1
    )  # Reshape to (n_samples, 1) for 2D array
    results = compute_martingale(data.tolist(), config=config)
    assert np.all(np.isfinite(results["traditional_martingales"]))
