# tests/test_changepoint/test_strangeness.py

"""Tests for strangeness computation in the changepoint detection module.

This module contains comprehensive tests for strangeness measures and p-value
computation, including configuration validation, input validation, edge cases,
and mathematical properties.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.strangeness import (
    StrangenessConfig,
    strangeness_point,
    get_pvalue,
)
from src.changepoint.distance import DistanceConfig


# Test StrangenessConfig validation
def test_strangeness_config_validation():
    """Test validation of StrangenessConfig parameters."""
    # Valid configurations
    StrangenessConfig()  # Test default values
    StrangenessConfig(n_clusters=3, batch_size=100, random_state=42)
    StrangenessConfig(
        n_clusters=2,
        distance_config=DistanceConfig(metric="euclidean"),
    )

    # Invalid n_clusters
    with pytest.raises(ValueError, match="n_clusters must be positive"):
        StrangenessConfig(n_clusters=0)
    with pytest.raises(ValueError, match="n_clusters must be positive"):
        StrangenessConfig(n_clusters=-1)

    # Invalid batch_size
    with pytest.raises(ValueError, match="batch_size must be positive"):
        StrangenessConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        StrangenessConfig(batch_size=-100)


# Test input validation for strangeness computation
def test_strangeness_input_validation():
    """Test input validation for strangeness computation."""
    # Empty input
    with pytest.raises(ValueError, match="Empty data sequence"):
        strangeness_point([])
    with pytest.raises(ValueError, match="Empty data sequence"):
        strangeness_point(np.array([]))

    # Invalid dimensions
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        strangeness_point(np.array([1, 2, 3, 4]))  # 1D array
    with pytest.raises(ValueError, match="Expected 2D or 3D array"):
        strangeness_point(np.array([[[[1]]]]))  # 4D array

    # Zero feature dimension
    with pytest.raises(ValueError, match="Feature dimension cannot be zero"):
        strangeness_point(np.array([[]]))  # Empty features

    # Too few samples for clustering
    data = np.random.rand(2, 3)  # 2 samples, 3 features
    config = StrangenessConfig(n_clusters=3)  # Requesting 3 clusters
    with pytest.raises(
        ValueError, match="Number of samples .* must be >= number of clusters"
    ):
        strangeness_point(data, config)


# Test strangeness computation with different data shapes
def test_strangeness_computation_shapes():
    """Test strangeness computation with different input shapes."""
    # 2D data (samples, features)
    data_2d = np.random.rand(10, 3)
    scores_2d = strangeness_point(data_2d)
    assert scores_2d.shape == (10,)
    assert np.all(scores_2d >= 0)

    # 3D data (samples, time, features)
    data_3d = np.random.rand(10, 5, 3)
    scores_3d = strangeness_point(data_3d)
    assert scores_3d.shape == (10,)
    assert np.all(scores_3d >= 0)


# Test strangeness computation with different metrics
def test_strangeness_different_metrics():
    """Test strangeness computation with different distance metrics."""
    data = np.random.rand(20, 3)
    metrics = ["euclidean", "manhattan", "cosine", "mahalanobis"]

    for metric in metrics:
        config = StrangenessConfig(
            n_clusters=2,
            distance_config=DistanceConfig(metric=metric),
        )
        scores = strangeness_point(data, config)
        assert scores.shape == (20,)
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))


# Test strangeness computation with different clustering parameters
def test_strangeness_clustering_params():
    """Test strangeness computation with different clustering parameters."""
    data = np.random.rand(100, 3)

    # Test different numbers of clusters
    for n_clusters in [1, 2, 5]:
        config = StrangenessConfig(n_clusters=n_clusters)
        scores = strangeness_point(data, config)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)

    # Test with and without mini-batch
    configs = [
        StrangenessConfig(n_clusters=3),  # Standard KMeans
        StrangenessConfig(n_clusters=3, batch_size=20),  # MiniBatchKMeans
    ]
    for config in configs:
        scores = strangeness_point(data, config)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)


# Test strangeness computation with known patterns
def test_strangeness_known_patterns():
    """Test strangeness computation with known data patterns."""
    # Create clusters with known structure
    cluster1 = np.random.normal(0, 0.1, (10, 2))  # Tight cluster around (0,0)
    cluster2 = np.random.normal(5, 0.1, (10, 2))  # Tight cluster around (5,5)
    outlier = np.array([[10, 10]])  # Clear outlier

    # Combine the data
    data = np.vstack([cluster1, cluster2, outlier])
    config = StrangenessConfig(n_clusters=2)  # Use 2 clusters

    # Compute strangeness scores
    scores = strangeness_point(data, config)

    # The outlier should have higher strangeness
    assert scores[-1] > np.mean(scores[:-1])

    # Points in clusters should have lower strangeness
    assert np.all(scores[:-1] < scores[-1])


# Test p-value computation
def test_pvalue_computation():
    """Test p-value computation from strangeness scores."""
    # Test basic functionality
    strangeness = [1.0, 2.0, 3.0, 4.0, 5.0]
    pvalue = get_pvalue(strangeness)
    assert 0 <= pvalue <= 1

    # Test with random state for reproducibility
    pvalue1 = get_pvalue(strangeness, random_state=42)
    pvalue2 = get_pvalue(strangeness, random_state=42)
    assert pvalue1 == pvalue2

    # Test with identical values
    strangeness = [1.0, 1.0, 1.0, 1.0]
    pvalue = get_pvalue(strangeness)
    assert 0 <= pvalue <= 1

    # Test input validation
    with pytest.raises(TypeError):
        get_pvalue("not a list")
    with pytest.raises(ValueError):
        get_pvalue([])


# Test edge cases
def test_strangeness_edge_cases():
    """Test strangeness computation with edge cases."""
    # All zero vectors
    data = np.zeros((10, 3))
    scores = strangeness_point(data)
    assert np.all(scores >= 0)
    assert np.all(np.isfinite(scores))

    # All identical points
    data = np.ones((10, 3))
    scores = strangeness_point(data)
    assert np.all(scores >= 0)
    assert np.all(np.isfinite(scores))

    # Very large values
    data = np.random.rand(10, 3) * 1e10
    scores = strangeness_point(data)
    assert np.all(np.isfinite(scores))

    # Very small values
    data = np.random.rand(10, 3) * 1e-10
    scores = strangeness_point(data)
    assert np.all(np.isfinite(scores))


# Test numerical stability
def test_numerical_stability():
    """Test numerical stability of strangeness computation."""
    # Generate data with increasing scales
    scales = [1e-10, 1.0, 1e10]
    for scale in scales:
        data = np.random.rand(20, 3) * scale
        scores = strangeness_point(data)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    # Test with mixed scales
    data = np.random.rand(20, 3)
    data[:10] *= 1e10  # First half large values
    data[10:] *= 1e-10  # Second half small values
    scores = strangeness_point(data)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)
