"""Tests for distance computation utilities in the changepoint detection module.

This module contains comprehensive tests for all distance metrics and their
configurations, edge cases, and mathematical properties.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.distance import (
    DistanceConfig,
    compute_pairwise_distances,
    compute_cluster_distances,
    VALID_METRICS,
)


# Test DistanceConfig validation
def test_distance_config_validation():
    """Test validation of DistanceConfig parameters."""
    # Valid configurations
    DistanceConfig()  # Test default values
    DistanceConfig(metric="euclidean", p=2.0, eps=1e-8)
    DistanceConfig(metric="manhattan", cov_reg=1e-6)
    DistanceConfig(metric="cosine", use_correlation=True)

    # Invalid metric
    with pytest.raises(ValueError, match="Invalid metric"):
        DistanceConfig(metric="invalid_metric")

    # Invalid p parameter
    with pytest.raises(ValueError, match="p must be positive"):
        DistanceConfig(metric="minkowski", p=0)
    with pytest.raises(ValueError, match="p must be positive"):
        DistanceConfig(metric="minkowski", p=-1)

    # Invalid eps parameter
    with pytest.raises(ValueError, match="eps must be positive"):
        DistanceConfig(eps=0)
    with pytest.raises(ValueError, match="eps must be positive"):
        DistanceConfig(eps=-1e-8)

    # Invalid cov_reg parameter
    with pytest.raises(ValueError, match="cov_reg must be non-negative"):
        DistanceConfig(cov_reg=-1e-6)


# Test input validation for pairwise distances
def test_pairwise_distances_input_validation():
    """Test input validation for pairwise distance computation."""
    # Valid input shapes
    x = np.random.rand(10, 3)
    y = np.random.rand(5, 3)
    compute_pairwise_distances(x, y)

    # Invalid input dimensions
    with pytest.raises(ValueError, match="Expected 2D arrays"):
        compute_pairwise_distances(np.random.rand(10), y)
    with pytest.raises(ValueError, match="Expected 2D arrays"):
        compute_pairwise_distances(x, np.random.rand(5))

    # Feature dimension mismatch
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        compute_pairwise_distances(x, np.random.rand(5, 4))


# Test 1D distance computation
def test_1d_distance_computation():
    """Test distance computation for 1D data."""
    x = np.array([[1], [2], [3]])
    y = np.array([[0], [2], [4]])

    # Test different metrics for 1D
    metrics = ["euclidean", "manhattan", "minkowski", "chebyshev", "mahalanobis"]
    for metric in metrics:
        config = DistanceConfig(metric=metric)
        distances = compute_pairwise_distances(x, y, config)
        assert distances.shape == (3, 3)
        assert np.all(distances >= 0)  # All distances should be non-negative
        assert np.all(np.isfinite(distances))  # All distances should be finite


# Test specific distance metrics
@pytest.mark.parametrize("metric", VALID_METRICS)
def test_distance_metrics(metric):
    """Test all distance metrics with random data."""
    x = np.random.rand(5, 3)
    y = np.random.rand(4, 3)
    config = DistanceConfig(metric=metric)

    distances = compute_pairwise_distances(x, y, config)

    # Check basic properties
    assert distances.shape == (5, 4)
    assert np.all(distances >= 0)  # Non-negativity
    assert np.all(np.isfinite(distances))  # Finiteness


def test_euclidean_distance():
    """Test Euclidean distance specific properties."""
    x = np.array([[0, 0], [1, 0], [0, 1]])
    y = np.array([[0, 0], [1, 1]])
    config = DistanceConfig(metric="euclidean")

    distances = compute_pairwise_distances(x, y, config)

    # Known distances
    np.testing.assert_almost_equal(distances[0, 0], 0)  # Same point
    np.testing.assert_almost_equal(distances[1, 1], 1)  # Distance to (1,1)
    np.testing.assert_almost_equal(distances[2, 1], 1)  # Distance to (1,1)


def test_manhattan_distance():
    """Test Manhattan distance specific properties."""
    x = np.array([[0, 0], [1, 0], [0, 1]])
    y = np.array([[0, 0], [1, 1]])
    config = DistanceConfig(metric="manhattan")

    distances = compute_pairwise_distances(x, y, config)

    # Known distances
    np.testing.assert_almost_equal(distances[0, 0], 0)  # Same point
    np.testing.assert_almost_equal(distances[1, 1], 1)  # |1-1| + |0-1| = 0 + 1 = 1
    np.testing.assert_almost_equal(distances[2, 1], 1)  # |0-1| + |1-1| = 1 + 0 = 1


def test_cosine_distance():
    """Test cosine distance specific properties."""
    x = np.array([[1, 0], [1, 1], [0, 1]])
    y = np.array([[1, 0], [0, 1]])
    config = DistanceConfig(metric="cosine")

    distances = compute_pairwise_distances(x, y, config)

    # Known distances
    np.testing.assert_almost_equal(distances[0, 0], 0)  # Same direction
    np.testing.assert_almost_equal(distances[0, 1], 1)  # Perpendicular
    np.testing.assert_almost_equal(distances[1, 0], 1 - 1 / np.sqrt(2))  # 45 degrees


def test_mahalanobis_distance():
    """Test Mahalanobis distance specific properties."""
    # Generate correlated data
    cov = np.array([[1, 0.5], [0.5, 1]])
    x = np.random.multivariate_normal([0, 0], cov, 10)
    y = np.random.multivariate_normal([0, 0], cov, 5)

    # Test with and without correlation
    configs = [
        DistanceConfig(metric="mahalanobis", use_correlation=False),
        DistanceConfig(metric="mahalanobis", use_correlation=True),
    ]

    for config in configs:
        distances = compute_pairwise_distances(x, y, config)
        assert distances.shape == (10, 5)
        assert np.all(distances >= 0)
        assert np.all(np.isfinite(distances))


# Test cluster distances
def test_cluster_distances():
    """Test computation of distances to cluster centers."""
    # Generate random data and fit KMeans
    data = np.random.rand(20, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data)

    # Test with different metrics
    for metric in VALID_METRICS:
        config = DistanceConfig(metric=metric)
        distances = compute_cluster_distances(data, kmeans, config)

        # Check basic properties
        assert distances.shape == (20, 3)  # (n_samples, n_clusters)
        assert np.all(distances >= 0)
        assert np.all(np.isfinite(distances))


def test_cluster_distances_input_validation():
    """Test input validation for cluster distance computation."""
    data = np.random.rand(20, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data)

    # Invalid input type
    with pytest.raises(TypeError):
        compute_cluster_distances([1, 2, 3], kmeans)

    # Invalid dimensions
    with pytest.raises(ValueError):
        compute_cluster_distances(data.reshape(-1), kmeans)

    # Feature dimension mismatch
    wrong_dim_data = np.random.rand(20, 4)
    with pytest.raises(ValueError):
        compute_cluster_distances(wrong_dim_data, kmeans)


# Test edge cases
def test_distance_edge_cases():
    """Test distance computation with edge cases."""
    # Zero vectors
    x = np.zeros((2, 3))
    y = np.random.rand(2, 3)

    for metric in VALID_METRICS:
        config = DistanceConfig(metric=metric)
        distances = compute_pairwise_distances(x, y, config)
        assert np.all(np.isfinite(distances))  # Should handle zero vectors

    # Identical points
    x = np.array([[1, 2, 3]])
    y = np.array([[1, 2, 3]])

    for metric in VALID_METRICS:
        config = DistanceConfig(metric=metric)
        distances = compute_pairwise_distances(x, y, config)
        np.testing.assert_almost_equal(
            distances[0, 0], 0
        )  # Distance to self should be 0


# Test numerical stability
def test_numerical_stability():
    """Test numerical stability of distance computations."""
    # Very large values
    x = np.random.rand(5, 3) * 1e10
    y = np.random.rand(4, 3) * 1e10

    for metric in VALID_METRICS:
        config = DistanceConfig(metric=metric)
        distances = compute_pairwise_distances(x, y, config)
        assert np.all(np.isfinite(distances))  # Should handle large values

    # Very small values
    x = np.random.rand(5, 3) * 1e-10
    y = np.random.rand(4, 3) * 1e-10

    for metric in VALID_METRICS:
        config = DistanceConfig(metric=metric)
        distances = compute_pairwise_distances(x, y, config)
        assert np.all(np.isfinite(distances))  # Should handle small values
