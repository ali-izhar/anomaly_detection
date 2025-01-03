# tests/test_strangeness.py

"""
Test the `src/changepoint/strangeness.py` module.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.changepoint.strangeness import strangeness_point


def test_strangeness_point_2d_single_cluster():
    """Test 2D data with n_clusters=1 (simple case)."""
    data = np.random.randn(10, 5)  # 10 samples, 5 features
    scores = strangeness_point(data, n_clusters=1)
    assert scores.shape == (10,), f"Expected shape (10,), got {scores.shape}"
    print("test_strangeness_point_2d_single_cluster passed!")


def test_strangeness_point_2d_multiple_clusters():
    """Test 2D data with n_clusters=3."""
    data = np.random.randn(8, 4)  # 8 samples, 4 features
    scores = strangeness_point(data, n_clusters=3)
    assert scores.shape == (8,), f"Expected shape (8,), got {scores.shape}"
    print("test_strangeness_point_2d_multiple_clusters passed!")


def test_strangeness_point_3d_mini_batch():
    """Test 3D data triggering MiniBatchKMeans."""
    data = np.random.randn(2, 10, 4)  # => (20, 4) after flatten
    # We'll set batch_size=10 so 20 > 10 => triggers MiniBatch
    scores = strangeness_point(data, n_clusters=2, batch_size=10)
    # Flatten => (2*10=20, 4 features)
    assert scores.shape == (20,), f"Expected shape (20,), got {scores.shape}"
    print("test_strangeness_point_3d_mini_batch passed!")


def test_strangeness_point_empty():
    """Test empty input raises ValueError."""
    try:
        _ = strangeness_point([])
        raise AssertionError("Expected ValueError for empty data, got none.")
    except ValueError:
        print("test_strangeness_point_empty passed (caught ValueError)!")


if __name__ == "__main__":
    test_strangeness_point_2d_single_cluster()
    test_strangeness_point_2d_multiple_clusters()
    test_strangeness_point_3d_mini_batch()
    test_strangeness_point_empty()
