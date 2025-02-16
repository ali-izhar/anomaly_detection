# tests/test_changepoint/test_threshold.py

"""Tests for threshold-based change point detection classifier.

This module contains comprehensive tests for the threshold-based classifier,
including configuration validation, decision rules, probability estimation,
and SHAP analysis functionality.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.threshold import (
    CustomThresholdModel,
    ShapConfig,
)


# Test ShapConfig validation
def test_shap_config_validation():
    """Test validation of ShapConfig parameters."""
    # Valid configurations
    ShapConfig()  # Test default values
    ShapConfig(window_size=10, test_size=0.3)
    ShapConfig(
        window_size=3,
        test_size=0.1,
        random_state=None,
        use_probabilities=True,
    )

    # Invalid window size
    with pytest.raises(ValueError, match="window_size must be positive"):
        ShapConfig(window_size=0)
    with pytest.raises(ValueError, match="window_size must be positive"):
        ShapConfig(window_size=-1)

    # Invalid test size
    with pytest.raises(ValueError, match="test_size must be in"):
        ShapConfig(test_size=0)
    with pytest.raises(ValueError, match="test_size must be in"):
        ShapConfig(test_size=1.0)
    with pytest.raises(ValueError, match="test_size must be in"):
        ShapConfig(test_size=-0.1)


# Test model initialization and validation
def test_model_initialization():
    """Test CustomThresholdModel initialization and validation."""
    # Valid threshold values
    CustomThresholdModel(threshold=0.5)
    CustomThresholdModel(threshold=1.0)
    CustomThresholdModel()  # Test default value

    # Invalid threshold values
    with pytest.raises(ValueError, match="Threshold must be positive"):
        CustomThresholdModel(threshold=0)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        CustomThresholdModel(threshold=-1.0)


# Test model fitting and prediction
def test_model_fit_predict():
    """Test basic model fitting and prediction functionality."""
    # Create synthetic data
    X = np.array([[1, 2], [3, 4], [0, 1], [5, 6]])
    y = np.array([0, 1, 0, 1])

    model = CustomThresholdModel(threshold=5.0)

    # Test fitting
    model.fit(X, y)
    assert model.n_features_in_ == 2
    assert np.array_equal(model.classes_, np.array([0, 1]))

    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert np.all(np.isin(predictions, [0, 1]))

    # Verify decision boundary
    sums = np.sum(X, axis=1)
    expected = (sums > model.threshold).astype(int)
    np.testing.assert_array_equal(predictions, expected)


# Test probability estimation
def test_probability_estimation():
    """Test probability estimation functionality."""
    X = np.array([[1, 2], [3, 4], [0, 1], [5, 6]])
    y = np.array([0, 1, 0, 1])

    model = CustomThresholdModel(threshold=5.0)
    model.fit(X, y)

    # Test probability computation
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert np.all((0 <= probs) & (probs <= 1))

    # Test probability formula
    sums = np.sum(X, axis=1)
    expected_probs = sums / (sums + model.threshold)
    np.testing.assert_array_almost_equal(probs, expected_probs)

    # Test negative class probabilities
    neg_probs = model.predict_proba(X, positive_class=False)
    np.testing.assert_array_almost_equal(neg_probs, 1 - probs)


# Test input validation
def test_input_validation():
    """Test input validation for various methods."""
    model = CustomThresholdModel()
    X_valid = np.array([[1, 2], [3, 4]])
    y_valid = np.array([0, 1])

    # Test fit input validation
    with pytest.raises(ValueError):
        model.fit(X_valid, y_valid[:1])  # Mismatched lengths
    with pytest.raises(ValueError):
        model.fit([], [])  # Empty input
    with pytest.raises(ValueError):
        model.fit(np.array([[]]), np.array([0]))  # Empty features

    # Fit valid data first
    model.fit(X_valid, y_valid)

    # Test predict input validation
    with pytest.raises(ValueError):
        model.predict(np.array([[1]]))  # Wrong number of features
    with pytest.raises(ValueError):
        model.predict_proba(np.array([[1]]))  # Wrong number of features
    with pytest.raises(ValueError):
        model.predict(np.array([[]]))  # Empty features


# Test SHAP value computation
def test_shap_computation():
    """Test SHAP value computation functionality."""
    # Create synthetic data
    X = np.random.rand(20, 3)
    change_points = [5, 15]
    sequence_length = len(X)

    model = CustomThresholdModel()

    # Test basic SHAP computation
    shap_values = model.compute_shap_values(
        X=X,
        change_points=change_points,
        sequence_length=sequence_length,
    )
    assert shap_values.shape == X.shape

    # Test with custom configuration
    config = ShapConfig(
        window_size=3,
        test_size=0.3,
        use_probabilities=True,
    )
    shap_values = model.compute_shap_values(
        X=X,
        change_points=change_points,
        sequence_length=sequence_length,
        config=config,
    )
    assert shap_values.shape == X.shape


# Test martingale SHAP analysis
def test_martingale_shap_analysis():
    """Test SHAP analysis for martingale contributions."""
    # Create synthetic martingale data
    martingales = {
        "degree": {"martingales": [0.5, 1.0, 1.5]},
        "density": {"martingales": [1.0, 1.5, 2.0]},
        "clustering": {"martingales": [0.8, 1.2, 1.6]},
    }
    change_points = [1]
    sequence_length = 3

    model = CustomThresholdModel()

    # Test martingale SHAP computation
    shap_values, feature_names = model.compute_martingale_shap_values(
        martingales=martingales,
        change_points=change_points,
        sequence_length=sequence_length,
    )

    assert len(feature_names) == 3  # Number of features
    assert shap_values.shape == (sequence_length, len(feature_names))

    # Test with invalid martingales
    with pytest.raises(ValueError, match="No valid features"):
        model.compute_martingale_shap_values(
            martingales={"combined": {"martingales": [1.0]}},
            change_points=change_points,
            sequence_length=sequence_length,
        )


# Test edge cases
def test_edge_cases():
    """Test model behavior with edge cases."""
    model = CustomThresholdModel(threshold=1.0)

    # Test with zero features
    X = np.zeros((5, 2))
    y = np.zeros(5)
    model.fit(X, y)
    predictions = model.predict(X)
    assert np.all(predictions == 0)  # All zeros should predict no change

    # Test with large values
    X_large = np.ones((5, 2)) * 1e10
    model.fit(X_large, y)
    probs = model.predict_proba(X_large)
    assert np.all(probs > 0.99)  # Should predict change with high probability

    # Test with small values
    X_small = np.ones((5, 2)) * 1e-10
    model.fit(X_small, y)
    probs = model.predict_proba(X_small)
    assert np.all(probs < 0.01)  # Should predict no change with high probability


# Test scikit-learn compatibility
def test_sklearn_compatibility():
    """Test scikit-learn API compatibility."""
    model = CustomThresholdModel()

    # Test get_params
    params = model.get_params()
    assert "threshold" in params

    # Test set_params
    model.set_params(threshold=2.0)
    assert model.threshold == 2.0

    # Test score method
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1


# Test feature importance
def test_feature_importance():
    """Test feature importance computation."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)

    model = CustomThresholdModel()
    model.fit(X, y)

    importances = model.get_feature_importances()
    assert len(importances) == 3
    assert np.allclose(np.sum(importances), 1.0)
    assert np.allclose(importances, 1 / 3)  # Should be uniform
