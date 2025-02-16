# src/changepoint/threshold.py

"""Threshold-based classifier for change point detection in graph sequences.

This module implements a threshold-based classifier for detecting change points in
graph sequences using a fixed decision rule. The classifier is designed to work with
both single-view and multiview martingale sequences and provides SHAP-based
explanations for detected changes.

Mathematical Framework:
---------------------
1. Decision Rule:
   For a feature vector x ∈ ℝᵈ, the classifier predicts:
   y = 1[∑ᵢ xᵢ > τ]
   where τ is the decision threshold.

2. Probability Estimation:
   P(change) = ∑ᵢ xᵢ / (∑ᵢ xᵢ + τ)
   This provides a continuous score in [0,1] for SHAP analysis.

3. SHAP Analysis:
   - Feature-level: Which graph properties contribute to changes
   - Martingale-level: How different views influence detection

Properties:
----------
1. Interpretable decision boundary
2. Fast computation (linear in feature dimension)
3. No parameter estimation required
4. Compatible with scikit-learn API
5. Supports both hard and soft predictions

References:
----------
[1] Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model
    Predictions." Advances in Neural Information Processing Systems 30.
[2] Ribeiro, M. T., et al. (2016). "Why Should I Trust You?: Explaining the
    Predictions of Any Classifier." KDD '16.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import shap

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

logger = logging.getLogger(__name__)

# Define a fixed feature order for consistent visualization and analysis in SHAP plots.
FEATURE_ORDER = [
    "degree",
    "density",
    "clustering",
    "betweenness",
    "eigenvector",
    "closeness",
    "singular_value",
    "laplacian",
]


@dataclass(frozen=True)
class ShapConfig:
    """Configuration for SHAP analysis.

    Attributes:
        window_size: Size of window around change points for labeling.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        use_probabilities: Whether to analyze probability outputs (via predict_proba).
        positive_class: Whether to analyze positive class predictions.
    """

    window_size: int = 5
    test_size: float = 0.2
    random_state: int = 42
    use_probabilities: bool = False
    positive_class: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0,1), got {self.test_size}")


class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    """Threshold-based classifier for change point detection.

    This classifier implements a simple fixed-threshold rule:
        y = 1[∑ᵢ xᵢ > τ]
    where x are the input features and τ is the decision threshold.

    The model is designed for:
      1. Binary classification of change points.
      2. Probability estimation for uncertain predictions.
      3. SHAP-based feature importance analysis.
      4. Martingale contribution analysis in multiview detection.

    Mathematical Properties:
    -------------------------
      - Linear decision boundary in feature space.
      - Monotonic in feature values.
      - Scale-invariant threshold.
      - Interpretable decision rule.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the model with a decision threshold.

        Args:
            threshold: Decision boundary τ. Predicts change if ∑ᵢ xᵢ > τ.

        Raises:
            ValueError: If threshold is not positive.
        """
        # Validate threshold: must be positive.
        if threshold <= 0:
            logger.error(f"Invalid threshold value: {threshold}")
            raise ValueError("Threshold must be positive")
        self.threshold = threshold
        logger.debug(f"Initialized model with threshold={threshold}")

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "CustomThresholdModel":
        """Fit the model (stores dimensions for validation).

        No parameter estimation is needed as the decision rule is fixed.
        This method only validates and stores input dimensions.

        Args:
            X: Feature matrix [n_samples × n_features].
            y: Binary labels (0: no change, 1: change).
            sample_weight: Optional sample weights (not used).

        Returns:
            self: The fitted model.
        """
        # Validate X and y using scikit-learn's utility.
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[
            1
        ]  # Store number of features for later validation.
        self.classes_ = np.unique(y)  # Save unique classes.
        logger.info(f"Model fitted with {self.n_features_in_} features")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change points using the threshold rule.

        Implements the decision rule:
            y = 1[∑ᵢ xᵢ > τ]

        Args:
            X: Feature matrix [n_samples × n_features].

        Returns:
            Binary predictions (0: no change, 1: change).

        Raises:
            ValueError: If X has wrong dimensions.
        """
        check_is_fitted(self)  # Ensure model has been fitted.
        X = check_array(X, accept_sparse=False)

        # Validate that the input feature dimension matches that seen during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Sum features and apply threshold to determine change.
        predictions = (np.sum(X, axis=1) > self.threshold).astype(int)
        n_changes = np.sum(predictions)
        logger.debug(f"Predicted {n_changes}/{len(predictions)} changes")
        return predictions

    def predict_proba(self, X: np.ndarray, positive_class: bool = True) -> np.ndarray:
        """Compute change point probabilities.

        Uses the formula:
            P(change) = ∑ᵢ xᵢ / (∑ᵢ xᵢ + τ)

        Args:
            X: Feature matrix [n_samples × n_features].
            positive_class: If True, return P(change).

        Returns:
            Probability array [n_samples].
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Validate input dimensions.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        sums = np.sum(X, axis=1)  # Sum over features for each sample.
        # Calculate probability using the given formula.
        probs_change = sums / (sums + self.threshold)
        return probs_change if positive_class else 1 - probs_change

    def compute_shap_values(
        self,
        X: np.ndarray,
        change_points: List[int],
        sequence_length: int,
        config: Optional[ShapConfig] = None,
    ) -> np.ndarray:
        """Compute SHAP values for feature importance analysis.

        Creates binary labels around change points and uses SHAP's
        KernelExplainer to compute feature contributions.

        Args:
            X: Feature matrix [n_samples × n_features].
            change_points: Indices of detected changes.
            sequence_length: Total sequence length.
            config: SHAP analysis configuration.

        Returns:
            SHAP values [n_samples × n_features].
        """
        config = config or ShapConfig()

        # Create binary labels where a window around each change point is marked as 1.
        y = np.zeros(sequence_length)
        for cp in change_points:
            # Label a window around each change point.
            start_idx = max(0, cp - config.window_size)
            end_idx = min(len(y), cp + config.window_size)
            y[start_idx:end_idx] = 1

        # Split the dataset for training and testing.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        # Fit the model on the training split.
        self.fit(X_train, y_train)

        try:
            # Choose the appropriate prediction function based on configuration.
            predict_fn = (
                self.predict_proba if config.use_probabilities else self.predict
            )

            # Use SHAP's KernelExplainer to compute SHAP values.
            explainer = shap.KernelExplainer(predict_fn, X_train)
            shap_values = explainer.shap_values(X)

            # If SHAP returns a list (one per class), select the desired class.
            if isinstance(shap_values, list):
                shap_values = (
                    shap_values[1] if config.positive_class else shap_values[0]
                )
            return shap_values

        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            # In case of failure, return a zero matrix.
            return np.zeros((len(X), X.shape[1]))

    def compute_martingale_shap_values(
        self,
        martingales: Dict[str, Dict[str, Any]],
        change_points: List[int],
        sequence_length: int,
        config: Optional[ShapConfig] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute SHAP values for martingale contributions.

        Analyzes how different features' martingales influence
        change point detection in multiview settings.

        Args:
            martingales: Martingale values per feature.
            change_points: True change points.
            sequence_length: Sequence length.
            config: SHAP analysis configuration.

        Returns:
            Tuple of (SHAP values, feature names).
        """
        # Convert the martingales dictionary into a feature matrix.
        feature_matrix = []
        feature_names = []

        for feature in FEATURE_ORDER:
            # Skip combined features and only process individual features.
            if feature in martingales and feature != "combined":
                martingales_array = np.array(
                    [
                        x.item() if isinstance(x, np.ndarray) else x
                        for x in martingales[feature]["martingales"]
                    ]
                )
                feature_matrix.append(martingales_array)
                feature_names.append(feature)

        if not feature_matrix:
            raise ValueError("No valid features in martingales dictionary")

        # Stack features column-wise to form a 2D matrix.
        X = np.vstack(feature_matrix).T

        # Compute SHAP values using the previously defined method.
        shap_values = self.compute_shap_values(
            X=X,
            change_points=change_points,
            sequence_length=sequence_length,
            config=config,
        )

        return shap_values, feature_names

    def get_feature_importances(self) -> np.ndarray:
        """Get uniform feature importances for SHAP analysis."""
        check_is_fitted(self)
        # Return equal importance for all features (since no training is done).
        return np.ones(self.n_features_in_) / self.n_features_in_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Accuracy score in [0,1].
        """
        return np.mean(self.predict(X) == y)

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters for scikit-learn compatibility."""
        return {"threshold": self.threshold}

    def set_params(self, **params: dict) -> "CustomThresholdModel":
        """Set model parameters for scikit-learn compatibility."""
        for param, value in params.items():
            logger.debug(f"Setting parameter {param}={value}")
            setattr(self, param, value)
        return self
