# src/changepoint/threshold.py

"""Threshold-based classifier for change point detection in graph sequences."""

import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import shap  # type: ignore
from typing import List
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    """Threshold-based classifier for change point detection in graph sequences.
    Decision rule: y = 1[sum_i x_i > tau] where:
    - x_i are feature values (centrality measures)
    - tau is the decision threshold
    - 1[·] is the indicator function

    Designed for SHAP analysis to explain which centrality measures
    contribute most to detected changes.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize model with decision threshold.

        Args:
            threshold: Decision boundary tau
                - y = 1 if sum_i x_i > tau (change point)
                - y = 0 if sum_i x_i <= tau (no change)
        """
        if threshold <= 0:
            logger.error(f"Invalid threshold value: {threshold}")
            raise ValueError("Threshold must be positive")

        self.threshold = threshold
        logger.debug(f"Initialized CustomThresholdModel with threshold={threshold}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomThresholdModel":
        """Store feature dimensions and class labels.
        No parameter estimation needed as model uses fixed threshold rule:
        y_hat = 1[sum_i x_i > tau]

        Args:
            X: Feature matrix [n_samples x n_features]
            y: Binary labels (0: no change, 1: change)

        Returns:
            self: Fitted model instance

        Raises:
            ValueError: If input dimensions mismatch or invalid
        """

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        logger.info(f"Model fitted with {self.n_features_in_} features")
        logger.debug(f"Unique classes: {self.classes_}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict change points using threshold rule.
        Computes y_hat = 1[sum_i x_i > tau] for each sample

        Args:
            X: Feature matrix [n_samples x n_features]

        Returns:
            Binary predictions [n_samples]
            - 1: Change point detected
            - 0: No change

        Raises:
            ValueError: If X has wrong dimensions
        """
        check_is_fitted(self, ["n_features_in_", "classes_"])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            logger.error(
                f"Feature dimension mismatch: got {X.shape[1]}, expected {self.n_features_in_}"
            )
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        logger.debug(f"Making predictions for {X.shape[0]} samples")
        predictions = (np.sum(X, axis=1) > self.threshold).astype(int)

        n_changes = np.sum(predictions)
        logger.debug(
            f"Predicted {n_changes} change points out of {len(predictions)} samples"
        )
        logger.debug(f"Change point ratio: {n_changes / len(predictions)}")

        return predictions

    def predict_proba(self, X: np.ndarray, positive_class: bool = True) -> np.ndarray:
        """Compute change point probabilities.
        P(change) = sum_i x_i / (sum_i x_i + tau)
        Used for SHAP value computation to get continuous predictions.

        Args:
            X: Feature matrix [n_samples x n_features]
            positive_class: If True, returns P(change), if False returns P(no change)

        Returns:
            Probability array [n_samples] containing either P(change) or P(no change)
            depending on positive_class parameter

        Raises:
            ValueError: If X has wrong dimensions
        """
        check_is_fitted(self, ["n_features_in_", "classes_"])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            logger.error(
                f"Feature dimension mismatch: got {X.shape[1]}, expected {self.n_features_in_}"
            )
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        logger.debug(f"Computing probabilities for {X.shape[0]} samples")

        # Compute change probabilities
        sums = np.sum(X, axis=1)
        probs_change = sums / (sums + self.threshold)

        # Return either P(change) or P(no change) based on positive_class parameter
        probs = probs_change if positive_class else 1 - probs_change

        logger.debug(f"Average probability: {np.mean(probs)}")
        return probs

    def compute_shap_values(
        self,
        X: np.ndarray,
        change_points: List[int],
        sequence_length: int,
        window_size: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        probs: bool = False,
        positive_class: bool = True,
    ) -> np.ndarray:
        """SHAP value computation including data preparation and model training.

        Args:
            X: Feature matrix [n_timesteps x n_features]
            change_points: List of change point indices
            sequence_length: Total length of the sequence
            window_size: Size of window around change points to mark as positive
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            probs: If True, returns SHAP values for P(change), if False returns SHAP values for P(no change)
            positive_class: If True, returns SHAP values for positive class, if False returns SHAP values for negative class

        Returns:
            SHAP values for the entire sequence
        """
        # Create binary labels based on change points
        y = np.zeros(sequence_length)
        for cp in change_points:
            # Mark a window around each change point as positive
            start_idx = max(0, cp - window_size)
            end_idx = min(len(y), cp + window_size)
            y[start_idx:end_idx] = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.fit(X_train, y_train)

        # Compute SHAP values using training data as background
        try:
            if probs:
                explainer = shap.KernelExplainer(self.predict_proba, X_train)
            else:
                explainer = shap.KernelExplainer(self.predict, X_train)

            shap_values = explainer.shap_values(X)

            # If shap_values is a list (for binary classification), take values for positive class by default
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if positive_class else shap_values[0]

            return shap_values

        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            return np.zeros((len(X), X.shape[1]))  # Return dummy values on error

    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances for SHAP analysis."""
        return np.ones(self.n_features_in_) / self.n_features_in_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy of change point predictions.
        accuracy = (TP + TN) / n_samples

        Args:
            X: Feature matrix [n_samples x n_features]
            y: True binary labels

        Returns:
            Classification accuracy in [0,1]
        """
        logger.debug(f"Computing score for {len(X)} samples")
        accuracy = np.mean(self.predict(X) == y)
        logger.info(f"Model accuracy: {accuracy}")
        return accuracy

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters for scikit-learn compatibility."""
        return {"threshold": self.threshold}

    def set_params(self, **params: dict) -> "CustomThresholdModel":
        """Set model parameters for scikit-learn compatibility."""
        for param, value in params.items():
            logger.debug(f"Setting parameter {param}={value}")
            setattr(self, param, value)
        return self
