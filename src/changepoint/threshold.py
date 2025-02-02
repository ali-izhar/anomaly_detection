# src/changepoint/threshold.py

"""Threshold-based classifier for change point detection in graph sequences.

This module defines a simple classifier that uses a fixed decision rule:
    y = 1[sum_i x_i > τ]
where x_i are the feature values (e.g. centrality measures extracted from graph snapshots)
and τ is the decision threshold. This classifier is also set up for SHAP analysis so that
one can explain which features (centrality measures) contribute most to the detected changes.
"""

from typing import List
import logging
import numpy as np
import shap

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

logger = logging.getLogger(__name__)


class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    """Threshold-based classifier for change point detection in graph sequences.

    This classifier uses a simple fixed-threshold rule applied to the sum of features.
    It predicts a change (label 1) if the sum of feature values is greater than the threshold,
    and no change (label 0) otherwise. It is also designed for SHAP analysis.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the model with a decision threshold.

        Args:
            threshold: The decision boundary τ. The model predicts:
                - 1 (change) if sum_i x_i > τ
                - 0 (no change) otherwise.

        Raises:
            ValueError: If threshold is not positive.
        """
        if threshold <= 0:
            logger.error(f"Invalid threshold value: {threshold}")
            raise ValueError("Threshold must be positive")
        self.threshold = threshold
        logger.debug(f"Initialized CustomThresholdModel with threshold={threshold}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomThresholdModel":
        """
        Fit the model to the provided data.

        For this classifier, no parameter estimation is necessary because the decision rule
        is fixed (i.e. y_hat = 1[sum_i x_i > τ]). This method only stores the input feature
        dimensions and the class labels for compatibility with scikit-learn.

        Args:
            X: Feature matrix of shape [n_samples x n_features]
            y: Binary labels (0: no change, 1: change)

        Returns:
            self: Fitted model instance.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # Store number of input features.
        self.classes_ = np.unique(y)  # Store unique class labels.
        logger.info(f"Model fitted with {self.n_features_in_} features")
        logger.debug(f"Unique classes: {self.classes_}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict change points using the fixed threshold rule.

        For each sample, the prediction is given by:
            y_hat = 1 if sum(x_i) > τ else 0

        Args:
            X: Feature matrix of shape [n_samples x n_features]

        Returns:
            Binary predictions: An array of 0s and 1s.

        Raises:
            ValueError: If X does not have the expected number of features.
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
        # Sum features and compare against threshold.
        predictions = (np.sum(X, axis=1) > self.threshold).astype(int)
        n_changes = np.sum(predictions)
        logger.debug(
            f"Predicted {n_changes} change points out of {len(predictions)} samples"
        )
        logger.debug(f"Change point ratio: {n_changes / len(predictions)}")
        return predictions

    def predict_proba(self, X: np.ndarray, positive_class: bool = True) -> np.ndarray:
        """
        Compute change point probabilities.

        This method computes a continuous probability of change using:
            P(change) = sum(x_i) / (sum(x_i) + τ)
        This continuous output is useful for SHAP analysis.

        Args:
            X: Feature matrix of shape [n_samples x n_features]
            positive_class: If True, return P(change); if False, return P(no change)

        Returns:
            A probability array of shape [n_samples].

        Raises:
            ValueError: If X does not have the expected dimensions.
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
        sums = np.sum(X, axis=1)
        # Compute probability of change.
        probs_change = sums / (sums + self.threshold)
        # Return probability for the positive class if requested.
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
        """
        Compute SHAP values to explain which features contribute to detected changes.

        This method creates binary labels around the provided change points and trains the model.
        It then uses SHAP's KernelExplainer to compute SHAP values for the input feature matrix.

        Args:
            X: Feature matrix [n_timesteps x n_features]
            change_points: List of indices where a change point was detected.
            sequence_length: Total length of the sequence.
            window_size: Size of the window around change points to mark as positive.
            test_size: Proportion of the data to use for testing.
            random_state: Random seed.
            probs: If True, compute SHAP values for the probability output.
            positive_class: If True, return SHAP values for the positive class.

        Returns:
            SHAP values for the entire sequence.
        """
        # Create binary labels: mark a window around each change point as positive.
        y = np.zeros(sequence_length)
        for cp in change_points:
            start_idx = max(0, cp - window_size)
            end_idx = min(len(y), cp + window_size)
            y[start_idx:end_idx] = 1

        # Split data for training the model.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.fit(X_train, y_train)

        try:
            # Choose the explainer based on whether we want probabilities.
            if probs:
                explainer = shap.KernelExplainer(self.predict_proba, X_train)
            else:
                explainer = shap.KernelExplainer(self.predict, X_train)
            shap_values = explainer.shap_values(X)
            # If multiple classes, choose the appropriate one.
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if positive_class else shap_values[0]
            return shap_values
        except Exception as e:
            logger.error(f"SHAP computation failed: {str(e)}")
            # Return a dummy array on error.
            return np.zeros((len(X), X.shape[1]))

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances for SHAP analysis.
        Here, all features are considered equally important.
        """
        return np.ones(self.n_features_in_) / self.n_features_in_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy of change point predictions.

        Accuracy is defined as (TP + TN) / n_samples.

        Args:
            X: Feature matrix [n_samples x n_features]
            y: True binary labels.

        Returns:
            Accuracy in [0,1].
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
