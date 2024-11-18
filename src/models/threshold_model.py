# src/models/models.py

import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute change point probabilities.
        P(change) = sum_i x_i / (sum_i x_i + tau)
        Used for SHAP value computation to get continuous predictions.

        Args:
            X: Feature matrix [n_samples x n_features]

        Returns:
            Probability matrix [n_samples x 2]
            - Column 0: P(no change)
            - Column 1: P(change)

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
        sums = np.sum(X, axis=1).reshape(-1, 1)
        probs_change = sums / (sums + self.threshold)

        # Return [P(no change), P(change)]
        probabilities = np.hstack([1 - probs_change, probs_change])

        logger.debug(f"Average change probability: {np.mean(probs_change)}")
        return probabilities

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
