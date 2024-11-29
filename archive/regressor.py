# archive/regressor.py

"""MartingalePredictor class that uses a Random Forest model to predict future martingale values based on:
- Historical feature vectors
- Recent martingale values
- Rate of change in martingale values

It uses a sliding window approach to prepare training data, where each input consists of:
- A window of feature vectors
- A window of martingale values
- First-order derivatives of martingale values

To make predictions, it:
- Forecasts future martingale values
- Predicts if/when the threshold will be exceeded
- Provides confidence score for prediction
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Configuration for anomaly prediction.

    Attributes:
        window_size: Size of historical window to consider
        forecast_horizon: Number of future timesteps to predict
        threshold: Martingale threshold for anomaly detection
        confidence_level: Required confidence for prediction (0-1)
    """

    window_size: int = 10
    forecast_horizon: int = 5
    threshold: float = 20.0
    confidence_level: float = 0.9


class MartingalePredictor:
    """Predicts future martingale values and forecasts anomalies before detection.

    Uses a sliding window approach to predict future martingale values based on:
    - Historical feature values
    - Current martingale values
    - Rate of change in martingale values
    """

    def __init__(self, config: PredictionConfig) -> None:
        """Initialize predictor with configuration parameters.

        Args:
            config: Prediction configuration parameters
        """
        self.config = config
        self._model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._feature_history: List[np.ndarray] = []
        self._martingale_history: List[float] = []
        self._is_fitted = False

        logger.info(f"Initialized MartingalePredictor with config: {config}")

    def prepare_training_data(
        self, features: np.ndarray, martingales: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sliding window training data.

        Args:
            features: Historical feature vectors [n_samples, n_features]
            martingales: Historical martingale values [n_samples]

        Returns:
            Tuple of (X_train, y_train) for model training
        """
        X, y = [], []
        window_size = self.config.window_size
        horizon = self.config.forecast_horizon

        for i in range(window_size, len(features) - horizon):
            # Feature window
            feature_window = features[i - window_size : i]

            # Martingale window and derivatives
            mart_window = martingales[i - window_size : i]
            mart_derivatives = np.diff(mart_window)

            # Combine features
            window_features = np.concatenate(
                [feature_window.flatten(), mart_window, mart_derivatives]
            )

            # Target is future martingale values
            target = martingales[i : i + horizon]

            X.append(window_features)
            y.append(target)

        return np.array(X), np.array(y)

    def fit(self, features: np.ndarray, martingales: np.ndarray) -> None:
        """Train the prediction model on historical data.

        Args:
            features: Historical feature vectors [n_samples, n_features]
            martingales: Historical martingale values [n_samples]
        """
        logger.info("Preparing training data for predictor")
        X_train, y_train = self.prepare_training_data(features, martingales)

        logger.info(f"Training predictor model with {len(X_train)} samples")
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(
        self, current_features: np.ndarray, current_martingales: np.ndarray
    ) -> Dict[str, Any]:
        """Predict future martingale values and potential anomalies.

        Args:
            current_features: Current feature window [window_size, n_features]
            current_martingales: Current martingale window [window_size]

        Returns:
            Dictionary containing:
            - predicted_martingales: Predicted future values
            - anomaly_predicted: Whether anomaly is predicted
            - predicted_detection_time: When anomaly threshold will be exceeded
            - confidence: Prediction confidence score

        Raises:
            ValueError: If model is not fitted or input dimensions are invalid
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare input features
        mart_derivatives = np.diff(current_martingales)
        input_features = np.concatenate(
            [current_features.flatten(), current_martingales, mart_derivatives]
        ).reshape(1, -1)

        # Make prediction
        predicted_marts = self._model.predict(input_features)[0]

        # Check if threshold will be exceeded
        threshold_exceeded = predicted_marts > self.config.threshold

        # Calculate prediction confidence
        predictions = []
        for estimator in self._model.estimators_:
            pred = estimator.predict(input_features)[0]
            predictions.append(pred)
        confidence = np.mean(
            [np.any(pred > self.config.threshold) for pred in predictions]
        )

        # Determine if anomaly should be predicted
        anomaly_predicted = confidence >= self.config.confidence_level and np.any(
            threshold_exceeded
        )

        # Find predicted detection time
        predicted_detection_time = None
        if anomaly_predicted:
            detection_times = np.where(threshold_exceeded)[0]
            if len(detection_times) > 0:
                predicted_detection_time = detection_times[0]

        logger.info(f"Prediction complete - Anomaly predicted: {anomaly_predicted}")
        if anomaly_predicted:
            logger.info(f"Predicted detection at t+{predicted_detection_time}")

        return {
            "predicted_martingales": predicted_marts,
            "anomaly_predicted": anomaly_predicted,
            "predicted_detection_time": predicted_detection_time,
            "confidence": confidence,
        }

    def update_history(self, features: np.ndarray, martingale: float) -> None:
        """Update historical data with new observations.

        Args:
            features: New feature vector
            martingale: New martingale value
        """
        self._feature_history.append(features)
        self._martingale_history.append(martingale)

        # Maintain window size
        if len(self._feature_history) > self.config.window_size:
            self._feature_history.pop(0)
            self._martingale_history.pop(0)


# if __name__ == "__main__":

#     # 1. Train the model on historical data
#     config = PredictionConfig(
#         window_size=10, forecast_horizon=5, threshold=20.0, confidence_level=0.9
#     )

#     predictor = MartingalePredictor(config)
#     predictor.fit(historical_features, historical_martingales)

#     # 2. Online prediction with new data
#     prediction = predictor.predict(current_features, current_martingales)
#     if prediction["anomaly_predicted"]:
#         print(
#             f"Anomaly predicted to occur at t+{prediction['predicted_detection_time']}"
#         )
