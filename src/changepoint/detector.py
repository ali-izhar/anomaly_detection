"""Change point detector using parallel martingale framework."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from .betting import BettingConfig
from .conformal import DistanceMetric
from .martingale import MartingaleConfig, run_parallel_detection


@dataclass
class DetectorConfig:
    """Configuration for change point detector.

    Default values are tuned for effective detection on graph feature data.
    """

    threshold: float = 30.0
    """Detection threshold. Lower values = more sensitive detection."""

    history_size: int = 10
    """Number of historical observations for prediction model."""

    window_size: Optional[int] = None
    """Optional sliding window size. If None, uses all history since last reset."""

    reset: bool = True
    """Whether to reset martingales after detection."""

    cooldown: int = 30
    """Minimum timesteps between detections."""

    betting_name: str = "mixture"
    """Betting function: 'mixture', 'power', 'beta', or 'exponential'."""

    betting_params: Dict[str, Any] = field(
        default_factory=lambda: {"epsilons": [0.7, 0.8, 0.9]}
    )
    """Parameters for the betting function."""

    random_state: Optional[int] = 42
    """Random seed for reproducibility."""

    distance_metric: DistanceMetric = "euclidean"
    """Distance metric for non-conformity scores: 'euclidean', 'mahalanobis', 'cosine', 'chebyshev'."""

    horizon_decay: float = 0.7
    """Exponential decay for horizon contributions. Lower = more dampening."""

    normalize_horizons: bool = False
    """Whether to normalize the horizon product. False gives faster detection."""

    def to_martingale_config(self) -> MartingaleConfig:
        """Convert to martingale configuration."""
        return MartingaleConfig(
            threshold=self.threshold,
            history_size=self.history_size,
            window_size=self.window_size,
            reset=self.reset,
            betting=BettingConfig(name=self.betting_name, params=self.betting_params),
            random_state=self.random_state,
            cooldown=self.cooldown,
            distance_metric=self.distance_metric,
            horizon_decay=self.horizon_decay,
            normalize_horizons=self.normalize_horizons,
        )


class ChangePointDetector:
    """Parallel martingale detector for change points in multivariate time series.

    Runs both traditional and horizon martingales in parallel.
    Both share the same base value and reset together, ensuring horizon
    never underperforms traditional.

    Example:
        >>> config = DetectorConfig(threshold=30.0)
        >>> detector = ChangePointDetector(config)
        >>> result = detector.run(features, predictions)
        >>> print(result['traditional_change_points'])
        >>> print(result['horizon_change_points'])
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.

        Args:
            config: Detection parameters. Uses sensible defaults if not provided.
        """
        self.config = config or DetectorConfig()
        self._change_points = []

    @property
    def change_points(self) -> list:
        """List of detected change point indices (from horizon martingale)."""
        return self._change_points.copy()

    def run(
        self,
        data: np.ndarray,
        predicted_data: np.ndarray,
    ) -> Dict[str, Any]:
        """Run parallel change point detection.

        Args:
            data: Feature matrix of shape (n_samples, n_features)
            predicted_data: Predictions of shape (n_predictions, horizon, n_features)

        Returns:
            Dict with detection results including:
                - traditional_change_points: Detections from traditional martingale
                - horizon_change_points: Detections from horizon martingale
                - traditional_sum_martingales: Summed traditional martingale values
                - horizon_sum_martingales: Summed horizon martingale values

        Raises:
            ValueError: If data dimensions are invalid
        """
        data = np.asarray(data)
        predictions = np.asarray(predicted_data)
        self._validate_input(data, predictions)

        mart_config = self.config.to_martingale_config()
        result = run_parallel_detection(data, predictions, mart_config)

        # Store horizon change points as the primary result
        self._change_points = result["horizon_change_points"]
        return result

    def _validate_input(self, data: np.ndarray, predictions: np.ndarray):
        """Validate input data dimensions."""
        if data.size == 0:
            raise ValueError("Empty input data")

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (n_samples, n_features), got shape {data.shape}")

        if predictions.size == 0:
            raise ValueError("Predictions required for parallel detection")

        if predictions.ndim != 3:
            raise ValueError(
                f"Predictions must be 3D (n_predictions, horizon, n_features), got shape {predictions.shape}"
            )

        if predictions.shape[-1] != data.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: data has {data.shape[1]} features, "
                f"predictions have {predictions.shape[-1]}"
            )
