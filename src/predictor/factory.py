"""Factory for creating predictors."""

from typing import Dict, Any, Optional
from .feature_predictor import FeaturePredictor, GraphFeaturePredictor


class PredictorFactory:
    """Factory for creating predictor instances.

    Uses FeaturePredictor (Holt's double exponential smoothing with trend).
    """

    DEFAULT_CONFIG = {
        "alpha": 0.3,
        "beta": 0.1,
        "n_history": 10,
    }

    @classmethod
    def create(
        cls,
        predictor_type: str = "feature",
        config: Optional[Dict[str, Any]] = None
    ) -> GraphFeaturePredictor:
        """Create predictor instance.

        Args:
            predictor_type: Predictor type (only 'feature' supported)
            config: Configuration parameters

        Returns:
            GraphFeaturePredictor instance
        """
        # Accept 'graph' for backwards compatibility, but use feature predictor
        if predictor_type not in ("feature", "graph"):
            raise ValueError(f"Unknown predictor type: {predictor_type}. Use 'feature'.")

        merged = cls.DEFAULT_CONFIG.copy()
        if config:
            # Filter out legacy graph predictor params
            valid_params = {"alpha", "beta", "n_history", "feature_names"}
            filtered = {k: v for k, v in config.items() if k in valid_params}
            merged.update(filtered)

        return GraphFeaturePredictor(**merged)
