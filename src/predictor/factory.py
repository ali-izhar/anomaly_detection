# src/predictor/factory.py

"""Factory for creating different types of network predictors."""

from typing import Dict, Any, Optional, Type
from .base import BasePredictor
from .adaptive import AdaptivePredictor
from .auto import AutoChangepointPredictor
from .weighted import EnhancedWeightedPredictor
from .statistical import StatisticalAdaptivePredictor
from .aware import ChangepointAwarePredictor


class PredictorFactory:
    """Factory class for creating network predictors.

    This factory provides a centralized way to create different types of predictors
    with appropriate configuration. It supports all predictor types that implement
    the BasePredictor interface.
    """

    # Registry of available predictor types
    PREDICTOR_TYPES = {
        "adaptive": AdaptivePredictor,
        "auto": AutoChangepointPredictor,
        "weighted": EnhancedWeightedPredictor,
        "statistical": StatisticalAdaptivePredictor,
        "aware": ChangepointAwarePredictor,
    }

    # Default configurations for each predictor type
    DEFAULT_CONFIGS = {
        "adaptive": {
            "k": 10,
            "alpha": 0.8,
            "initial_gamma": 0.1,
            "initial_beta": 0.5,
            "error_window": 5,
        },
        "auto": {"alpha": 0.85, "min_phase_length": 40},
        "weighted": {
            "n_history": 5,
            "adaptive": True,
            "enforce_connectivity": True,
            "binary": True,
            "spectral_reg": 0.4,
            "community_reg": 0.4,
            "n_communities": 2,
            "temporal_window": 10,
            "distribution_reg": 0.3,
        },
        "statistical": {
            "alpha": 0.8,
            "change_threshold": 5,
            "min_phase_length": 40,
            "history_size": 40,
        },
        "aware": {
            "n_nodes": 50,
            "alpha": 0.99,
            "min_phase": 40,
        },
    }

    @classmethod
    def register_predictor(
        cls,
        name: str,
        predictor_class: Type[BasePredictor],
        default_config: Dict[str, Any],
    ) -> None:
        """Register a new predictor type with the factory.

        Parameters
        ----------
        name : str
            Name to register the predictor under
        predictor_class : Type[BasePredictor]
            The predictor class (must implement BasePredictor)
        default_config : Dict[str, Any]
            Default configuration for the predictor
        """
        if not issubclass(predictor_class, BasePredictor):
            raise ValueError(f"Predictor class must implement BasePredictor interface")

        cls.PREDICTOR_TYPES[name] = predictor_class
        cls.DEFAULT_CONFIGS[name] = default_config

    @classmethod
    def create(
        cls, predictor_type: str, config: Optional[Dict[str, Any]] = None
    ) -> BasePredictor:
        """Create a predictor instance of the specified type.

        Parameters
        ----------
        predictor_type : str
            Type of predictor to create. Must be one of:
            - "adaptive": Adaptive predictor with changepoint detection
            - "auto": Auto-changepoint predictor with temporal memory
            - "weighted": Enhanced weighted predictor with distribution awareness
            - "adaptive_rnn": Graph Adaptive RNN Predictor
            - "hbtgn": Hierarchical Bayesian Temporal Graph Network Predictor
        config : Optional[Dict[str, Any]]
            Configuration overrides for the predictor.
            If None, uses default configuration.

        Returns
        -------
        BasePredictor
            Instantiated predictor of the requested type

        Raises
        ------
        ValueError
            If predictor_type is not recognized
        """
        if predictor_type not in cls.PREDICTOR_TYPES:
            valid_types = list(cls.PREDICTOR_TYPES.keys())
            raise ValueError(
                f"Unknown predictor type: {predictor_type}. Must be one of {valid_types}"
            )

        # Get the predictor class and default config
        predictor_class = cls.PREDICTOR_TYPES[predictor_type]
        default_config = cls.DEFAULT_CONFIGS[predictor_type].copy()

        # Update with any provided config overrides
        if config:
            default_config.update(config)

        # Instantiate the predictor
        return predictor_class(**default_config)

    @classmethod
    def get_available_types(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about available predictor types.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping predictor names to their default configurations
        """
        return {
            name: {
                "class": predictor_class,
                "default_config": cls.DEFAULT_CONFIGS[name],
            }
            for name, predictor_class in cls.PREDICTOR_TYPES.items()
        }
