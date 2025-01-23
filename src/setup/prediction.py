# src/setup/prediction.py

"""Network prediction and model management."""

from typing import Dict, List, Any, Optional
import numpy as np
import networkx as nx
import logging
from dataclasses import dataclass

from predictor.weighted import WeightedPredictor
from predictor.hybrid import (
    BAPredictor,
    SBMPredictor,
    WSPredictor,
    ERPredictor,
)

logger = logging.getLogger(__name__)

# Model to predictor mapping
PREDICTOR_MAP = {
    "weighted": WeightedPredictor,
    "hybrid": {
        "ba": BAPredictor,
        "ws": WSPredictor,
        "er": ERPredictor,
        "sbm": SBMPredictor,
    },
}

# Graph model name mapping
GRAPH_MODELS = {
    "barabasi_albert": "ba",
    "watts_strogatz": "ws",
    "erdos_renyi": "er",
    "stochastic_block_model": "sbm",
    "ba": "ba",
    "ws": "ws",
    "er": "er",
    "sbm": "sbm",
}

# Model predictor recommendations
MODEL_PREDICTOR_RECOMMENDATIONS = {
    "ba": ["weighted", "hybrid"],
    "ws": ["weighted", "hybrid"],
    "er": ["weighted", "hybrid"],
    "sbm": ["weighted", "hybrid"],
}


@dataclass
class PredictionResult:
    """Container for prediction results."""

    time: int
    adjacency: np.ndarray
    graph: nx.Graph
    metrics: Dict[str, float]
    history_size: int


class PredictorFactory:
    """Creates and manages network predictors."""

    @staticmethod
    def create_predictor(
        predictor_type: Optional[str] = None, model: Optional[str] = None, **kwargs
    ) -> Any:
        """Create predictor based on type and model."""
        if not predictor_type:
            logger.info("Using default weighted predictor")
            return WeightedPredictor()

        if predictor_type == "weighted":
            logger.info("Using weighted predictor")
            return WeightedPredictor()
        elif predictor_type == "hybrid":
            predictor_class = PREDICTOR_MAP["hybrid"].get(model)
            if predictor_class is None:
                raise ValueError(f"No hybrid predictor for model type {model}")
            logger.info(f"Using hybrid predictor for {model} model")
            return predictor_class(**kwargs)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")


class NetworkPredictor:
    """Generates and validates network predictions."""

    def __init__(
        self,
        predictor: Any,
        feature_extractor: Any,
        min_history: int,
        prediction_window: int,
    ):
        """Initialize network predictor."""
        self.predictor = predictor
        self.feature_extractor = feature_extractor
        self.min_history = min_history
        self.prediction_window = prediction_window

    def generate_predictions(
        self,
        network_series: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate predictions for network evolution."""
        predictions = []
        seq_len = len(network_series)

        # Step 1: Generate predictions for each timestep
        for t in range(self.min_history, seq_len):
            # Step 2: Get history window
            history = network_series[:t]

            # Step 3: Generate prediction
            predicted_adjs = self.predictor.predict(
                history, horizon=self.prediction_window
            )

            # Step 4: Convert to graph and compute metrics
            pred_graph = nx.from_numpy_array(predicted_adjs[0])
            predictions.append(
                {
                    "time": t,
                    "adjacency": predicted_adjs[0],
                    "graph": pred_graph,
                    "metrics": self.feature_extractor.get_all_metrics(
                        pred_graph
                    ).__dict__,
                    "history_size": len(history),
                }
            )

        return predictions

    def validate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        actual_graphs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate predictions against actual evolution."""
        validation_metrics = {
            "structural": [],  # Graph structure similarity
            "feature": [],  # Feature-based similarity
            "temporal": [],  # Temporal accuracy
        }

        # Step 1: Compare each prediction with actual
        for pred, actual in zip(predictions, actual_graphs[self.min_history :]):
            # Step 2: Compute structural similarity
            edit_distance = nx.graph_edit_distance(
                pred["graph"], actual["graph"], timeout=2  # Limit computation time
            )
            validation_metrics["structural"].append(
                1.0 / (1.0 + (edit_distance or 0))  # Handle None result
            )

            # Step 3: Compare features
            pred_features = pred["metrics"]
            actual_features = actual["metrics"]
            feature_diff = {
                k: abs(pred_features[k] - actual_features[k])
                for k in pred_features.keys()
                if k in actual_features
            }
            validation_metrics["feature"].append(
                1.0 - np.mean(list(feature_diff.values()))
            )

            # Step 4: Check temporal accuracy
            temporal_score = 1.0 if actual.get("is_change_point", False) else 0.0
            validation_metrics["temporal"].append(temporal_score)

        # Step 5: Compute summary statistics
        summary = {
            metric: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }
            for metric, scores in validation_metrics.items()
        }

        return {
            "detailed": validation_metrics,
            "summary": summary,
        }

    def assess_prediction_confidence(
        self, predictions: List[Dict[str, Any]], threshold: float = 0.8
    ) -> Dict[str, List[float]]:
        """Assess confidence in predictions."""
        confidence_scores = []
        confidence_flags = []

        # Step 1: Compute confidence for each prediction
        for pred in predictions:
            # Step 2: Calculate confidence factors
            history_factor = min(1.0, pred["history_size"] / 100)
            density = nx.density(pred["graph"])
            clustering = nx.average_clustering(pred["graph"])

            # Step 3: Combine factors
            confidence = np.mean(
                [
                    history_factor,
                    1.0 - abs(0.5 - density),  # Penalize extreme densities
                    clustering,  # Higher clustering -> more stable
                ]
            )

            confidence_scores.append(float(confidence))
            confidence_flags.append(confidence >= threshold)

        return {
            "scores": confidence_scores,
            "flags": confidence_flags,
            "threshold": threshold,
            "mean_confidence": float(np.mean(confidence_scores)),
        }
