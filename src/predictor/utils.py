# src/predictor/utils.py

"""Utility functions for network prediction and analysis."""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import networkx as nx
import logging

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from changepoint.detector import ChangePointDetector
from changepoint.threshold import CustomThresholdModel

logger = logging.getLogger(__name__)

#######################
# Data Generation
#######################


def generate_network_series(config: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
    """Generate a time series of evolving networks.

    Returns:
        Dict containing:
        - graphs: List of network states
        - change_points: List of change point indices
        - parameters: List of parameters for each segment
        - metadata: Additional information
        - model: Model type
        - num_changes: Number of changes
        - n: Number of nodes
        - sequence_length: Total sequence length
    """
    if seed is not None:
        np.random.seed(seed)

    generator = GraphGenerator()
    feature_extractor = NetworkFeatureExtractor()

    # Generate base sequence
    result = generator.generate_sequence(
        model=config["model"], params=config["params"], seed=seed
    )

    # Map parameters to timesteps
    param_map = {}
    current_params = result["parameters"][0]
    for i in range(config["params"].seq_len):
        if i in result["change_points"]:
            current_params = result["parameters"][result["change_points"].index(i) + 1]
        param_map[i] = current_params

    # Convert to network states
    network_states = []
    for i, adj in enumerate(result["graphs"]):
        G = nx.from_numpy_array(adj)
        network_states.append(
            {
                "time": i,
                "adjacency": adj,
                "graph": G,
                "metrics": feature_extractor.get_all_metrics(G).__dict__,
                "params": param_map[i],
                "is_change_point": i in result["change_points"],
            }
        )

    # Return in expected format
    return {
        "graphs": network_states,
        "change_points": result["change_points"],
        "parameters": result["parameters"],
        "metadata": result["metadata"],
        "model": result["model"],
        "num_changes": result["num_changes"],
        "n": result["n"],
        "sequence_length": result["sequence_length"],
    }


#######################
# Prediction
#######################


def generate_predictions(
    network_series: List[Dict[str, Any]],
    predictor: Any,
    min_history: int,
    seq_len: int,
    prediction_window: int,
) -> List[Dict[str, Any]]:
    """Generate predictions using the given predictor."""
    predictions = []
    feature_extractor = NetworkFeatureExtractor()

    for t in range(min_history, seq_len):
        history = network_series[:t]
        predicted_adjs = predictor.predict(history, horizon=prediction_window)

        pred_graph = nx.from_numpy_array(predicted_adjs[0])
        predictions.append(
            {
                "time": t,
                "adjacency": predicted_adjs[0],
                "graph": pred_graph,
                "metrics": feature_extractor.get_all_metrics(pred_graph).__dict__,
                "history_size": len(history),
            }
        )

    return predictions


#######################
# Metrics Computation
#######################


def compute_network_features(graphs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Compute network features for analysis."""
    feature_extractor = NetworkFeatureExtractor()
    features = {
        "degree": [],
        "clustering": [],
        "betweenness": [],
        "closeness": [],
    }

    for state in graphs:
        G = state["graph"]
        metrics = feature_extractor.get_all_metrics(G)
        features["degree"].append(metrics.avg_degree)
        features["clustering"].append(metrics.clustering)
        features["betweenness"].append(metrics.avg_betweenness)
        features["closeness"].append(metrics.avg_closeness)

    return {k: np.array(v) for k, v in features.items()}


def analyze_prediction_phases(
    predictions: List[Dict[str, Any]],
    actual_series: List[Dict[str, Any]],
    min_history: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Analyze prediction accuracy across different phases."""
    feature_extractor = NetworkFeatureExtractor()

    if not predictions or not actual_series:
        logger.warning("No predictions or actual series to analyze")
        return {"phases": {}, "error": "No data to analyze"}

    # Define phases
    total_predictions = len(predictions)
    if total_predictions < 3:
        phases = [(0, total_predictions, "All predictions")]
    else:
        third = total_predictions // 3
        phases = [
            (0, third, "First third"),
            (third, 2 * third, "Middle third"),
            (2 * third, None, "Last third"),
        ]

    results = {"phases": {}}

    # Analyze each phase
    for start, end, phase_name in phases:
        logger.info(f"\nAnalyzing {phase_name}:")
        phase_predictions = predictions[start:end]
        phase_actuals = actual_series[
            min_history + start : min_history + (end if end else len(predictions))
        ]

        # Calculate errors
        all_errors = []
        for pred, actual in zip(phase_predictions, phase_actuals):
            pred_metrics = feature_extractor.get_all_metrics(pred["graph"]).__dict__
            actual_metrics = feature_extractor.get_all_metrics(actual["graph"]).__dict__
            all_errors.append(calculate_error_metrics(actual_metrics, pred_metrics))

        if not all_errors:
            logger.warning(f"No errors calculated for {phase_name}")
            continue

        # Average errors
        avg_errors = {
            metric: np.mean([e[metric] for e in all_errors])
            for metric in all_errors[0].keys()
        }

        results["phases"][phase_name] = {
            "start": start,
            "end": end,
            "errors": avg_errors,
        }

        for metric, error in avg_errors.items():
            logger.info(f"Average MAE for {metric}: {error:.3f}")

    return results


#######################
# Martingale Analysis
#######################


def compute_martingales(
    features: Dict[str, np.ndarray],
    detector: ChangePointDetector,
    threshold: float = 30.0,
    epsilon: float = 0.8,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compute martingales for network features."""
    martingales = {"reset": {}, "cumulative": {}}

    for feature_name, feature_data in features.items():
        # Reset martingales
        reset_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=True,
        )
        martingales["reset"][feature_name] = {
            "martingales": reset_results["martingale_values"],
            "change_detected_instant": reset_results["change_points"],
        }

        # Cumulative martingales
        cumul_results = detector.detect_changes(
            data=feature_data.reshape(-1, 1),
            threshold=threshold,
            epsilon=epsilon,
            reset=False,
        )
        martingales["cumulative"][feature_name] = {
            "martingales": cumul_results["martingale_values"],
            "change_detected_instant": cumul_results["change_points"],
        }

    return martingales


def compute_shap_values(
    martingales: Dict[str, Dict[str, Dict[str, Any]]],
    features: Dict[str, np.ndarray],
    threshold: float = 30.0,
    window_size: int = 5,
) -> np.ndarray:
    """Compute SHAP values for martingale analysis."""
    model = CustomThresholdModel(threshold=threshold)

    feature_matrix = np.column_stack(
        [martingales["reset"][feature]["martingales"] for feature in features.keys()]
    )

    change_points = sorted(
        list(
            set(
                cp
                for m in martingales["reset"].values()
                for cp in m["change_detected_instant"]
            )
        )
    )

    return model.compute_shap_values(
        X=feature_matrix,
        change_points=change_points,
        sequence_length=len(feature_matrix),
        window_size=window_size,
    )
