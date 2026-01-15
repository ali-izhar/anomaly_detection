#!/usr/bin/env python3
"""MIT Reality Mining dataset processing and change detection."""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint import ChangePointDetector, DetectorConfig
from src.graph import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.utils import normalize_features, normalize_predictions, OutputManager, prepare_result_data
from src.predictor import PredictorFactory

logger = logging.getLogger(__name__)

# Known MIT Reality events (approximate dates)
KNOWN_EVENTS = {
    "ColumbusDay": {"date": "2008-10-12", "description": "Columbus Day / Fall break"},
    "Thanksgiving": {"date": "2008-11-26", "description": "Thanksgiving holiday"},
    "Christmas": {"date": "2008-12-25", "description": "Christmas holiday"},
    "NewYear": {"date": "2009-01-01", "description": "New Year holiday"},
    "SpringBreak": {"date": "2009-03-15", "description": "Spring break"},
    "EndOfSemester": {"date": "2009-05-15", "description": "End of spring semester"},
}


def load_proximity_data(file_path: str) -> pd.DataFrame:
    """Load and clean MIT Reality proximity data."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.str.strip().str.strip('"').str.replace("\ufeff", "", regex=False)
    df = df.rename(columns={
        "user.id": "user_id", "remote.user.id.if.known": "remote_user_id",
        "time": "timestamp", "prob2": "probability"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.strip().str.strip('"'), errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["remote_user_id"] = pd.to_numeric(df["remote_user_id"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df = df.dropna(subset=["timestamp", "probability"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} records, {df['user_id'].nunique()} users, {df['date'].min()} to {df['date'].max()}")
    return df


def create_daily_graphs(df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, nx.Graph]:
    """Create daily graph snapshots from proximity data."""
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date
    all_users = set(df["user_id"].unique()) | set(df["remote_user_id"].unique())
    daily_graphs = {}

    for date in sorted(df["date"].unique()):
        day_df = df[(df["date"] == date) & (df["probability"] >= threshold)]
        if len(day_df) <= 5:
            continue
        G = nx.Graph()
        G.add_nodes_from(all_users)
        G.add_edges_from(day_df[["user_id", "remote_user_id"]].values)
        if G.number_of_edges() > 1:
            daily_graphs[date.isoformat()] = G

    logger.info(f"Created {len(daily_graphs)} daily graphs")
    return daily_graphs


def extract_features(
    adjacency_matrices: List[np.ndarray],
    feature_names: Optional[List[str]] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Extract network features from adjacency matrices."""
    if feature_names is None:
        feature_names = [
            "mean_degree", "density", "mean_clustering", "mean_betweenness",
            "mean_eigenvector", "mean_closeness", "max_singular_value", "min_nonzero_laplacian",
        ]

    extractor = NetworkFeatureExtractor()
    features = []
    for adj in adjacency_matrices:
        graph = adjacency_to_graph(adj)
        numeric = extractor.get_numeric_features(graph)
        features.append([numeric[name] for name in feature_names])

    features_array = np.array(features)
    result = {"features_numeric": features_array, "feature_names": feature_names}

    if normalize:
        normalized, means, stds = normalize_features(features_array)
        result.update({"features_normalized": normalized, "feature_means": means, "feature_stds": stds})

    return result


def find_event_indices(dates: List[str], tolerance: int = 10) -> Dict[str, Dict]:
    """Map known events to date indices."""
    date_to_idx = {d: i for i, d in enumerate(dates)}
    events = {}

    for name, info in KNOWN_EVENTS.items():
        target = pd.to_datetime(info["date"])
        closest = min(dates, key=lambda x: abs(pd.to_datetime(x) - target))
        events[name] = {
            "idx": date_to_idx[closest],
            "date": closest,
            "description": info["description"],
        }

    return events


def run_mit_detection(
    features_normalized: np.ndarray,
    adjacency_matrices: List[np.ndarray],
    dates: List[str],
    true_cps: List[int],
    config: Dict[str, Any],
    output_dir: str,
    enable_prediction: bool = True,
) -> Dict[str, Any]:
    """Run change point detection on MIT Reality features."""
    os.makedirs(output_dir, exist_ok=True)
    n_trials = config.get("trials", {}).get("n_trials", 10)
    seeds = [42, 142, 241, 341, 443, 542, 642, 743, 842, 1043][:n_trials]

    # Initialize predictor and generate predictions (required for parallel detection)
    predicted_normalized = None
    if enable_prediction:
        try:
            pred_config = config["model"]["predictor"]
            predictor = PredictorFactory.create(pred_config["type"], pred_config["config"])
            horizon = config["detection"].get("prediction_horizon", 5)
            n_history = pred_config["config"].get("n_history", 5)

            predicted_features = []
            extractor = NetworkFeatureExtractor()
            for t in range(n_history, len(adjacency_matrices)):
                history = [{"adjacency": g} for g in adjacency_matrices[t-n_history:t]]
                preds = predictor.predict(history, horizon=horizon)
                timestep_features = []
                for pred_adj in preds:
                    graph = adjacency_to_graph(pred_adj)
                    numeric = extractor.get_numeric_features(graph)
                    timestep_features.append([numeric[n] for n in config.get("features", [])])
                predicted_features.append(timestep_features)

            if predicted_features:
                _, means, stds = normalize_features(features_normalized)
                predicted_normalized = normalize_predictions(np.array(predicted_features), means, stds)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            raise RuntimeError("Predictions required for parallel martingale detection")

    if predicted_normalized is None:
        raise RuntimeError("Predictions required for parallel martingale detection")

    # Run detection trials
    results = []
    det_config = config["detection"]
    betting_name = det_config["betting_func_config"]["name"]
    betting_params = det_config["betting_func_config"].get(betting_name, {})
    distance_metric = det_config.get("distance", {}).get("measure", "mahalanobis")

    for seed in seeds:
        detector_cfg = DetectorConfig(
            threshold=det_config["threshold"],
            history_size=config["model"]["predictor"]["config"].get("n_history", 5),
            window_size=det_config.get("max_window"),
            reset=det_config.get("reset", True),
            cooldown=det_config.get("cooldown", 30),
            betting_name=betting_name,
            betting_params=betting_params,
            random_state=seed,
            distance_metric=distance_metric,
        )
        detector = ChangePointDetector(detector_cfg)
        result = detector.run(data=features_normalized, predicted_data=predicted_normalized)
        if result:
            results.append(result)

    if not results:
        raise RuntimeError("All detection trials failed")

    # Export results
    if config.get("execution", {}).get("save_csv", True):
        try:
            csv_dir = os.path.join(output_dir, "csv")
            manager = OutputManager(csv_dir, config)
            manager.export_to_csv(results[0], true_cps, individual_trials=results)
        except Exception as e:
            logger.error(f"CSV export failed: {e}")

    return {
        "individual_trials": results,
        "aggregated": results[0],
        "dates": dates,
        "change_points": {
            "traditional": results[0].get("traditional_change_points", []),
            "horizon": results[0].get("horizon_change_points", []),
        },
    }


def evaluate_detections(detected: List[int], true_cps: List[int], tolerance: int = 10) -> Dict[str, Any]:
    """Evaluate detection performance."""
    if not true_cps:
        return {"precision": 0, "recall": 0, "f1": 0}

    matches = []
    for d in detected:
        closest = min(true_cps, key=lambda x: abs(x - d))
        if abs(closest - d) <= tolerance:
            matches.append((d, closest, abs(d - closest)))

    tp = len(matches)
    fp = len(detected) - tp
    fn = len(true_cps) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {"precision": precision, "recall": recall, "f1": f1, "matches": matches}


def process_dataset(
    file_path: str,
    threshold: float = 0.3,
    output_dir: str = "results/mit_reality",
    do_detection: bool = True,
    enable_prediction: bool = True,
    detection_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Process MIT Reality dataset end-to-end."""
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    df = load_proximity_data(file_path)
    daily_graphs = create_daily_graphs(df, threshold=threshold)
    dates = sorted(daily_graphs.keys())
    adj_matrices = [nx.to_numpy_array(daily_graphs[d]) for d in dates]

    # Extract features
    features = extract_features(adj_matrices, normalize=True)

    # Find event indices
    events = find_event_indices(dates)
    potential_cps = sorted(set(e["idx"] for e in events.values()))
    logger.info(f"Potential change points: {potential_cps}")

    results = {
        "adjacency_matrices": adj_matrices,
        "dates": dates,
        "features": features,
        "events": events,
        "potential_change_points": potential_cps,
    }

    # Run detection
    if do_detection:
        n_timesteps = len(adj_matrices)
        config = {
            "trials": {"n_trials": 10},
            "detection": {
                "threshold": 20.0,  # Lower threshold for MIT per paper
                "reset": True,
                "max_window": None,
                "prediction_horizon": 5,
                "cooldown": 30,
                "betting_func_config": {"name": "mixture", "mixture": {"epsilons": [0.7, 0.8, 0.9]}},
                "distance": {"measure": "mahalanobis"},
            },
            "model": {
                "type": "multiview",
                "predictor": {"type": "feature", "config": {"n_history": 5, "alpha": 0.3, "beta": 0.1}},
            },
            "features": features["feature_names"],
            "params": {"seq_len": n_timesteps},
            "execution": {"enable_prediction": enable_prediction, "save_csv": True},
        }
        if detection_config:
            for k, v in detection_config.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v

        detection_results = run_mit_detection(
            features["features_normalized"], adj_matrices, dates, potential_cps,
            config, os.path.join(output_dir, "detection"), enable_prediction,
        )
        results["detection"] = detection_results

        # Evaluate
        if "traditional" in detection_results["change_points"]:
            trad_cps = detection_results["change_points"]["traditional"]
            results["evaluation"] = evaluate_detections(trad_cps, potential_cps)
            logger.info(f"Evaluation: {results['evaluation']}")

    logger.info(f"Processed {len(dates)} days")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    results = process_dataset(
        "data/mit_reality/Proximity.csv",
        threshold=0.3,
        do_detection=True,
        enable_prediction=True,  # Required for parallel martingale
    )

    if "detection" in results:
        cps = results["detection"]["change_points"]
        logger.info(f"Traditional CPs: {cps.get('traditional', [])}")
        logger.info(f"Horizon CPs: {cps.get('horizon', [])}")
        if "evaluation" in results:
            logger.info(f"F1: {results['evaluation']['f1']:.3f}")


if __name__ == "__main__":
    main()
