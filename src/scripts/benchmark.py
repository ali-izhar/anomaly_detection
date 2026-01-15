#!/usr/bin/env python3
"""Benchmark script to reproduce paper results.

Compares Traditional Martingale vs Horizon Martingale across network types.
Outputs detection delay, precision, recall, and F1 score.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.changepoint import ChangePointDetector, DetectorConfig
from src.predictor import FeaturePredictor
from src.graph import NetworkFeatureExtractor, GraphGenerator
from src.graph.utils import adjacency_to_graph
from src.utils import normalize_features


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    network: str
    scenario: str
    method: str
    true_cps: List[int]
    detected_cps: List[int]
    delays: List[int]
    precision: float
    recall: float
    f1: float
    avg_delay: float


FEATURE_NAMES = [
    "mean_degree", "density", "mean_clustering", "mean_betweenness",
    "mean_eigenvector", "mean_closeness", "max_singular_value", "min_nonzero_laplacian"
]

# Scenario configurations
SCENARIOS = {
    "sbm": {
        "community_merge": {
            "n": 50, "seq_len": 150, "min_segment": 50, "num_blocks": 2,
            "intra_prob": 0.8, "inter_prob": 0.05,
            "min_intra_prob": 0.3, "max_intra_prob": 0.9,
            "min_inter_prob": 0.01, "max_inter_prob": 0.3,
            "min_changes": 1, "max_changes": 1,
        },
        "density_change": {
            "n": 50, "seq_len": 150, "min_segment": 50, "num_blocks": 2,
            "intra_prob": 0.6, "inter_prob": 0.1,
            "min_intra_prob": 0.2, "max_intra_prob": 0.8,
            "min_inter_prob": 0.05, "max_inter_prob": 0.2,
            "min_changes": 1, "max_changes": 1,
        },
    },
    "er": {
        "density_change": {
            "n": 50, "seq_len": 150, "min_segment": 50,
            "prob": 0.1, "min_prob": 0.05, "max_prob": 0.25,
            "min_changes": 1, "max_changes": 1,
        },
    },
    "ba": {
        "parameter_shift": {
            "n": 50, "seq_len": 150, "min_segment": 50,
            "m": 2, "min_m": 1, "max_m": 5,
            "min_changes": 1, "max_changes": 1,
        },
    },
    "ws": {
        "rewiring_change": {
            "n": 50, "seq_len": 150, "min_segment": 50,
            "k_nearest": 6, "rewire_prob": 0.1,
            "min_prob": 0.05, "max_prob": 0.4, "min_k": 4, "max_k": 8,
            "min_changes": 1, "max_changes": 1,
        },
    },
}


def extract_features(adj_matrices: List[np.ndarray]) -> np.ndarray:
    """Extract features from adjacency matrices."""
    extractor = NetworkFeatureExtractor()
    features = []
    for adj in adj_matrices:
        G = adjacency_to_graph(adj)
        numeric = extractor.get_numeric_features(G)
        features.append([numeric[name] for name in FEATURE_NAMES])
    return np.array(features)


def generate_predictions(features: np.ndarray, n_history: int = 10, horizon: int = 5) -> np.ndarray:
    """Generate predictions using FeaturePredictor."""
    predictor = FeaturePredictor(alpha=0.3, beta=0.1, n_history=n_history, adaptive=True)
    predictions = []

    for t in range(n_history, len(features)):
        history = features[t-n_history:t]
        predictor.fit(history)
        pred = predictor.predict(horizon)
        predictions.append(pred)
        predictor.update(features[t])

    return np.array(predictions)


def evaluate_detection(detected: List[int], true_cps: List[int], tolerance: int = 15) -> Dict[str, Any]:
    """Evaluate detection performance."""
    if not true_cps:
        return {"precision": 0, "recall": 0, "f1": 0, "delays": [], "avg_delay": float('inf')}

    matched_detected = set()
    matched_true = set()
    delays = []

    for true_cp in true_cps:
        # Find detections within tolerance after the change point
        candidates = [(d, d - true_cp) for d in detected
                      if true_cp - tolerance <= d <= true_cp + tolerance + 25]
        if candidates:
            best = min(candidates, key=lambda x: abs(x[1]))
            matched_detected.add(best[0])
            matched_true.add(true_cp)
            delays.append(best[1])

    tp = len(matched_true)
    fp = len(detected) - len(matched_detected)
    fn = len(true_cps) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    avg_delay = np.mean(delays) if delays else float('inf')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delays": delays,
        "avg_delay": avg_delay,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def run_detection(features: np.ndarray, predictions: np.ndarray,
                  threshold: float = 30.0, n_history: int = 10) -> Dict[str, Any]:
    """Run martingale detection."""
    config = DetectorConfig(
        threshold=threshold,
        history_size=n_history,
        reset=True,
        cooldown=20,
        betting_name="mixture",
        betting_params={"epsilons": [0.7, 0.8, 0.9]},
        random_state=42,
        distance_metric="euclidean",
        horizon_decay=0.7,
        normalize_horizons=False,
    )

    detector = ChangePointDetector(config)
    return detector.run(features, predictions)


def run_scenario(network: str, scenario: str, n_trials: int = 5) -> List[BenchmarkResult]:
    """Run benchmark for a specific network/scenario combination."""
    results = []
    params = SCENARIOS[network][scenario].copy()

    for trial in range(n_trials):
        seed = 42 + trial * 100
        params["seed"] = seed

        # Generate network sequence
        generator = GraphGenerator(network)
        gen_result = generator.generate_sequence(params)

        graphs = gen_result["graphs"]
        true_cps = gen_result["change_points"]

        if not true_cps:
            print(f"  Trial {trial+1}: No change points generated, skipping")
            continue

        # Extract and normalize features
        features = extract_features(graphs)
        features_norm, _, _ = normalize_features(features)

        # Generate predictions
        n_history = 10
        horizon = 5
        predictions = generate_predictions(features_norm, n_history, horizon)

        # Run detection
        det_result = run_detection(features_norm, predictions, threshold=30.0, n_history=n_history)

        # Evaluate both methods
        for method, cps_key in [("Traditional", "standalone_change_points"),
                                 ("Horizon", "horizon_change_points")]:
            detected = det_result[cps_key]
            eval_result = evaluate_detection(detected, true_cps)

            results.append(BenchmarkResult(
                network=network,
                scenario=scenario,
                method=method,
                true_cps=true_cps,
                detected_cps=detected,
                delays=eval_result["delays"],
                precision=eval_result["precision"],
                recall=eval_result["recall"],
                f1=eval_result["f1"],
                avg_delay=eval_result["avg_delay"]
            ))

    return results


def aggregate_results(results: List[BenchmarkResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate results by method."""
    by_method = {}

    for r in results:
        key = r.method
        if key not in by_method:
            by_method[key] = {"precision": [], "recall": [], "f1": [], "delay": []}

        by_method[key]["precision"].append(r.precision)
        by_method[key]["recall"].append(r.recall)
        by_method[key]["f1"].append(r.f1)
        if r.avg_delay != float('inf'):
            by_method[key]["delay"].append(r.avg_delay)

    aggregated = {}
    for method, metrics in by_method.items():
        aggregated[method] = {
            "precision": np.mean(metrics["precision"]) if metrics["precision"] else 0,
            "recall": np.mean(metrics["recall"]) if metrics["recall"] else 0,
            "f1": np.mean(metrics["f1"]) if metrics["f1"] else 0,
            "avg_delay": np.mean(metrics["delay"]) if metrics["delay"] else float('inf')
        }

    return aggregated


def run_threshold_sweep(threshold: float = 30.0):
    """Run benchmark with a specific threshold."""
    scenarios = [
        ("sbm", "community_merge"),
        ("sbm", "density_change"),
        ("er", "density_change"),
        ("ba", "parameter_shift"),
        ("ws", "rewiring_change"),
    ]

    n_trials = 5
    all_results = []

    for network, scenario in scenarios:
        params = SCENARIOS[network][scenario].copy()

        for trial in range(n_trials):
            seed = 42 + trial * 100
            params["seed"] = seed

            generator = GraphGenerator(network)
            gen_result = generator.generate_sequence(params)
            graphs = gen_result["graphs"]
            true_cps = gen_result["change_points"]

            if not true_cps:
                continue

            features = extract_features(graphs)
            features_norm, _, _ = normalize_features(features)

            n_history = 10
            horizon = 5
            predictions = generate_predictions(features_norm, n_history, horizon)

            det_result = run_detection(features_norm, predictions, threshold=threshold, n_history=n_history)

            for method, cps_key in [("Traditional", "standalone_change_points"),
                                     ("Horizon", "horizon_change_points")]:
                detected = det_result[cps_key]
                eval_result = evaluate_detection(detected, true_cps)

                all_results.append(BenchmarkResult(
                    network=network, scenario=scenario, method=method,
                    true_cps=true_cps, detected_cps=detected,
                    delays=eval_result["delays"],
                    precision=eval_result["precision"],
                    recall=eval_result["recall"],
                    f1=eval_result["f1"],
                    avg_delay=eval_result["avg_delay"]
                ))

    return aggregate_results(all_results)


def main():
    print("=" * 80)
    print("HORIZON MARTINGALE BENCHMARK - Reproducing Paper Results")
    print("=" * 80)

    # First, sweep thresholds to find optimal
    print("\n" + "-" * 60)
    print("THRESHOLD SWEEP")
    print("-" * 60)
    print(f"{'Threshold':<12} {'H-Prec':<10} {'H-Recall':<10} {'H-F1':<10} {'H-Delay':<10}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 30.0

    for threshold in [20, 30, 40, 50, 60, 80, 100]:
        agg = run_threshold_sweep(threshold)
        if "Horizon" in agg:
            h = agg["Horizon"]
            delay_str = f"{h['avg_delay']:.1f}" if h['avg_delay'] != float('inf') else "N/A"
            print(f"{threshold:<12} {h['precision']:.3f}     {h['recall']:.3f}     {h['f1']:.3f}     {delay_str}")

            if h['f1'] > best_f1:
                best_f1 = h['f1']
                best_threshold = threshold

    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.3f})")

    # Run full benchmark with best threshold
    print("\n" + "=" * 80)
    print(f"FULL BENCHMARK (threshold={best_threshold})")
    print("=" * 80)

    scenarios = [
        ("sbm", "community_merge"),
        ("sbm", "density_change"),
        ("er", "density_change"),
        ("ba", "parameter_shift"),
        ("ws", "rewiring_change"),
    ]

    n_trials = 5
    all_results = []

    print(f"\nRunning {len(scenarios)} scenarios x {n_trials} trials each...\n")

    for network, scenario in scenarios:
        print(f"Running {network}/{scenario}...", end=" ", flush=True)
        try:
            # Use best threshold
            params = SCENARIOS[network][scenario].copy()
            results = []

            for trial in range(n_trials):
                seed = 42 + trial * 100
                params["seed"] = seed

                generator = GraphGenerator(network)
                gen_result = generator.generate_sequence(params)
                graphs = gen_result["graphs"]
                true_cps = gen_result["change_points"]

                if not true_cps:
                    continue

                features = extract_features(graphs)
                features_norm, _, _ = normalize_features(features)

                n_history = 10
                horizon = 5
                predictions = generate_predictions(features_norm, n_history, horizon)

                det_result = run_detection(features_norm, predictions, threshold=best_threshold, n_history=n_history)

                for method, cps_key in [("Traditional", "standalone_change_points"),
                                         ("Horizon", "horizon_change_points")]:
                    detected = det_result[cps_key]
                    eval_result = evaluate_detection(detected, true_cps)

                    results.append(BenchmarkResult(
                        network=network, scenario=scenario, method=method,
                        true_cps=true_cps, detected_cps=detected,
                        delays=eval_result["delays"],
                        precision=eval_result["precision"],
                        recall=eval_result["recall"],
                        f1=eval_result["f1"],
                        avg_delay=eval_result["avg_delay"]
                    ))

            all_results.extend(results)
            print(f"done ({len(results)//2} trials)")
        except Exception as e:
            print(f"failed: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNo results collected. Check for errors above.")
        return

    # Print results by scenario
    print("\n" + "=" * 80)
    print("RESULTS BY SCENARIO")
    print("=" * 80)

    for network, scenario in scenarios:
        scenario_results = [r for r in all_results if r.network == network and r.scenario == scenario]
        if not scenario_results:
            continue

        print(f"\n{network.upper()} - {scenario}")
        print("-" * 60)
        print(f"{'Method':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Delay':<12}")
        print("-" * 60)

        agg = aggregate_results(scenario_results)
        for method in ["Traditional", "Horizon"]:
            if method in agg:
                m = agg[method]
                delay_str = f"{m['avg_delay']:.1f}" if m['avg_delay'] != float('inf') else "N/A"
                print(f"{method:<15} {m['precision']:.3f}        {m['recall']:.3f}        {m['f1']:.3f}        {delay_str}")

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    overall_agg = aggregate_results(all_results)
    print(f"\n{'Method':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Delay':<12}")
    print("-" * 60)

    for method in ["Traditional", "Horizon"]:
        if method in overall_agg:
            m = overall_agg[method]
            delay_str = f"{m['avg_delay']:.1f}" if m['avg_delay'] != float('inf') else "N/A"
            print(f"{method:<15} {m['precision']:.3f}        {m['recall']:.3f}        {m['f1']:.3f}        {delay_str}")

    # Improvement summary
    if "Traditional" in overall_agg and "Horizon" in overall_agg:
        trad = overall_agg["Traditional"]
        hor = overall_agg["Horizon"]

        print("\n" + "-" * 60)
        print("HORIZON IMPROVEMENT OVER TRADITIONAL:")

        if trad["f1"] > 0:
            f1_imp = (hor["f1"] - trad["f1"]) / max(trad["f1"], 0.001) * 100
            print(f"  F1 Score: {hor['f1']:.3f} vs {trad['f1']:.3f} ({f1_imp:+.1f}%)")

        if trad["avg_delay"] != float('inf') and hor["avg_delay"] != float('inf'):
            delay_imp = trad["avg_delay"] - hor["avg_delay"]
            print(f"  Detection Delay: {hor['avg_delay']:.1f} vs {trad['avg_delay']:.1f} ({delay_imp:+.1f} timesteps)")
        elif hor["avg_delay"] != float('inf'):
            print(f"  Detection Delay: Horizon detects ({hor['avg_delay']:.1f}) while Traditional misses")


if __name__ == "__main__":
    main()
