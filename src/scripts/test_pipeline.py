#!/usr/bin/env python3
"""Full pipeline test with synthetic network data.

Generates temporal network graphs with known change points and tests:
1. Perfect predictions (baseline)
2. FeaturePredictor (Holt's smoothing)
3. Baseline (exp avg) (exponential averaging)
"""

import numpy as np
import networkx as nx
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.changepoint import ChangePointDetector, DetectorConfig
from src.predictor.feature_predictor import FeaturePredictor
from src.graph import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.utils import normalize_features


def generate_er_graph(n_nodes, p):
    """Generate Erdos-Renyi random graph."""
    return nx.erdos_renyi_graph(n_nodes, p)


def generate_network_sequence(n_timesteps=100, n_nodes=50, change_points=[35, 70]):
    """Generate sequence of network graphs with change points.

    Regime changes affect edge probability (density).
    """
    np.random.seed(42)
    graphs = []
    params = []

    # Initial density
    base_p = 0.1

    for t in range(n_timesteps):
        if t < change_points[0]:
            # Regime 1: baseline
            p = base_p + np.random.normal(0, 0.01)
        elif t < change_points[1]:
            # Regime 2: higher density
            p = base_p + 0.08 + np.random.normal(0, 0.01)
        else:
            # Regime 3: lower density
            p = base_p - 0.03 + np.random.normal(0, 0.01)

        p = np.clip(p, 0.02, 0.5)
        G = generate_er_graph(n_nodes, p)
        graphs.append(G)
        params.append(p)

    return graphs, params


def extract_features(graphs, feature_names):
    """Extract features from graph sequence."""
    extractor = NetworkFeatureExtractor()
    features = []

    for G in graphs:
        numeric = extractor.get_numeric_features(G)
        features.append([numeric[name] for name in feature_names])

    return np.array(features)


def generate_perfect_predictions(features, n_history, horizon, noise_std=0.05):
    """Generate near-perfect predictions."""
    n_samples = len(features)
    n_features = features.shape[1]

    predictions = []
    for t in range(n_history, n_samples):
        timestep_preds = []
        for h in range(horizon):
            future_idx = t + h + 1
            if future_idx < n_samples:
                pred = features[future_idx] + np.random.randn(n_features) * noise_std
            else:
                pred = features[-1] + np.random.randn(n_features) * noise_std
            timestep_preds.append(pred)
        predictions.append(timestep_preds)

    return np.array(predictions)


def generate_feature_predictions(features, n_history, horizon):
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


def generate_baseline_predictions(features, n_history, horizon):
    """Generate predictions using Baseline (exp avg) (exponential averaging)."""
    predictions = []
    for t in range(n_history, len(features)):
        history = features[t-n_history:t]

        # Exponential weighted average
        weights = np.exp(-0.5 * np.arange(len(history))[::-1])
        weights /= weights.sum()
        weighted_mean = np.average(history, axis=0, weights=weights)

        timestep_preds = [weighted_mean.copy() for _ in range(horizon)]
        predictions.append(timestep_preds)

    return np.array(predictions)


def run_detection(features, predictions, threshold, n_history):
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


def compute_detection_delay(detected, true_cps, tolerance=15):
    """Compute average detection delay for matched change points."""
    if not detected or not true_cps:
        return float('inf'), []

    delays = []
    for true_cp in true_cps:
        # Find first detection at or after the change point
        candidates = [d for d in detected if d >= true_cp]
        if candidates:
            closest = min(candidates)
            delay = closest - true_cp
            if delay <= tolerance + 10:  # Allow reasonable delay
                delays.append(delay)

    return np.mean(delays) if delays else float('inf'), delays


def main():
    print("=" * 70)
    print("FULL PIPELINE TEST: Synthetic Network Data")
    print("=" * 70)

    # Parameters
    n_timesteps = 120
    n_nodes = 50
    n_history = 10
    horizon = 5
    true_cps = [40, 80]
    threshold = 30.0

    feature_names = [
        "mean_degree", "density", "mean_clustering", "mean_betweenness",
        "mean_eigenvector", "mean_closeness", "max_singular_value", "min_nonzero_laplacian"
    ]

    print(f"\nConfiguration:")
    print(f"  - {n_timesteps} timesteps, {n_nodes} nodes per graph")
    print(f"  - True change points: {true_cps}")
    print(f"  - {len(feature_names)} features, {n_history} history, {horizon} horizon")
    print(f"  - Detection threshold: {threshold}")

    # Generate network sequence
    print("\nGenerating synthetic network sequence...")
    graphs, params = generate_network_sequence(n_timesteps, n_nodes, true_cps)

    # Extract features
    print("Extracting network features...")
    features = extract_features(graphs, feature_names)
    features_norm, means, stds = normalize_features(features)

    # Test each predictor
    results = {}

    print("\n" + "-" * 70)
    print("1. PERFECT PREDICTIONS (oracle baseline)")
    print("-" * 70)

    perfect_preds = generate_perfect_predictions(features_norm, n_history, horizon, noise_std=0.05)
    result_perfect = run_detection(features_norm, perfect_preds, threshold, n_history)

    trad_delay, _ = compute_detection_delay(result_perfect["traditional_change_points"], true_cps)
    hor_delay, _ = compute_detection_delay(result_perfect["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_perfect['traditional_change_points']}")
    print(f"  Avg delay: {trad_delay:.1f} timesteps")
    print(f"Horizon:     {result_perfect['horizon_change_points']}")
    print(f"  Avg delay: {hor_delay:.1f} timesteps")

    results["perfect"] = {"traditional": trad_delay, "horizon": hor_delay,
                          "trad_cps": result_perfect["traditional_change_points"],
                          "hor_cps": result_perfect["horizon_change_points"]}

    print("\n" + "-" * 70)
    print("2. FEATURE PREDICTOR (Holt's exponential smoothing with trend)")
    print("-" * 70)

    feature_preds = generate_feature_predictions(features_norm, n_history, horizon)
    result_feature = run_detection(features_norm, feature_preds, threshold, n_history)

    trad_delay_f, _ = compute_detection_delay(result_feature["traditional_change_points"], true_cps)
    hor_delay_f, _ = compute_detection_delay(result_feature["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_feature['traditional_change_points']}")
    print(f"  Avg delay: {trad_delay_f:.1f} timesteps")
    print(f"Horizon:     {result_feature['horizon_change_points']}")
    print(f"  Avg delay: {hor_delay_f:.1f} timesteps")

    results["feature"] = {"traditional": trad_delay_f, "horizon": hor_delay_f,
                          "trad_cps": result_feature["traditional_change_points"],
                          "hor_cps": result_feature["horizon_change_points"]}

    print("\n" + "-" * 70)
    print("3. BASELINE PREDICTOR (exponential weighted averaging)")
    print("-" * 70)

    hybrid_preds = generate_baseline_predictions(features_norm, n_history, horizon)
    result_hybrid = run_detection(features_norm, hybrid_preds, threshold, n_history)

    trad_delay_h, _ = compute_detection_delay(result_hybrid["traditional_change_points"], true_cps)
    hor_delay_h, _ = compute_detection_delay(result_hybrid["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_hybrid['traditional_change_points']}")
    print(f"  Avg delay: {trad_delay_h:.1f} timesteps")
    print(f"Horizon:     {result_hybrid['horizon_change_points']}")
    print(f"  Avg delay: {hor_delay_h:.1f} timesteps")

    results["hybrid"] = {"traditional": trad_delay_h, "horizon": hor_delay_h,
                         "trad_cps": result_hybrid["traditional_change_points"],
                         "hor_cps": result_hybrid["horizon_change_points"]}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Detection Delay Comparison")
    print("=" * 70)
    print(f"\n{'Predictor':<30} {'Traditional':<12} {'Horizon':<12} {'Horizon Gain':<15}")
    print("-" * 70)

    for name, key in [("Perfect (oracle)", "perfect"),
                      ("FeaturePredictor", "feature"),
                      ("Baseline (exp avg)", "hybrid")]:
        r = results[key]
        t_str = f"{r['traditional']:.1f}" if r['traditional'] != float('inf') else "N/A"
        h_str = f"{r['horizon']:.1f}" if r['horizon'] != float('inf') else "N/A"

        if r['traditional'] != float('inf') and r['horizon'] != float('inf'):
            gain = r['traditional'] - r['horizon']
            gain_str = f"{gain:+.1f} steps"
        elif r['horizon'] != float('inf'):
            gain_str = "Horizon only"
        else:
            gain_str = "N/A"

        print(f"{name:<30} {t_str:<12} {h_str:<12} {gain_str:<15}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare FeaturePredictor vs Baseline (exp avg) horizon detection
    if results["feature"]["horizon"] != float('inf') and results["hybrid"]["horizon"] != float('inf'):
        improvement = results["hybrid"]["horizon"] - results["feature"]["horizon"]
        if improvement > 0:
            print(f"\n1. FeaturePredictor enables {improvement:.1f} timesteps FASTER horizon detection")
            print("   than Baseline (exp avg).")
        else:
            print(f"\n1. Both predictors achieve similar horizon detection timing.")

    # Compare to perfect baseline
    if results["feature"]["horizon"] != float('inf') and results["perfect"]["horizon"] != float('inf'):
        gap = results["feature"]["horizon"] - results["perfect"]["horizon"]
        print(f"\n2. FeaturePredictor horizon is {gap:.1f} steps from perfect baseline.")

    # Horizon vs Traditional gain
    if results["perfect"]["traditional"] != float('inf') and results["perfect"]["horizon"] != float('inf'):
        gain = results["perfect"]["traditional"] - results["perfect"]["horizon"]
        print(f"\n3. With perfect predictions, horizon detects {gain:.1f} steps earlier than traditional.")


if __name__ == "__main__":
    main()
