#!/usr/bin/env python3
"""Test FeaturePredictor integration with martingale framework.

Compares detection timing between:
1. Perfect predictions (baseline)
2. FeaturePredictor (Holt's smoothing with trend)
3. Baseline (exp avg) (exponential averaging)

Goal: Show that better predictions enable faster horizon martingale detection.
"""

import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.changepoint import ChangePointDetector, DetectorConfig
from src.predictor.feature_predictor import FeaturePredictor
from src.predictor import PredictorFactory
from src.graph import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.utils import normalize_features, normalize_predictions


def generate_test_data(n_timesteps=100, n_features=4, change_points=[35, 70]):
    """Generate synthetic feature data with known change points."""
    np.random.seed(42)

    features = np.zeros((n_timesteps, n_features))

    # Segment 1: baseline
    features[:change_points[0]] = np.random.randn(change_points[0], n_features) * 0.3

    # Segment 2: shifted mean
    features[change_points[0]:change_points[1]] = (
        np.random.randn(change_points[1] - change_points[0], n_features) * 0.3 + 1.5
    )

    # Segment 3: different variance
    features[change_points[1]:] = (
        np.random.randn(n_timesteps - change_points[1], n_features) * 0.8 - 0.5
    )

    return features, change_points


def generate_perfect_predictions(features, n_history=10, horizon=5, noise_std=0.1):
    """Generate near-perfect predictions (actual future values + small noise)."""
    n_samples = len(features)
    n_features = features.shape[1]

    predictions = []
    for t in range(n_history, n_samples):
        timestep_preds = []
        for h in range(horizon):
            future_idx = t + h + 1
            if future_idx < n_samples:
                # Perfect prediction with small noise
                pred = features[future_idx] + np.random.randn(n_features) * noise_std
            else:
                # Extrapolate for out-of-bounds
                pred = features[-1] + np.random.randn(n_features) * noise_std
            timestep_preds.append(pred)
        predictions.append(timestep_preds)

    return np.array(predictions)


def generate_feature_predictor_predictions(features, n_history=10, horizon=5):
    """Generate predictions using FeaturePredictor (Holt's smoothing)."""
    n_samples = len(features)
    predictor = FeaturePredictor(alpha=0.3, beta=0.1, n_history=n_history, adaptive=True)

    predictions = []
    for t in range(n_history, n_samples):
        history = features[t-n_history:t]
        predictor.fit(history)
        pred = predictor.predict(horizon)
        predictions.append(pred)
        # Update with actual observation for next iteration
        predictor.update(features[t])

    return np.array(predictions)


def generate_baseline_predictions(features, n_history=10, horizon=5):
    """Generate predictions using Baseline (exp avg) (exponential averaging)."""
    n_samples = len(features)

    # Simulate Baseline (exp avg) behavior (exponential weighted average)
    predictions = []
    for t in range(n_history, n_samples):
        history = features[t-n_history:t]

        # Exponential weighted average (same as Baseline (exp avg))
        weights = np.exp(-0.5 * np.arange(len(history))[::-1])
        weights /= weights.sum()
        weighted_mean = np.average(history, axis=0, weights=weights)

        # All horizons get same prediction (no trend extrapolation)
        timestep_preds = [weighted_mean.copy() for _ in range(horizon)]
        predictions.append(timestep_preds)

    return np.array(predictions)


def run_detection(features, predictions, threshold=30.0, n_history=10):
    """Run martingale detection and return results."""
    config = DetectorConfig(
        threshold=threshold,
        history_size=n_history,
        reset=True,
        cooldown=15,
        betting_name="mixture",
        betting_params={"epsilons": [0.7, 0.8, 0.9]},
        random_state=42,
        distance_metric="euclidean",
        horizon_decay=0.7,
        normalize_horizons=False,
    )

    detector = ChangePointDetector(config)
    result = detector.run(features, predictions)
    return result


def evaluate_detection(detected_cps, true_cps, tolerance=10):
    """Evaluate detection performance."""
    if not detected_cps:
        return {"detected": [], "delays": [], "avg_delay": float('inf'), "missed": true_cps}

    delays = []
    matched = []

    for true_cp in true_cps:
        # Find closest detection after the change point
        candidates = [d for d in detected_cps if d >= true_cp - tolerance]
        if candidates:
            closest = min(candidates, key=lambda x: abs(x - true_cp))
            if abs(closest - true_cp) <= tolerance + 20:  # Allow some delay
                delay = closest - true_cp
                delays.append(delay)
                matched.append((true_cp, closest, delay))

    missed = [cp for cp in true_cps if cp not in [m[0] for m in matched]]

    return {
        "detected": detected_cps,
        "matches": matched,
        "delays": delays,
        "avg_delay": np.mean(delays) if delays else float('inf'),
        "missed": missed,
    }


def main():
    print("=" * 70)
    print("PREDICTOR INTEGRATION TEST: Martingale Detection Comparison")
    print("=" * 70)

    # Generate test data
    n_timesteps = 100
    n_history = 10
    horizon = 5
    threshold = 15.0  # Lower threshold to trigger traditional martingale
    true_cps = [35, 70]

    print(f"\nTest setup:")
    print(f"  - {n_timesteps} timesteps, {n_history} history, {horizon} horizon")
    print(f"  - True change points: {true_cps}")
    print(f"  - Detection threshold: {threshold}")

    features, _ = generate_test_data(n_timesteps, n_features=4, change_points=true_cps)

    # Normalize features
    features_norm, means, stds = normalize_features(features)

    print("\n" + "-" * 70)
    print("1. PERFECT PREDICTIONS (baseline)")
    print("-" * 70)

    perfect_preds = generate_perfect_predictions(features_norm, n_history, horizon, noise_std=0.05)
    result_perfect = run_detection(features_norm, perfect_preds, threshold, n_history)

    trad_eval = evaluate_detection(result_perfect["traditional_change_points"], true_cps)
    hor_eval = evaluate_detection(result_perfect["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_perfect['traditional_change_points']}")
    print(f"  Avg delay: {trad_eval['avg_delay']:.1f} timesteps")
    print(f"Horizon:     {result_perfect['horizon_change_points']}")
    print(f"  Avg delay: {hor_eval['avg_delay']:.1f} timesteps")

    if trad_eval['avg_delay'] != float('inf') and hor_eval['avg_delay'] != float('inf'):
        improvement = trad_eval['avg_delay'] - hor_eval['avg_delay']
        print(f"\n  => Horizon detects {improvement:.1f} timesteps EARLIER with perfect predictions")

    print("\n" + "-" * 70)
    print("2. FEATURE PREDICTOR (Holt's double exponential smoothing)")
    print("-" * 70)

    feature_preds = generate_feature_predictor_predictions(features_norm, n_history, horizon)
    result_feature = run_detection(features_norm, feature_preds, threshold, n_history)

    trad_eval_f = evaluate_detection(result_feature["traditional_change_points"], true_cps)
    hor_eval_f = evaluate_detection(result_feature["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_feature['traditional_change_points']}")
    print(f"  Avg delay: {trad_eval_f['avg_delay']:.1f} timesteps")
    print(f"Horizon:     {result_feature['horizon_change_points']}")
    print(f"  Avg delay: {hor_eval_f['avg_delay']:.1f} timesteps")

    if trad_eval_f['avg_delay'] != float('inf') and hor_eval_f['avg_delay'] != float('inf'):
        improvement = trad_eval_f['avg_delay'] - hor_eval_f['avg_delay']
        print(f"\n  => Horizon detects {improvement:.1f} timesteps EARLIER with FeaturePredictor")

    print("\n" + "-" * 70)
    print("3. BASELINE PREDICTOR (exponential weighted averaging)")
    print("-" * 70)

    hybrid_preds = generate_baseline_predictions(features_norm, n_history, horizon)
    result_hybrid = run_detection(features_norm, hybrid_preds, threshold, n_history)

    trad_eval_h = evaluate_detection(result_hybrid["traditional_change_points"], true_cps)
    hor_eval_h = evaluate_detection(result_hybrid["horizon_change_points"], true_cps)

    print(f"\nTraditional: {result_hybrid['traditional_change_points']}")
    print(f"  Avg delay: {trad_eval_h['avg_delay']:.1f} timesteps")
    print(f"Horizon:     {result_hybrid['horizon_change_points']}")
    print(f"  Avg delay: {hor_eval_h['avg_delay']:.1f} timesteps")

    if trad_eval_h['avg_delay'] != float('inf') and hor_eval_h['avg_delay'] != float('inf'):
        improvement = trad_eval_h['avg_delay'] - hor_eval_h['avg_delay']
        print(f"\n  => Horizon detects {improvement:.1f} timesteps EARLIER with Baseline (exp avg)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Detection Delay Comparison (lower is better)")
    print("=" * 70)
    print(f"\n{'Predictor':<25} {'Traditional':<15} {'Horizon':<15} {'Improvement':<15}")
    print("-" * 70)

    predictors = [
        ("Perfect (baseline)", trad_eval, hor_eval),
        ("FeaturePredictor", trad_eval_f, hor_eval_f),
        ("Baseline (exp avg)", trad_eval_h, hor_eval_h),
    ]

    for name, trad, hor in predictors:
        trad_delay = f"{trad['avg_delay']:.1f}" if trad['avg_delay'] != float('inf') else "N/A"
        hor_delay = f"{hor['avg_delay']:.1f}" if hor['avg_delay'] != float('inf') else "N/A"

        if trad['avg_delay'] != float('inf') and hor['avg_delay'] != float('inf'):
            imp = trad['avg_delay'] - hor['avg_delay']
            imp_str = f"{imp:+.1f} steps"
        else:
            imp_str = "N/A"

        print(f"{name:<25} {trad_delay:<15} {hor_delay:<15} {imp_str:<15}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compare FeaturePredictor vs Baseline (exp avg)
    if (hor_eval_f['avg_delay'] != float('inf') and
        hor_eval_h['avg_delay'] != float('inf')):

        fp_improvement = hor_eval_h['avg_delay'] - hor_eval_f['avg_delay']
        if fp_improvement > 0:
            print(f"\nFeaturePredictor enables {fp_improvement:.1f} timesteps FASTER detection")
            print("than Baseline (exp avg) with the horizon martingale.")
        else:
            print(f"\nBoth predictors achieve similar detection timing.")

    # Compare to perfect baseline
    if (hor_eval_f['avg_delay'] != float('inf') and
        hor_eval['avg_delay'] != float('inf')):

        gap = hor_eval_f['avg_delay'] - hor_eval['avg_delay']
        print(f"\nFeaturePredictor is {gap:.1f} timesteps slower than perfect predictions.")
        print("(Gap represents room for further predictor improvement)")


if __name__ == "__main__":
    main()
