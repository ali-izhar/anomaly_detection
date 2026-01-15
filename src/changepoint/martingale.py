# src/changepoint/martingale.py

"""Martingale-based change point detection.

Implements Algorithm 1 (Parallel Horizon Martingale Detection) from the ICDM 2025 paper.

Key equations:
- Definition 6: M_t^{(k)} = M_{t-1}^{(k)} * g(p_t^{(k)})  (Traditional)
- Definition 7: M_{t,h}^{(k)} = M_{t-1}^{(k)} * g(p_{t,h}^{(k)})  (Horizon)
- Aggregation: M_t^A = sum_k M_t^{(k)}

IMPORTANT DESIGN NOTE - Why We Track "Standalone Traditional":
==============================================================

The paper's Algorithm 1 specifies that Traditional and Horizon martingales share
the same base value M_{t-1} and reset together when EITHER crosses the threshold.
This is the "shared" tracking mode.

Problem: With shared tracking, Horizon always detects first (since Horizon >= Traditional
by construction). When Horizon detects and triggers a reset, Traditional never gets
a chance to accumulate enough evidence to cross the threshold. This makes it appear
that Traditional "fails" when in reality it would have detected - just slower.

Example:
  - Change at t=100, threshold=100
  - At t=122: Horizon=109.8 (crosses), Traditional=26.6 (still building)
  - Shared reset triggered at t=122
  - Traditional resets to 1.0, never crosses threshold
  - Result: Horizon detects at t=122, Traditional "fails"

  But if Traditional ran independently:
  - At t=129: Traditional=148.2 (crosses threshold)
  - Result: Traditional detects at t=129, Horizon at t=122
  - Horizon is 7 timesteps faster (the correct comparison!)

Solution: We track THREE martingales:
  1. Traditional (shared) - Uses shared base, resets when either detects
  2. Horizon (shared) - Uses shared base * multiplier, resets when either detects
  3. Standalone Traditional - Completely independent, own history, never affected by Horizon

For benchmarking/comparison, use "standalone_change_points" for Traditional timing.
The shared Traditional is still tracked for completeness but will often show no detections.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from .betting import BettingConfig, create_betting_function
from .conformal import compute_nonconformity_score, compute_pvalue, DistanceMetric


@dataclass
class MartingaleConfig:
    """Configuration for parallel martingale detection."""

    threshold: float = 30.0
    """Detection threshold. When sum of martingales exceeds this, a change point is declared."""

    history_size: int = 10
    """Number of historical observations used for predictions."""

    window_size: Optional[int] = None
    """Optional sliding window size. If None, uses all history since last reset."""

    reset: bool = True
    """Whether to reset martingales after detection."""

    cooldown: int = 30
    """Minimum timesteps between detections to avoid duplicate alerts."""

    betting: BettingConfig = field(
        default_factory=lambda: BettingConfig(name="mixture", params={"epsilons": [0.7, 0.8, 0.9]})
    )
    """Betting function configuration."""

    random_state: Optional[int] = None
    """Random seed for reproducibility of p-value tie-breaking."""

    distance_metric: DistanceMetric = "euclidean"
    """Distance metric for non-conformity scores."""

    horizon_decay: float = 0.7
    """Exponential decay for horizon contributions. Each horizon h is weighted by decay^h.
    Lower values = more dampening. Set to 1.0 for no dampening."""

    normalize_horizons: bool = False
    """Whether to normalize the horizon product by taking geometric mean.
    False gives faster detection; True is more conservative."""


def run_parallel_detection(
    data: np.ndarray,
    predictions: np.ndarray,
    config: MartingaleConfig,
) -> Dict[str, Any]:
    """Run parallel martingale detection with both shared and standalone tracking.

    Implements Algorithm 1 from the paper with an additional "standalone" Traditional
    martingale for fair comparison. See module docstring for detailed explanation.

    Tracking modes:
        - Shared (Traditional + Horizon): Both use same base, reset together
        - Standalone Traditional: Completely independent, for fair timing comparison

    Args:
        data: Feature matrix of shape (n_samples, n_features)
        predictions: Predicted features of shape (n_predictions, horizon, n_features)
        config: Martingale configuration

    Returns:
        Dict containing:
            Shared tracking (Algorithm 1):
            - traditional_change_points: Detections from shared Traditional (often empty)
            - horizon_change_points: Detections from shared Horizon
            - traditional_sum_martingales: Summed Traditional martingale values
            - horizon_sum_martingales: Summed Horizon martingale values

            Standalone tracking (for fair comparison):
            - standalone_change_points: Detections from independent Traditional
            - standalone_sum_martingales: Summed standalone martingale values

            Per-feature breakdowns:
            - individual_*_martingales: Per-feature martingale trajectories

    Note:
        For benchmarking Traditional vs Horizon detection timing, use
        "standalone_change_points" instead of "traditional_change_points".
    """
    data = np.asarray(data)
    predictions = np.asarray(predictions)
    n_samples, n_features = data.shape
    n_horizons = predictions.shape[1] if len(predictions) > 0 else 0

    betting_fn = create_betting_function(config.betting)

    # SHARED observation history (for Traditional and Horizon, resets together)
    windows = [[] for _ in range(n_features)]
    scores = [[] for _ in range(n_features)]

    # SHARED base values (both martingales use the same base, reset together)
    shared_values = [1.0] * n_features

    # Traditional martingale tracking
    trad_individual = [[1.0] for _ in range(n_features)]
    trad_sum = [float(n_features)]
    trad_cps = []

    # Horizon martingale tracking
    hor_individual = [[1.0] for _ in range(n_features)]
    hor_sum = [float(n_features)]
    hor_cps = []

    # STANDALONE Traditional - Completely independent tracking for fair comparison
    #
    # WHY THIS EXISTS:
    # With shared tracking, Horizon always detects first and resets both martingales.
    # This prevents Traditional from ever crossing the threshold, making it look like
    # Traditional "fails" when it would have detected - just slower.
    #
    # Standalone Traditional has:
    # - Its own observation history (standalone_windows) that NEVER resets
    # - Its own non-conformity scores (standalone_scores) that NEVER reset
    # - Its own martingale values that only reset when IT detects
    #
    # Use "standalone_change_points" in results for fair Traditional vs Horizon comparison.
    standalone_windows = [[] for _ in range(n_features)]
    standalone_scores = [[] for _ in range(n_features)]
    standalone_values = [1.0] * n_features
    standalone_individual = [[1.0] for _ in range(n_features)]
    standalone_sum = [float(n_features)]
    standalone_cps = []
    standalone_last_detection = -config.cooldown - 1

    # Shared cooldown for main detection
    last_detection = -config.cooldown - 1

    for i in range(n_samples):
        trad_feature_vals = []
        hor_feature_vals = []
        standalone_feature_vals = []

        # Maintain sliding window if configured
        for k in range(n_features):
            if config.window_size and len(windows[k]) >= config.window_size:
                windows[k] = windows[k][-config.window_size:]
                scores[k] = scores[k][-config.window_size:]

        for k in range(n_features):
            # ============ COMPUTE P-VALUE (shared) ============
            if len(windows[k]) < 2:
                pv = 0.5
                current_score = 0.0
            else:
                history = np.array(windows[k]).reshape(-1, 1)
                current_score = compute_nonconformity_score(
                    history, np.array([data[i, k]]), config.distance_metric
                )
                pv = compute_pvalue(np.array(scores[k]), current_score, config.random_state)

            # ============ TRADITIONAL (shared base with Horizon) ============
            trad_val = betting_fn(shared_values[k], pv)
            trad_feature_vals.append(trad_val)
            trad_individual[k].append(trad_val)

            # ============ HORIZON (shared base * multiplier) ============
            pred_idx = i - config.history_size
            if pred_idx >= 0 and pred_idx < len(predictions) and len(windows[k]) >= 2:
                history = np.array(windows[k]).reshape(-1, 1)

                log_product = 0.0
                total_weight = 0.0

                for h in range(n_horizons):
                    if h >= predictions.shape[1]:
                        continue

                    pred_value = predictions[pred_idx, h, k]
                    pred_score = compute_nonconformity_score(
                        history, np.array([pred_value]), config.distance_metric
                    )
                    pred_pv = compute_pvalue(np.array(scores[k]), pred_score, config.random_state)
                    betting_multiplier = betting_fn(1.0, pred_pv)

                    weight = config.horizon_decay ** h
                    log_product += weight * np.log(max(betting_multiplier, 1e-10))
                    total_weight += weight

                if total_weight > 0:
                    if config.normalize_horizons:
                        horizon_multiplier = np.exp(log_product / total_weight)
                    else:
                        horizon_multiplier = np.exp(log_product)
                else:
                    horizon_multiplier = 1.0

                hor_val = trad_val * horizon_multiplier
            else:
                hor_val = trad_val

            hor_feature_vals.append(hor_val)
            hor_individual[k].append(hor_val)

            # ============ STANDALONE TRADITIONAL (fully independent) ============
            # Uses its own observation history that NEVER resets
            if len(standalone_windows[k]) < 2:
                standalone_pv = 0.5
                standalone_score = 0.0
            else:
                standalone_history = np.array(standalone_windows[k]).reshape(-1, 1)
                standalone_score = compute_nonconformity_score(
                    standalone_history, np.array([data[i, k]]), config.distance_metric
                )
                standalone_pv = compute_pvalue(
                    np.array(standalone_scores[k]), standalone_score, config.random_state
                )

            standalone_val = betting_fn(standalone_values[k], standalone_pv)
            standalone_feature_vals.append(standalone_val)
            standalone_individual[k].append(standalone_val)

            # Store score for standalone (never resets)
            if len(standalone_windows[k]) >= 2:
                standalone_scores[k].append(standalone_score)

            # Store score for next iteration
            if len(windows[k]) >= 2:
                scores[k].append(current_score)

        # Aggregate
        trad_total = sum(trad_feature_vals)
        hor_total = sum(hor_feature_vals)
        standalone_total = sum(standalone_feature_vals)
        trad_sum.append(trad_total)
        hor_sum.append(hor_total)
        standalone_sum.append(standalone_total)

        # ============ DETECTION (shared for Trad/Horizon, independent for Standalone) ============
        detected = False
        if i - last_detection >= config.cooldown:
            # Traditional detection (from shared tracking)
            if trad_total > config.threshold:
                trad_cps.append(i)
                detected = True
            # Horizon detection (from shared tracking)
            if hor_total > config.threshold:
                hor_cps.append(i)
                detected = True

        # Standalone Traditional detection (fully independent)
        if i - standalone_last_detection >= config.cooldown:
            if standalone_total > config.threshold:
                standalone_cps.append(i)
                standalone_last_detection = i
                if config.reset:
                    standalone_values = [1.0] * n_features
                else:
                    standalone_values = standalone_feature_vals.copy()
            else:
                standalone_values = standalone_feature_vals.copy()
        else:
            standalone_values = standalone_feature_vals.copy()

        # Always update standalone observation history (NEVER resets)
        for k in range(n_features):
            standalone_windows[k].append(data[i, k])

        # ============ SHARED RESET (Trad and Horizon reset together) ============
        if detected:
            last_detection = i
            if config.reset:
                windows = [[] for _ in range(n_features)]
                scores = [[] for _ in range(n_features)]
                shared_values = [1.0] * n_features
        else:
            shared_values = trad_feature_vals.copy()
            for k in range(n_features):
                windows[k].append(data[i, k])

    # Return results with clear documentation of each field
    return {
        # === SHARED TRACKING (Algorithm 1 as written) ===
        # Traditional and Horizon share base value, reset together when either detects.
        # NOTE: traditional_change_points will often be empty because Horizon detects first.
        "traditional_change_points": trad_cps,
        "traditional_sum_martingales": np.array(trad_sum[1:]),
        "individual_traditional_martingales": [np.array(m[1:]) for m in trad_individual],

        # Horizon = Traditional * prediction_multiplier (always >= Traditional)
        "horizon_change_points": hor_cps,
        "horizon_sum_martingales": np.array(hor_sum[1:]),
        "individual_horizon_martingales": [np.array(m[1:]) for m in hor_individual],

        # === STANDALONE TRADITIONAL (for fair comparison) ===
        # Runs completely independently - use this to compare detection timing fairly.
        # See module docstring for why this is necessary.
        "standalone_change_points": standalone_cps,
        "standalone_sum_martingales": np.array(standalone_sum[1:]),
        "individual_standalone_martingales": [np.array(m[1:]) for m in standalone_individual],
    }


def _compute_horizon_weights(n_horizons: int, decay: float = 0.8) -> List[float]:
    """Compute exponentially decaying weights for combining horizon martingales.

    Args:
        n_horizons: Number of prediction horizons
        decay: Exponential decay rate (higher = faster decay)

    Returns:
        Normalized weights summing to 1.0
    """
    if n_horizons == 0:
        return []
    weights = [np.exp(-decay * h) for h in range(n_horizons)]
    total = sum(weights)
    return [w / total for w in weights]
