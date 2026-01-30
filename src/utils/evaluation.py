"""Metrics calculation and report generation for change detection."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tabulate import tabulate

# Paper parameters (Section V)
DEFAULT_MAX_DELAY = 20  # Δ in the paper
DEFAULT_SEQ_LENGTH = 200  # T for synthetic networks


def calculate_metrics(
    detected: List[int],
    true_cps: List[int],
    total_steps: int = DEFAULT_SEQ_LENGTH,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> Dict[str, float]:
    """Calculate detection metrics per paper Equations 26-28.

    Args:
        detected: List of detection times {τ_j}
        true_cps: List of true change point times {t_i}
        total_steps: Total sequence length T
        max_delay: Maximum acceptable delay Δ (default 20 per paper)

    Returns:
        Dict with tpr, fpr, avg_delay (ADD)
    """
    if not true_cps:
        return {"tpr": 0.0, "fpr": 0.0, "avg_delay": 0.0}

    detected = sorted(set(detected))
    true_cps = sorted(true_cps)

    # TPR (Eq. 26): fraction of CPs detected within [t_i, t_i + Δ]
    tp = 0
    delays = []
    matched_detections = set()

    for cp in true_cps:
        # Find detections in window [cp, cp + max_delay]
        valid = [d for d in detected if cp <= d <= cp + max_delay]
        if valid:
            tp += 1
            earliest = min(valid)
            delays.append(earliest - cp)
            matched_detections.update(valid)

    tpr = tp / len(true_cps)

    # FPR (Eq. 27): false detections / non-change timesteps
    # Numerator: detections NOT in any [t_i, t_i + Δ] window
    false_positives = [d for d in detected if d not in matched_detections]

    # Denominator: T - |{t_i}| * (Δ + 1) per paper
    # This is the number of timesteps NOT in any detection window
    non_change_steps = total_steps - len(true_cps) * (max_delay + 1)
    fpr = len(false_positives) / non_change_steps if non_change_steps > 0 else 0.0

    # ADD (Eq. 28): average delay for matched detections
    avg_delay = np.mean(delays) if delays else 0.0

    return {"tpr": tpr, "fpr": fpr, "avg_delay": avg_delay}


def calculate_full_metrics(
    trad_detected: List[int],
    horizon_detected: List[int],
    true_cps: List[int],
    total_steps: int = DEFAULT_SEQ_LENGTH,
    max_delay: int = DEFAULT_MAX_DELAY,
) -> Dict[str, Any]:
    """Calculate full metrics for both traditional and horizon martingales.

    Returns metrics matching Table IV format from paper.
    """
    trad = calculate_metrics(trad_detected, true_cps, total_steps, max_delay)
    horizon = calculate_metrics(horizon_detected, true_cps, total_steps, max_delay)

    # Compute improvements
    delay_reduction = 0.0
    if trad["avg_delay"] > 0:
        delay_reduction = (trad["avg_delay"] - horizon["avg_delay"]) / trad["avg_delay"]

    tpr_improvement = horizon["tpr"] - trad["tpr"]

    return {
        "traditional": trad,
        "horizon": horizon,
        "delay_reduction": delay_reduction,
        "tpr_improvement": tpr_improvement,
    }


def find_detection_delay(
    cp: int, detections: List[int], max_delay: int = DEFAULT_MAX_DELAY
) -> Tuple[Optional[int], Optional[int]]:
    """Find delay between change point and its detection.

    Args:
        cp: True change point time
        detections: List of detection times
        max_delay: Maximum acceptable delay Δ

    Returns:
        (delay, detection_time) or (None, None) if not detected
    """
    if not detections:
        return None, None
    # Valid detection: within [cp, cp + max_delay]
    valid = [d for d in detections if cp <= d <= cp + max_delay]
    if valid:
        earliest = min(valid)
        return earliest - cp, earliest
    return None, None


def analyze_results(
    results: Dict[str, Any],
    fmt: str = "rounded_grid",
    total_steps: int = DEFAULT_SEQ_LENGTH,
) -> str:
    """Generate analysis report for detection results."""
    true_cps = list(results.get("true_change_points", []))
    trials = results.get("individual_trials", [])

    if trials and len(trials) > 1:
        return _analyze_multiple_trials(results, fmt, total_steps)

    trad = list(results.get("traditional_change_points", []))
    horizon = list(results.get("horizon_change_points", []))

    rows = []
    for cp in true_cps:
        td, t_det = find_detection_delay(cp, trad)
        hd, h_det = find_detection_delay(cp, horizon)
        reduction = _delay_reduction(td, hd)
        rows.append([cp, t_det or "-", td if td is not None else "-",
                     h_det or "-", hd if hd is not None else "-", reduction])

    headers = ["True CP", "Trad Det", "Delay", "Horizon Det", "Delay", "Reduction"]
    table = tabulate(rows, headers=headers, tablefmt=fmt)

    # Full metrics per paper
    trad_metrics = calculate_metrics(trad, true_cps, total_steps)
    hor_metrics = calculate_metrics(horizon, true_cps, total_steps)

    avg_td = trad_metrics["avg_delay"]
    avg_hd = hor_metrics["avg_delay"]
    avg_red = (avg_td - avg_hd) / avg_td if avg_td and avg_hd and avg_td > 0 else None

    summary = [
        ["TPR", f"{trad_metrics['tpr']:.2%}", f"{hor_metrics['tpr']:.2%}"],
        ["FPR", f"{trad_metrics['fpr']:.3f}", f"{hor_metrics['fpr']:.3f}"],
        ["ADD", f"{avg_td:.2f}" if avg_td else "N/A", f"{avg_hd:.2f}" if avg_hd else "N/A"],
        ["Delay Reduction", "", f"{avg_red:.1%}" if avg_red else "N/A"],
    ]
    summary_table = tabulate(summary, headers=["Metric", "Traditional", "Horizon"], tablefmt=fmt)

    return f"Change Point Detection Analysis\n{'='*30}\n\nDetails:\n{table}\n\nSummary (Δ={DEFAULT_MAX_DELAY}):\n{summary_table}\n"


def _analyze_multiple_trials(
    results: Dict[str, Any],
    fmt: str,
    total_steps: int = DEFAULT_SEQ_LENGTH,
) -> str:
    """Analyze multiple trial results with full metrics."""
    true_cps = list(results.get("true_change_points", []))
    trials = results.get("individual_trials", [])

    # Aggregate all detections across trials for metrics
    all_trad_detections = []
    all_hor_detections = []
    all_td, all_hd = [], []
    rows = []

    for i, trial in enumerate(trials):
        trad = trial.get("traditional_change_points", [])
        horizon = trial.get("horizon_change_points", [])
        all_trad_detections.extend(trad)
        all_hor_detections.extend(horizon)

        for cp in true_cps:
            td, t_det = find_detection_delay(cp, trad)
            hd, h_det = find_detection_delay(cp, horizon)
            if td is not None:
                all_td.append(td)
            if hd is not None:
                all_hd.append(hd)
            rows.append([f"T{i+1}", cp, t_det or "-", td if td is not None else "-",
                        h_det or "-", hd if hd is not None else "-", _delay_reduction(td, hd)])

    # Compute per-trial averaged metrics
    n_trials = len(trials)
    trad_tpr_sum, trad_fpr_sum, trad_add_sum = 0.0, 0.0, 0.0
    hor_tpr_sum, hor_fpr_sum, hor_add_sum = 0.0, 0.0, 0.0

    for trial in trials:
        trad = trial.get("traditional_change_points", [])
        horizon = trial.get("horizon_change_points", [])
        tm = calculate_metrics(trad, true_cps, total_steps)
        hm = calculate_metrics(horizon, true_cps, total_steps)
        trad_tpr_sum += tm["tpr"]
        trad_fpr_sum += tm["fpr"]
        trad_add_sum += tm["avg_delay"]
        hor_tpr_sum += hm["tpr"]
        hor_fpr_sum += hm["fpr"]
        hor_add_sum += hm["avg_delay"]

    # Averaged metrics
    trad_tpr = trad_tpr_sum / n_trials
    trad_fpr = trad_fpr_sum / n_trials
    trad_add = trad_add_sum / n_trials
    hor_tpr = hor_tpr_sum / n_trials
    hor_fpr = hor_fpr_sum / n_trials
    hor_add = hor_add_sum / n_trials

    delay_red = (trad_add - hor_add) / trad_add if trad_add > 0 else 0.0
    tpr_imp = hor_tpr - trad_tpr

    rows.append(["Avg", "-",
                 f"{trad_tpr:.0%}", f"{trad_add:.1f}",
                 f"{hor_tpr:.0%}", f"{hor_add:.1f}",
                 f"{delay_red:.1%}"])

    headers = ["Trial", "CP", "Trad Det", "Delay", "Horizon Det", "Delay", "Reduction"]

    # Summary table matching Table IV format
    summary = [
        ["TPR", f"{trad_tpr:.2%}", f"{hor_tpr:.2%}", f"+{tpr_imp:.1%}" if tpr_imp > 0 else f"{tpr_imp:.1%}"],
        ["FPR", f"{trad_fpr:.3f}", f"{hor_fpr:.3f}", ""],
        ["ADD", f"{trad_add:.1f}", f"{hor_add:.1f}", f"{delay_red:.1%} ↓"],
    ]

    detail_table = tabulate(rows, headers=headers, tablefmt=fmt)
    summary_table = tabulate(summary, headers=["Metric", "Traditional", "Horizon", "Improvement"], tablefmt=fmt)

    return f"""Detection Analysis ({n_trials} Trials, Δ={DEFAULT_MAX_DELAY})
{'='*50}

{detail_table}

Summary (Table IV format):
{summary_table}
"""


def _delay_reduction(td: Optional[int], hd: Optional[int]) -> str:
    """Compute delay reduction string."""
    if td is None:
        return "-"
    if hd is None:
        return "-"
    if td == 0:
        return "0%"
    red = (td - hd) / td
    return f"{red:.0%}" if red >= 0 else f"{red:.0%} (worse)"


def print_report(results: Dict[str, Any], fmt: str = "rounded_grid") -> None:
    """Print analysis report to stdout."""
    print(analyze_results(results, fmt))
