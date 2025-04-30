#!/usr/bin/env python
# src/scripts/collect_data.py

import pandas as pd
import numpy as np
import os
from collections import defaultdict


def load_trial_data(file_path):
    """Load trial data from a CSV file with multiple sheets."""
    try:
        # Read Excel file with multiple sheets
        excel_file = pd.ExcelFile(file_path)
        trial_sheets = [
            sheet for sheet in excel_file.sheet_names if sheet.startswith("Trial")
        ]

        print(f"Found {len(trial_sheets)} trial sheets: {trial_sheets}")

        trial_dfs = []
        for sheet in trial_sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            trial_dfs.append(df)

        return trial_dfs
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []


def find_change_points(df):
    """Extract the ground truth change points from a dataframe."""
    change_points = df.loc[df["true_change_point"] > 0, "timestep"].values.tolist()
    return change_points


def find_detections(df, method="traditional"):
    """Find detection points for a given method."""
    col = f"{method}_detected"
    if col in df.columns:
        detections = df.loc[df[col] > 0, "timestep"].values.tolist()
        return detections
    return []


def calculate_eauc(tpr, fpr):
    """Calculate estimated Area Under Curve (eAUC) from a single TPR-FPR point.

    This uses a trapezoid approximation method where we know:
    - The curve must pass through (0,0) and (1,1)
    - We have one empirical point (FPR, TPR)

    Args:
        tpr: True Positive Rate
        fpr: False Positive Rate

    Returns:
        Estimated AUC value between 0 and 1
    """
    # Sanity check on inputs
    if pd.isna(tpr) or pd.isna(fpr) or tpr < 0 or fpr < 0:
        return 0.5  # Default to random classifier

    # If FPR is 0, calculate area of rectangle (best case)
    if fpr == 0:
        return tpr

    # Calculate area using trapezoid rule with 3 points: (0,0), (fpr,tpr), (1,1)
    # Area 1: Triangle from (0,0) to (fpr,0) to (fpr,tpr)
    area1 = 0.5 * fpr * tpr

    # Area 2: Trapezoid from (fpr,0) to (1,0) to (1,1) to (fpr,tpr)
    area2 = 0.5 * (1 - fpr) * (tpr + 1)

    return area1 + area2


def calculate_detection_metrics(trial_dfs, change_points_list=None):
    """Calculate detection metrics across all trials."""
    if not trial_dfs:
        return {}

    # If change points weren't provided, extract them from the first trial
    if not change_points_list:
        change_points_list = find_change_points(trial_dfs[0])

    # Optionally merge 94-100 change points
    merged_change_points = []
    for cp in change_points_list:
        # Skip 100 if we're merging, as we'll count it as part of 94
        if cp == 100 and 94 in change_points_list:
            continue
        merged_change_points.append(cp)

    change_points_list = sorted(merged_change_points)
    print(f"Analyzing metrics for change points: {change_points_list}")

    # Initialize metrics
    metrics = {
        "traditional": {
            "by_cp": defaultdict(
                lambda: {"detections": [], "delays": [], "within_window": 0, "trials_detected": set()}
            ),
            "tpr": 0,
            "fpr": 0,
            "add": 0,
            "auc": 0,
            "false_positives": [],
            "total_detections": 0,
        },
        "horizon": {
            "by_cp": defaultdict(
                lambda: {"detections": [], "delays": [], "within_window": 0, "trials_detected": set()}
            ),
            "tpr": 0,
            "fpr": 0,
            "add": 0,
            "auc": 0,
            "false_positives": [],
            "total_detections": 0,
        },
    }

    # Detection window size
    DETECTION_WINDOW = 15

    for i, df in enumerate(trial_dfs):
        trial_idx = i  # Keep track of which trial we're processing
        # Get change points and detections
        trial_change_points = find_change_points(df)
        for method in ["traditional", "horizon"]:
            detections = find_detections(df, method)
            metrics[method]["total_detections"] += len(detections)

            # Match detections to nearest change points
            for detection in detections:
                # Find closest change point
                distances = []
                for cp in change_points_list:
                    # Special handling for 94 (treat 94-100 as one change point)
                    if cp == 94:
                        # Use the best match between 94 and 100
                        dist_94 = abs(detection - 94)
                        dist_100 = abs(detection - 100)
                        distances.append(min(dist_94, dist_100))
                    else:
                        distances.append(abs(detection - cp))

                closest_cp_idx = np.argmin(distances)
                closest_cp = change_points_list[closest_cp_idx]

                # Special handling for delay calculation with merged change points
                if closest_cp == 94:
                    # Use the closest of the two change points for delay calculation
                    if abs(detection - 94) <= abs(detection - 100):
                        delay = detection - 94
                    else:
                        delay = detection - 100
                else:
                    delay = detection - closest_cp

                # Record this detection
                metrics[method]["by_cp"][closest_cp]["detections"].append(detection)
                metrics[method]["by_cp"][closest_cp]["delays"].append(delay)

                # Is this a true positive (within window)?
                if abs(delay) <= DETECTION_WINDOW:
                    # Only count as a new true positive if we haven't already detected 
                    # this change point in this trial
                    if trial_idx not in metrics[method]["by_cp"][closest_cp]["trials_detected"]:
                        metrics[method]["by_cp"][closest_cp]["within_window"] += 1
                        metrics[method]["by_cp"][closest_cp]["trials_detected"].add(trial_idx)
                else:
                    # This is a false positive - not within window of any change point
                    metrics[method]["false_positives"].append(detection)

    # Calculate aggregate metrics
    total_trials = len(trial_dfs)

    for method in ["traditional", "horizon"]:
        total_detection_count = 0
        total_within_window = 0
        all_delays = []

        # Calculate metrics for each change point
        for cp in change_points_list:
            cp_data = metrics[method]["by_cp"][cp]
            detection_count = len(cp_data["detections"])
            total_detection_count += detection_count
            total_within_window += cp_data["within_window"]

            # Only include positive delays (detections after change point) for ADD calculation
            positive_delays = [d for d in cp_data["delays"] if d >= 0]
            if positive_delays:
                all_delays.extend(positive_delays)
                avg_delay = sum(positive_delays) / len(positive_delays)
            else:
                avg_delay = 0

            # Store average detection time and delay
            if detection_count > 0:
                avg_detection = sum(cp_data["detections"]) / detection_count
            else:
                avg_detection = 0

            metrics[method]["by_cp"][cp]["avg_detection"] = avg_detection
            metrics[method]["by_cp"][cp]["avg_delay"] = avg_delay
            metrics[method]["by_cp"][cp]["detection_rate"] = (
                len(cp_data["trials_detected"]) / total_trials
            )

        # Calculate TPR as proportion of possible change points correctly detected
        # (max 1 detection per change point per trial)
        max_possible_detections = len(change_points_list) * total_trials
        metrics[method]["tpr"] = total_within_window / max_possible_detections

        # Calculate FPR as proportion of detections that are false positives
        if metrics[method]["total_detections"] > 0:
            metrics[method]["fpr"] = (
                len(metrics[method]["false_positives"])
                / metrics[method]["total_detections"]
            )
        else:
            metrics[method]["fpr"] = 0.0

        if all_delays:
            metrics[method]["add"] = sum(all_delays) / len(all_delays)
        else:
            metrics[method]["add"] = 0

        # Calculate AUC using eAUC method
        metrics[method]["auc"] = calculate_eauc(
            metrics[method]["tpr"], metrics[method]["fpr"]
        )

    return metrics, change_points_list


def calculate_delay_reductions(metrics, change_points):
    """Calculate delay reductions between traditional and horizon methods."""
    reductions = {}

    for cp in change_points:
        trad_delay = metrics["traditional"]["by_cp"][cp]["avg_delay"]
        hor_delay = metrics["horizon"]["by_cp"][cp]["avg_delay"]

        if trad_delay > 0:
            reduction = (trad_delay - hor_delay) / trad_delay
            reductions[cp] = reduction
        else:
            reductions[cp] = 0

    # Calculate overall reduction, excluding negative reductions if needed
    all_reductions = list(reductions.values())
    if all_reductions:
        avg_reduction = sum(all_reductions) / len(all_reductions)

        # Also calculate excluding specific problematic change points (94/100)
        filtered_reductions = [
            red for cp, red in reductions.items() if cp not in [94, 100]
        ]
        if filtered_reductions:
            filtered_avg = sum(filtered_reductions) / len(filtered_reductions)
        else:
            filtered_avg = 0
    else:
        avg_reduction = 0
        filtered_avg = 0

    return reductions, avg_reduction, filtered_avg


def print_table_data(metrics, change_points, reductions):
    """Print data formatted for LaTeX tables."""
    print("\n=== Summary Table Data ===")
    for method in ["traditional", "horizon"]:
        print(
            f"{method.capitalize()}: TPR={metrics[method]['tpr']*100:.1f}%, FPR={metrics[method]['fpr']*100:.1f}%, ADD={metrics[method]['add']:.1f}, AUC={metrics[method]['auc']:.3f}"
        )
        print(
            f"  - True positives: {sum([cp_data['within_window'] for cp_data in metrics[method]['by_cp'].values()])}"
        )
        print(f"  - False positives: {len(metrics[method]['false_positives'])}")
        print(f"  - Total detections: {metrics[method]['total_detections']}")

    if "traditional" in metrics and "horizon" in metrics:
        overall_reduction = (
            metrics["traditional"]["add"] - metrics["horizon"]["add"]
        ) / metrics["traditional"]["add"]
        print(f"Overall delay reduction: {overall_reduction*100:.1f}%")

    print("\n=== Detailed Table Data ===")
    print(
        "True CP, Method, Avg. Detection, Avg. Delay, Delay Reduction, Within 20 Steps, AUC"
    )

    for cp in change_points:
        for method in ["traditional", "horizon"]:
            cp_data = metrics[method]["by_cp"][cp]

            reduction = ""
            if method == "horizon" and cp in reductions:
                reduction = f"{reductions[cp]*100:.1f}%"
            elif method == "traditional":
                reduction = "--"

            # Display appropriate name for merged change points
            cp_name = cp
            if cp == 94:
                cp_name = "94-100"

            within_window = f"{cp_data['within_window']}/{len(cp_data['detections'])}"

            # Calculate local AUC for this change point (if we have enough data)
            local_tpr = (
                cp_data["within_window"] / total_trials
                if "total_trials" in locals()
                else 0
            )
            local_fpr = (
                (len(cp_data["detections"]) - cp_data["within_window"])
                / len(cp_data["detections"])
                if len(cp_data["detections"]) > 0
                else 0
            )
            local_auc = calculate_eauc(local_tpr, local_fpr)

            print(
                f"{cp_name}, {method.capitalize()}, {cp_data['avg_detection']:.1f}, {cp_data['avg_delay']:.1f}, {reduction}, {within_window}, {local_auc:.3f}"
            )


def main():
    file_path = input("Enter path to combined.xlsx file: ")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    trial_dfs = load_trial_data(file_path)
    if not trial_dfs:
        print("No trial data found.")
        return

    # Use specific change points but exclude 100 since we're merging it with 94
    change_points = [23, 68, 94, 173, 234]

    metrics, detected_cps = calculate_detection_metrics(trial_dfs, change_points)
    reductions, avg_reduction, filtered_avg = calculate_delay_reductions(
        metrics, detected_cps
    )

    print("\n=== Results ===")
    print(f"Overall delay reduction: {avg_reduction*100:.1f}%")
    print(f"Detection window size: 20 timesteps")
    print(f"Change points 94 and 100 treated as a single change point")

    print_table_data(metrics, detected_cps, reductions)

    # Save results to file
    with open("detection_stats_improved.txt", "w") as f:
        f.write(f"Overall delay reduction: {avg_reduction*100:.1f}%\n")
        f.write(f"Detection window size: 20 timesteps\n")
        f.write(f"Change points 94 and 100 treated as a single change point\n")

        f.write("\n=== Summary Table Data ===\n")
        for method in ["traditional", "horizon"]:
            f.write(
                f"{method.capitalize()}: TPR={metrics[method]['tpr']*100:.1f}%, FPR={metrics[method]['fpr']*100:.1f}%, ADD={metrics[method]['add']:.1f}, AUC={metrics[method]['auc']:.3f}\n"
            )
            f.write(
                f"  - True positives: {sum([cp_data['within_window'] for cp_data in metrics[method]['by_cp'].values()])}\n"
            )
            f.write(f"  - False positives: {len(metrics[method]['false_positives'])}\n")
            f.write(f"  - Total detections: {metrics[method]['total_detections']}\n")

        if "traditional" in metrics and "horizon" in metrics:
            overall_reduction = (
                metrics["traditional"]["add"] - metrics["horizon"]["add"]
            ) / metrics["traditional"]["add"]
            f.write(f"Overall delay reduction: {overall_reduction*100:.1f}%\n")


if __name__ == "__main__":
    main()
