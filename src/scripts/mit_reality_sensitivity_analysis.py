#!/usr/bin/env python3
"""MIT Reality Sensitivity Analysis Script"""

import os
import re
import yaml
import pandas as pd
from typing import Dict, Tuple, Optional
from collections import defaultdict


def parse_mit_experiment_name(exp_name: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse MIT Reality experiment name to extract group and parameters.

    Returns:
        Tuple of (group, parameters_dict)
    """
    # Power betting: mit_reality_power_{epsilon}_{distance}
    power_pattern = r"^mit_reality_power_([0-9.]+)_(\w+)$"
    power_match = re.match(power_pattern, exp_name)
    if power_match:
        epsilon, distance = power_match.groups()
        return "power_betting", {
            "dataset": "mit_reality",
            "epsilon": epsilon,
            "distance": distance,
        }

    # Mixture betting: mit_reality_mixture_{distance}
    mixture_pattern = r"^mit_reality_mixture_(\w+)$"
    mixture_match = re.match(mixture_pattern, exp_name)
    if mixture_match:
        distance = mixture_match.groups()[0]
        return "mixture_betting", {"dataset": "mit_reality", "distance": distance}

    # Beta betting: mit_reality_beta_{beta_a}_{beta_b}_{distance}
    beta_pattern = r"^mit_reality_beta_([0-9.]+)_([0-9.]+)_(\w+)$"
    beta_match = re.match(beta_pattern, exp_name)
    if beta_match:
        beta_a, beta_b, distance = beta_match.groups()
        return "beta_betting", {
            "dataset": "mit_reality",
            "beta_a": beta_a,
            "beta_b": beta_b,
            "distance": distance,
        }

    # Threshold analysis: mit_reality_threshold_{threshold}_{distance}
    threshold_pattern = r"^mit_reality_threshold_(\d+)_(\w+)$"
    threshold_match = re.match(threshold_pattern, exp_name)
    if threshold_match:
        threshold, distance = threshold_match.groups()
        return "threshold_analysis", {
            "dataset": "mit_reality",
            "threshold": threshold,
            "distance": distance,
        }

    return "unknown", {}


def load_experiment_config(config_path: str) -> Optional[Dict]:
    """Load and parse the experiment configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None


def extract_config_parameters(config: Dict) -> Dict:
    """Extract key parameters from the experiment configuration."""
    params = {}

    # Dataset type
    if "model" in config and "dataset" in config["model"]:
        params["dataset"] = config["model"]["dataset"]

    # Distance measure
    if "detection" in config and "distance" in config["detection"]:
        params["distance_measure"] = config["detection"]["distance"]["measure"]
        params["distance_p"] = config["detection"]["distance"].get("p", 2.0)

    # Betting function configuration
    if "detection" in config and "betting_func_config" in config["detection"]:
        betting_config = config["detection"]["betting_func_config"]
        params["betting_function"] = betting_config["name"]

        if betting_config["name"] == "power" and "power" in betting_config:
            params["epsilon"] = betting_config["power"]["epsilon"]
        elif betting_config["name"] == "beta" and "beta" in betting_config:
            params["beta_a"] = betting_config["beta"]["a"]
            params["beta_b"] = betting_config["beta"]["b"]
        elif betting_config["name"] == "mixture" and "mixture" in betting_config:
            params["mixture_epsilons"] = betting_config["mixture"]["epsilons"]

    # Detection threshold
    if "detection" in config and "threshold" in config["detection"]:
        params["threshold"] = config["detection"]["threshold"]

    # Other detection parameters
    if "detection" in config:
        params["batch_size"] = config["detection"].get("batch_size", None)
        params["cooldown_period"] = config["detection"].get("cooldown_period", None)

    return params


def get_detection_window_for_changepoint(changepoint: int) -> tuple:
    """
    Get the detection window for each specific changepoint in MIT Reality.

    Updated logic: detections between changepoints are considered valid (late detections for earlier CPs):
    - CP 23: window from 23 to 67 (until next CP region)
    - CP 68: window from 68 to 93 (until next CP region)
    - CP 94: window from 94 to 99 (until next CP)
    - CP 100: window from 100 to 172 (until next CP region)
    - CP 173: window from 173 to 233 (until next CP region)
    - CP 234: window from 234 to 270 (end of sequence)
    """
    windows = {
        23: (23, 67),  # CP 23: detect from 23 to 67
        68: (68, 93),  # CP 68: detect from 68 to 93
        94: (94, 99),  # CP 94: detect from 94 to 99
        100: (100, 172),  # CP 100: detect from 100 to 172
        173: (173, 233),  # CP 173: detect from 173 to 233
        234: (234, 270),  # CP 234: detect from 234 to 270
    }

    return windows.get(changepoint, (changepoint, changepoint + 30))


def compute_metrics_for_changepoint(
    detection_df: pd.DataFrame,
    changepoint: int,
    window_size: int = 30,
    n_trials: int = 10,
) -> Dict:
    """
    Compute TPR, FPR, and ADD for a specific changepoint in MIT Reality data.

    Uses specific detection windows for each changepoint:
    - CP 23: detections from 23 to 53 are TP
    - CP 68: detections from 68 to 88 are TP
    - CP 94: detections from 94 to 100 are TP (limited by next CP)
    - CP 100: detections from 100 to 130 are TP
    - CP 173: detections from 173 to 213 are TP
    - CP 234: detections from 234 to 274 are TP

    Args:
        detection_df: DataFrame with detection details
        changepoint: The true changepoint location
        window_size: Window size for considering a detection as TP (default: 30, but overridden by specific windows)
        n_trials: Total number of trials run (default: 10)

    Returns:
        Dictionary with TPR, FPR, ADD metrics
    """
    # Get the specific detection window for this changepoint
    window_start, window_end = get_detection_window_for_changepoint(changepoint)

    # Get ALL detections in the dataset (not just those attributed to this CP)
    all_detections = detection_df.copy()

    # Find detections that fall within this changepoint's detection window
    tp_detections = all_detections[
        (all_detections["Detection Index"] >= window_start)
        & (all_detections["Detection Index"] <= window_end)
    ].copy()

    # Count trials with at least one TP detection for this changepoint
    trials_with_tp = tp_detections["Trial"].nunique() if len(tp_detections) > 0 else 0

    # False Negatives: trials with no TP detection for this changepoint
    false_negatives = n_trials - trials_with_tp

    # Calculate TPR (no FPR per changepoint - FPR is calculated globally)
    tpr = trials_with_tp / n_trials if n_trials > 0 else 0.0

    # Calculate ADD (Average Detection Delay) for TRUE POSITIVE detections only
    if len(tp_detections) > 0:
        # Calculate delay as detection index - changepoint
        delays = tp_detections["Detection Index"] - changepoint
        add = delays.mean()
    else:
        add = float("inf")  # No true positives means infinite delay

    return {
        "changepoint": changepoint,
        "window_start": window_start,
        "window_end": window_end,
        "total_trials": n_trials,
        "tpr": tpr,
        "fpr": 0.0,  # FPR calculated separately at global level
        "add": add,
        "true_positives": trials_with_tp,
        "false_positives": 0,  # FP calculated separately
        "false_negatives": false_negatives,
        "trials_with_detections": (
            tp_detections["Trial"].nunique() if len(tp_detections) > 0 else 0
        ),
        "total_tp_detections": len(tp_detections),
        "total_fp_detections": 0,  # FP calculated separately
    }


def compute_global_fpr(detection_df: pd.DataFrame, n_trials: int = 10) -> float:
    """
    Compute global False Positive Rate for MIT Reality data.

    FPR = (number of trials with at least one FP detection) / total trials

    False positives are detections in gaps between changepoint regions:
    - 0-22 (before first CP)
    - 68-67 (gap - none)
    - 94-93 (gap - none)
    - 173-172 (gap - none)
    - 271+ (after last CP)

    Valid detection windows (NOT false positives):
    - CP 23: 23 to 67
    - CP 68: 68 to 93
    - CP 94: 94 to 99
    - CP 100: 100 to 172
    - CP 173: 173 to 233
    - CP 234: 234 to 270
    """
    if len(detection_df) == 0:
        return 0.0

    # Define all valid detection windows (detections in these ranges are NOT false positives)
    valid_windows = [
        (23, 67),  # CP 23
        (68, 93),  # CP 68
        (94, 99),  # CP 94
        (100, 172),  # CP 100
        (173, 233),  # CP 173
        (234, 270),  # CP 234
    ]

    # Find trials that have at least one FP detection
    trials_with_fp = set()

    for _, detection in detection_df.iterrows():
        detection_index = detection["Detection Index"]
        trial = detection["Trial"]
        is_fp = True

        # Check if detection falls within any valid window
        for window_start, window_end in valid_windows:
            if window_start <= detection_index <= window_end:
                is_fp = False
                break

        if is_fp:
            trials_with_fp.add(trial)

    # Calculate FPR as proportion of trials with at least one FP
    fpr = len(trials_with_fp) / n_trials if n_trials > 0 else 0.0

    return fpr


def analyze_experiment_folder(folder_path: str) -> Optional[Dict]:
    """
    Analyze a single MIT Reality experiment folder and extract configuration and metrics.

    Args:
        folder_path: Path to the experiment folder

    Returns:
        Dictionary with experiment analysis results or None if failed
    """
    config_path = os.path.join(folder_path, "config.yaml")
    results_path = os.path.join(
        folder_path, "detection", "csv", "detection_results.xlsx"
    )

    # Check if required files exist
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return None

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return None

    try:
        # Load configuration
        config = load_experiment_config(config_path)
        if config is None:
            return None

        # Extract configuration parameters
        config_params = extract_config_parameters(config)

        # Load Excel file
        excel_file = pd.ExcelFile(results_path)

        # Check if required sheets exist
        required_sheets = ["ChangePointMetadata", "Detection Details"]
        missing_sheets = [
            sheet for sheet in required_sheets if sheet not in excel_file.sheet_names
        ]

        if missing_sheets:
            print(f"Missing required sheets in {results_path}: {missing_sheets}")
            return None

        # Load the required sheets
        changepoint_metadata = pd.read_excel(
            results_path, sheet_name="ChangePointMetadata"
        )
        detection_details = pd.read_excel(results_path, sheet_name="Detection Details")

        # Get all changepoints
        changepoints = changepoint_metadata["change_point"].tolist()

        # Get number of trials from config
        n_trials = config.get("trials", {}).get("n_trials", 10)

        # Compute metrics for each changepoint
        changepoint_metrics = []
        for cp in changepoints:
            metrics = compute_metrics_for_changepoint(
                detection_details, cp, n_trials=n_trials
            )
            changepoint_metrics.append(metrics)

        # Compute global FPR (detections outside all valid windows)
        global_fpr = compute_global_fpr(detection_details, n_trials=n_trials)

        # Compute overall metrics (averaged across all changepoints)
        if changepoint_metrics:
            overall_tpr = sum(m["tpr"] for m in changepoint_metrics) / len(
                changepoint_metrics
            )
            overall_fpr = global_fpr  # Use global FPR instead of averaging per-CP FPR

            # For ADD, exclude infinite values and compute mean
            finite_adds = [
                m["add"] for m in changepoint_metrics if m["add"] != float("inf")
            ]
            overall_add = (
                sum(finite_adds) / len(finite_adds) if finite_adds else float("inf")
            )
        else:
            overall_tpr = 0.0
            overall_fpr = global_fpr
            overall_add = float("inf")

        return {
            "folder_path": folder_path,
            "config_parameters": config_params,
            "changepoints": changepoints,
            "changepoint_metrics": changepoint_metrics,
            "overall_metrics": {
                "tpr": overall_tpr,
                "fpr": overall_fpr,
                "add": overall_add,
                "num_changepoints": len(changepoints),
            },
        }

    except Exception as e:
        print(f"Error analyzing folder {folder_path}: {e}")
        return None


def collect_all_mit_experiment_data(
    base_dir: str = "results/mit_reality_sensitivity_analysis",
) -> Dict:
    """
    Collect and analyze data from all MIT Reality experiment folders.

    Returns:
        Dictionary with all experiment results organized by parameter type
    """
    if not os.path.exists(base_dir):
        print(f"Base directory does not exist: {base_dir}")
        return {}

    all_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    print(f"Found {len(all_dirs)} MIT Reality experiment directories")

    # Organize results by parameter configuration
    results = defaultdict(list)
    failed_experiments = []

    for i, dirname in enumerate(all_dirs):
        if (i + 1) % 20 == 0:
            print(f"Processing experiment {i+1}/{len(all_dirs)}")

        folder_path = os.path.join(base_dir, dirname)

        # Extract experiment name from directory name
        parts = dirname.split("_")
        timestamp_idx = None
        for j, part in enumerate(parts):
            if len(part) == 8 and part.isdigit() and part.startswith("20"):
                timestamp_idx = j
                break

        if timestamp_idx is None:
            failed_experiments.append(dirname)
            continue

        exp_name = "_".join(parts[:timestamp_idx])
        group, params = parse_mit_experiment_name(exp_name)

        if group == "unknown":
            failed_experiments.append(dirname)
            continue

        # Analyze the experiment
        analysis_result = analyze_experiment_folder(folder_path)

        if analysis_result is None:
            failed_experiments.append(dirname)
            continue

        # Store result organized by parameter configuration
        results[group].append(
            {
                "exp_name": exp_name,
                "dirname": dirname,
                "group": group,
                "parameters": params,
                "analysis": analysis_result,
            }
        )

    print(
        f"Successfully processed {sum(len(results[grp]) for grp in results)} experiments"
    )
    print(f"Failed to process {len(failed_experiments)} experiments")

    return dict(results)


def create_mit_sensitivity_table(experiment_data: Dict) -> pd.DataFrame:
    """
    Create the MIT Reality parameter sensitivity table.

    Args:
        experiment_data: Dictionary with all experiment results

    Returns:
        DataFrame with the sensitivity analysis table
    """
    # Initialize results structure
    table_data = []

    # Initialize row for MIT Reality dataset
    row = {"Dataset": "MIT Reality", "Metric": None}

    # Process Power Betting experiments
    power_experiments = experiment_data.get("power_betting", [])
    power_results = defaultdict(list)

    for exp in power_experiments:
        epsilon = exp["parameters"]["epsilon"]
        distance = exp["parameters"]["distance"]
        metrics = exp["analysis"]["overall_metrics"]

        key = f"epsilon_{epsilon}"
        power_results[key].append(metrics)

    # Process Mixture Betting experiments
    mixture_experiments = experiment_data.get("mixture_betting", [])
    mixture_results = []

    for exp in mixture_experiments:
        metrics = exp["analysis"]["overall_metrics"]
        mixture_results.append(metrics)

    # Process Beta Betting experiments
    beta_experiments = experiment_data.get("beta_betting", [])
    beta_results = defaultdict(list)

    for exp in beta_experiments:
        beta_a = exp["parameters"]["beta_a"]
        beta_b = exp["parameters"]["beta_b"]
        metrics = exp["analysis"]["overall_metrics"]

        key = f"beta_{beta_a}_{beta_b}"
        beta_results[key].append(metrics)

    # Process Threshold experiments
    threshold_experiments = experiment_data.get("threshold_analysis", [])
    threshold_results = defaultdict(list)

    for exp in threshold_experiments:
        threshold = exp["parameters"]["threshold"]
        metrics = exp["analysis"]["overall_metrics"]

        key = f"threshold_{threshold}"
        threshold_results[key].append(metrics)

    # Aggregate distance metric results across all experiment types
    distance_results = defaultdict(list)

    for group_name, group_data in experiment_data.items():
        for exp in group_data:
            distance = exp["parameters"]["distance"]
            metrics = exp["analysis"]["overall_metrics"]
            distance_results[distance].append(metrics)

    # Create rows for TPR, FPR, ADD
    for metric in ["TPR", "FPR", "ADD"]:
        metric_row = row.copy()
        metric_row["Metric"] = metric

        # Power Betting columns
        for epsilon in ["0.2", "0.5", "0.7", "0.9"]:
            key = f"epsilon_{epsilon}"
            if key in power_results and power_results[key]:
                values = [m[metric.lower()] for m in power_results[key]]
                # Filter out infinite values for ADD
                if metric == "ADD":
                    values = [v for v in values if v != float("inf")]
                avg_value = sum(values) / len(values) if values else 0.0
                metric_row[f"Power_ε={epsilon}"] = avg_value
            else:
                metric_row[f"Power_ε={epsilon}"] = 0.0

        # Mixture Betting column
        if mixture_results:
            values = [m[metric.lower()] for m in mixture_results]
            if metric == "ADD":
                values = [v for v in values if v != float("inf")]
            avg_value = sum(values) / len(values) if values else 0.0
            metric_row["Mixture"] = avg_value
        else:
            metric_row["Mixture"] = 0.0

        # Beta Betting columns
        for beta_combo in ["0.2_2.5", "0.4_1.8", "0.6_1.2"]:
            key = f"beta_{beta_combo}"
            if key in beta_results and beta_results[key]:
                values = [m[metric.lower()] for m in beta_results[key]]
                if metric == "ADD":
                    values = [v for v in values if v != float("inf")]
                avg_value = sum(values) / len(values) if values else 0.0
                metric_row[f"Beta_{beta_combo}"] = avg_value
            else:
                metric_row[f"Beta_{beta_combo}"] = 0.0

        # Distance Metric columns
        for distance in ["euclidean", "mahalanobis", "cosine", "chebyshev"]:
            if distance in distance_results and distance_results[distance]:
                values = [m[metric.lower()] for m in distance_results[distance]]
                if metric == "ADD":
                    values = [v for v in values if v != float("inf")]
                avg_value = sum(values) / len(values) if values else 0.0
                metric_row[f"Dist_{distance}"] = avg_value
            else:
                metric_row[f"Dist_{distance}"] = 0.0

        # Threshold columns
        for threshold in ["20", "50", "100"]:
            key = f"threshold_{threshold}"
            if key in threshold_results and threshold_results[key]:
                values = [m[metric.lower()] for m in threshold_results[key]]
                if metric == "ADD":
                    values = [v for v in values if v != float("inf")]
                avg_value = sum(values) / len(values) if values else 0.0
                metric_row[f"Threshold_{threshold}"] = avg_value
            else:
                metric_row[f"Threshold_{threshold}"] = 0.0

        table_data.append(metric_row)

    return pd.DataFrame(table_data)


def format_mit_sensitivity_table_for_paper(df: pd.DataFrame) -> str:
    """
    Format the MIT Reality sensitivity table for LaTeX/paper presentation.

    Args:
        df: DataFrame with sensitivity analysis results

    Returns:
        Formatted string for paper presentation
    """
    output = []
    output.append("MIT REALITY PARAMETER SENSITIVITY ANALYSIS")
    output.append("=" * 80)
    output.append("")

    # Create formatted table for MIT Reality
    metrics = ["TPR", "FPR", "ADD"]

    for metric in metrics:
        metric_row = df[df["Metric"] == metric].iloc[0]

        output.append(f"{metric}:")

        # Power Betting
        power_values = []
        for epsilon in ["0.2", "0.5", "0.7", "0.9"]:
            col = f"Power_ε={epsilon}"
            if col in metric_row:
                value = metric_row[col]
                if metric == "ADD" and value == 0.0:
                    power_values.append("--")
                elif metric in ["TPR", "FPR"]:
                    power_values.append(f"{value:.3f}")
                else:
                    power_values.append(f"{value:.1f}")
            else:
                power_values.append("--")

        mixture_val = metric_row.get("Mixture", 0.0)
        if metric == "ADD" and mixture_val == 0.0:
            mixture_str = "--"
        elif metric in ["TPR", "FPR"]:
            mixture_str = f"{mixture_val:.3f}"
        else:
            mixture_str = f"{mixture_val:.1f}"

        output.append(
            f"  Power Betting: e=0.2:{power_values[0]} | e=0.5:{power_values[1]} | e=0.7:{power_values[2]} | e=0.9:{power_values[3]} | Mixture:{mixture_str}"
        )

        # Beta Betting
        beta_values = []
        for beta_combo in ["0.2_2.5", "0.4_1.8", "0.6_1.2"]:
            col = f"Beta_{beta_combo}"
            if col in metric_row:
                value = metric_row[col]
                if metric == "ADD" and value == 0.0:
                    beta_values.append("--")
                elif metric in ["TPR", "FPR"]:
                    beta_values.append(f"{value:.3f}")
                else:
                    beta_values.append(f"{value:.1f}")
            else:
                beta_values.append("--")

        output.append(
            f"  Beta Betting: (0.2,2.5):{beta_values[0]} | (0.4,1.8):{beta_values[1]} | (0.6,1.2):{beta_values[2]}"
        )

        # Distance Metrics
        dist_values = []
        dist_names = ["euclidean", "mahalanobis", "cosine", "chebyshev"]
        dist_abbrev = ["Euc", "Mah", "Cos", "Cheb"]

        for dist in dist_names:
            col = f"Dist_{dist}"
            if col in metric_row:
                value = metric_row[col]
                if metric == "ADD" and value == 0.0:
                    dist_values.append("--")
                elif metric in ["TPR", "FPR"]:
                    dist_values.append(f"{value:.3f}")
                else:
                    dist_values.append(f"{value:.1f}")
            else:
                dist_values.append("--")

        output.append(
            f"  Distance: {dist_abbrev[0]}:{dist_values[0]} | {dist_abbrev[1]}:{dist_values[1]} | {dist_abbrev[2]}:{dist_values[2]} | {dist_abbrev[3]}:{dist_values[3]}"
        )

        # Thresholds
        thresh_values = []
        for threshold in ["20", "50", "100"]:
            col = f"Threshold_{threshold}"
            if col in metric_row:
                value = metric_row[col]
                if metric == "ADD" and value == 0.0:
                    thresh_values.append("--")
                elif metric in ["TPR", "FPR"]:
                    thresh_values.append(f"{value:.3f}")
                else:
                    thresh_values.append(f"{value:.1f}")
            else:
                thresh_values.append("--")

        output.append(
            f"  Threshold: L=20:{thresh_values[0]} | L=50:{thresh_values[1]} | L=100:{thresh_values[2]}"
        )
        output.append("")

    return "\n".join(output)


def create_latex_mit_sensitivity_table(df: pd.DataFrame) -> str:
    """
    Create LaTeX table for MIT Reality sensitivity analysis matching Table III format.

    Args:
        df: DataFrame with MIT Reality sensitivity analysis results

    Returns:
        LaTeX table string
    """
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append(
        "\\caption{MIT Reality parameter sensitivity analysis showing relative effectiveness of different configurations. Values represent performance metrics for each parameter choice on real-world academic collaboration data.}"
    )
    latex_lines.append("\\label{tab:mit_reality_parameter_sensitivity}")
    latex_lines.append("\\scriptsize")

    # Column specification
    latex_lines.append(
        "\\begin{tabular}{|l|c|cccc>{\\columncolor{blue!5}}c|ccc|c>{\\columncolor{green!5}}ccc|ccc|}"
    )
    latex_lines.append("\\hline")

    # Multi-row header
    latex_lines.append(
        "\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Metric}} & \\multicolumn{5}{c|}{\\textbf{Power Betting}} & \\multicolumn{3}{c|}{\\textbf{Beta Betting $(a,b)$}} & \\multicolumn{4}{c|}{\\textbf{Distance Metric}} & \\multicolumn{3}{c|}{\\textbf{Threshold ($\\lambda$)}} \\\\"
    )
    latex_lines.append("\\cline{3-17}")
    latex_lines.append(
        " & & $\\epsilon$=0.2 & $\\epsilon$=0.5 & $\\epsilon$=0.7 & $\\epsilon$=0.9 & \\textbf{Mixture} & (0.2,2.5) & (0.4,1.8) & (0.6,1.2) & \\textbf{Euc.} & \\textbf{Mah.} & \\textbf{Cos.} & \\textbf{Cheb.} & \\textbf{20} & \\textbf{50} & \\textbf{100} \\\\"
    )
    latex_lines.append("\\hline")

    # Process MIT Reality data
    dataset_data = df[df["Dataset"] == "MIT Reality"]

    if len(dataset_data) == 0:
        latex_lines.append("\\multicolumn{17}{|c|}{No MIT Reality data available} \\\\")
        latex_lines.append("\\hline")
    else:
        # Process each metric
        metrics = ["TPR", "FPR", "ADD"]

        for i, metric in enumerate(metrics):
            metric_row = dataset_data[dataset_data["Metric"] == metric]

            if len(metric_row) == 0:
                continue

            metric_row = metric_row.iloc[0]

            # Start row with dataset name (only for first metric)
            if i == 0:
                row_start = f"\\multirow{{3}}{{*}}{{MIT Reality}} & {metric}"
            else:
                row_start = f" & {metric}"

            # Power Betting columns
            power_values = []
            for epsilon in ["0.2", "0.5", "0.7", "0.9"]:
                col = f"Power_ε={epsilon}"
                if col in metric_row:
                    value = metric_row[col]
                    if metric == "ADD" and value == 0.0:
                        power_values.append("--")
                    elif metric in ["TPR", "FPR"]:
                        power_values.append(f"{value:.3f}")
                    else:
                        power_values.append(f"{value:.1f}")
                else:
                    power_values.append("0.000" if metric in ["TPR", "FPR"] else "--")

            # Mixture value
            mixture_val = metric_row.get("Mixture", 0.0)
            if metric == "ADD" and mixture_val == 0.0:
                mixture_str = "--"
            elif metric in ["TPR", "FPR"]:
                mixture_str = f"{mixture_val:.3f}"
            else:
                mixture_str = f"{mixture_val:.1f}"

            # Beta Betting columns
            beta_values = []
            for beta_combo in ["0.2_2.5", "0.4_1.8", "0.6_1.2"]:
                col = f"Beta_{beta_combo}"
                if col in metric_row:
                    value = metric_row[col]
                    if metric == "ADD" and value == 0.0:
                        beta_values.append("--")
                    elif metric in ["TPR", "FPR"]:
                        beta_values.append(f"{value:.3f}")
                    else:
                        beta_values.append(f"{value:.1f}")
                else:
                    beta_values.append("0.000" if metric in ["TPR", "FPR"] else "--")

            # Distance Metric columns
            dist_values = []
            dist_names = ["euclidean", "mahalanobis", "cosine", "chebyshev"]

            for dist in dist_names:
                col = f"Dist_{dist}"
                if col in metric_row:
                    value = metric_row[col]
                    if metric == "ADD" and value == 0.0:
                        dist_values.append("--")
                    elif metric in ["TPR", "FPR"]:
                        dist_values.append(f"{value:.3f}")
                    else:
                        dist_values.append(f"{value:.1f}")
                else:
                    dist_values.append("0.000" if metric in ["TPR", "FPR"] else "--")

            # Threshold columns
            thresh_values = []
            for threshold in ["20", "50", "100"]:
                col = f"Threshold_{threshold}"
                if col in metric_row:
                    value = metric_row[col]
                    if metric == "ADD" and value == 0.0:
                        thresh_values.append("--")
                    elif metric in ["TPR", "FPR"]:
                        thresh_values.append(f"{value:.3f}")
                    else:
                        thresh_values.append(f"{value:.1f}")
                else:
                    thresh_values.append("0.000" if metric in ["TPR", "FPR"] else "--")

            # Construct the complete row
            row = f"{row_start} & {power_values[0]} & {power_values[1]} & {power_values[2]} & {power_values[3]} & {mixture_str} & {beta_values[0]} & {beta_values[1]} & {beta_values[2]} & {dist_values[0]} & {dist_values[1]} & {dist_values[2]} & {dist_values[3]} & {thresh_values[0]} & {thresh_values[1]} & {thresh_values[2]} \\\\"

            latex_lines.append(row)

        latex_lines.append("\\hline")

    # Table footer
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)


def verify_mit_villes_inequality(experiment_data: Dict):
    """
    Verify that False Positive Rates respect Ville's inequality: FPR ≤ 1/λ
    for MIT Reality experiments.

    Args:
        experiment_data: Dictionary with all experiment results
    """
    print("\n" + "=" * 80)
    print("VERIFYING VILLE'S INEQUALITY FOR MIT REALITY: FPR ≤ 1/λ")
    print("=" * 80)

    # Define thresholds used in experiments
    thresholds = [20.0, 50.0, 100.0]

    violations = []
    total_checks = 0

    print(f"\nMIT REALITY DATASET:")
    print("-" * 40)

    # Check threshold experiments specifically
    threshold_experiments = experiment_data.get("threshold_analysis", [])

    for exp in threshold_experiments:
        threshold = float(exp["parameters"]["threshold"])
        metrics = exp["analysis"]["overall_metrics"]
        fpr = metrics["fpr"]

        # Ville's inequality bound
        ville_bound = 1.0 / threshold

        total_checks += 1

        # Check if FPR violates the bound
        if fpr > ville_bound:
            violations.append(
                {
                    "dataset": "mit_reality",
                    "threshold": threshold,
                    "fpr": fpr,
                    "ville_bound": ville_bound,
                    "violation": fpr - ville_bound,
                }
            )
            status = "❌ VIOLATION"
        else:
            status = "✅ SATISFIED"

        print(f"  λ={threshold:3.0f}: FPR={fpr:.4f} ≤ 1/λ={ville_bound:.4f} {status}")

    # Also check other experiment types with default threshold (60.0)
    default_threshold = 60.0
    ville_bound_default = 1.0 / default_threshold

    for group_name, group_experiments in experiment_data.items():
        if group_name == "threshold_analysis":
            continue  # Already checked above

        if group_experiments:
            # Average FPR for this group
            fprs = [
                exp["analysis"]["overall_metrics"]["fpr"] for exp in group_experiments
            ]
            avg_fpr = sum(fprs) / len(fprs)

            total_checks += 1

            if avg_fpr > ville_bound_default:
                violations.append(
                    {
                        "dataset": "mit_reality",
                        "group": group_name,
                        "threshold": default_threshold,
                        "fpr": avg_fpr,
                        "ville_bound": ville_bound_default,
                        "violation": avg_fpr - ville_bound_default,
                    }
                )
                status = "❌ VIOLATION"
            else:
                status = "✅ SATISFIED"

            print(
                f"  {group_name} (λ=60): FPR={avg_fpr:.4f} ≤ 1/λ={ville_bound_default:.4f} {status}"
            )

    # Summary
    print("\n" + "=" * 80)
    print("MIT REALITY VILLE'S INEQUALITY VERIFICATION SUMMARY")
    print("=" * 80)

    print(f"Total checks performed: {total_checks}")
    print(f"Violations found: {len(violations)}")
    print(
        f"Compliance rate: {((total_checks - len(violations)) / total_checks * 100):.1f}%"
    )

    if violations:
        print(f"\n❌ VIOLATIONS DETECTED:")
        print("-" * 50)
        for v in violations:
            if "group" in v:
                print(
                    f"  MIT Reality - {v['group']}: FPR={v['fpr']:.4f} > 1/λ={v['ville_bound']:.4f} (excess: {v['violation']:.4f})"
                )
            else:
                print(
                    f"  MIT Reality - λ={v['threshold']}: FPR={v['fpr']:.4f} > 1/λ={v['ville_bound']:.4f} (excess: {v['violation']:.4f})"
                )
    else:
        print(f"\n🎉 ALL CHECKS PASSED!")
        print(
            f"   False positive rates are well-controlled and respect Ville's inequality."
        )

    return violations


def analyze_mit_fpr_distribution(experiment_data: Dict):
    """
    Analyze the distribution of FPR values across all MIT Reality experiments.

    Args:
        experiment_data: Dictionary with all experiment results
    """
    print("\n" + "=" * 80)
    print("MIT REALITY FALSE POSITIVE RATE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    all_fprs = []
    fpr_by_threshold = {20.0: [], 50.0: [], 100.0: [], 60.0: []}  # 60.0 is default

    for group_name, group_experiments in experiment_data.items():
        for exp in group_experiments:
            fpr = exp["analysis"]["overall_metrics"]["fpr"]
            all_fprs.append(fpr)

            # Categorize by threshold
            if group_name == "threshold_analysis":
                threshold = float(exp["parameters"]["threshold"])
                if threshold in fpr_by_threshold:
                    fpr_by_threshold[threshold].append(fpr)
            else:
                # Default threshold for other experiments
                fpr_by_threshold[60.0].append(fpr)

    print(f"Total MIT Reality experiments analyzed: {len(all_fprs)}")
    print(f"Overall FPR statistics:")
    print(f"  Mean: {sum(all_fprs)/len(all_fprs):.4f}")
    print(f"  Min:  {min(all_fprs):.4f}")
    print(f"  Max:  {max(all_fprs):.4f}")

    # Count zero FPRs
    zero_fprs = sum(1 for fpr in all_fprs if fpr == 0.0)
    print(
        f"  Zero FPR experiments: {zero_fprs}/{len(all_fprs)} ({zero_fprs/len(all_fprs)*100:.1f}%)"
    )

    print(f"\nFPR by threshold:")
    for threshold, fprs in fpr_by_threshold.items():
        if fprs:
            ville_bound = 1.0 / threshold
            mean_fpr = sum(fprs) / len(fprs)
            max_fpr = max(fprs)
            violations = sum(1 for fpr in fprs if fpr > ville_bound)

            print(
                f"  λ={threshold:3.0f}: n={len(fprs):3d}, mean={mean_fpr:.4f}, max={max_fpr:.4f}, bound={ville_bound:.4f}, violations={violations}"
            )


def test_mit_metrics_on_sample_experiment():
    """
    Test our metrics calculation on a sample MIT Reality experiment to verify correctness.
    """
    print("🧪 TESTING MIT REALITY METRICS CALCULATION ON SAMPLE EXPERIMENT")
    print("=" * 80)

    # Get first experiment directory
    base_dir = "results/mit_reality_sensitivity_analysis"

    if not os.path.exists(base_dir):
        print(f"MIT Reality results directory not found: {base_dir}")
        return

    all_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not all_dirs:
        print("No MIT Reality experiment directories found!")
        return

    # Find an experiment with actual detections
    for dirname in all_dirs[:10]:  # Check first 10
        folder_path = os.path.join(base_dir, dirname)
        results_path = os.path.join(
            folder_path, "detection", "csv", "detection_results.xlsx"
        )

        if not os.path.exists(results_path):
            continue

        try:
            # Load detection details
            detection_details = pd.read_excel(
                results_path, sheet_name="Detection Details"
            )
            changepoint_metadata = pd.read_excel(
                results_path, sheet_name="ChangePointMetadata"
            )

            if len(detection_details) > 0:  # Found experiment with detections
                print(f"📁 Analyzing MIT Reality experiment: {dirname}")

                changepoints = changepoint_metadata["change_point"].tolist()
                print(f"📍 Expected changepoints: {changepoints}")
                print(f"📊 Total detections found: {len(detection_details)}")

                # Show sample of detection data
                print(f"\n📋 Sample detection data:")
                print(
                    detection_details[
                        [
                            "Trial",
                            "Detection Index",
                            "Nearest True CP",
                            "Distance to CP",
                            "Is Within 30 Steps",
                        ]
                    ]
                    .head(10)
                    .to_string()
                )

                # Test metrics for each changepoint
                print(f"\n🔍 DETECTION WINDOW ANALYSIS:")
                for cp in changepoints:
                    window_start, window_end = get_detection_window_for_changepoint(cp)
                    print(
                        f"   CP {cp}: detection window [{window_start}, {window_end}]"
                    )

                # Compute global FPR
                global_fpr = compute_global_fpr(detection_details, n_trials=10)
                print(f"\n📊 GLOBAL FPR: {global_fpr:.3f}")

                for cp in changepoints:
                    computed_metrics = compute_metrics_for_changepoint(
                        detection_details, cp, n_trials=10
                    )

                    print(f"\n🔍 METRICS FOR CHANGEPOINT {cp}:")
                    print(
                        f"   Detection window: [{computed_metrics['window_start']}, {computed_metrics['window_end']}]"
                    )
                    print(
                        f"   TPR: {computed_metrics['tpr']:.3f} ({computed_metrics['true_positives']}/{computed_metrics['total_trials']} trials)"
                    )
                    print(f"   ADD: {computed_metrics['add']:.2f}")
                    print(
                        f"   TP detections in window: {computed_metrics['total_tp_detections']}"
                    )
                    print(f"   FN trials: {computed_metrics['false_negatives']}")

                return  # Only test first experiment with detections

        except Exception as e:
            print(f"Error processing {dirname}: {e}")
            continue

    print("No MIT Reality experiments with detections found in first 10 directories")


def main():
    """Main function to create MIT Reality parameter sensitivity analysis."""
    print("Creating MIT Reality Parameter Sensitivity Analysis Table...")
    print("=" * 80)

    # First, test our metrics calculation
    print("Step 0: Verifying MIT Reality metrics calculation...")
    test_mit_metrics_on_sample_experiment()

    # Collect all experiment data
    print("\nStep 1: Collecting data from all MIT Reality experiments...")
    experiment_data = collect_all_mit_experiment_data()

    if not experiment_data:
        print("No MIT Reality experiment data found!")
        return

    # Verify Ville's inequality
    print("\nStep 2: Verifying Ville's inequality for MIT Reality (FPR ≤ 1/λ)...")
    violations = verify_mit_villes_inequality(experiment_data)
    analyze_mit_fpr_distribution(experiment_data)

    # Create the sensitivity table
    print("\nStep 3: Creating MIT Reality parameter sensitivity table...")
    sensitivity_df = create_mit_sensitivity_table(experiment_data)

    # Format and display the table
    print("\nStep 4: Formatting MIT Reality results...")
    formatted_table = format_mit_sensitivity_table_for_paper(sensitivity_df)
    latex_table = create_latex_mit_sensitivity_table(sensitivity_df)

    print("\n" + "=" * 80)
    print("MIT REALITY PARAMETER SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(formatted_table)

    # Save results to files
    output_dir = "results/mit_reality_sensitivity_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Save text version
    text_file = os.path.join(output_dir, "mit_reality_parameter_sensitivity_table.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(formatted_table)

    # Save LaTeX version
    latex_file = os.path.join(output_dir, "mit_reality_parameter_sensitivity_table.tex")
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    # Save CSV version for further analysis
    csv_file = os.path.join(output_dir, "mit_reality_parameter_sensitivity_table.csv")
    sensitivity_df.to_csv(csv_file, index=False)

    print(f"\nMIT Reality results saved to:")
    print(f"  Text version: {text_file}")
    print(f"  LaTeX version: {latex_file}")
    print(f"  CSV version: {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
