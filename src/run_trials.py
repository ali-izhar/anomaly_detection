# src/run_trials.py

"""Script for running multiple trials of network forecasting experiments with comprehensive data collection."""

import sys
from pathlib import Path
import argparse
import json
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from itertools import product

sys.path.append(str(Path(__file__).parent.parent))

from main import ExperimentRunner

logger = logging.getLogger(__name__)

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

MODEL_PREDICTOR_RECOMMENDATIONS = {
    "ba": ["weighted", "hybrid"],
    "ws": ["weighted", "hybrid"],
    "er": ["weighted", "hybrid"],
    "sbm": ["weighted", "hybrid"],
}


def parse_trial_args():
    """Parse command line arguments with extended parameter ranges."""
    parser = argparse.ArgumentParser(
        description="Multi-trial network prediction experiments"
    )

    # Basic parameters
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials per parameter combination",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=list(GRAPH_MODELS.keys()),
        default="ba",
        help="Type of network model",
    )
    parser.add_argument(
        "-p",
        "--predictor",
        type=str,
        choices=["weighted", "hybrid"],
        help="Type of predictor",
    )

    # Network parameters with ranges
    parser.add_argument(
        "--n-nodes-range",
        type=int,
        nargs=3,
        default=[50, 200, 50],
        help="Range for number of nodes [start, end, step]",
    )
    parser.add_argument(
        "--seq-len-range",
        type=int,
        nargs=3,
        default=[100, 300, 50],
        help="Range for sequence length [start, end, step]",
    )
    parser.add_argument(
        "--min-changes-range",
        type=int,
        nargs=3,
        default=[2, 5, 1],
        help="Range for min change points [start, end, step]",
    )
    parser.add_argument(
        "--max-changes-range",
        type=int,
        nargs=3,
        default=[2, 5, 1],
        help="Range for max change points [start, end, step]",
    )
    parser.add_argument(
        "--min-segment-range",
        type=int,
        nargs=3,
        default=[30, 70, 10],
        help="Range for min segment length [start, end, step]",
    )

    # Fixed parameters
    parser.add_argument(
        "-w", "--prediction-window", type=int, default=5, help="Prediction steps"
    )
    parser.add_argument(
        "-mh", "--min-history", type=int, default=10, help="Min history length"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()
    args.model = GRAPH_MODELS[args.model]

    if args.predictor is None:
        args.predictor = MODEL_PREDICTOR_RECOMMENDATIONS[args.model][0]
        logger.info(f"Using recommended predictor for {args.model}: {args.predictor}")

    return args


def generate_parameter_combinations(args):
    """Generate all parameter combinations to test."""
    n_nodes_range = range(
        args.n_nodes_range[0], args.n_nodes_range[1] + 1, args.n_nodes_range[2]
    )
    seq_len_range = range(
        args.seq_len_range[0], args.seq_len_range[1] + 1, args.seq_len_range[2]
    )
    min_changes_range = range(
        args.min_changes_range[0],
        args.min_changes_range[1] + 1,
        args.min_changes_range[2],
    )
    max_changes_range = range(
        args.max_changes_range[0],
        args.max_changes_range[1] + 1,
        args.max_changes_range[2],
    )
    min_segment_range = range(
        args.min_segment_range[0],
        args.min_segment_range[1] + 1,
        args.min_segment_range[2],
    )

    combinations = []
    for n, l, min_c, max_c, seg in product(
        n_nodes_range,
        seq_len_range,
        min_changes_range,
        max_changes_range,
        min_segment_range,
    ):
        if min_c <= max_c and seg * max_c < l:  # Ensure valid combinations
            combinations.append(
                {
                    "n_nodes": n,
                    "seq_len": l,
                    "min_changes": min_c,
                    "max_changes": max_c,
                    "min_segment": seg,
                }
            )

    return combinations


def extract_trial_metrics(results):
    """Extract comprehensive metrics from a single trial's results."""
    try:
        actual_martingales = results["actual_metrics"][1]
        pred_martingales = results["forecast_metrics"][2]
        change_points = results["ground_truth"]["change_points"]

        # Extract feature-wise metrics
        feature_metrics = {}
        for feature_type in ["degree", "clustering", "betweenness", "closeness"]:
            if feature_type in actual_martingales["reset"]:
                feature_metrics[feature_type] = {
                    "actual_values": actual_martingales["reset"][feature_type][
                        "martingales"
                    ],
                    "pred_values": pred_martingales["reset"][feature_type][
                        "martingales"
                    ],
                    "weight": actual_martingales["reset"][feature_type]["weight"],
                }

        # Compute detection metrics
        actual_delay, actual_false_alarms = compute_detection_metrics(
            actual_martingales, change_points, 50.0, 10
        )
        pred_delay, pred_false_alarms = compute_detection_metrics(
            pred_martingales, change_points, 50.0, 10
        )

        return {
            "actual_delay": actual_delay,
            "pred_delay": pred_delay,
            "actual_false_alarms": actual_false_alarms,
            "pred_false_alarms": pred_false_alarms,
            "change_points": change_points,
            "feature_metrics": feature_metrics,
        }

    except Exception as e:
        logger.error(f"Error in extract_trial_metrics: {str(e)}")
        raise


def plot_comprehensive_results(all_results, output_dir):
    """Create comprehensive visualization of results."""
    # Create results directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(all_results)

    # 1. Detection Delay Distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=pd.melt(
            df[["actual_delay", "pred_delay"]], value_name="Delay", var_name="Type"
        )
    )
    plt.title("Detection Delay Distribution")
    plt.savefig(plots_dir / "detection_delays.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. False Alarms vs Network Size
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="n_nodes", y="actual_false_alarms", label="Actual")
    sns.scatterplot(data=df, x="n_nodes", y="pred_false_alarms", label="Predicted")
    plt.title("False Alarms vs Network Size")
    plt.savefig(plots_dir / "false_alarms_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Detection Performance vs Sequence Length
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="seq_len", y="actual_delay", label="Actual Delay")
    sns.lineplot(data=df, x="seq_len", y="pred_delay", label="Predicted Delay")
    plt.title("Detection Performance vs Sequence Length")
    plt.savefig(plots_dir / "performance_vs_seqlen.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Feature-wise Performance
    if "feature_metrics" in df.columns:
        feature_df = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "weight": metrics["weight"],
                    "actual_avg": np.mean(metrics["actual_values"]),
                    "pred_avg": np.mean(metrics["pred_values"]),
                }
                for result in df["feature_metrics"]
                for feature, metrics in result.items()
            ]
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_df, x="feature", y="actual_avg", label="Actual")
        sns.barplot(data=feature_df, x="feature", y="pred_avg", label="Predicted")
        plt.title("Feature-wise Performance Comparison")
        plt.savefig(plots_dir / "feature_performance.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_comprehensive_results(all_results, output_dir):
    """Save detailed results and statistics."""
    results_df = pd.DataFrame(all_results)

    # Calculate summary statistics
    summary_stats = {
        "actual_delay": {
            "mean": results_df["actual_delay"].mean(),
            "std": results_df["actual_delay"].std(),
            "median": results_df["actual_delay"].median(),
        },
        "pred_delay": {
            "mean": results_df["pred_delay"].mean(),
            "std": results_df["pred_delay"].std(),
            "median": results_df["pred_delay"].median(),
        },
        "actual_false_alarms": {
            "mean": results_df["actual_false_alarms"].mean(),
            "std": results_df["actual_false_alarms"].std(),
        },
        "pred_false_alarms": {
            "mean": results_df["pred_false_alarms"].mean(),
            "std": results_df["pred_false_alarms"].std(),
        },
        "parameter_effects": {
            "n_nodes_correlation": results_df["n_nodes"].corr(
                results_df["actual_delay"]
            ),
            "seq_len_correlation": results_df["seq_len"].corr(
                results_df["actual_delay"]
            ),
        },
    }

    # Save results
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=4)


def extract_martingale_values(martingale_dict):
    """Extract and aggregate martingale values from the dictionary structure.

    The martingale dictionary has the structure:
    {
        'reset': {
            'feature1': {'martingales': [...], ...},
            'feature2': {'martingales': [...], ...},
            ...
        },
        'cumulative': {
            'feature1': {'martingales': [...], ...},
            'feature2': {'martingales': [...], ...},
            ...
        }
    }
    """
    if isinstance(martingale_dict, (list, np.ndarray)):
        return martingale_dict

    if isinstance(martingale_dict, dict):
        # Check if it's the top-level martingale dictionary
        if "reset" in martingale_dict and "cumulative" in martingale_dict:
            # Use reset martingales as they're more sensitive to changes
            reset_martingales = martingale_dict["reset"]

            # Aggregate martingales from all features
            feature_martingales = []
            for feature_data in reset_martingales.values():
                if isinstance(feature_data, dict) and "martingales" in feature_data:
                    feature_martingales.append(feature_data["martingales"])

            if feature_martingales:
                # Sum all feature martingales
                return np.sum(feature_martingales, axis=0)

        # Try common keys that might contain the values
        for key in ["martingales", "martingale_values", "values"]:
            if key in martingale_dict:
                return martingale_dict[key]

        # If no direct values found, try nested dictionaries
        for value in martingale_dict.values():
            if isinstance(value, dict):
                try:
                    return extract_martingale_values(value)
                except ValueError:
                    continue
            elif isinstance(value, (list, np.ndarray)):
                return value

    raise ValueError(
        f"Could not extract martingale values from structure: {type(martingale_dict)}"
    )


def compute_detection_metrics(martingales, change_points, threshold, min_history):
    """Compute detection delay and false alarms.

    Args:
        martingales: Dictionary or array of martingale values
        change_points: List of actual change point indices
        threshold: Detection threshold value
        min_history: Minimum history length before detection starts

    Returns:
        tuple: (average_delay, false_alarms)
            - average_delay: Mean delay in detecting true changes
            - false_alarms: Number of false positive detections
    """
    try:
        # Extract martingale values if it's a dictionary
        martingale_values = extract_martingale_values(martingales)
        logger.debug(
            f"Extracted martingale values shape: {np.array(martingale_values).shape}"
        )

        # Ensure it's a numpy array
        martingale_values = np.array(martingale_values)

        # Get detection points
        detections = np.where(martingale_values > threshold)[0]

        if len(detections) == 0:
            return float("inf"), 0  # No detections

        # Calculate detection delays
        delays = []
        false_alarms = 0
        detected_changes = set()

        for detection in detections:
            # Adjust detection time by min_history
            actual_time = detection + min_history

            # Find the closest change point
            if len(change_points) == 0:
                false_alarms += 1
                continue

            closest_cp = min(change_points, key=lambda x: abs(x - actual_time))

            # If detection is within a reasonable window of a change point (e.g., 20 steps)
            # and this change point hasn't been detected before
            if (
                abs(actual_time - closest_cp) <= 20
                and closest_cp not in detected_changes
            ):
                delays.append(max(0, actual_time - closest_cp))
                detected_changes.add(closest_cp)
            else:
                false_alarms += 1

        avg_delay = np.mean(delays) if delays else float("inf")
        return avg_delay, false_alarms

    except Exception as e:
        logger.error(f"Error in compute_detection_metrics: {str(e)}")
        logger.error(f"Martingales type: {type(martingales)}")
        if isinstance(martingales, dict):
            logger.error(f"Martingales keys: {martingales.keys()}")
            if "reset" in martingales:
                logger.error(f"Reset keys: {martingales['reset'].keys()}")
        raise


def main():
    """Main execution function for comprehensive parameter sweep experiments."""
    args = parse_trial_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("experiment_comprehensive.log"),
            logging.StreamHandler(),
        ],
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        f"results/comprehensive_{args.model}_{args.predictor}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(args)
    logger.info(f"Generated {len(param_combinations)} parameter combinations")

    # Store all results
    all_results = []

    # Run experiments for each parameter combination
    for params in tqdm(param_combinations, desc="Parameter combinations"):
        # Update args with current parameters
        for key, value in params.items():
            setattr(args, key, value)

        # Run trials for current parameter combination
        for trial in range(args.n_trials):
            try:
                trial_seed = args.seed + trial if args.seed is not None else trial
                trial_runner = ExperimentRunner(
                    args,
                    output_dir=output_dir
                    / f"params_{params['n_nodes']}_{params['seq_len']}"
                    / f"trial_{trial}",
                    seed=trial_seed,
                )

                results = trial_runner.run_single_experiment()
                metrics = extract_trial_metrics(results)

                # Add parameter information to metrics
                metrics.update(params)
                metrics["trial"] = trial
                all_results.append(metrics)

            except Exception as e:
                logger.error(
                    f"Trial failed for params {params}, trial {trial}: {str(e)}"
                )
                continue

    # Generate visualizations and save results
    plot_comprehensive_results(all_results, output_dir)
    save_comprehensive_results(all_results, output_dir)

    print(f"\nComprehensive results saved to: {output_dir}")
    print("Check experiment_comprehensive.log for detailed execution information.")


if __name__ == "__main__":
    main()
