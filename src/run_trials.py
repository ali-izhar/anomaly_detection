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
    """Parse command line arguments with fixed parameters."""
    parser = argparse.ArgumentParser(
        description="Multi-trial network prediction experiments"
    )

    # Basic parameters
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
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

    # Fixed network parameters
    parser.add_argument("-n", "--n-nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument(
        "-l", "--seq-len", type=int, default=100, help="Sequence length"
    )
    parser.add_argument(
        "--min-changes", type=int, default=1, help="Number of change points"
    )
    parser.add_argument(
        "--max-changes", type=int, default=1, help="Number of change points"
    )
    parser.add_argument(
        "-s", "--min-segment", type=int, default=30, help="Min segment length"
    )
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
    """Generate single parameter combination with fixed values."""
    return [
        {
            "n_nodes": args.n_nodes,
            "seq_len": args.seq_len,
            "min_changes": args.min_changes,
            "max_changes": args.max_changes,
            "min_segment": args.min_segment,
        }
    ]


def extract_trial_metrics(results):
    """Extract comprehensive metrics from a single trial's results."""
    try:
        actual_martingales = results["actual_metrics"][1]
        pred_martingales = results["forecast_metrics"][2]
        change_points = results["ground_truth"]["change_points"]
        if isinstance(change_points, np.ndarray):
            change_points = change_points.tolist()

        # Extract feature-wise metrics
        feature_metrics = {}
        for feature_type in ["degree", "clustering", "betweenness", "closeness"]:
            if feature_type in actual_martingales["reset"]:
                actual_values = actual_martingales["reset"][feature_type]["martingales"]
                pred_values = pred_martingales["reset"][feature_type]["martingales"]
                weight = actual_martingales["reset"][feature_type]["weight"]

                # Convert numpy arrays to lists
                if isinstance(actual_values, np.ndarray):
                    actual_values = actual_values.tolist()
                if isinstance(pred_values, np.ndarray):
                    pred_values = pred_values.tolist()
                if isinstance(weight, (np.floating, np.integer)):
                    weight = float(weight)

                feature_metrics[feature_type] = {
                    "actual_values": actual_values,
                    "pred_values": pred_values,
                    "weight": weight,
                }

        # Compute detection metrics
        actual_delay, actual_false_alarms = compute_detection_metrics(
            actual_martingales, change_points, 50.0, 10
        )
        pred_delay, pred_false_alarms = compute_detection_metrics(
            pred_martingales, change_points, 50.0, 10
        )

        # Convert any numpy types to Python types
        if isinstance(actual_delay, (np.floating, np.integer)):
            actual_delay = float(actual_delay)
        if isinstance(pred_delay, (np.floating, np.integer)):
            pred_delay = float(pred_delay)
        if isinstance(actual_false_alarms, (np.floating, np.integer)):
            actual_false_alarms = int(actual_false_alarms)
        if isinstance(pred_false_alarms, (np.floating, np.integer)):
            pred_false_alarms = int(pred_false_alarms)

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
        import traceback

        logger.error(traceback.format_exc())
        raise


def plot_comprehensive_results(all_results, output_dir):
    """Create comprehensive visualization of results."""
    # Create results directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in all_results]
    )

    # 1. Detection Delay Distribution
    plt.figure(figsize=(12, 6))
    delay_data = pd.DataFrame(
        {
            "Type": ["Actual"] * len(df) + ["Predicted"] * len(df),
            "Delay": list(df["actual_delay"]) + list(df["pred_delay"]),
        }
    )
    sns.boxplot(data=delay_data, x="Type", y="Delay")
    plt.title("Detection Delay Distribution")
    plt.savefig(plots_dir / "detection_delays.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. False Alarms Distribution
    plt.figure(figsize=(12, 6))
    false_alarms_data = pd.DataFrame(
        {
            "Type": ["Actual"] * len(df) + ["Predicted"] * len(df),
            "False Alarms": list(df["actual_false_alarms"])
            + list(df["pred_false_alarms"]),
        }
    )
    sns.boxplot(data=false_alarms_data, x="Type", y="False Alarms")
    plt.title("False Alarms Distribution")
    plt.savefig(plots_dir / "false_alarms.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Feature-wise Performance
    feature_data = []
    for result in all_results:
        if "feature_metrics" in result:
            for feature, metrics in result["feature_metrics"].items():
                try:
                    # Handle ragged arrays by flattening all values
                    actual_values = [
                        val
                        for sublist in metrics["actual_values"]
                        for val in (
                            sublist
                            if isinstance(sublist, (list, np.ndarray))
                            else [sublist]
                        )
                    ]
                    pred_values = [
                        val
                        for sublist in metrics["pred_values"]
                        for val in (
                            sublist
                            if isinstance(sublist, (list, np.ndarray))
                            else [sublist]
                        )
                    ]

                    # Convert to numpy arrays and filter finite values
                    actual_values = np.array(
                        [v for v in actual_values if np.isfinite(float(v))]
                    )
                    pred_values = np.array(
                        [v for v in pred_values if np.isfinite(float(v))]
                    )

                    if len(actual_values) > 0 and len(pred_values) > 0:
                        feature_data.append(
                            {
                                "Feature": feature,
                                "Weight": float(metrics["weight"]),
                                "Actual": float(np.mean(actual_values)),
                                "Predicted": float(np.mean(pred_values)),
                                "Trial": result.get("trial", 0),
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Error processing feature {feature} in trial {result.get('trial', 0)}: {str(e)}"
                    )
                    continue

    if feature_data:
        feature_df = pd.DataFrame(feature_data)

        # Plot average feature values
        plt.figure(figsize=(12, 6))
        feature_means = feature_df.groupby("Feature")[["Actual", "Predicted"]].mean()
        feature_means.plot(
            kind="bar",
            yerr=feature_df.groupby("Feature")[["Actual", "Predicted"]].std(),
        )
        plt.title("Feature-wise Performance Comparison")
        plt.xlabel("Feature")
        plt.ylabel("Average Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_performance.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Plot feature weights
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_df, x="Feature", y="Weight")
        plt.title("Feature Weights")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_weights.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Trial Summary
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["trial"], df["actual_delay"], "b-", label="Actual")
    plt.plot(df["trial"], df["pred_delay"], "r--", label="Predicted")
    plt.title("Detection Delays Across Trials")
    plt.xlabel("Trial")
    plt.ylabel("Delay")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["trial"], df["actual_false_alarms"], "b-", label="Actual")
    plt.plot(df["trial"], df["pred_false_alarms"], "r--", label="Predicted")
    plt.title("False Alarms Across Trials")
    plt.xlabel("Trial")
    plt.ylabel("False Alarms")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "trial_summary.png", dpi=300, bbox_inches="tight")
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


def serialize_for_json(obj):
    """Helper function to serialize objects for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__"):  # Handle custom objects
        return {
            "class": obj.__class__.__name__,
            "attributes": serialize_for_json(obj.__dict__),
        }
    return obj


class MultiTrialExperimentRunner:
    """Class to handle multiple trials with consistent parameters."""

    def __init__(self, args, output_dir):
        """Initialize runner with fixed parameters."""
        self.args = args
        self.output_dir = output_dir
        self.base_seed = (
            args.seed if args.seed is not None else np.random.randint(0, 10000)
        )

        # Initialize single experiment runner with fixed parameters
        self.experiment_runner = ExperimentRunner(
            args, output_dir=output_dir, seed=self.base_seed
        )

        # Store initial network parameters
        self.initial_params = self.experiment_runner.config
        logger.info(f"Initialized with parameters: {self.initial_params}")

    def run_trials(self, n_trials):
        """Run multiple trials with the same parameters."""
        all_results = []

        # Save experiment configuration first
        config = {
            "model": self.args.model,
            "predictor": self.args.predictor,
            "parameters": serialize_for_json(self.initial_params),
            "base_seed": self.base_seed,
            "args": serialize_for_json(vars(self.args)),
        }
        with open(self.output_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=4)

        for trial in tqdm(range(n_trials), desc="Running trials"):
            try:
                # Create trial directory
                trial_dir = self.output_dir / f"trial_{trial}"
                trial_dir.mkdir(parents=True, exist_ok=True)

                # Update seed for this trial while keeping other parameters fixed
                trial_seed = self.base_seed + trial
                self.experiment_runner.seed = trial_seed

                # Run single trial
                results = self.experiment_runner.run_single_experiment()

                # Extract and process metrics
                metrics = extract_trial_metrics(results)

                # Add trial information
                metrics.update(
                    {
                        "trial": trial,
                        "seed": trial_seed,
                        "n_nodes": self.args.n_nodes,
                        "seq_len": self.args.seq_len,
                        "min_changes": self.args.min_changes,
                        "max_changes": self.args.max_changes,
                        "min_segment": self.args.min_segment,
                    }
                )

                # Save individual trial results with proper serialization
                serialized_metrics = serialize_for_json(metrics)
                with open(trial_dir / "trial_results.json", "w") as f:
                    json.dump(serialized_metrics, f, indent=4)

                all_results.append(metrics)

            except Exception as e:
                logger.error(f"Trial {trial} failed: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        return all_results


def main():
    """Main execution function for multiple trials with consistent parameters."""
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
    output_dir = Path(f"results/experiment_{args.model}_{args.predictor}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize multi-trial runner
    multi_runner = MultiTrialExperimentRunner(args, output_dir)

    # Run all trials
    all_results = multi_runner.run_trials(args.n_trials)

    if not all_results:
        logger.error(
            "All trials failed. Check experiment_comprehensive.log for details."
        )
        return

    # Generate visualizations and save results
    plot_comprehensive_results(all_results, output_dir)
    save_comprehensive_results(all_results, output_dir)

    print(f"\nResults saved to: {output_dir}")
    print("Check experiment_comprehensive.log for detailed execution information.")


if __name__ == "__main__":
    main()
