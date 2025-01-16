# src/run_trials.py

"""Script for running multiple trials of network forecasting experiments."""

import sys
from pathlib import Path
import argparse
import json
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from main import ExperimentRunner, convert_to_serializable

logger = logging.getLogger(__name__)

GRAPH_MODELS = {
    "barabasi_albert": "ba",
    "watts_strogatz": "ws",
    "erdos_renyi": "er",
    "stochastic_block_model": "sbm",
    "random_core_periphery": "rcp",
    "lfr_benchmark": "lfr",
    "ba": "ba",
    "ws": "ws",
    "er": "er",
    "sbm": "sbm",
    "rcp": "rcp",
    "lfr": "lfr",
}

MODEL_PREDICTOR_RECOMMENDATIONS = {
    "ba": ["weighted", "hybrid"],
    "ws": ["weighted", "hybrid"],
    "er": ["weighted", "hybrid"],
    "sbm": ["weighted", "hybrid"],
    "rcp": ["weighted", "hybrid"],
    "lfr": ["weighted", "hybrid"],
}


def parse_trial_args():
    """Parse command line arguments, extending the original parser."""
    parser = argparse.ArgumentParser(
        description="Multi-trial network prediction experiments"
    )

    # Add number of trials parameter first
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of trials to run"
    )

    # Add model argument
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=list(GRAPH_MODELS.keys()),
        default="ba",
        help="Type of network model",
    )

    # Add predictor argument
    parser.add_argument(
        "-p",
        "--predictor",
        type=str,
        choices=["weighted", "hybrid"],
        help="Type of predictor",
    )

    # Add other arguments
    parser.add_argument("-n", "--n-nodes", type=int, default=50, help="Number of nodes")
    parser.add_argument(
        "-l", "--seq-len", type=int, default=100, help="Sequence length"
    )
    parser.add_argument("--min-changes", type=int, default=2, help="Min change points")
    parser.add_argument("--max-changes", type=int, default=2, help="Max change points")
    parser.add_argument(
        "-s", "--min-segment", type=int, default=50, help="Min segment length"
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
    """Compute detection delay and false alarms."""
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


def extract_trial_metrics(results):
    """Extract detection metrics from a single trial's results."""
    try:
        actual_martingales = results["actual_metrics"][
            1
        ]  # Index 1 contains martingales
        pred_martingales = results["forecast_metrics"][
            2
        ]  # Index 2 contains martingales
        change_points = results["ground_truth"]["change_points"]

        logger.debug(f"Actual martingales type: {type(actual_martingales)}")
        logger.debug(
            f"Actual martingales structure: {actual_martingales.keys() if isinstance(actual_martingales, dict) else 'not a dict'}"
        )
        logger.debug(f"Pred martingales type: {type(pred_martingales)}")
        logger.debug(
            f"Pred martingales structure: {pred_martingales.keys() if isinstance(pred_martingales, dict) else 'not a dict'}"
        )

        # Compute detection metrics
        actual_delay, actual_false_alarms = compute_detection_metrics(
            actual_martingales,
            change_points,
            50.0,  # THRESHOLD
            10,  # min_history
        )

        pred_delay, pred_false_alarms = compute_detection_metrics(
            pred_martingales,
            change_points,
            50.0,  # THRESHOLD
            10,  # min_history
        )

        return {
            "actual_delay": actual_delay,
            "pred_delay": pred_delay,
            "actual_false_alarms": actual_false_alarms,
            "pred_false_alarms": pred_false_alarms,
            "change_points": change_points,
        }

    except Exception as e:
        logger.error(f"Error in extract_trial_metrics: {str(e)}")
        logger.error(f"Results keys: {results.keys()}")
        logger.error(f"Actual metrics type: {type(results['actual_metrics'])}")
        logger.error(f"Forecast metrics type: {type(results['forecast_metrics'])}")
        raise


def plot_aggregate_results(metrics, output_dir):
    """Plot aggregate results across all trials."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Detection Delays
    plt.subplot(2, 1, 1)
    data = [
        [m["actual_delay"] for m in metrics if m["actual_delay"] != float("inf")],
        [m["pred_delay"] for m in metrics if m["pred_delay"] != float("inf")],
    ]
    plt.boxplot(data, labels=["Actual", "Predicted"])
    plt.title("Detection Delays Across Trials")
    plt.ylabel("Time Steps")

    # Plot 2: False Alarms
    plt.subplot(2, 1, 2)
    data = [
        [m["actual_false_alarms"] for m in metrics],
        [m["pred_false_alarms"] for m in metrics],
    ]
    plt.boxplot(data, labels=["Actual", "Predicted"])
    plt.title("False Alarms Across Trials")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_results.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_aggregate_results(metrics, output_dir):
    """Save aggregate statistics to a JSON file."""
    stats = {
        "actual_delay": {
            "mean": float(
                np.mean(
                    [
                        m["actual_delay"]
                        for m in metrics
                        if m["actual_delay"] != float("inf")
                    ]
                )
            ),
            "std": float(
                np.std(
                    [
                        m["actual_delay"]
                        for m in metrics
                        if m["actual_delay"] != float("inf")
                    ]
                )
            ),
        },
        "pred_delay": {
            "mean": float(
                np.mean(
                    [
                        m["pred_delay"]
                        for m in metrics
                        if m["pred_delay"] != float("inf")
                    ]
                )
            ),
            "std": float(
                np.std(
                    [
                        m["pred_delay"]
                        for m in metrics
                        if m["pred_delay"] != float("inf")
                    ]
                )
            ),
        },
        "actual_false_alarms": {
            "mean": float(np.mean([m["actual_false_alarms"] for m in metrics])),
            "std": float(np.std([m["actual_false_alarms"] for m in metrics])),
        },
        "pred_false_alarms": {
            "mean": float(np.mean([m["pred_false_alarms"] for m in metrics])),
            "std": float(np.std([m["pred_false_alarms"] for m in metrics])),
        },
        "raw_metrics": convert_to_serializable(
            metrics
        ),  # Save raw metrics for reference
    }

    with open(output_dir / "aggregate_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def main():
    """Main execution function for multiple trials."""
    args = parse_trial_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("experiment_debug.log"), logging.StreamHandler()],
    )

    # Create output directory for multiple trials
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/multi_trial_{args.model}_{args.predictor}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run trials
    trial_metrics = []
    for trial in tqdm(range(args.n_trials), desc="Running trials"):
        try:
            # Create runner for this trial with a unique seed
            trial_seed = args.seed + trial if args.seed is not None else trial
            trial_runner = ExperimentRunner(
                args, output_dir=output_dir / f"trial_{trial}", seed=trial_seed
            )

            # Run trial
            results = trial_runner.run_single_experiment()

            # Extract metrics
            metrics = extract_trial_metrics(results)
            trial_metrics.append(metrics)

            # Save individual trial results
            trial_runner.save_results(results)

        except Exception as e:
            logger.error(f"Trial {trial} failed: {str(e)}")
            continue

    if not trial_metrics:
        logger.error("All trials failed. Check experiment_debug.log for details.")
        return

    # Plot and save aggregate results
    plot_aggregate_results(trial_metrics, output_dir)
    save_aggregate_results(trial_metrics, output_dir)

    print(f"\nAggregate results saved to: {output_dir}")
    print("Check experiment_debug.log for detailed execution information.")


if __name__ == "__main__":
    main()
