"""Script to run parameter studies for network prediction experiments."""

import itertools
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any
import numpy as np
from datetime import datetime
import json

from runner import ExperimentRunner, ExperimentConfig
from config.graph_configs import GRAPH_CONFIGS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Parameter combinations to test
NETWORK_PARAMS = {
    "model": ["sbm"],  # Fixed to SBM
    "nodes": [50],  # Fixed moderate size
    "sequence_length": [100],  # Fixed length for 1 change
    "min_changes": [1],  # Fixed to 1 change
    "max_changes": [1],  # Fixed to 1 change
    "min_segment": [50],  # Fixed segment length
}

PREDICTION_PARAMS = {
    "prediction_window": [3],  # Two window sizes
    "min_history": [10],  # Fixed history length
}

DETECTION_PARAMS = {
    "martingale_threshold": [30.0],  # Two threshold levels
    "martingale_epsilon": [0.7],  # Two epsilon values
}


def generate_parameter_combinations():
    """Generate all combinations of parameters to test."""
    # Generate network parameter combinations
    network_combinations = [
        dict(zip(NETWORK_PARAMS.keys(), v))
        for v in itertools.product(*NETWORK_PARAMS.values())
    ]

    # Generate prediction parameter combinations
    prediction_combinations = [
        dict(zip(PREDICTION_PARAMS.keys(), v))
        for v in itertools.product(*PREDICTION_PARAMS.values())
    ]

    # Generate detection parameter combinations
    detection_combinations = [
        dict(zip(DETECTION_PARAMS.keys(), v))
        for v in itertools.product(*DETECTION_PARAMS.values())
    ]

    return network_combinations, prediction_combinations, detection_combinations


def create_experiment_config(
    network_params: Dict[str, Any],
    prediction_params: Dict[str, Any],
    detection_params: Dict[str, Any],
) -> ExperimentConfig:
    """Create ExperimentConfig from parameter combinations."""
    # Get graph configuration
    graph_config = GRAPH_CONFIGS[network_params["model"]](
        n=network_params["nodes"],
        seq_len=network_params["sequence_length"],
        min_segment=network_params["min_segment"],
        min_changes=(
            1 if network_params["sequence_length"] == 100 else 2
        ),  # Force changes based on sequence length
        max_changes=(
            1 if network_params["sequence_length"] == 100 else 2
        ),  # Force changes based on sequence length
    )

    # Create config
    config = ExperimentConfig(
        model=network_params["model"],
        params=graph_config["params"],
        min_history=prediction_params["min_history"],
        prediction_window=prediction_params["prediction_window"],
        martingale_threshold=detection_params["martingale_threshold"],
        martingale_epsilon=detection_params["martingale_epsilon"],
        n_runs=1,  # Run each combination 5 times for statistical significance
        save_individual=False,
        visualize_individual=False,
    )

    return config


def extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant metrics from experiment results."""
    metrics = {}

    try:
        # Get actual change points - convert to native Python list
        actual_cps = [int(cp) for cp in result["ground_truth"]["change_points"]]
        metrics["actual_cps"] = actual_cps

        # Get detection and prediction delays
        if "delays" in result:
            detection_delays = [
                float(d["mean"]) for d in result["delays"]["detection"].values()
            ]
            prediction_delays = [
                float(d["mean"]) for d in result["delays"]["prediction"].values()
            ]

            # Store all delays and their corresponding CPs
            metrics.update(
                {
                    "avg_detection_delay": (
                        float(np.mean(detection_delays)) if detection_delays else np.nan
                    ),
                    "std_detection_delay": (
                        float(np.std(detection_delays)) if detection_delays else np.nan
                    ),
                    "avg_prediction_delay": (
                        float(np.mean(prediction_delays))
                        if prediction_delays
                        else np.nan
                    ),
                    "std_prediction_delay": (
                        float(np.std(prediction_delays))
                        if prediction_delays
                        else np.nan
                    ),
                    "detection_delays_per_cp": {
                        int(cp): {"mean": float(d["mean"]), "std": float(d["std"])}
                        for cp, d in result["delays"]["detection"].items()
                    },
                    "prediction_delays_per_cp": {
                        int(cp): {"mean": float(d["mean"]), "std": float(d["std"])}
                        for cp, d in result["delays"]["prediction"].items()
                    },
                }
            )

            # Calculate average detection and prediction times for each CP
            detection_times = {}
            prediction_times = {}

            for cp in actual_cps:
                cp = int(cp)  # Ensure CP is an integer
                if cp in result["delays"]["detection"]:
                    det_delay = float(result["delays"]["detection"][cp]["mean"])
                    detection_times[cp] = float(cp + det_delay)
                else:
                    detection_times[cp] = np.nan

                if cp in result["delays"]["prediction"]:
                    pred_delay = float(result["delays"]["prediction"][cp]["mean"])
                    prediction_times[cp] = float(cp + pred_delay)
                else:
                    prediction_times[cp] = np.nan

            metrics.update(
                {
                    "actual_cp_times": actual_cps,
                    "detection_times": detection_times,
                    "prediction_times": prediction_times,
                }
            )

        # Get distribution analysis metrics
        if "distribution_analysis" in result:
            dist_analysis = result["distribution_analysis"]
            metrics.update(
                {
                    "kl_div_hist": float(dist_analysis["kl_div_hist"]),
                    "kl_div_kde": float(dist_analysis["kl_div_kde"]),
                    "js_div": float(dist_analysis["js_div"]),
                    "correlation": float(dist_analysis["correlation"]),
                }
            )
        else:
            # If distribution analysis is not available, add default values
            metrics.update(
                {
                    "kl_div_hist": np.nan,
                    "kl_div_kde": np.nan,
                    "js_div": np.nan,
                    "correlation": np.nan,
                }
            )

        # Debug log the extracted metrics
        logger.debug(f"Extracted metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error extracting metrics: {str(e)}")
        metrics = {
            "actual_cps": [],
            "avg_detection_delay": np.nan,
            "std_detection_delay": np.nan,
            "avg_prediction_delay": np.nan,
            "std_prediction_delay": np.nan,
            "detection_delays_per_cp": {},
            "prediction_delays_per_cp": {},
            "actual_cp_times": [],
            "detection_times": {},
            "prediction_times": {},
            "kl_div_hist": np.nan,
            "kl_div_kde": np.nan,
            "js_div": np.nan,
            "correlation": np.nan,
        }

    return metrics


def run_parameter_study():
    """Run the parameter study and collect results."""
    # Generate parameter combinations
    network_combinations, prediction_combinations, detection_combinations = (
        generate_parameter_combinations()
    )

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = Path("results")
    base_results_dir.mkdir(
        exist_ok=True
    )  # Create base results directory if it doesn't exist

    results_dir = base_results_dir / f"parameter_study_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store all results
    all_results = []
    total_combinations = (
        len(network_combinations)
        * len(prediction_combinations)
        * len(detection_combinations)
    )

    logger.info(f"Starting parameter study with {total_combinations} combinations")

    for i, (net_params, pred_params, det_params) in enumerate(
        itertools.product(
            network_combinations, prediction_combinations, detection_combinations
        )
    ):

        # Create combination directory
        combination_dir = results_dir / f"combination_{i+1}"
        combination_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = create_experiment_config(net_params, pred_params, det_params)

        # Log progress
        logger.info(f"\nRunning combination {i+1}/{total_combinations}")
        logger.info(f"Network params: {net_params}")
        logger.info(f"Prediction params: {pred_params}")
        logger.info(f"Detection params: {det_params}")

        try:
            # Run experiment
            runner = ExperimentRunner(config=config, output_dir=combination_dir)
            result = runner.run()

            # Extract metrics
            metrics = extract_metrics(result)

            # Store results
            experiment_result = {
                **net_params,
                **pred_params,
                **det_params,
                **metrics,
                "distribution_analysis": result.get("distribution_analysis", {}),
            }
            all_results.append(experiment_result)

            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(results_dir / "parameter_study_results.csv", index=False)

            # Save individual combination results
            with open(combination_dir / "combination_params.json", "w") as f:
                json.dump(
                    {
                        "network_params": net_params,
                        "prediction_params": pred_params,
                        "detection_params": det_params,
                        "metrics": metrics,
                        "distribution_analysis": result.get(
                            "distribution_analysis", {}
                        ),
                    },
                    f,
                    indent=4,
                )

        except Exception as e:
            logger.error(f"Error running combination {i+1}: {str(e)}")
            # Save error information
            with open(combination_dir / "error.txt", "w") as f:
                f.write(f"Error: {str(e)}")
            continue

    # Create final DataFrame
    df = pd.DataFrame(all_results)

    if not df.empty:
        # Save final results
        df.to_csv(results_dir / "parameter_study_results.csv", index=False)
        logger.info(f"\nParameter study completed. Results saved to {results_dir}")
    else:
        logger.error("No results were generated!")

    return df


def analyze_results(df: pd.DataFrame):
    """Analyze the parameter study results with focus on key parameters."""
    if df.empty:
        logger.warning("No results to analyze!")
        return pd.DataFrame()

    logger.info(f"Available columns in results: {df.columns.tolist()}")

    # Focus on key parameters
    key_params = {
        "sequence_length": "Sequence Length",
        "prediction_window": "Prediction Window",
        "martingale_threshold": "Detection Threshold",
        "martingale_epsilon": "Martingale Epsilon",
    }

    summary_table = []

    for param, param_name in key_params.items():
        if param not in df.columns:
            logger.warning(f"Parameter {param} not found in results, skipping...")
            continue

        try:
            grouped = (
                df.groupby(param)
                .agg(
                    {
                        "avg_detection_delay": ["mean", "std"],
                        "avg_prediction_delay": ["mean", "std"],
                        "kl_div_hist": "mean",
                        "kl_div_kde": "mean",
                        "js_div": "mean",
                        "correlation": "mean",
                    }
                )
                .round(3)
            )

            for value in grouped.index:
                # Get subset of data for this parameter value
                subset = df[df[param] == value]

                # Calculate CP timing statistics
                cp_stats = []
                for _, row in subset.iterrows():
                    for cp in row["actual_cps"]:
                        cp_stat = {
                            "actual": cp,
                            "detected": row["detection_times"].get(cp, np.nan),
                            "predicted": row["prediction_times"].get(cp, np.nan),
                        }
                        cp_stats.append(cp_stat)

                # Average CP timings
                avg_cp_stats = {}
                for cp in set([stat["actual"] for stat in cp_stats]):
                    relevant_stats = [s for s in cp_stats if s["actual"] == cp]
                    avg_cp_stats[cp] = {
                        "actual": cp,
                        "detected": np.nanmean([s["detected"] for s in relevant_stats]),
                        "predicted": np.nanmean(
                            [s["predicted"] for s in relevant_stats]
                        ),
                    }

                # Format CP timing information
                cp_timing_str = "; ".join(
                    [
                        f"CP{cp}: {stats['actual']}/{stats['detected']:.1f}/{stats['predicted']:.1f}"
                        for cp, stats in avg_cp_stats.items()
                    ]
                )

                if param == "sequence_length":
                    value_display = f"{value} ({1 if value == 100 else 2} changes)"
                else:
                    value_display = value

                row = {
                    "Parameter": param_name,
                    "Value": value_display,
                    "CP Timings (Actual/Detected/Predicted)": cp_timing_str,
                    "Detection Delay": f"{grouped.loc[value, ('avg_detection_delay', 'mean')]:.2f} ± {grouped.loc[value, ('avg_detection_delay', 'std')]:.2f}",
                    "Prediction Delay": f"{grouped.loc[value, ('avg_prediction_delay', 'mean')]:.2f} ± {grouped.loc[value, ('avg_prediction_delay', 'std')]:.2f}",
                    "KL Div (Hist)": f"{grouped.loc[value, ('kl_div_hist', 'mean')]:.3f}",
                    "KL Div (KDE)": f"{grouped.loc[value, ('kl_div_kde', 'mean')]:.3f}",
                    "JS Div": f"{grouped.loc[value, ('js_div', 'mean')]:.3f}",
                    "Correlation": f"{grouped.loc[value, ('correlation', 'mean')]:.3f}",
                }
                summary_table.append(row)

        except Exception as e:
            logger.error(f"Error analyzing parameter {param}: {str(e)}")
            continue

    return pd.DataFrame(summary_table)


if __name__ == "__main__":
    # Run parameter study
    results_df = run_parameter_study()

    # Create publication-ready analysis
    summary_table = analyze_results(results_df)

    if not summary_table.empty:
        # Print summary table
        print("\nParameter Study Summary:")
        print(summary_table.to_string(index=False))
    else:
        print("\nNo results to display.")
