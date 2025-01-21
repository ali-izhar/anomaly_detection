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
    "sequence_length": [100, 200],  # Fixed length for 1 change
    "min_changes": [1],  # Fixed to 1 change
    "max_changes": [2],  # Fixed to 1 change
    "min_segment": [50],  # Fixed segment length
}

PREDICTION_PARAMS = {
    "prediction_window": [5, 10, 15],  # Two window sizes
    "min_history": [10],  # Fixed history length
}

DETECTION_PARAMS = {
    "martingale_threshold": [50.0, 100.0],  # Two threshold levels
    "martingale_epsilon": [0.5, 0.7, 0.9],  # Two epsilon values
}

N_RUNS = 2


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
        min_changes=network_params["min_changes"],
        max_changes=network_params["max_changes"],
    )

    # Create config with fixed seed for reproducibility
    config = ExperimentConfig(
        model=network_params["model"],
        params=graph_config["params"],
        min_history=prediction_params["min_history"],
        prediction_window=prediction_params["prediction_window"],
        martingale_threshold=detection_params["martingale_threshold"],
        martingale_epsilon=detection_params["martingale_epsilon"],
        n_runs=N_RUNS,
        save_individual=False,
        visualize_individual=False,
    )

    return config


def extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant metrics from experiment results."""
    metrics = {}

    try:
        # Handle both single and multiple run results
        if "ground_truth" in result:
            # Single run case
            actual_cps = [int(cp) for cp in result["ground_truth"]["change_points"]]
            metrics["actual_cps"] = actual_cps

            if "delays" in result:
                # Get raw delays
                detection_delays = [
                    float(d["mean"]) for d in result["delays"]["detection"].values()
                ]
                prediction_delays = [
                    float(d["mean"]) for d in result["delays"]["prediction"].values()
                ]

                # Calculate statistics
                metrics.update(
                    {
                        "avg_detection_delay": (
                            round(float(np.mean(detection_delays)), 2)
                            if detection_delays
                            else np.nan
                        ),
                        "std_detection_delay": (
                            round(float(np.std(detection_delays)), 2)
                            if detection_delays
                            else np.nan
                        ),
                        "avg_prediction_delay": (
                            round(float(np.mean(prediction_delays)), 2)
                            if prediction_delays
                            else np.nan
                        ),
                        "std_prediction_delay": (
                            round(float(np.std(prediction_delays)), 2)
                            if prediction_delays
                            else np.nan
                        ),
                    }
                )

                # Store per-CP delays
                metrics["delays_per_cp"] = {
                    int(cp): {
                        "detection": (
                            round(float(result["delays"]["detection"][cp]["mean"]), 2)
                            if cp in result["delays"]["detection"]
                            else np.nan
                        ),
                        "prediction": (
                            round(float(result["delays"]["prediction"][cp]["mean"]), 2)
                            if cp in result["delays"]["prediction"]
                            else np.nan
                        ),
                    }
                    for cp in actual_cps
                }
        else:
            # Multiple runs case
            if "all_results" in result:
                # Collect delays from all runs
                all_detection_delays = []
                all_prediction_delays = []
                all_delays_per_cp = {}

                for run_result in result["all_results"]:
                    if "delays" in run_result:
                        # Collect detection delays
                        detection_delays = [
                            float(d["mean"])
                            for d in run_result["delays"]["detection"].values()
                        ]
                        all_detection_delays.extend(detection_delays)

                        # Collect prediction delays
                        prediction_delays = [
                            float(d["mean"])
                            for d in run_result["delays"]["prediction"].values()
                        ]
                        all_prediction_delays.extend(prediction_delays)

                        # Collect per-CP delays
                        for cp, delays in run_result["delays"]["detection"].items():
                            if cp not in all_delays_per_cp:
                                all_delays_per_cp[cp] = {
                                    "detection": [],
                                    "prediction": [],
                                }
                            all_delays_per_cp[cp]["detection"].append(
                                float(delays["mean"])
                            )
                            if cp in run_result["delays"]["prediction"]:
                                all_delays_per_cp[cp]["prediction"].append(
                                    float(
                                        run_result["delays"]["prediction"][cp]["mean"]
                                    )
                                )

                # Calculate aggregate metrics
                metrics.update(
                    {
                        "avg_detection_delay": (
                            round(float(np.mean(all_detection_delays)), 2)
                            if all_detection_delays
                            else np.nan
                        ),
                        "std_detection_delay": (
                            round(float(np.std(all_detection_delays)), 2)
                            if all_detection_delays
                            else np.nan
                        ),
                        "avg_prediction_delay": (
                            round(float(np.mean(all_prediction_delays)), 2)
                            if all_prediction_delays
                            else np.nan
                        ),
                        "std_prediction_delay": (
                            round(float(np.std(all_prediction_delays)), 2)
                            if all_prediction_delays
                            else np.nan
                        ),
                    }
                )

                # Calculate per-CP averages
                metrics["delays_per_cp"] = {
                    int(cp): {
                        "detection": (
                            round(float(np.mean(delays["detection"])), 2)
                            if delays["detection"]
                            else np.nan
                        ),
                        "prediction": (
                            round(float(np.mean(delays["prediction"])), 2)
                            if delays["prediction"]
                            else np.nan
                        ),
                    }
                    for cp, delays in all_delays_per_cp.items()
                }

                # Get actual CPs from first run (they should be the same across runs)
                if result["all_results"]:
                    metrics["actual_cps"] = [
                        int(cp)
                        for cp in result["all_results"][0]["ground_truth"][
                            "change_points"
                        ]
                    ]

    except Exception as e:
        logger.error(f"Error extracting metrics: {str(e)}")
        metrics = {
            "actual_cps": [],
            "avg_detection_delay": np.nan,
            "std_detection_delay": np.nan,
            "avg_prediction_delay": np.nan,
            "std_prediction_delay": np.nan,
            "delays_per_cp": {},
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
        logger.info(f"\nRunning combination {i+1}/{total_combinations}")
        combination_dir = results_dir / f"combination_{i+1}"
        combination_dir.mkdir(exist_ok=True)

        try:
            # Create experiment config
            config = create_experiment_config(net_params, pred_params, det_params)

            # Create runner with fixed seed for reproducibility
            runner = ExperimentRunner(
                config=config, output_dir=combination_dir, seed=42
            )

            # Run experiment
            results = runner.run()
            metrics = extract_metrics(results)

            # Add parameters to metrics
            metrics.update(net_params)
            metrics.update(pred_params)
            metrics.update(det_params)
            all_results.append(metrics)

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
                        delays = row["delays_per_cp"].get(str(cp), {})
                        cp_stat = {
                            "actual": cp,
                            "detection": delays.get("detection", np.nan),
                            "prediction": delays.get("prediction", np.nan),
                        }
                        cp_stats.append(cp_stat)

                # Average CP timings
                avg_cp_stats = {}
                for cp in set([stat["actual"] for stat in cp_stats]):
                    relevant_stats = [s for s in cp_stats if s["actual"] == cp]
                    avg_cp_stats[cp] = {
                        "actual": cp,
                        "detection": np.nanmean(
                            [s["detection"] for s in relevant_stats]
                        ),
                        "prediction": np.nanmean(
                            [s["prediction"] for s in relevant_stats]
                        ),
                    }

                # Format CP timing information
                cp_timing_str = "; ".join(
                    [
                        f"CP{cp}: {stats['actual']}/{stats['detection']:.1f}/{stats['prediction']:.1f}"
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
