# src/run.py

"""
Main script to run the forecast-based martingale detection algorithm on network sequences.
Implements the experimental setup from Section 6 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

Usage:
    python src/run.py <model_alias>
"""

import sys
import argparse
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.algorithm import run_forecast_martingale_detection
from src.configs.loader import get_config

from src.changepoint.visualizer import MartingaleVisualizer
from src.graph.generator import GraphGenerator

# from src.graph.visualizer import NetworkVisualizer


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Default parameters
DEFAULT_PARAMS = {
    "horizon": 5,
    "threshold": 60.0,
    "epsilon": 0.7,
    "window_size": 10,
    "predictor_type": "adaptive",
}


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def run_detection(model_alias: str, output_dir: str = "results"):
    """Run the forecast-based martingale detection algorithm on a network sequence."""
    # 1. Setup
    model_name = get_full_model_name(model_alias)
    logger.info(f"Running detection on {model_name} network sequence...")

    # 2. Generate Network Sequence
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__

    # test: minimize params
    params.update(
        {
            "n": 30,
            "seq_len": 50,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 20,
        }
    )

    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    true_change_points = result["change_points"]

    # 3. Run Detection Algorithm
    logger.info("Running forecast-based martingale detection...")
    pbar = tqdm(total=len(graphs), desc="Running detection", unit="step")

    detection_results = run_forecast_martingale_detection(
        graph_sequence=[nx.from_numpy_array(g) for g in graphs],
        horizon=DEFAULT_PARAMS["horizon"],
        threshold=DEFAULT_PARAMS["threshold"],
        epsilon=DEFAULT_PARAMS["epsilon"],
        window_size=DEFAULT_PARAMS["window_size"],
        predictor_type=DEFAULT_PARAMS["predictor_type"],
        random_state=42,
        progress_callback=lambda t: pbar.update(1),
    )

    pbar.close()

    # 4. Analyze Results
    traditional_changes = detection_results["traditional_changes"]
    horizon_changes = detection_results["horizon_changes"]
    detected_points = detection_results["change_points"]

    logger.info(f"\nResults Summary:")
    logger.info(f"True change points: {true_change_points}")
    logger.info(f"Traditional martingale detections: {traditional_changes}")
    logger.info(f"Horizon martingale detections: {horizon_changes}")
    logger.info(f"Combined detected points: {detected_points}")

    if true_change_points:
        # Analyze traditional martingale performance
        if traditional_changes:
            trad_errors = []
            for true_cp in true_change_points:
                closest_trad = min(traditional_changes, key=lambda x: abs(x - true_cp))
                error = abs(closest_trad - true_cp)
                trad_errors.append(error)
            avg_trad_error = np.mean(trad_errors)
            logger.info(
                f"Traditional martingale average detection delay: {avg_trad_error:.2f} time steps"
            )

        # Analyze horizon martingale performance
        if horizon_changes:
            horizon_errors = []
            for true_cp in true_change_points:
                closest_horizon = min(horizon_changes, key=lambda x: abs(x - true_cp))
                error = abs(closest_horizon - true_cp)
                horizon_errors.append(error)
            avg_horizon_error = np.mean(horizon_errors)
            logger.info(
                f"Horizon martingale average detection delay: {avg_horizon_error:.2f} time steps"
            )

        # Compare detection speeds
        if traditional_changes and horizon_changes:
            for true_cp in true_change_points:
                closest_trad = min(traditional_changes, key=lambda x: abs(x - true_cp))
                closest_horizon = min(horizon_changes, key=lambda x: abs(x - true_cp))
                if closest_horizon < closest_trad:
                    speedup = closest_trad - closest_horizon
                    logger.info(
                        f"Horizon martingale detected change at t={true_cp} faster by {speedup} steps"
                    )
                elif closest_horizon > closest_trad:
                    delay = closest_horizon - closest_trad
                    logger.info(
                        f"Traditional martingale was faster at t={true_cp} by {delay} steps"
                    )
                else:
                    logger.info(
                        f"Both martingales detected change at t={true_cp} simultaneously"
                    )

    # # 5. Visualize Results
    # logger.info("Creating visualizations...")

    # # Prepare feature martingales for visualization
    # feature_names = [
    #     "degree",
    #     "density",
    #     "clustering",
    #     "betweenness",
    #     "eigenvector",
    #     "closeness",
    #     "singular_value",
    #     "laplacian",
    # ]

    # # Create martingales dictionary for visualization
    # feature_martingales = {}
    # for i, feature in enumerate(feature_names):
    #     feature_martingales[feature] = {
    #         "martingales": detection_results["individual_martingales_obs"][i],
    #         "p_values": [1.0]
    #         * len(detection_results["individual_martingales_obs"][i]),  # Placeholder
    #         "strangeness": [0.0]
    #         * len(detection_results["individual_martingales_obs"][i]),  # Placeholder
    #     }

    # # Add combined martingales
    # feature_martingales["combined"] = {
    #     "martingales": detection_results["M_observed"],
    #     "p_values": [1.0] * len(detection_results["M_observed"]),  # Placeholder
    #     "strangeness": [0.0] * len(detection_results["M_observed"]),  # Placeholder
    #     "martingale_sum": detection_results["M_observed"],
    #     "martingale_avg": detection_results["M_predicted"],
    # }

    # # Create visualizations using MartingaleVisualizer
    # martingale_viz = MartingaleVisualizer(
    #     martingales=feature_martingales,
    #     change_points=true_change_points,
    #     threshold=DEFAULT_PARAMS["threshold"],
    #     epsilon=DEFAULT_PARAMS["epsilon"],
    #     output_dir=output_dir,
    # )
    # martingale_viz.create_visualization()

    # logger.info(f"Results saved to {output_dir}/")


def main():
    """Run detection based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run change point detection on network evolution."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to analyze: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )

    args = parser.parse_args()
    run_detection(args.model, args.output_dir)


if __name__ == "__main__":
    main()
