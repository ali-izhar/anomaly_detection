# src/run.py

"""
Main script to run the forecast-based martingale detection algorithm on network sequences.
Implements the experimental setup from Section 6 of the paper:
'Faster Structural Change Detection in Dynamic Networks via Statistical Forecasting'

The script follows the experimental methodology:
1. Generate synthetic network sequences with known change points (Section 6.1)
2. Extract multiview features as defined in Section 4
3. Run Algorithm 1 for detection (Section 5)
4. Evaluate and visualize results (Section 6.2)

Usage:
    python src/run.py <model_alias>
"""

import os
import sys
import argparse
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.graph.visualizer import NetworkVisualizer
from src.predictor.adaptive import GraphPredictor
from src.configs.loader import get_config
from src.algorithm import (
    extract_numeric_features,
    run_forecast_martingale_detection,
    DEFAULT_PARAMS,
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_full_model_name(alias: str) -> str:
    """Get full model name from alias."""
    REVERSE_ALIASES = {
        "ba": "barabasi_albert",
        "ws": "watts_strogatz",
        "er": "erdos_renyi",
        "sbm": "stochastic_block_model",
    }
    return REVERSE_ALIASES.get(alias, alias)


def visualize_results(
    model_name: str,
    graphs: list,
    features_raw: list,
    true_change_points: list,
    detected_change_points: list,
    martingale_results: dict,
    output_dir: str = "results",
):
    """
    Create visualizations of network states and detection results.
    Follows visualization approach from Section 6.2 of the paper:
    1. Network states at key points (initial, change points, final)
    2. Martingale evolution showing:
       - M_t (observed martingale) from Algorithm 1, Line 6
       - Mhat_t (horizon martingale) from Algorithm 1, Line 7
       - Combined evidence max(M_t, Mhat_t) for detection
    3. Individual feature martingales to analyze sensitivity
    """
    os.makedirs(output_dir, exist_ok=True)
    viz = NetworkVisualizer()

    # 1. Network State Visualization
    key_points = sorted(
        list(set([0] + true_change_points + detected_change_points + [len(graphs) - 1]))
    )
    n_points = len(key_points)

    fig, axes = plt.subplots(
        n_points,
        2,
        figsize=(viz.SINGLE_COLUMN_WIDTH, viz.STANDARD_HEIGHT * n_points / 2),
    )
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} Network States",
        fontsize=viz.TITLE_SIZE,
        y=0.98,
    )

    for i, time_idx in enumerate(key_points):
        point_type = (
            "Initial State"
            if time_idx == 0
            else (
                "Final State"
                if time_idx == len(graphs) - 1
                else (
                    "True Change Point"
                    if time_idx in true_change_points
                    else (
                        "Detected Change Point"
                        if time_idx in detected_change_points
                        else "State"
                    )
                )
            )
        )

        # Plot network state
        viz.plot_network(
            graphs[time_idx],
            ax=axes[i, 0],
            title=f"Network {point_type} at t={time_idx}",
            layout="spring",
        )

        # Plot adjacency matrix
        viz.plot_adjacency(
            graphs[time_idx], ax=axes[i, 1], title=f"Adjacency Matrix at t={time_idx}"
        )

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_states.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 2. Martingale Evolution
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot observed martingales
    time = range(len(martingale_results["M_observed"]))
    ax1.plot(
        time,
        martingale_results["M_observed"],
        label="Observed",
        color="blue",
        linewidth=2,
    )
    ax1.axhline(
        y=DEFAULT_PARAMS["threshold"], color="r", linestyle="--", label="Threshold"
    )
    for cp in true_change_points:
        ax1.axvline(
            x=cp,
            color="g",
            linestyle=":",
            alpha=0.5,
            label="True Change" if cp == true_change_points[0] else "",
        )
    for cp in detected_change_points:
        ax1.axvline(
            x=cp,
            color="purple",
            linestyle="-.",
            alpha=0.5,
            label="Detected" if cp == detected_change_points[0] else "",
        )
    ax1.set_title("Observed Martingale Evolution (M_t)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Martingale Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot predicted martingales
    ax2.plot(
        time,
        martingale_results["M_predicted"],
        label="Predicted",
        color="orange",
        linewidth=2,
    )
    ax2.axhline(
        y=DEFAULT_PARAMS["threshold"], color="r", linestyle="--", label="Threshold"
    )
    for cp in true_change_points:
        ax2.axvline(
            x=cp,
            color="g",
            linestyle=":",
            alpha=0.5,
            label="True Change" if cp == true_change_points[0] else "",
        )
    for cp in detected_change_points:
        ax2.axvline(
            x=cp,
            color="purple",
            linestyle="-.",
            alpha=0.5,
            label="Detected" if cp == detected_change_points[0] else "",
        )
    ax2.set_title("Horizon Martingale Evolution (Mhat_t)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Martingale Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot combined evidence
    ax3.plot(
        time,
        martingale_results["M_observed"],
        label="M_t (Observed)",
        color="blue",
        alpha=0.5,
    )
    ax3.plot(
        time,
        martingale_results["M_predicted"],
        label="Mhat_t (Predicted)",
        color="orange",
        alpha=0.5,
    )
    ax3.plot(
        time,
        np.maximum(martingale_results["M_observed"], martingale_results["M_predicted"]),
        label="Max(M_t, Mhat_t)",
        color="red",
        linewidth=2,
    )
    ax3.axhline(
        y=DEFAULT_PARAMS["threshold"], color="r", linestyle="--", label="Threshold"
    )
    for cp in true_change_points:
        ax3.axvline(
            x=cp,
            color="g",
            linestyle=":",
            alpha=0.5,
            label="True Change" if cp == true_change_points[0] else "",
        )
    for cp in detected_change_points:
        ax3.axvline(
            x=cp,
            color="purple",
            linestyle="-.",
            alpha=0.5,
            label="Detected" if cp == detected_change_points[0] else "",
        )
    ax3.set_title(
        "Combined Evidence (Detection when either M_t or Mhat_t exceeds threshold)"
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Martingale Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_martingales.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 3. Individual Feature Martingales
    if "individual_martingales_obs" in martingale_results:
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        feature_names = [
            "Degree",
            "Density",
            "Clustering",
            "Betweenness",
            "Eigenvector",
            "Closeness",
            "Singular Value",
            "Laplacian",
        ]

        for i, (name, m_obs, m_pred) in enumerate(
            zip(
                feature_names,
                martingale_results["individual_martingales_obs"],
                martingale_results["individual_martingales_pred"],
            )
        ):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            ax.plot(time, m_obs, label="Observed", color="blue", alpha=0.7)
            ax.plot(time, m_pred, label="Predicted", color="orange", alpha=0.7)
            ax.axhline(
                y=DEFAULT_PARAMS["threshold"] / 8,
                color="r",
                linestyle="--",
                label="Threshold/8",
            )

            for cp in true_change_points:
                ax.axvline(x=cp, color="g", linestyle=":", alpha=0.5)

            ax.set_title(f"{name} Feature Martingales")
            ax.set_xlabel("Time")
            ax.set_ylabel("Martingale Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_feature_martingales.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def run_detection(model_alias: str, output_dir: str = "results"):
    """
    Run the forecast-based martingale detection algorithm on a network sequence.
    Implements the experimental methodology from Section 6 of the paper.

    Steps (with paper section references):
    1. Generate synthetic network sequence with changes (Section 6.1)
    2. Extract multiview features from each snapshot (Section 4)
    3. Initialize predictor with optimal parameters (Section 6.3)
    4. Run Algorithm 1 for detection (Section 5)
    5. Analyze results using metrics from Section 6.2
    """
    # 1. Setup - Get full model name for config loading
    model_name = get_full_model_name(model_alias)
    logger.info(f"Running detection on {model_name} network sequence...")

    # 2. Generate Network Sequence (Section 6.1)
    generator = GraphGenerator(model_alias)

    # Load base configuration from YAML
    config = get_config(model_name)

    # Override experiment-specific parameters
    params = config["params"].__dict__
    params.update(
        {
            "n": 50,  # Number of nodes (as in Section 6.1)
            "seq_len": 200,  # Sequence length matching paper experiments
            "min_changes": 1,  # Minimum structural changes
            "max_changes": 2,  # Maximum structural changes
            "min_segment": 40,  # Minimum segment length (Section 6.1)
        }
    )

    # Generate sequence using parameters
    result = generator.generate_sequence(params)
    graphs = result["graphs"]
    true_change_points = result["change_points"]

    # 3. Extract Features (Section 4)
    logger.info("Extracting network features...")
    feature_extractor = NetworkFeatureExtractor()
    features_raw = []  # Store raw feature dictionaries for visualization
    features_numeric = []  # Store numeric features for detection

    for adj_matrix in tqdm(graphs, desc="Extracting features", unit="graph"):
        graph = nx.from_numpy_array(adj_matrix)
        feature_dict = feature_extractor.get_features(graph)
        features_raw.append(feature_dict)
        numeric_features = extract_numeric_features(feature_dict)
        features_numeric.append(numeric_features)

    # 4. Initialize Predictor (Section 5.1)
    predictor = GraphPredictor(
        k=DEFAULT_PARAMS["window_size"],  # Window size k=10 from Section 6.3
        alpha=0.8,  # Decay factor for temporal patterns (Section 5.1)
        initial_gamma=0.1,  # Weight for structural constraints (Section 5.1)
        initial_beta=0.5,  # Initial mixing parameter (adapted online)
    )

    # 5. Run Detection Algorithm (Algorithm 1)
    logger.info("Running forecast-based martingale detection...")

    # Create a progress bar for the detection process
    pbar = tqdm(total=len(graphs), desc="Running detection", unit="step")

    # Create a callback to update the progress bar
    def progress_callback(t):
        pbar.update(1)

    detection_results = run_forecast_martingale_detection(
        graph_sequence=[nx.from_numpy_array(g) for g in graphs],
        horizon=DEFAULT_PARAMS["horizon"],  # h=5 from Section 6.3
        threshold=DEFAULT_PARAMS["threshold"],  # λ=60 from Section 6.3
        epsilon=DEFAULT_PARAMS["epsilon"],  # ε=0.7 from Section 6.3
        window_size=DEFAULT_PARAMS["window_size"],
        predictor=predictor,
        random_state=42,
        progress_callback=progress_callback,  # Add callback for progress updates
    )

    pbar.close()

    # 6. Analyze Results (Section 6.2)
    detected_points = detection_results["change_points"]
    logger.info(f"\nResults Summary:")
    logger.info(f"True change points: {true_change_points}")
    logger.info(f"Detected change points: {detected_points}")

    if true_change_points and detected_points:
        # Compute detection delay (key metric from Section 6.2)
        errors = []
        for true_cp in true_change_points:
            closest_detected = min(detected_points, key=lambda x: abs(x - true_cp))
            error = abs(closest_detected - true_cp)
            errors.append(error)
        avg_error = np.mean(errors)
        logger.info(f"Average detection delay: {avg_error:.2f} time steps")

    # 7. Visualize Results (Section 6.2)
    logger.info("Creating visualizations...")
    visualize_results(
        model_name=model_name,
        graphs=graphs,
        features_raw=features_raw,
        true_change_points=true_change_points,
        detected_change_points=detected_points,
        martingale_results=detection_results,
        output_dir=output_dir,
    )

    logger.info(f"Results saved to {output_dir}/")


def main():
    """Run detection based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run forecast-based martingale detection on network evolution."
    )
    parser.add_argument(
        "model",
        choices=["ba", "ws", "er", "sbm"],
        help="Model to analyze: ba (Barabási-Albert), ws (Watts-Strogatz), "
        "er (Erdős-Rényi), sbm (Stochastic Block Model)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)",
    )

    args = parser.parse_args()
    run_detection(args.model, args.output_dir)


if __name__ == "__main__":
    main()
