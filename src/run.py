# src/run.py

"""
Main script to run martingale-based change detection on network sequences.

Usage:
    python src/run.py <model_alias>
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.configs.loader import get_config
from src.graph.generator import GraphGenerator
from src.predictor.factory import PredictorFactory
from src.changepoint.visualizer import MartingaleVisualizer
from src.graph.visualizer import NetworkVisualizer
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph

# Setup logging with debug level
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
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


def run_detection(
    model_alias: str,
    threshold: float = 60.0,
    epsilon: float = 0.7,
    batch_size: int = 1000,
    max_martingale: float = None,
    reset: bool = True,
    max_window: int = None,
    random_state: int = 42,
    prediction_horizon: int = 5,
) -> Dict[str, Any]:
    """
    ALGORITHM: Forecast-based Graph Structural Change Detection using Martingale

    INPUT:
    - Network sequence {G_t}_{t=1}^n (generated from model_alias)
    - Threshold λ (threshold parameter)
    - Epsilon ε (sensitivity parameter)
    - Batch size (for multiview processing)
    - Max window (for historical context)
    - Random state (for reproducibility)
    - Prediction horizon h (number of steps to predict ahead)

    OUTPUT:
    - Change points τ and detection statistics
    """

    # ========================================================================== #
    # =================== STEP 1: Setup and Initialization ===================== #
    # ========================================================================== #
    model_name = get_full_model_name(model_alias)
    logger.info(f"Starting detection process for model: {model_name}")

    # Initialize predictor using factory
    predictor_config = {
        "n_history": max_window if max_window else 10,
        "adaptive": True,
        "enforce_connectivity": True,
        "binary": True,
        "spectral_reg": 0.4,
        "community_reg": 0.4,
        "n_communities": 2,
        "temporal_window": 10,
        "distribution_reg": 0.3,
    }
    predictor = PredictorFactory.create("adaptive", predictor_config)
    logger.info(f"Initialized predictor with history_size={predictor.history_size}")
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 2: Generate Network Sequence ==================== #
    # ========================================================================== #
    generator = GraphGenerator(model_alias)
    config = get_config(model_name)
    params = config["params"].__dict__
    # Override for testing
    params.update(
        {
            "n": 50,
            "seq_len": 100,
            "min_changes": 1,
            "max_changes": 1,
            "min_segment": 40,
        }
    )

    # Generate sequence with ground truth change points
    logger.info("Generating graph sequence...")
    result = generator.generate_sequence(params)
    graphs = result["graphs"]  # List of adjacency matrices
    true_change_points = result["change_points"]

    # Log sequence dimensions
    logger.info("Graph Sequence Dimensions:")
    logger.info(f"- Sequence length (T): {len(graphs)}")
    logger.info(f"- Individual graph shape (NxN): {graphs[0].shape}")
    logger.info(f"- Memory footprint: {sum(g.nbytes for g in graphs) / 1024:.2f} KB")
    logger.info(f"- True change points: {true_change_points}")
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 3: Extract Features ============================ #
    # ========================================================================== #
    logger.info("Extracting features from graph sequence...")

    # Extract features from the graph sequence
    features_raw = []
    features_numeric = []

    # Process first graph to get feature dimensions
    first_graph = adjacency_to_graph(graphs[0])
    first_raw_features = NetworkFeatureExtractor().get_features(first_graph)
    first_numeric_features = extract_numeric_features(
        first_raw_features, feature_set="all"
    )

    logger.info("Feature Extraction Dimensions:")
    logger.info(f"- Number of raw feature types: {len(first_raw_features)}")
    logger.info(f"- Numeric feature vector length: {len(first_numeric_features)}")

    # Process all graphs
    for adj_matrix in graphs:
        graph = adjacency_to_graph(adj_matrix)
        raw_features = NetworkFeatureExtractor().get_features(graph)
        numeric_features = extract_numeric_features(raw_features, feature_set="all")
        features_raw.append(raw_features)
        features_numeric.append(numeric_features)

    features_numeric = np.array(features_numeric)
    logger.info("Final Feature Sequence Dimensions:")
    logger.info(f"- Shape (TxF): {features_numeric.shape}")  # T timesteps × F features
    logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 4: Generate Predictions ========================= #
    # ========================================================================== #
    logger.info("Generating predictions...")
    predicted_graphs = []

    # Process predictions
    for t in range(len(graphs)):
        current = graphs[t]
        history_start = max(0, t - predictor.history_size)
        history = [{"adjacency": g} for g in graphs[history_start:t]]

        if t >= predictor.history_size:
            predictions = predictor.predict(history, horizon=prediction_horizon)
            predicted_graphs.append(predictions)

    # Process predictions if they exist
    predicted_features_numeric = None
    if predicted_graphs:
        logger.info("Prediction Sequence Dimensions:")
        logger.info(f"- Number of prediction timesteps (T'): {len(predicted_graphs)}")
        logger.info(f"- Predictions per timestep (H): {len(predicted_graphs[0])}")
        logger.info(
            f"- Individual prediction shape (NxN): {predicted_graphs[0][0].shape}"
        )

        # Process features from predictions
        predicted_features_raw = []
        predicted_features_numeric = []

        for predictions in predicted_graphs:
            timestep_features = []
            for pred_adj in predictions:
                graph = adjacency_to_graph(pred_adj)
                raw_features = NetworkFeatureExtractor().get_features(graph)
                numeric_features = extract_numeric_features(
                    raw_features, feature_set="all"
                )
                timestep_features.append(numeric_features)
            predicted_features_numeric.append(timestep_features)

        predicted_features_numeric = np.array(predicted_features_numeric)
        logger.info("Predicted Features Dimensions:")
        logger.info(
            f"- Shape (T'xHxF): {predicted_features_numeric.shape}"
        )  # T' prediction timesteps × H horizon × F features
        logger.info("-" * 50)

    # ========================================================================== #
    # =================== STEP 5: Initialize Detector ========================== #
    # ========================================================================== #
    logger.info("Initializing change point detector...")

    detector = ChangePointDetector(
        martingale_method="multiview",
        history_size=predictor.history_size,
        threshold=threshold,
        epsilon=epsilon,
        random_state=random_state,
        feature_set="all",
        batch_size=batch_size,
        max_martingale=max_martingale,
        reset=reset,
        max_window=max_window,
    )

    # Run detection with extracted features
    logger.info("Running detection with extracted features...")
    detection_result = detector.run(
        data=features_numeric,
        predicted_data=predicted_features_numeric,
    )

    # First check if detection_result exists
    if detection_result is None:
        logger.error("Detection result is None")
        return

    logger.info("-" * 50)
    logger.info("Change Point Detection Analysis:")

    # Get detections from both streams
    traditional_changes = detection_result["change_points"]
    horizon_changes = detection_result.get("horizon_change_points", [])

    # Basic detection summary
    logger.info("\nDetection Summary:")
    logger.info(f"True change points: {true_change_points}")
    logger.info(f"Traditional martingale detections: {traditional_changes}")
    logger.info(f"Horizon martingale detections: {horizon_changes}")

    # Detailed detection analysis
    if detector.method == "single_view":
        martingales = detection_result.get("martingales", [])
        pred_martingales = detection_result.get("prediction_martingales", [])

        logger.info("\nDetailed Detection Analysis:")
        logger.info("Traditional Martingale:")
        logger.info(f"- Number of detections: {len(traditional_changes)}")
        if traditional_changes:
            logger.info(f"- Detection times: {traditional_changes}")
            logger.info(
                f"- Martingale values at detection: {[martingales[i] for i in traditional_changes]}"
            )

        logger.info("\nHorizon Martingale:")
        logger.info(f"- Number of detections: {len(horizon_changes)}")
        if horizon_changes:
            logger.info(f"- Detection times: {horizon_changes}")
            logger.info(
                f"- Martingale values at detection: {[pred_martingales[i] for i in horizon_changes]}"
            )

    else:  # multiview
        mart_sum = detection_result.get("martingale_sum", [])
        pred_sum = detection_result.get("prediction_martingale_sum", [])

        logger.info("\nDetailed Detection Analysis:")
        logger.info("Traditional Martingale (Sum):")
        logger.info(f"- Number of detections: {len(traditional_changes)}")
        if traditional_changes:
            logger.info(f"- Detection times: {traditional_changes}")
            logger.info(
                f"- Sum martingale values at detection: {[mart_sum[i] for i in traditional_changes]}"
            )

        logger.info("\nHorizon Martingale (Sum):")
        logger.info(f"- Number of detections: {len(horizon_changes)}")
        if horizon_changes:
            logger.info(f"- Detection times: {horizon_changes}")
            logger.info(
                f"- Sum martingale values at detection: {[pred_sum[i] for i in horizon_changes]}"
            )

    # Detection Delay Analysis
    logger.info("\nDetection Delay Analysis:")
    for tcp in true_change_points:
        logger.info(f"\nAnalyzing true change point at t={tcp}:")

        # Traditional delays
        trad_delays = [d - tcp for d in traditional_changes if d >= tcp]
        if trad_delays:
            min_delay = min(trad_delays)
            detection_time = tcp + min_delay
            logger.info(f"Traditional Martingale:")
            logger.info(f"- Detected after {min_delay} steps (at t={detection_time})")
            if detector.method == "single_view":
                logger.info(f"- Detection value: {martingales[detection_time]:.4f}")
            else:
                logger.info(f"- Detection value: {mart_sum[detection_time]:.4f}")
        else:
            logger.info("Traditional Martingale: No detection")

        # Horizon delays
        horizon_delays = [d - tcp for d in horizon_changes if d >= tcp]
        if horizon_delays:
            min_delay = min(horizon_delays)
            detection_time = tcp + min_delay
            logger.info(f"Horizon Martingale:")
            logger.info(f"- Detected after {min_delay} steps (at t={detection_time})")
            if detector.method == "single_view":
                logger.info(
                    f"- Detection value: {pred_martingales[detection_time]:.4f}"
                )
            else:
                logger.info(f"- Detection value: {pred_sum[detection_time]:.4f}")
        else:
            logger.info("Horizon Martingale: No detection")

    # Early/Late Detection Analysis
    logger.info("\nEarly/Late Detection Analysis:")
    for cp in sorted(set(traditional_changes + horizon_changes)):
        trad_detected = cp in traditional_changes
        horizon_detected = cp in horizon_changes

        if trad_detected and horizon_detected:
            logger.info(f"Time t={cp}: Detected by both martingales")
        elif trad_detected:
            logger.info(f"Time t={cp}: Detected only by traditional martingale")
        elif horizon_detected:
            logger.info(f"Time t={cp}: Detected only by horizon martingale")

    logger.info("-" * 50)

    # Add detailed logging of detection results
    logger.info("-" * 50)
    logger.info("Detection Results Analysis:")

    # Log basic detection results
    logger.info(f"Detection result keys: {detection_result.keys()}")
    logger.info(
        f"Number of change points detected: {len(detection_result.get('change_points', []))}"
    )
    logger.info(
        f"Change points detected at: {detection_result.get('change_points', [])}"
    )

    # Log martingale sequences
    if detector.method == "single_view":
        logger.info("\nSingle-view Martingale Analysis:")
        martingales = detection_result.get("martingales", [])
        pred_martingales = detection_result.get("prediction_martingales", [])
        logger.info(
            f"Traditional martingale sequence length: {len(martingales) if martingales is not None else 0}"
        )
        logger.info(
            f"Prediction martingale sequence length: {len(pred_martingales) if pred_martingales is not None else 0}"
        )

        # Log final values
        final_trad = martingales[-1] if martingales and len(martingales) > 0 else "N/A"
        final_pred = (
            pred_martingales[-1]
            if pred_martingales and len(pred_martingales) > 0
            else "N/A"
        )
        logger.info(f"Final traditional martingale value: {final_trad}")
        logger.info(f"Final prediction martingale value: {final_pred}")
    else:  # multiview
        logger.info("\nMultiview Martingale Analysis:")
        mart_sum = detection_result.get("martingale_sum", [])
        mart_avg = detection_result.get("martingale_avg", [])
        indiv_marts = detection_result.get("individual_martingales", [])

        logger.info(
            f"Martingale sum sequence length: {len(mart_sum) if mart_sum is not None else 0}"
        )
        logger.info(
            f"Martingale avg sequence length: {len(mart_avg) if mart_avg is not None else 0}"
        )
        logger.info(
            f"Individual martingales count: {len(indiv_marts) if indiv_marts is not None else 0}"
        )

        # Log prediction martingales
        pred_sum = detection_result.get("prediction_martingale_sum", [])
        pred_avg = detection_result.get("prediction_martingale_avg", [])
        pred_indiv = detection_result.get("prediction_individual_martingales", [])

        logger.info(
            f"Prediction martingale sum sequence length: {len(pred_sum) if pred_sum is not None else 0}"
        )
        logger.info(
            f"Prediction martingale avg sequence length: {len(pred_avg) if pred_avg is not None else 0}"
        )
        logger.info(
            f"Prediction individual martingales count: {len(pred_indiv) if pred_indiv is not None else 0}"
        )

    # Log strangeness and p-values
    logger.info("\nStrangeness and P-value Analysis:")
    logger.info(
        f"Traditional strangeness sequence length: {len(detection_result.get('strangeness', []))}"
    )
    # Fix p-values length reporting
    pvalues_length = (
        len(detection_result["pvalues"])
        if detection_result["pvalues"] is not None
        else 0
    )
    logger.info(f"Traditional p-values sequence length: {pvalues_length}")

    if "prediction_strangeness" in detection_result:
        logger.info(
            f"Prediction strangeness sequence length: {len(detection_result['prediction_strangeness'])}"
        )
    if "prediction_pvalues" in detection_result:
        logger.info(
            f"Prediction p-values sequence length: {len(detection_result['prediction_pvalues'])}"
        )

    # Log feature information if available
    if "features_raw" in detection_result:
        logger.info("\nFeature Information:")
        logger.info(
            f"Raw features sequence length: {len(detection_result['features_raw'])}"
        )
    if "features_numeric" in detection_result:
        logger.info(
            f"Numeric features sequence length: {len(detection_result['features_numeric'])}"
        )
        if len(detection_result["features_numeric"]) > 0:
            logger.info(
                f"Feature vector dimension: {detection_result['features_numeric'][0].shape}"
            )

    logger.info("-" * 50)

    # After getting detection results, add this analysis:
    logger.info("\nDetection Time Analysis:")
    if detector.method == "single_view":
        martingales = detection_result.get("martingales", [])
        pred_martingales = detection_result.get("prediction_martingales", [])

        # Find when each martingale crosses threshold
        trad_detections = [i for i, m in enumerate(martingales) if m >= threshold]
        pred_detections = [i for i, m in enumerate(pred_martingales) if m >= threshold]

        logger.info("Traditional Martingale Detections:")
        logger.info(f"- Detection times: {trad_detections}")
        if trad_detections:
            logger.info(
                f"- Values at detection: {[martingales[i] for i in trad_detections]}"
            )

        logger.info("\nHorizon Martingale Detections:")
        logger.info(f"- Detection times: {pred_detections}")
        if pred_detections:
            logger.info(
                f"- Values at detection: {[pred_martingales[i] for i in pred_detections]}"
            )

    else:  # multiview
        mart_sum = detection_result.get("martingale_sum", [])
        pred_sum = detection_result.get("prediction_martingale_sum", [])

        # Find when each martingale crosses threshold
        trad_detections = [i for i, m in enumerate(mart_sum) if m >= threshold]
        pred_detections = [i for i, m in enumerate(pred_sum) if m >= threshold]

        logger.info("Traditional Martingale (Sum) Detections:")
        logger.info(f"- Detection times: {trad_detections}")
        if trad_detections:
            logger.info(
                f"- Values at detection: {[mart_sum[i] for i in trad_detections]}"
            )

        logger.info("\nHorizon Martingale (Sum) Detections:")
        logger.info(f"- Detection times: {pred_detections}")
        if pred_detections:
            logger.info(
                f"- Values at detection: {[pred_sum[i] for i in pred_detections]}"
            )

    # Compare with true change points
    if true_change_points:
        logger.info("\nDelay Analysis:")
        logger.info(f"True change points: {true_change_points}")

        for tcp in true_change_points:
            # Traditional delays
            trad_delays = [d - tcp for d in trad_detections if d >= tcp]
            if trad_delays:
                logger.info(
                    f"Traditional detection delay for change at t={tcp}: {min(trad_delays)} steps"
                )
            else:
                logger.info(f"Traditional martingale did not detect change at t={tcp}")

            # Prediction delays
            pred_delays = [d - tcp for d in pred_detections if d >= tcp]
            if pred_delays:
                logger.info(
                    f"Horizon detection delay for change at t={tcp}: {min(pred_delays)} steps"
                )
            else:
                logger.info(f"Horizon martingale did not detect change at t={tcp}")

    # ========================================================================== #
    # =================== STEP 6: Create Visualizations ============================ #
    # ========================================================================== #
    logger.info("\nCreating visualizations...")
    output_dir = f"examples/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Visualize network states at key points
    viz = NetworkVisualizer()
    key_points = sorted(
        list(
            set(
                [0]
                + true_change_points
                + detection_result["change_points"]
                + [len(graphs) - 1]
            )
        )
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

    # Prepare node colors for SBM
    node_color = None
    if "stochastic_block_model" in model_name.lower():
        graph = nx.from_numpy_array(graphs[0])
        n = graph.number_of_nodes()
        num_blocks = int(np.sqrt(n))
        block_sizes = [n // num_blocks] * (num_blocks - 1)
        block_sizes.append(n - sum(block_sizes))
        node_color = []
        for j, size in enumerate(block_sizes):
            node_color.extend([f"C{j}"] * size)

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
                        if time_idx in detection_result["change_points"]
                        else "State"
                    )
                )
            )
        )

        viz.plot_network(
            graphs[time_idx],
            ax=axes[i, 0],
            title=f"Network {point_type} at t={time_idx}",
            layout="spring",
            node_color=node_color,
        )

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

    # 2. Visualize feature evolution
    features_raw = features_raw  # Use the original features
    detected_change_points = detection_result["change_points"]

    fig = plt.figure(figsize=(viz.SINGLE_COLUMN_WIDTH, viz.STANDARD_HEIGHT * 2))
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(1.0)

    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    feature_names = [
        "degrees",
        "density",
        "clustering",
        "betweenness",
        "eigenvector",
        "closeness",
        "singular_values",
        "laplacian_eigenvalues",
    ]

    colors = {
        "actual": "#1f77b4",  # Blue
        "change_point": "#FF9999",  # Light red
        "detected": "#ff7f0e",  # Orange
    }

    for i, feature_name in enumerate(feature_names):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        time = range(len(features_raw))

        if isinstance(features_raw[0][feature_name], list):
            mean_values = [
                np.mean(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features_raw
            ]
            std_values = [
                np.std(f[feature_name]) if len(f[feature_name]) > 0 else 0
                for f in features_raw
            ]

            ax.plot(
                time,
                mean_values,
                color=colors["actual"],
                alpha=0.8,
                linewidth=1.0,
            )
            ax.fill_between(
                time,
                np.array(mean_values) - np.array(std_values),
                np.array(mean_values) + np.array(std_values),
                color=colors["actual"],
                alpha=0.1,
            )
        else:
            values = [f[feature_name] for f in features_raw]
            ax.plot(
                time,
                values,
                color=colors["actual"],
                alpha=0.8,
                linewidth=1.0,
            )

        for cp in true_change_points:
            ax.axvline(
                x=cp,
                color=colors["change_point"],
                linestyle="--",
                alpha=0.5,
                linewidth=0.8,
            )

        for cp in detected_change_points:
            if isinstance(features_raw[0][feature_name], list):
                y_val = mean_values[cp]
            else:
                y_val = features_raw[cp][feature_name]
            ax.plot(
                cp,
                y_val,
                "o",
                color=colors["detected"],
                markersize=6,
                alpha=0.8,
                markeredgewidth=1,
            )

        ax.set_title(
            feature_name.replace("_", " ").title(),
            fontsize=viz.TITLE_SIZE,
            pad=4,
        )
        ax.set_xlabel("Time" if row == 3 else "", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.set_ylabel("Value" if col == 0 else "", fontsize=viz.LABEL_SIZE, labelpad=2)
        ax.tick_params(labelsize=viz.TICK_SIZE, pad=1)
        ax.grid(True, alpha=0.15, linewidth=0.5, linestyle=":")

        ax.set_xticks(np.arange(0, len(time), 10))
        ax.set_xlim(0, len(time))

    plt.suptitle(
        f"{model_name.replace('_', ' ').title()} Feature Evolution\n"
        + "True Change Points (Red), Detected Change Points (Orange)",
        fontsize=viz.TITLE_SIZE,
        y=0.98,
    )
    plt.tight_layout(pad=0.3)
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_features.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.02,
    )
    plt.close()

    # 3. Create martingale visualization
    logger.info("\nAnalyzing detection results before visualization:")
    logger.info("-" * 50)
    logger.info("Available keys in detection_result:")
    logger.info(f"- {list(detection_result.keys())}")

    if "individual_martingales" in detection_result:
        logger.info("\nIndividual Martingales Analysis:")
        logger.info(
            f"- Number of individual martingales: {len(detection_result['individual_martingales'])}"
        )
        if detection_result["individual_martingales"]:
            logger.info(
                f"- Length of each individual martingale: {len(detection_result['individual_martingales'][0])}"
            )

    if "prediction_individual_martingales" in detection_result:
        logger.info("\nPrediction Individual Martingales Analysis:")
        logger.info(
            f"- Number of prediction martingales: {len(detection_result['prediction_individual_martingales'])}"
        )
        if detection_result["prediction_individual_martingales"]:
            logger.info(
                f"- Length of each prediction martingale: {len(detection_result['prediction_individual_martingales'][0])}"
            )

    logger.info("\nCombined Martingales Analysis:")
    logger.info(
        f"- Martingale sum length: {len(detection_result.get('martingale_sum', []))}"
    )
    logger.info(
        f"- Prediction martingale sum length: {len(detection_result.get('prediction_martingale_sum', []))}"
    )
    logger.info("-" * 50)

    martingale_data = {}
    if detector.method == "single_view":
        martingale_data = {
            "combined": {
                # Traditional martingales
                "martingales": detection_result.get("martingales", []),
                "pvalues": detection_result.get("pvalues", []),
                "strangeness": detection_result.get("strangeness", []),
                # Prediction martingales
                "prediction_martingales": detection_result.get(
                    "prediction_martingales", []
                ),
                "prediction_pvalues": detection_result.get("prediction_pvalues", []),
                "prediction_strangeness": detection_result.get(
                    "prediction_strangeness", []
                ),
                # Change points and features
                "change_points": detection_result.get("change_points", []),
                "features_raw": features_raw,
            }
        }
    else:  # multiview
        # First get individual feature martingales
        individual_martingales = detection_result.get("individual_martingales", [])
        prediction_individual_martingales = detection_result.get(
            "prediction_individual_martingales", []
        )

        # Create feature-specific entries
        features = [
            "degree",
            "density",
            "clustering",
            "betweenness",
            "eigenvector",
            "closeness",
            "singular_value",
            "laplacian",
        ]

        feature_data = {}
        for i, feature in enumerate(features):
            if i < len(individual_martingales):
                feature_data[feature] = {
                    # Traditional martingales for this feature
                    "martingales": individual_martingales[i],
                    "pvalues": detection_result.get("pvalues", [None] * len(features))[
                        i
                    ],
                    "strangeness": detection_result.get(
                        "strangeness", [None] * len(features)
                    )[i],
                }

                # Add prediction martingales for this feature
                if prediction_individual_martingales:
                    num_horizons = len(prediction_individual_martingales) // len(
                        features
                    )
                    feature_start_idx = i * num_horizons
                    feature_end_idx = (i + 1) * num_horizons

                    if feature_start_idx < len(prediction_individual_martingales):
                        feature_predictions = prediction_individual_martingales[
                            feature_start_idx:feature_end_idx
                        ]
                        if feature_predictions:
                            feature_data[feature].update(
                                {
                                    # Store all horizon predictions separately
                                    "prediction_martingales_horizons": feature_predictions,
                                    # Also store the sum for backward compatibility
                                    "prediction_martingales": (
                                        np.sum(
                                            [
                                                np.array(m)
                                                for m in feature_predictions
                                                if len(m) > 0
                                            ],
                                            axis=0,
                                        )
                                        if feature_predictions
                                        and any(len(m) > 0 for m in feature_predictions)
                                        else []
                                    ),
                                    "prediction_pvalues": detection_result.get(
                                        "prediction_pvalues", [None] * len(features)
                                    )[i],
                                    "prediction_strangeness": detection_result.get(
                                        "prediction_strangeness", [None] * len(features)
                                    )[i],
                                }
                            )

        martingale_data = {
            # Add feature-specific data
            **feature_data,
            # Add combined data
            "combined": {
                # Traditional martingales
                "martingales": detection_result.get(
                    "martingale_sum", []
                ),  # Use sum as main martingale
                "martingale_sum": detection_result.get("martingale_sum", []),
                "martingale_avg": detection_result.get("martingale_avg", []),
                "individual_martingales": detection_result.get(
                    "individual_martingales", []
                ),
                # Prediction martingales
                "prediction_martingales": detection_result.get(
                    "prediction_martingale_sum", []
                ),  # Use sum as main prediction
                "prediction_martingale_sum": detection_result.get(
                    "prediction_martingale_sum", []
                ),
                "prediction_martingale_avg": detection_result.get(
                    "prediction_martingale_avg", []
                ),
                "prediction_individual_martingales": detection_result.get(
                    "prediction_individual_martingales", []
                ),
                # P-values and strangeness
                "pvalues": detection_result.get("pvalues", []),
                "strangeness": detection_result.get("strangeness", []),
                "prediction_pvalues": detection_result.get("prediction_pvalues", []),
                "prediction_strangeness": detection_result.get(
                    "prediction_strangeness", []
                ),
                # Change points and features
                "change_points": detection_result.get("change_points", []),
                "features_raw": features_raw,
            },
        }

    martingale_viz = MartingaleVisualizer(
        martingales=martingale_data,
        change_points=true_change_points,
        threshold=threshold,
        epsilon=epsilon,
        output_dir=output_dir,
        skip_shap=False,
        method=detector.method,
    )
    martingale_viz.create_visualization()

    logger.info(f"Visualizations saved to {output_dir}/")

    # ========================================================================== #
    # =================== STEP 7: Return Complete Results ======================== #
    # ========================================================================== #
    results = {
        "true_change_points": true_change_points,
        "model_name": model_name,
        "params": params,
        "detected_changes": detection_result["change_points"],
        # Feature information
        "pvalues": detection_result.get("pvalues", []),  # Fixed key name
        "strangeness": detection_result.get("strangeness", []),
        "features_raw": features_raw,  # Use original features
        "features_numeric": features_numeric,  # Use original features
        # Prediction data
        "predicted_graphs": predicted_graphs,
        "predictor_states": predictor.get_state(),
        # Additional prediction statistics
        "prediction_pvalues": detection_result.get("prediction_pvalues", []),
        "prediction_strangeness": detection_result.get("prediction_strangeness", []),
    }

    # Add method-specific results
    if detector.method == "single_view":
        results.update(
            {
                "martingales": detection_result.get("martingales", []),
                "prediction_martingales": detection_result.get(
                    "prediction_martingales", []
                ),
                "traditional_delays": [
                    d - tcp
                    for tcp in true_change_points
                    for d in detection_result["change_points"]
                    if d >= tcp
                ],
                "traditional_detection_times": detection_result["change_points"],
                "prediction_delays": [
                    d - tcp
                    for tcp in true_change_points
                    for d in detection_result.get("horizon_change_points", [])
                    if d >= tcp
                ],
                "prediction_detection_times": detection_result.get(
                    "horizon_change_points", []
                ),
            }
        )
    else:  # multiview
        results.update(
            {
                "martingale_sum": detection_result.get("martingale_sum", []),
                "martingale_avg": detection_result.get("martingale_avg", []),
                "individual_martingales": detection_result.get(
                    "individual_martingales", []
                ),
                "prediction_martingale_sum": detection_result.get(
                    "prediction_martingale_sum", []
                ),
                "prediction_martingale_avg": detection_result.get(
                    "prediction_martingale_avg", []
                ),
                "prediction_individual_martingales": detection_result.get(
                    "prediction_individual_martingales", []
                ),
                "traditional_delays": [
                    d - tcp
                    for tcp in true_change_points
                    for d in detection_result["change_points"]
                    if d >= tcp
                ],
                "traditional_detection_times": detection_result["change_points"],
                "prediction_delays": [
                    d - tcp
                    for tcp in true_change_points
                    for d in detection_result.get("horizon_change_points", [])
                    if d >= tcp
                ],
                "prediction_detection_times": detection_result.get(
                    "horizon_change_points", []
                ),
            }
        )

    return results


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
        "--threshold",
        type=float,
        default=60.0,
        help="Detection threshold (default: 60.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7,
        help="Sensitivity parameter (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for multiview (default: 1000)",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=None,
        help="Maximum window size (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    results = run_detection(
        model_alias=args.model,
        threshold=args.threshold,
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        max_window=args.max_window,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
