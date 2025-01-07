# src/compare.py

"""Compare performance of weighted and hybrid SBM predictors."""

import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Any
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor
from predictor.weighted import WeightedPredictor
from predictor.hybrid.sbm_predictor import SBMPredictor
from config.graph_configs import GRAPH_CONFIGS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare predictor performance")
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of runs for averaging results"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def generate_network_series(
    config: Dict[str, Any], seed: int = None
) -> List[Dict[str, Any]]:
    """Generate a time series of evolving networks."""
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")

    # Initialize generator and feature extractor
    generator = GraphGenerator()
    feature_extractor = NetworkFeatureExtractor()
    logger.debug("Initialized generator and feature extractor")

    # Generate sequence with model-specific parameters
    logger.info(f"Generating sequence with model {config['model']}")
    result = generator.generate_sequence(
        model=config["model"], params=config["params"], seed=seed
    )
    logger.debug(f"Generated sequence with {len(result['graphs'])} graphs")

    # Create a mapping of parameters for each time step
    param_map = {}
    current_params = result["parameters"][0]
    for i in range(config["params"].seq_len):
        if i in result["change_points"]:
            current_params = result["parameters"][result["change_points"].index(i) + 1]
            logger.debug(f"Parameter change at step {i}: {current_params}")
        param_map[i] = current_params

    # Convert to list of network states
    network_series = []
    for i, adj in enumerate(result["graphs"]):
        G = nx.from_numpy_array(adj)
        network_series.append(
            {
                "time": i,
                "adjacency": adj,
                "graph": G,
                "metrics": feature_extractor.get_all_metrics(G).__dict__,
                "params": param_map[i],
                "is_change_point": i in result["change_points"],
            }
        )
        logger.debug(
            f"Processed graph {i}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
        )

    logger.info(f"Generated sequence of {len(network_series)} networks")
    return network_series


def compute_metrics(true_adj: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute coverage and FPR metrics."""
    # Convert networkx graph to numpy array if needed
    if isinstance(true_adj, nx.Graph):
        true_adj = nx.to_numpy_array(true_adj)
        logger.debug("Converted networkx graph to numpy array")

    # Compute coverage
    true_edges = np.where(true_adj > 0)
    correct = np.sum(pred[true_edges] > 0)
    total_edges = np.sum(true_adj > 0)
    coverage = correct / total_edges if total_edges > 0 else 0.0
    logger.debug(f"Coverage: {coverage:.3f} (correct={correct}, total={total_edges})")

    # Compute FPR
    non_edges = np.where(true_adj == 0)
    false_pos = np.sum(pred[non_edges] > 0)
    total_non_edges = len(non_edges[0])
    fpr = false_pos / total_non_edges if total_non_edges > 0 else 0
    logger.debug(f"FPR: {fpr:.3f} (false_pos={false_pos}, total_non={total_non_edges})")

    score = coverage - fpr
    logger.debug(f"Final score: {score:.3f}")

    return {"coverage": coverage, "fpr": fpr, "score": score}


def evaluate_predictor(
    predictor, history: List[Dict[str, Any]], min_history: int = 10
) -> List[Dict[str, float]]:
    """Evaluate predictor performance over the entire sequence."""
    logger.info(
        f"Evaluating {predictor.__class__.__name__} with {len(history)} timesteps"
    )
    metrics_list = []

    # Start predictions after min_history steps
    for i in range(min_history, len(history) - 1):
        logger.debug(f"Making prediction for step {i}")
        current_history = history[:i]
        true_next = history[i]["graph"]

        try:
            # Make prediction
            pred = predictor.predict(current_history, horizon=1)[0]
            logger.debug(f"Made prediction for step {i}, shape: {pred.shape}")

            # Compute metrics
            metrics = compute_metrics(true_next, pred)
            metrics_list.append(metrics)
            logger.debug(
                f"Step {i} metrics: coverage={metrics['coverage']:.3f}, fpr={metrics['fpr']:.3f}, score={metrics['score']:.3f}"
            )

        except Exception as e:
            logger.error(f"Error making prediction for step {i}: {str(e)}")
            continue

    logger.info(f"Completed evaluation with {len(metrics_list)} valid predictions")
    return metrics_list


def main():
    """Run comparison between weighted and hybrid predictors."""
    args = get_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Get SBM config
    config = GRAPH_CONFIGS["sbm"]()
    logger.info("Loaded SBM config")

    # Initialize results storage
    results = defaultdict(list)

    logger.info(f"Running comparison over {args.n_runs} runs...")

    for run in range(args.n_runs):
        logger.info(f"\nStarting run {run + 1}/{args.n_runs}")

        try:
            # Generate network sequence
            history = generate_network_series(config, seed=args.seed + run)

            # Initialize predictors
            logger.debug("Initializing predictors")
            weighted_predictor = WeightedPredictor()
            hybrid_predictor = SBMPredictor()

            # Evaluate both predictors
            logger.info("Evaluating weighted predictor")
            weighted_metrics = evaluate_predictor(weighted_predictor, history)
            logger.info("Evaluating hybrid predictor")
            hybrid_metrics = evaluate_predictor(hybrid_predictor, history)

            # Store results
            results["weighted"].append(weighted_metrics)
            results["hybrid"].append(hybrid_metrics)

            # Print run summary
            for predictor in ["weighted", "hybrid"]:
                metrics = results[predictor][-1]
                avg_coverage = np.mean([m["coverage"] for m in metrics])
                avg_fpr = np.mean([m["fpr"] for m in metrics])
                avg_score = np.mean([m["score"] for m in metrics])

                logger.info(f"\n{predictor.capitalize()} Predictor - Run {run + 1}:")
                logger.info(f"Average Coverage: {avg_coverage:.3f}")
                logger.info(f"Average FPR: {avg_fpr:.3f}")
                logger.info(f"Average Score: {avg_score:.3f}")

        except Exception as e:
            logger.error(f"Error in run {run + 1}: {str(e)}", exc_info=True)
            continue

    if not any(results.values()):
        logger.error("No successful runs completed")
        return

    # Print final aggregated results
    logger.info("\nComputing final aggregated results...")
    for predictor in ["weighted", "hybrid"]:
        # Compute means across all runs and time steps
        all_coverage = [[m["coverage"] for m in run] for run in results[predictor]]
        all_fpr = [[m["fpr"] for m in run] for run in results[predictor]]
        all_score = [[m["score"] for m in run] for run in results[predictor]]

        mean_coverage = np.mean([np.mean(run) for run in all_coverage])
        std_coverage = np.std([np.mean(run) for run in all_coverage])
        mean_fpr = np.mean([np.mean(run) for run in all_fpr])
        std_fpr = np.std([np.mean(run) for run in all_fpr])
        mean_score = np.mean([np.mean(run) for run in all_score])
        std_score = np.std([np.mean(run) for run in all_score])

        logger.info(f"\n{predictor.capitalize()} Predictor Overall:")
        logger.info(f"Coverage: {mean_coverage:.3f} ± {std_coverage:.3f}")
        logger.info(f"FPR: {mean_fpr:.3f} ± {std_fpr:.3f}")
        logger.info(f"Score: {mean_score:.3f} ± {std_score:.3f}")


if __name__ == "__main__":
    main()
