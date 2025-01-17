# src/predictor/run_grid_search.py

"""Grid search script for network forecasting."""

import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Any
import numpy as np
import networkx as nx

sys.path.append(str(Path(__file__).parent.parent))

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor
from config.graph_configs import GRAPH_CONFIGS

from weighted import WeightedPredictor
from grid_search import GridSearch
from hybrid.sbm_predictor import SBMPredictor
from hybrid.ba_predictor import BAPredictor
from hybrid.er_predictor import ERPredictor
from hybrid.ws_predictor import WSPredictor


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Map of predictor types and their graph model variants
PREDICTOR_MAP = {
    "weighted": WeightedPredictor,
    "hybrid": {
        "sbm": SBMPredictor,
        "ba": BAPredictor,
        "er": ERPredictor,
        "ws": WSPredictor,
    },
}

# Define parameter grids for each predictor type
PARAM_GRIDS = {
    "weighted": {
        "n_history": [3, 4, 5, 6, 7],
        "weights": [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.5, 0.3, 0.2]),
            np.array([0.4, 0.3, 0.3]),
            np.array([0.4, 0.4, 0.2]),
            np.array([0.34, 0.33, 0.33]),
        ],
        "adaptive": [True, False],
        "enforce_connectivity": [True, False],
        "binary": [True, False],
    },
    "hybrid": {
        "sbm": {
            "intra_threshold": np.linspace(0.05, 0.15, 6),
            "inter_threshold": np.linspace(0.1, 0.4, 7),
            "intra_hist_weight": np.linspace(0.05, 0.5, 6),
            "inter_hist_weight": np.array([0.0, 0.02, 0.05, 0.08, 0.1]),
        },
        "ba": {
            "attachment_threshold": np.linspace(0.05, 0.4, 8),
            "hist_weight": np.linspace(0.1, 0.7, 7),
        },
        "er": {
            "prob_threshold": np.linspace(0.05, 0.4, 8),
            "hist_weight": np.linspace(0.1, 0.7, 7),
        },
        "rcp": {
            "connection_threshold": np.linspace(0.05, 0.4, 8),
            "hist_weight": np.linspace(0.1, 0.7, 7),
        },
        "ws": {
            "rewiring_threshold": np.linspace(0.05, 0.4, 8),
            "hist_weight": np.linspace(0.1, 0.7, 7),
        },
    },
}


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Grid search for network prediction")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="sbm",
        choices=list(GRAPH_CONFIGS.keys()),
        help="Network model to use for generating data",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="hybrid",
        choices=["weighted", "hybrid"],
        help="Predictor type to use (weighted or hybrid)",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="sbm",
        choices=["sbm", "ba", "er", "ws"],
        help="Graph model to use for hybrid predictor",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_runs", type=int, default=5, help="Number of runs for averaging results"
    )
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

    # Generate sequence with model-specific parameters
    logger.info(f"Generating sequence with model {config['model']}")
    result = generator.generate_sequence(
        model=config["model"], params=config["params"], seed=seed
    )

    # Create a mapping of parameters for each time step
    param_map = {}
    current_params = result["parameters"][0]
    for i in range(config["params"].seq_len):
        # Update parameters at change points
        if i in result["change_points"]:
            current_params = result["parameters"][result["change_points"].index(i) + 1]
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

    logger.info(f"Generated sequence of {len(network_series)} networks")
    return network_series


def main():
    """Run grid search to find optimal hyperparameters."""
    args = get_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Get model config and predictor class
    config = GRAPH_CONFIGS[args.model]()
    logger.info(f"Loaded config for model {args.model}")

    # Get predictor class and parameters based on type
    if args.predictor == "weighted":
        predictor_class = PREDICTOR_MAP["weighted"]
        param_grid = PARAM_GRIDS["weighted"]
    else:  # hybrid predictor
        predictor_class = PREDICTOR_MAP["hybrid"][args.graph_type]
        param_grid = PARAM_GRIDS["hybrid"][args.graph_type]

    logger.info(f"Using predictor class: {predictor_class.__name__}")
    logger.debug(f"Parameter grid: {param_grid}")

    # Initialize results storage
    all_results = []

    predictor_desc = (
        f"{args.predictor}"
        if args.predictor == "weighted"
        else f"{args.predictor}_{args.graph_type}"
    )
    logger.info(
        f"Running grid search for {args.model} data with {predictor_desc} predictor ({args.n_runs} runs)..."
    )

    for run in range(args.n_runs):
        logger.info(f"\nStarting run {run + 1}/{args.n_runs}")

        try:
            # Generate network sequence
            history = generate_network_series(config, seed=args.seed + run)

            # Initialize grid search
            grid_search = GridSearch(predictor_class, param_grid)

            # Run grid search
            best_params, run_results = grid_search.run_grid_search(history)

            # Store results
            all_results.append(
                {"run": run, "best_params": best_params, "results": run_results}
            )

            # Print run results
            logger.info(f"Run {run + 1} best parameters:")
            logger.info(best_params)
            metrics = run_results[str(best_params)]
            logger.info(f"Coverage: {metrics['coverage']:.3f}")
            logger.info(f"FPR: {metrics['fpr']:.3f}")
            logger.info(f"Score: {metrics['score']:.3f}")

        except Exception as e:
            logger.error(f"Error in run {run + 1}: {str(e)}", exc_info=True)
            continue

    if not all_results:
        logger.error("No successful runs completed.")
        return

    # Analyze results across runs
    logger.info("\nAggregating results across runs...")

    # Collect all parameter combinations that were best in any run
    best_param_sets = {}
    for run_result in all_results:
        param_key = str(run_result["best_params"])
        if param_key not in best_param_sets:
            best_param_sets[param_key] = {
                "params": run_result["best_params"],
                "counts": 0,
                "coverage": [],
                "fpr": [],
                "scores": [],
            }

        best_param_sets[param_key]["counts"] += 1
        metrics = run_result["results"][param_key]
        best_param_sets[param_key]["coverage"].append(metrics["coverage"])
        best_param_sets[param_key]["fpr"].append(metrics["fpr"])
        best_param_sets[param_key]["scores"].append(metrics["score"])

    # Print summary of best parameter sets
    logger.info("\nBest parameter combinations across runs:")
    for param_key, stats in sorted(
        best_param_sets.items(), key=lambda x: np.mean(x[1]["scores"]), reverse=True
    ):
        logger.info(f"\nParameters: {stats['params']}")
        logger.info(f"Times selected as best: {stats['counts']}/{args.n_runs}")
        logger.info(
            f"Average Coverage: {np.mean(stats['coverage']):.3f} ± {np.std(stats['coverage']):.3f}"
        )
        logger.info(
            f"Average FPR: {np.mean(stats['fpr']):.3f} ± {np.std(stats['fpr']):.3f}"
        )
        logger.info(
            f"Average Score: {np.mean(stats['scores']):.3f} ± {np.std(stats['scores']):.3f}"
        )


if __name__ == "__main__":
    main()
