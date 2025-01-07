# src/predictor/grid_search.py

import numpy as np
from typing import Dict, List, Tuple
import itertools
import networkx as nx
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GridSearch:
    def __init__(self, predictor_class, param_grid: Dict):
        self.predictor_class = predictor_class
        self.param_grid = param_grid
        logger.info(f"Initialized GridSearch with predictor {predictor_class.__name__}")
        logger.info(f"Parameter grid: {param_grid}")

    def evaluate_params(self, args) -> Tuple[Dict, Dict[str, float]]:
        """Evaluate a single parameter combination."""
        param_dict, history = args
        try:
            # Create predictor with these parameters
            predictor = self.predictor_class(**param_dict)

            # Make prediction
            pred = predictor.predict(history, horizon=1)[0]

            # Get the true adjacency matrix
            true_G = history[-1]["graph"]
            true_adj = nx.to_numpy_array(true_G)

            # Evaluate metrics
            coverage = self.compute_coverage(true_adj, pred)
            fpr = self.compute_fpr(true_adj, pred)
            score = coverage - fpr

            return param_dict, {
                "coverage": coverage,
                "fpr": fpr,
                "score": score,
            }
        except Exception as e:
            logger.error(f"Error evaluating parameters {param_dict}: {str(e)}")
            return None

    def run_grid_search(self, history: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Run parallel grid search over hyperparameters.

        Args:
            history: List of historical network states

        Returns:
            best_params: Dictionary of best parameters
            results: Dictionary of all results
        """
        logger.info("Starting grid search...")
        logger.info(f"History length: {len(history)}")

        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        param_dicts = [dict(zip(param_names, params)) for params in param_combinations]

        n_combinations = len(param_combinations)
        logger.info(f"Testing {n_combinations} parameter combinations")

        # Prepare arguments for parallel processing
        args = [(params, history) for params in param_dicts]

        # Use half of available CPUs to avoid overloading
        n_processes = max(1, cpu_count() // 2)
        logger.info(f"Using {n_processes} processes for parallel grid search")

        best_score = float("-inf")
        best_params = None
        results = {}

        # Run parallel grid search with progress bar
        with Pool(n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(self.evaluate_params, args),
                total=n_combinations,
                desc="Grid Search Progress",
            ):
                if result is None:
                    continue

                param_dict, metrics = result
                param_key = str(param_dict)
                results[param_key] = {"params": param_dict, **metrics}

                # Update best if better
                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    best_params = param_dict
                    logger.info(
                        f"New best score: {best_score:.3f} with params: {param_dict}"
                    )

        if best_params is None:
            raise RuntimeError("No valid parameter combinations found")

        logger.info("Grid search completed")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.3f}")

        return best_params, results

    def compute_coverage(self, true_adj: np.ndarray, pred: np.ndarray) -> float:
        """Compute edge coverage."""
        true_edges = np.where(true_adj > 0)
        correct = np.sum(pred[true_edges] > 0)
        total_edges = np.sum(true_adj > 0)
        return correct / total_edges if total_edges > 0 else 0.0

    def compute_fpr(self, true_adj: np.ndarray, pred: np.ndarray) -> float:
        """Compute false positive rate."""
        non_edges = np.where(true_adj == 0)
        false_pos = np.sum(pred[non_edges] > 0)
        return false_pos / len(non_edges[0]) if len(non_edges[0]) > 0 else 0.0


if __name__ == "__main__":

    param_grid = {
        "intra_threshold": np.linspace(0.08, 0.12, 5),  # Fine-tune around 0.1
        "inter_threshold": np.linspace(0.1, 0.3, 5),  # Explore lower values
        "intra_hist_weight": np.linspace(0.1, 0.4, 4),  # Explore lower values
        "inter_hist_weight": np.array([0.0, 0.02, 0.05]),  # Focus on very low values
    }

    # grid_search = GridSearch(HybridPredictor, param_grid)
    # best_params, results = grid_search.run_grid_search(history)
    # print(best_params)
    # print(results)
