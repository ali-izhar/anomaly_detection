# src/compare.py

"""Network predictor comparison framework.

This module provides functionality to compare different network predictors
across various network types and metrics. It generates synthetic networks
using different models, applies multiple predictors, and evaluates their
performance using various metrics.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Type
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import networkx as nx

from predictor import (
    BasePredictor,
    WeightedPredictor,
    SpectralPredictor,
    EmbeddingPredictor,
    DynamicalPredictor,
    EnsemblePredictor,
    AdaptivePredictor,
    HybridPredictor,
)
from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from config.graph_configs import GRAPH_CONFIGS

logger = logging.getLogger(__name__)

# Mapping from short names to full model names
MODEL_NAME_MAP = {
    "ba": "barabasi_albert",
    "er": "erdos_renyi",
    "ws": "watts_strogatz",
    "sbm": "stochastic_block_model",
    "rcp": "random_core_periphery",
    # "lfr": "lfr_benchmark",
}


class PredictorComparison:
    """Framework for comparing network predictors.

    Parameters
    ----------
    predictors : Optional[List[Type[BasePredictor]]], optional
        List of predictor classes to compare, by default None
    n_trials : int, optional
        Number of trials per configuration, by default 5
    sequence_length : int, optional
        Length of network sequence to generate, by default 200
    prediction_steps : int, optional
        Number of steps to predict ahead, by default 5
    save_dir : Optional[str], optional
        Directory to save results, by default None
    """

    def __init__(
        self,
        predictors: Optional[List[Type[BasePredictor]]] = None,
        n_trials: int = 5,
        sequence_length: int = 200,
        prediction_steps: int = 5,
        save_dir: Optional[str] = None,
    ):
        """Initialize comparison framework."""
        if predictors is None:
            predictors = [
                WeightedPredictor,
                SpectralPredictor,
                EmbeddingPredictor,
                DynamicalPredictor,
                EnsemblePredictor,
                AdaptivePredictor,
                HybridPredictor,
            ]

        self.predictors = predictors
        self.n_trials = n_trials
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.save_dir = Path(save_dir) if save_dir else None

        # Initialize components
        self.generator = GraphGenerator()
        self.feature_extractor = NetworkFeatureExtractor()

        # Store results
        self.results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    def run_comparison(
        self,
        model_types: Optional[List[str]] = None,
        n_nodes: int = 100,
        min_changes: int = 2,
        max_changes: int = 5,
        min_segment: int = 20,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
        """Run comparison experiments.

        Parameters
        ----------
        model_types : Optional[List[str]], optional
            Graph models to test (short names), by default None
        n_nodes : int, optional
            Number of nodes in networks, by default 100
        min_changes : int, optional
            Minimum number of change points, by default 2
        max_changes : int, optional
            Maximum number of change points, by default 5
        min_segment : int, optional
            Minimum length between changes, by default 20
        seed : Optional[int], optional
            Random seed for reproducibility, by default None

        Returns
        -------
        Dict[str, Dict[str, List[Dict[str, float]]]]
            Results dictionary
        """
        if model_types is None:
            model_types = list(MODEL_NAME_MAP.keys())

        # Initialize results structure
        self.results = {
            model: {pred.__name__: [] for pred in self.predictors}
            for model in model_types
        }

        # Run experiments for each model type
        for model in model_types:
            logger.info(f"Testing {model} networks...")

            # Get model configuration
            config = GRAPH_CONFIGS[MODEL_NAME_MAP[model]](
                n=n_nodes,
                seq_len=self.sequence_length,
                min_segment=min_segment,
                min_changes=min_changes,
                max_changes=max_changes,
            )

            # Run multiple trials
            for trial in range(self.n_trials):
                logger.info(f"Trial {trial + 1}/{self.n_trials}")

                # Generate network sequence
                sequence = generate_network_series(config, seed=seed)

                # Test each predictor
                for predictor_class in self.predictors:
                    metrics = self._evaluate_predictor(
                        predictor_class,
                        sequence,
                    )
                    self.results[model][predictor_class.__name__].append(metrics)

        # Save results if directory specified
        if self.save_dir:
            self._save_results()

        return self.results

    def _get_model_config(self, model: str, n_nodes: int) -> Dict[str, Any]:
        """Get configuration for graph model.

        Parameters
        ----------
        model : str
            Graph model type (short name)
        n_nodes : int
            Number of nodes

        Returns
        -------
        Dict[str, Any]
            Model configuration
        """
        # Convert short name to full name
        if model not in MODEL_NAME_MAP:
            raise ValueError(f"Unknown model type: {model}")

        full_name = MODEL_NAME_MAP[model]
        if full_name not in GRAPH_CONFIGS:
            raise ValueError(f"No configuration found for model: {full_name}")

        # Get configuration by calling the config function
        config = GRAPH_CONFIGS[full_name](n_nodes)
        return config

    def _evaluate_predictor(
        self,
        predictor_class: Type[BasePredictor],
        sequence: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluate a predictor on a network sequence.

        Parameters
        ----------
        predictor_class : Type[BasePredictor]
            Predictor class to evaluate
        sequence : List[Dict[str, Any]]
            Network sequence to test on

        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        predictor = predictor_class()
        all_errors = []

        # Split sequence into history and future
        history = sequence[: -self.prediction_steps]
        future = sequence[-self.prediction_steps :]

        # Make predictions
        predicted_adjs = predictor.predict(
            history,
            horizon=self.prediction_steps,
        )

        # Compute errors for each prediction
        for pred_adj, true in zip(predicted_adjs, future):
            # Convert predicted adjacency to graph
            pred_graph = nx.from_numpy_array(pred_adj)

            # Get network features
            pred_features = self.feature_extractor.get_all_metrics(pred_graph)
            true_features = self.feature_extractor.get_all_metrics(true["graph"])

            # Convert to dictionaries
            pred_dict = {k: float(v) for k, v in pred_features.__dict__.items()}
            true_dict = {k: float(v) for k, v in true_features.__dict__.items()}

            # Calculate errors
            errors = calculate_error_metrics(true_dict, pred_dict)
            all_errors.append(errors)

        # Compute average errors across predictions
        avg_errors = {}
        for metric in all_errors[0].keys():
            values = [e[metric] for e in all_errors]
            avg_errors[metric] = float(np.mean(values))

        return avg_errors

    def _save_results(self) -> None:
        """Save results to files."""
        # Create save directory if needed
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.save_dir / f"comparison_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Generate and save plots
        self._plot_results(timestamp)

    def _plot_results(self, timestamp: str) -> None:
        """Generate comparison plots.

        Parameters
        ----------
        timestamp : str
            Timestamp for file naming
        """
        # Plot settings
        plt.style.use("seaborn-v0_8-darkgrid")

        # Get metrics from first result
        first_model = next(iter(self.results))
        first_predictor = next(iter(self.results[first_model]))
        first_trial = self.results[first_model][first_predictor][0]
        metrics = list(first_trial.keys())

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Prepare data for plotting
            models = list(self.results.keys())
            x = np.arange(len(models))
            width = 0.8 / len(self.predictors)

            for i, pred_class in enumerate(self.predictors):
                pred_name = pred_class.__name__
                means = []
                stds = []

                for model in models:
                    values = [r[metric] for r in self.results[model][pred_name]]
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                # Plot bars with error bars
                ax.bar(
                    x + i * width - width * len(self.predictors) / 2,
                    means,
                    width,
                    yerr=stds,
                    label=pred_name,
                    capsize=3,
                )

            # Customize plot
            ax.set_ylabel(metric)
            ax.set_title(f"Comparison of {metric} Across Models")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_NAME_MAP[m] for m in models])
            ax.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(
                self.save_dir / f"comparison_{metric}_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def statistical_analysis(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Perform statistical analysis of results.

        Returns
        -------
        Dict[str, Dict[str, Dict[str, float]]]
            Dictionary of p-values for predictor comparisons
        """
        analysis = {}

        for model in self.results:
            analysis[model] = {}

            # Get all metrics from first result
            metrics = list(
                next(iter(next(iter(self.results.values())).values()))[0].keys()
            )

            for metric in metrics:
                analysis[model][metric] = {}

                # Compare each pair of predictors
                for i, pred1 in enumerate(self.predictors):
                    name1 = pred1.__name__

                    for pred2 in self.predictors[i + 1 :]:
                        name2 = pred2.__name__

                        # Get metric values for both predictors
                        values1 = [r[metric] for r in self.results[model][name1]]
                        values2 = [r[metric] for r in self.results[model][name2]]

                        # Perform t-test
                        _, p_value = ttest_ind(values1, values2)

                        analysis[model][metric][f"{name1}_vs_{name2}"] = float(p_value)

        return analysis


def generate_network_series(
    config: Dict[str, Any], seed: int = None
) -> List[Dict[str, Any]]:
    """Generate network time series.

    Parameters
    ----------
    config : Dict[str, Any]
        Network configuration
    seed : int, optional
        Random seed for reproducibility, by default None

    Returns
    -------
    List[Dict[str, Any]]
        List of network states
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info("Extracting parameters...")
    # Extract parameters
    model = config["model"]
    params = config["params"]  # This is already a parameter object

    logger.info("Generating network sequence...")
    # Generate the entire sequence at once
    generator = GraphGenerator()
    result = generator.generate_sequence(model, params, seed=seed)

    # Process each graph in the sequence
    network_series = []
    for t, adj_matrix in enumerate(result["graphs"]):
        # Convert adjacency matrix to graph
        graph = nx.from_numpy_array(adj_matrix)

        # Determine if this is a change point
        is_change_point = t in result["change_points"]

        # Find current segment
        segment = next(
            (i for i, cp in enumerate(result["change_points"]) if t < cp),
            len(result["change_points"]),
        )

        network_series.append(
            {
                "time": t,
                "graph": graph,
                "adjacency": adj_matrix,
                "is_change_point": is_change_point,
                "segment": segment,
                "params": result["parameters"][
                    segment
                ],  # Parameters are already dictionaries
                "metadata": result["metadata"][segment] if result["metadata"] else {},
            }
        )

    logger.info(f"Generated network sequence with {len(network_series)} states")
    return network_series


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize comparison framework
    comparison = PredictorComparison(
        predictors=[
            WeightedPredictor,
            SpectralPredictor,
            EmbeddingPredictor,
            DynamicalPredictor,
            EnsemblePredictor,
            AdaptivePredictor,
            HybridPredictor,
        ],
        n_trials=2,  # Back to 5 trials
        sequence_length=100,  # Back to normal sequence length
        prediction_steps=5,
        save_dir="results/comparison",
    )

    # Run comparison
    logger.info("Starting comparison across all models...")
    results = comparison.run_comparison(
        model_types=["ba", "er", "ws", "sbm", "rcp"],  # Test all models
        n_nodes=50,  # Back to normal node count
        min_changes=1,
        max_changes=2,  # Back to normal max changes
        min_segment=30,  # Back to normal segment length
    )

    # Print summary
    for model in results:
        print(f"\nResults for {model} networks:")
        for pred_name, metrics_list in results[model].items():
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()
            }
            print(f"\n{pred_name}:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

    # Print statistical analysis
    analysis = comparison.statistical_analysis()
    print("\nStatistical Analysis:")
    for model in analysis:
        print(f"\n{model} networks:")
        for metric in analysis[model]:
            print(f"\n  {metric}:")
            for comparison_name, p_value in analysis[model][metric].items():
                print(f"    {comparison_name}: p = {p_value:.4f}")
