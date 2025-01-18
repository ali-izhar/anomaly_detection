"""Utility functions for network prediction and analysis."""

from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import networkx as nx
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor, calculate_error_metrics
from changepoint.detector import ChangePointDetector
from changepoint.threshold import CustomThresholdModel
from predictor.visualizer import Visualizer

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for network experiments."""
    model: str
    params: Dict[str, Any]
    min_history: int
    prediction_window: int
    martingale_threshold: float = 100.0
    martingale_epsilon: float = 0.4
    shap_threshold: float = 30.0
    shap_window_size: int = 5
    n_runs: int = 1
    save_individual: bool = False
    visualize_individual: bool = False

class ExperimentRunner:
    """Class to run network prediction and analysis experiments."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Optional[Path] = None, seed: Optional[int] = None):
        """Initialize the experiment runner.
        
        Args:
            config: ExperimentConfig object containing experiment parameters
            output_dir: Optional output directory for results
            seed: Optional random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.output_dir = output_dir or self._create_output_dir()
        
        # Initialize components
        self.generator = GraphGenerator()
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()
        
        # Feature weights for martingale analysis
        self.feature_weights = {
            "betweenness": 1.0,  # Most reliable
            "clustering": 0.85,   # Most consistent
            "closeness": 0.7,     # Moderate reliability
            "degree": 0.5,        # Least reliable
        }

    def _create_output_dir(self) -> Path:
        """Create and return output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config.n_runs > 1:
            base_dir = Path(f"results/multi_{self.config.model}_{timestamp}")
        else:
            base_dir = Path(f"results/{self.config.model}_{timestamp}")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def run(self) -> Dict[str, Any]:
        """Run the experiment(s).
        
        This is the main entry point that handles both single and multiple runs.
        
        Returns:
            Dict containing results (single run) or aggregated results (multiple runs)
        """
        if self.config.n_runs > 1:
            return self._run_multiple()
        else:
            return self._run_single()

    def _run_single(self) -> Dict[str, Any]:
        """Run a single experiment.
        
        Returns:
            Dict containing experiment results
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate data
        graphs, ground_truth = self._generate_data()

        # Initialize predictor
        predictor = self._initialize_predictor()

        # Compute metrics
        actual_metrics = self._compute_actual_metrics(graphs)
        forecast_metrics = self._compute_forecast_metrics(graphs, predictor)

        results = {
            "graphs": graphs,
            "ground_truth": ground_truth,
            "actual_metrics": actual_metrics,
            "forecast_metrics": forecast_metrics,
            "seed": self.seed,
            "config": self.config,
        }

        self._save_results(results)
        self.visualize_results(results)

        return results

    def _run_multiple(self) -> Dict[str, Any]:
        """Run multiple experiments.
        
        Returns:
            Dict containing aggregated results
        """
        all_results = []
        base_seed = self.seed if self.seed is not None else np.random.randint(0, 10000)

        for i in range(self.config.n_runs):
            logger.info(f"\nRunning experiment {i+1}/{self.config.n_runs}")
            run_seed = base_seed + i if base_seed is not None else None
            
            # Create run-specific output directory if needed
            if self.config.save_individual:
                run_dir = self.output_dir / f"run_{i+1}"
                run_dir.mkdir(exist_ok=True)
            else:
                run_dir = None

            # Create runner for this iteration
            runner = ExperimentRunner(
                config=self.config,
                output_dir=run_dir,
                seed=run_seed
            )
            
            # Disable saving/visualization for individual runs unless specified
            results = runner._run_single()
            all_results.append(results)

        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save aggregated results
        self._save_aggregated_results(aggregated)
        
        return aggregated

    def _generate_data(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate synthetic network data."""
        logger.info(f"Generating {self.config.model} network time series...")
        result = self.generator.generate_sequence(
            model=self.config.model,
            params=self.config.params,
            seed=self.seed
        )

        # Map parameters to timesteps
        param_map = {}
        current_params = result["parameters"][0]
        for i in range(self.config.params.seq_len):
            if i in result["change_points"]:
                current_params = result["parameters"][result["change_points"].index(i) + 1]
            param_map[i] = current_params

        # Convert to network states
        network_states = []
        for i, adj in enumerate(result["graphs"]):
            G = nx.from_numpy_array(adj)
            network_states.append({
                "time": i,
                "adjacency": adj,
                "graph": G,
                "metrics": self.feature_extractor.get_all_metrics(G).__dict__,
                "params": param_map[i],
                "is_change_point": i in result["change_points"],
            })

        ground_truth = {
            "change_points": result["change_points"],
            "parameters": result["parameters"],
            "metadata": result["metadata"],
            "model": result["model"],
            "num_changes": result["num_changes"],
            "n": result["n"],
            "sequence_length": result["sequence_length"],
        }

        return network_states, ground_truth

    def _initialize_predictor(self):
        """Initialize the appropriate predictor based on configuration.
        
        Returns:
            Initialized predictor object
        """
        from predictor.weighted import WeightedPredictor
        from predictor.hybrid import (
            BAPredictor,
            SBMPredictor,
            WSPredictor,
            ERPredictor,
        )

        PREDICTOR_MAP = {
            "weighted": WeightedPredictor,
            "hybrid": {
                "ba": BAPredictor,
                "ws": WSPredictor,
                "er": ERPredictor,
                "sbm": SBMPredictor,
            },
        }

        if not hasattr(self.config, "predictor_type"):
            # Default to weighted predictor if not specified
            logger.info("No predictor type specified, using weighted predictor")
            return WeightedPredictor()

        if self.config.predictor_type == "weighted":
            return WeightedPredictor()
        elif self.config.predictor_type == "hybrid":
            # Get the appropriate hybrid predictor for the model type
            predictor_class = PREDICTOR_MAP["hybrid"].get(self.config.model)
            if predictor_class is None:
                raise ValueError(f"No hybrid predictor available for model type {self.config.model}")
            return predictor_class(config=self.config)
        else:
            raise ValueError(f"Unknown predictor type: {self.config.predictor_type}")

    def _compute_actual_metrics(self, graphs: List[Dict[str, Any]]) -> tuple:
        """Compute actual network metrics."""
        features = self.compute_network_features(graphs[self.config.min_history:])
        martingales = self.compute_martingales(features)
        shap_values = self.compute_shap_values(martingales, features)
        return features, martingales, shap_values

    def _compute_forecast_metrics(self, graphs: List[Dict[str, Any]], predictor) -> tuple:
        """Compute forecasting metrics."""
        predictions = self.generate_predictions(graphs, predictor)
        features = self.compute_network_features(predictions)
        martingales = self.compute_martingales(features)
        shap_values = self.compute_shap_values(martingales, features)
        return predictions, features, martingales, shap_values

    def _save_results(self, results: Dict[str, Any]):
        """Save individual experiment results."""
        if not self.output_dir:
            return

        try:
            serializable_data = {
                "model": self.config.model,
                "parameters": self._convert_to_serializable(self.config),
                "ground_truth": self._convert_to_serializable(results["ground_truth"]),
                "seed": self.seed,
            }

            with open(self.output_dir / "results.json", "w") as f:
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            # Save in a simpler format
            simple_data = {
                "model": self.config.model,
                "seed": self.seed,
                "error": str(e)
            }
            with open(self.output_dir / "results_error.json", "w") as f:
                json.dump(simple_data, f, indent=4)

    def _save_aggregated_results(self, aggregated: Dict[str, Any]):
        """Save aggregated results from multiple runs."""
        with open(self.output_dir / "aggregated_results.json", "w") as f:
            json.dump(self._convert_to_serializable(aggregated), f, indent=4)

        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Average number of change points: {aggregated['summary']['change_points']['mean']:.2f} ± {aggregated['summary']['change_points']['std']:.2f}")
        print(f"Average prediction accuracy: {aggregated['summary']['prediction_accuracy']['mean']:.2f} ± {aggregated['summary']['prediction_accuracy']['std']:.2f}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy types and custom objects to Python native types for JSON serialization."""
        if obj is None:
            return None
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        elif hasattr(obj, '__dict__'):  # Handle custom classes and dataclasses
            return {
                key: ExperimentRunner._convert_to_serializable(value)
                for key, value in obj.__dict__.items()
                if not key.startswith('_')  # Skip private attributes
            }
        elif isinstance(obj, dict):
            return {
                str(key): ExperimentRunner._convert_to_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [ExperimentRunner._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [ExperimentRunner._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return obj.tolist()
        elif hasattr(obj, '__str__'):  # Fallback for other objects
            return str(obj)
        return obj

    def _aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple runs."""
        aggregated = {
            "model": all_results[0]["ground_truth"]["model"],
            "n_runs": len(all_results),
            "metrics": {
                "change_points": [],
                "prediction_accuracy": [],
                "martingale_stats": {
                    "reset": {},
                    "cumulative": {},
                }
            }
        }
        
        for results in all_results:
            # Collect change points
            aggregated["metrics"]["change_points"].append(
                results["ground_truth"]["change_points"]
            )
            
            # Collect prediction accuracy
            phase_analysis = self.analyze_results(results)
            aggregated["metrics"]["prediction_accuracy"].append(phase_analysis)
            
            # Collect martingale statistics
            for feature in results["forecast_metrics"][2]["reset"]:
                if feature not in aggregated["metrics"]["martingale_stats"]["reset"]:
                    aggregated["metrics"]["martingale_stats"]["reset"][feature] = []
                    aggregated["metrics"]["martingale_stats"]["cumulative"][feature] = []
                
                aggregated["metrics"]["martingale_stats"]["reset"][feature].append(
                    results["forecast_metrics"][2]["reset"][feature]["martingales"]
                )
                aggregated["metrics"]["martingale_stats"]["cumulative"][feature].append(
                    results["forecast_metrics"][2]["cumulative"][feature]["martingales"]
                )
        
        # Compute summary statistics
        aggregated["summary"] = {
            "change_points": {
                "mean": np.mean([len(cp) for cp in aggregated["metrics"]["change_points"]]),
                "std": np.std([len(cp) for cp in aggregated["metrics"]["change_points"]]),
            },
            "prediction_accuracy": {
                "mean": np.mean([
                    run["phases"]["All predictions"]["errors"]["overall"]
                    for run in aggregated["metrics"]["prediction_accuracy"]
                ]),
                "std": np.std([
                    run["phases"]["All predictions"]["errors"]["overall"]
                    for run in aggregated["metrics"]["prediction_accuracy"]
                ]),
            },
            "martingale_stats": {
                "reset": {
                    feature: {
                        "mean": np.mean(values, axis=0),
                        "std": np.std(values, axis=0),
                    }
                    for feature, values in aggregated["metrics"]["martingale_stats"]["reset"].items()
                },
                "cumulative": {
                    feature: {
                        "mean": np.mean(values, axis=0),
                        "std": np.std(values, axis=0),
                    }
                    for feature, values in aggregated["metrics"]["martingale_stats"]["cumulative"].items()
                }
            }
        }
        
        return aggregated

    def generate_predictions(
        self,
        network_series: List[Dict[str, Any]],
        predictor: Any,
    ) -> List[Dict[str, Any]]:
        """Generate predictions using the given predictor.
        
        Args:
            network_series: List of network states
            predictor: Network predictor object
            
        Returns:
            List of prediction results
        """
        predictions = []
        seq_len = len(network_series)  # Get sequence length from input data

        for t in range(self.config.min_history, seq_len):
            history = network_series[:t]
            predicted_adjs = predictor.predict(history, horizon=self.config.prediction_window)

            pred_graph = nx.from_numpy_array(predicted_adjs[0])
            predictions.append({
                "time": t,
                "adjacency": predicted_adjs[0],
                "graph": pred_graph,
                "metrics": self.feature_extractor.get_all_metrics(pred_graph).__dict__,
                "history_size": len(history),
            })

        return predictions

    def compute_network_features(self, graphs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Compute network features for analysis.
        
        Args:
            graphs: List of network states
            
        Returns:
            Dict of computed features
        """
        features = {
            "degree": [],
            "clustering": [],
            "betweenness": [],
            "closeness": [],
        }

        for state in graphs:
            G = state["graph"]
            metrics = self.feature_extractor.get_all_metrics(G)
            features["degree"].append(metrics.avg_degree)
            features["clustering"].append(metrics.clustering)
            features["betweenness"].append(metrics.avg_betweenness)
            features["closeness"].append(metrics.avg_closeness)

        return {k: np.array(v) for k, v in features.items()}

    def analyze_results(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze prediction accuracy across different phases.
        
        Args:
            results: Dict containing experiment results
            
        Returns:
            Dict containing analysis results
        """
        if not results["forecast_metrics"]:
            logger.warning("No forecast metrics to analyze")
            return {"phases": {}, "error": "No data to analyze"}

        # Define phases
        total_predictions = len(results["forecast_metrics"])
        if total_predictions < 3:
            phases = [(0, total_predictions, "All predictions")]
        else:
            third = total_predictions // 3
            phases = [
                (0, third, "First third"),
                (third, 2 * third, "Middle third"),
                (2 * third, None, "Last third"),
            ]

        results = {"phases": {}}

        # Analyze each phase
        for start, end, phase_name in phases:
            logger.info(f"\nAnalyzing {phase_name}:")
            phase_predictions = results["forecast_metrics"][start:end]
            phase_actuals = results["graphs"][
                self.config.min_history + start : self.config.min_history + (end if end else len(results["forecast_metrics"]))
            ]

            # Calculate errors
            all_errors = []
            for pred, actual in zip(phase_predictions, phase_actuals):
                pred_metrics = self.feature_extractor.get_all_metrics(pred["graph"]).__dict__
                actual_metrics = self.feature_extractor.get_all_metrics(actual["graph"]).__dict__
                all_errors.append(calculate_error_metrics(actual_metrics, pred_metrics))

            if not all_errors:
                logger.warning(f"No errors calculated for {phase_name}")
                continue

            # Average errors
            avg_errors = {
                metric: np.mean([e[metric] for e in all_errors])
                for metric in all_errors[0].keys()
            }

            results["phases"][phase_name] = {
                "start": start,
                "end": end,
                "errors": avg_errors,
            }

            for metric, error in avg_errors.items():
                logger.info(f"Average MAE for {metric}: {error:.3f}")

        return results

    def compute_martingales(
        self,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute martingales for network features.
        
        Args:
            features: Dict of network features
            
        Returns:
            Dict containing martingale analysis results
        """
        martingales = {"reset": {}, "cumulative": {}}

        for feature_name, feature_data in features.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            adjusted_threshold = self.config.martingale_threshold / weight

            # Reset martingales
            reset_results = self.detector.detect_changes(
                data=feature_data.reshape(-1, 1),
                threshold=adjusted_threshold,
                epsilon=self.config.martingale_epsilon,
                reset=True,
            )

            weighted_martingales = reset_results["martingale_values"] * weight
            martingales["reset"][feature_name] = {
                "martingales": weighted_martingales,
                "change_detected_instant": reset_results["change_points"],
                "weight": weight,
                "adjusted_threshold": adjusted_threshold,
            }

            # Cumulative martingales
            cumul_results = self.detector.detect_changes(
                data=feature_data.reshape(-1, 1),
                threshold=adjusted_threshold,
                epsilon=self.config.martingale_epsilon,
                reset=False,
            )

            weighted_cumul_martingales = cumul_results["martingale_values"] * weight
            martingales["cumulative"][feature_name] = {
                "martingales": weighted_cumul_martingales,
                "change_detected_instant": cumul_results["change_points"],
                "weight": weight,
                "adjusted_threshold": adjusted_threshold,
            }

        return martingales

    def compute_shap_values(
        self,
        martingales: Dict[str, Dict[str, Dict[str, Any]]],
        features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute SHAP values for martingale analysis.
        
        Args:
            martingales: Dict of martingale results
            features: Dict of network features
            
        Returns:
            numpy array of SHAP values
        """
        model = CustomThresholdModel(threshold=self.config.shap_threshold)

        feature_matrix = np.column_stack(
            [martingales["reset"][feature]["martingales"] for feature in features.keys()]
        )

        change_points = sorted(list(set(
            cp for m in martingales["reset"].values()
            for cp in m["change_detected_instant"]
        )))

        return model.compute_shap_values(
            X=feature_matrix,
            change_points=change_points,
            sequence_length=len(feature_matrix),
            window_size=self.config.shap_window_size,
        )

    def visualize_results(self, results: Dict[str, Any]):
        """Generate visualizations for the results."""
        visualizer = Visualizer()

        # 1. Metric evolution
        plt.figure(figsize=(12, 8))
        visualizer.plot_metric_evolution(
            results["graphs"],
            results["forecast_metrics"][0],  # predictions
            self.config.min_history,
            model_type=self.config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "metric_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Performance extremes
        plt.figure(figsize=(20, 15))
        visualizer.plot_performance_extremes(
            results["graphs"][
                self.config.min_history : self.config.min_history
                + len(results["forecast_metrics"][0])
            ],
            results["forecast_metrics"][0],
            min_history=self.config.min_history,
            model_type=self.config.model,
            change_points=results["ground_truth"]["change_points"],
        )
        plt.savefig(
            self.output_dir / "performance_extremes.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Martingale comparison dashboard
        visualizer.create_martingale_comparison_dashboard(
            network_series=results["graphs"],
            actual_martingales=results["actual_metrics"][1],
            pred_martingales=results["forecast_metrics"][2],
            actual_shap=results["actual_metrics"][2],
            pred_shap=results["forecast_metrics"][3],
            output_path=self.output_dir / "martingale_comparison_dashboard.png",
            threshold=self.config.martingale_threshold,
            epsilon=self.config.martingale_epsilon,
            change_points=results["ground_truth"]["change_points"],
            prediction_window=self.config.prediction_window,
        )
