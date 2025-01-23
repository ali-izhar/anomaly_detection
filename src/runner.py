# src/runner.py

"""Experiment runner for network prediction and analysis."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import networkx as nx
import logging
import json
from datetime import datetime

from graph.generator import GraphGenerator
from graph.features import NetworkFeatureExtractor
from changepoint.detector import ChangePointDetector
from src.setup.visualization import Visualizer
from src.setup.metrics import MetricComputer
from src.setup.config import (
    ExperimentConfig,
    OutputConfig,
    VisualizationConfig,
    PreprocessingConfig,
    LoggingConfig,
)
from src.setup.prediction import PredictorFactory, NetworkPredictor
from src.setup.aggregate import ResultAggregator

# Setup logging
logging.basicConfig(
    level=getattr(logging, LoggingConfig().level),
    format=LoggingConfig().format,
    datefmt=LoggingConfig().date_format,
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs network prediction and analysis experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_config: Optional[OutputConfig] = None,
        vis_config: Optional[VisualizationConfig] = None,
        preproc_config: Optional[PreprocessingConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize with experiment configuration."""
        # Core config
        self.config = config
        self.output_config = output_config or OutputConfig()
        self.vis_config = vis_config or VisualizationConfig()
        self.preproc_config = preproc_config or PreprocessingConfig()
        self.seed = seed
        self.output_dir = self.output_config.output_dir or self._create_output_dir()

        # Initialize components
        self.generator = GraphGenerator()
        self.feature_extractor = NetworkFeatureExtractor()
        self.detector = ChangePointDetector()
        self.metric_computer = MetricComputer(self.feature_extractor, self.detector)
        self.visualizer = Visualizer(
            vis_config=self.vis_config,
            output_config=self.output_config,
            metric_computer=self.metric_computer,
        )
        self.result_aggregator = ResultAggregator(
            n_runs=self.config.n_runs,
            min_history=self.config.min_history,
        )

        # Initialize predictor
        predictor = PredictorFactory.create_predictor(
            predictor_type=self.config.predictor_type,
            model=self.config.model,
            min_history=self.config.min_history,
            prediction_window=self.config.prediction_window,
        )
        self.network_predictor = NetworkPredictor(
            predictor=predictor,
            feature_extractor=self.feature_extractor,
            min_history=self.config.min_history,
            prediction_window=self.config.prediction_window,
        )

        # Set feature weights from preprocessing config
        self.feature_weights = self.preproc_config.feature_weights

    def _create_output_dir(self) -> Path:
        """Create timestamped output directory."""
        timestamp = datetime.now().strftime(self.output_config.timestamp_format)
        if self.config.n_runs > 1:
            base_dir = Path(
                f"{self.output_config.results_dir}/multi_{self.config.model}_{timestamp}"
            )
        else:
            base_dir = Path(
                f"{self.output_config.results_dir}/{self.config.model}_{timestamp}"
            )

        base_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_dir = base_dir
        return base_dir

    def run(self) -> Dict[str, Any]:
        """Run single or multiple experiments based on config."""
        if self.config.n_runs > 1:
            return self._run_multiple()
        else:
            return self._run_single()

    def _run_single(
        self, run_number: Optional[int] = None, reuse_dir: bool = False
    ) -> Dict[str, Any]:
        """Run a single experiment."""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Step 1: Generate synthetic network data
        graphs, ground_truth = self._generate_data()

        # Step 2: Compute metrics on actual data
        actual_metrics = self._compute_actual_metrics(graphs)

        # Step 3: Generate predictions and compute forecast metrics
        forecast_metrics = self._compute_forecast_metrics(graphs)

        # Step 4: Calculate delays for each change point
        delays = {"detection": {}, "prediction": {}}
        change_points = ground_truth["change_points"]

        for cp in change_points:
            # Find first detection after change point
            for t in range(
                cp,
                min(
                    cp + self.config.params.min_segment,
                    len(actual_metrics[1]["reset"]["degree"]["martingale_values"]),
                ),
            ):
                if any(
                    m["martingale_values"][t - self.config.min_history]
                    > self.config.martingale_threshold
                    for m in actual_metrics[1]["reset"].values()
                ):
                    delays["detection"][cp] = {"mean": float(t - cp), "std": 0.0}
                    break

            # Find first prediction after change point
            for t in range(
                cp - self.config.prediction_window,
                min(
                    cp + self.config.params.min_segment - self.config.prediction_window,
                    len(forecast_metrics[2]["reset"]["degree"]["martingale_values"]),
                ),
            ):
                if any(
                    m["martingale_values"][
                        t - self.config.min_history + self.config.prediction_window
                    ]
                    > self.config.martingale_threshold
                    for m in forecast_metrics[2]["reset"].values()
                ):
                    delays["prediction"][cp] = {
                        "mean": float(t - cp + self.config.prediction_window),
                        "std": 0.0,
                    }
                    break

        # Step 5: Compile results
        results = {
            "graphs": graphs,
            "ground_truth": ground_truth,
            "actual_metrics": actual_metrics,
            "forecast_metrics": forecast_metrics,
            "delays": delays,
            "seed": self.seed,
            "run_number": run_number,
        }

        # Step 6: Save and visualize if needed
        if run_number is None or self.config.save_individual:
            self._save_results(results)
        if run_number is None or self.config.visualize_individual:
            self.visualizer.visualize_results(results, self.output_dir, self.config)

        return results

    def _run_multiple(self) -> Dict[str, Any]:
        """Run multiple experiments and aggregate results."""
        # Step 1: Run multiple experiments
        all_results = []
        base_seed = self.seed if self.seed is not None else np.random.randint(0, 10000)

        for i in range(self.config.n_runs):
            logger.info(f"\nRunning experiment {i+1}/{self.config.n_runs}")
            run_seed = base_seed + i if base_seed is not None else None

            # Run single experiment with current seed
            results = self._run_single(run_number=i + 1, reuse_dir=True)
            all_results.append(results)

        # Step 2: Aggregate results
        aggregated = self.result_aggregator.aggregate_results(all_results)

        # Step 3: Save and visualize aggregated results
        self.result_aggregator.save_aggregated_results(aggregated, self.output_dir)
        self.visualizer.visualize_aggregated_results(
            aggregated, self.output_dir, self.config
        )

        return aggregated

    def _get_detected_change_points(self, martingales: Dict[str, Dict]) -> List[int]:
        """Extract unique change points detected across features."""
        all_cps = []
        for feature_data in martingales.values():
            if "change_detected_instant" in feature_data:
                all_cps.extend(feature_data["change_detected_instant"])
        return sorted(list(set(all_cps)))

    def _compute_actual_metrics(self, graphs: List[Dict[str, Any]]) -> tuple:
        """Compute metrics on actual network data."""
        metrics = self.metric_computer.compute_metrics(
            graphs=graphs[self.config.min_history :],
            min_history=self.config.min_history,
            martingale_threshold=self.config.martingale_threshold,
            martingale_epsilon=self.config.martingale_epsilon,
            shap_threshold=self.config.shap_threshold,
            shap_window_size=self.config.shap_window_size,
        )
        return metrics.features, metrics.martingales, metrics.shap_values

    def _compute_forecast_metrics(self, graphs: List[Dict[str, Any]]) -> tuple:
        """Generate predictions and compute forecast metrics."""
        predictions = self.network_predictor.generate_predictions(graphs)
        metrics = self.metric_computer.compute_metrics(
            graphs=predictions,
            min_history=self.config.min_history,
            martingale_threshold=self.config.martingale_threshold,
            martingale_epsilon=self.config.martingale_epsilon,
            shap_threshold=self.config.shap_threshold,
            shap_window_size=self.config.shap_window_size,
        )
        return predictions, metrics.features, metrics.martingales, metrics.shap_values

    def _generate_data(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate synthetic network data sequence."""
        logger.info(f"Generating {self.config.model} network time series...")
        result = self.generator.generate_sequence(
            model=self.config.model, params=self.config.params, seed=self.seed
        )

        # Map parameters to timesteps
        param_map = {}
        current_params = result["parameters"][0]
        for i in range(self.config.params.seq_len):
            if i in result["change_points"]:
                current_params = result["parameters"][
                    result["change_points"].index(i) + 1
                ]
            param_map[i] = current_params

        # Convert to network states
        network_states = []
        for i, adj in enumerate(result["graphs"]):
            G = nx.from_numpy_array(adj)
            network_states.append(
                {
                    "time": i,
                    "adjacency": adj,
                    "graph": G,
                    "metrics": self.feature_extractor.get_all_metrics(G).__dict__,
                    "params": param_map[i],
                    "is_change_point": i in result["change_points"],
                }
            )

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

    def compute_network_features(
        self, graphs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Compute network features for analysis."""
        return self.metric_computer.compute_network_features(graphs)

    def analyze_results(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze prediction accuracy across different phases."""
        return self.metric_computer.analyze_results(
            results=results,
            min_history=self.config.min_history,
        )

    def compute_martingales(
        self,
        features: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Compute martingales for network features."""
        return self.metric_computer.compute_martingales(
            features=features,
            threshold=self.config.martingale_threshold,
            epsilon=self.config.martingale_epsilon,
        )

    def compute_shap_values(
        self,
        martingales: Dict[str, Dict[str, Dict[str, Any]]],
        features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute SHAP values for martingale analysis."""
        return self.metric_computer.compute_shap_values(
            martingales=martingales,
            features=features,
            threshold=self.config.shap_threshold,
            window_size=self.config.shap_window_size,
        )

    def visualize_results(self, results: Dict[str, Any]):
        """Generate visualizations for results."""
        self.visualizer.visualize_results(results, self.output_dir, self.config)

    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        if not self.output_dir:
            return

        try:
            serializable_data = {
                "model": self.config.model,
                "parameters": self.result_aggregator._convert_to_serializable(
                    self.config
                ),
                "ground_truth": self.result_aggregator._convert_to_serializable(
                    results["ground_truth"]
                ),
                "seed": self.seed,
            }

            # For multiple runs with save_individual flag, save in numbered subdirectories
            if results.get("run_number") is not None and self.config.save_individual:
                run_dir = self.output_dir / f"run_{results['run_number']}"
                run_dir.mkdir(parents=True, exist_ok=True)
                save_path = run_dir / "results.json"
            else:
                save_path = self.output_dir / "results.json"

            with open(save_path, "w") as f:
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            # Save in a simpler format
            simple_data = {
                "model": self.config.model,
                "seed": self.seed,
                "error": str(e),
            }
            error_path = (
                self.output_dir / f"run_{results['run_number']}" / "results_error.json"
                if results.get("run_number") is not None and self.config.save_individual
                else self.output_dir / "results_error.json"
            )
            with open(error_path, "w") as f:
                json.dump(simple_data, f, indent=4)
