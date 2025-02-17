# src/algorithm.py

"""Forecast-based Graph Structural Change Detection Algorithm."""

from pathlib import Path
from typing import Dict, Any, List, Optional

import logging
import sys
import yaml
import numpy as np
import time
import os

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint.detector import ChangePointDetector
from src.changepoint.visualizer import MartingaleVisualizer
from src.configs import get_config, get_full_model_name
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.predictor import PredictorFactory
from src.changepoint.detector import DetectorConfig

logger = logging.getLogger(__name__)


class GraphChangeDetection:
    """Main algorithm class for graph change point detection."""

    def __init__(self, config_path: str):
        """Initialize the algorithm with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate the loaded configuration."""
        required_sections = ["model", "detection", "features", "output", "trials"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate trials configuration
        trials_config = self.config["trials"]
        if trials_config["n_trials"] < 1:
            raise ValueError("Number of trials must be at least 1")

        # Validate random seeds configuration
        if trials_config["random_seeds"] is not None:
            if isinstance(trials_config["random_seeds"], (list, tuple)):
                if len(trials_config["random_seeds"]) != trials_config["n_trials"]:
                    raise ValueError(
                        "Number of random seeds must match number of trials"
                    )
            elif not isinstance(trials_config["random_seeds"], (int, float)):
                raise ValueError(
                    "random_seeds must be null, an integer, or a list of integers"
                )

        # Validate model configuration
        model_config = self.config["model"]
        if model_config["type"] not in ["multiview", "single_view"]:
            raise ValueError("Model type must be 'multiview' or 'single_view'")
        if model_config["network"] not in ["ba", "ws", "er", "sbm"]:
            raise ValueError("Invalid network model specified")

        # Validate predictor configuration
        predictor_config = model_config["predictor"]
        if predictor_config["type"] not in ["adaptive", "auto", "statistical"]:
            raise ValueError("Invalid predictor type specified")

        # Validate detection configuration
        det_config = self.config["detection"]
        betting_config = det_config.get("betting_func_config", {})
        if not betting_config or "name" not in betting_config:
            raise ValueError("Betting function configuration must specify a 'name'")

        valid_betting_functions = [
            "power",
            "exponential",
            "mixture",
            "constant",
            "beta",
            "kernel",
        ]
        if betting_config["name"] not in valid_betting_functions:
            raise ValueError(
                f"Invalid betting function specified. Must be one of: {valid_betting_functions}"
            )

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run(self) -> Dict[str, Any]:
        """Run the complete detection algorithm with multiple trials."""
        logger.info(f"Starting {self.config['name']} ...")
        logger.info(f"Description: {self.config['description']}")

        try:
            # Create timestamped output directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            network_type = self.config["model"]["network"]
            betting_function = self.config["detection"]["betting_func_config"]["name"]
            predictor_type = self.config["model"]["predictor"]["type"]
            distance_measure = self.config["detection"]["distance"]["measure"]
            self.config["output"]["directory"] = os.path.join(
                self.config["output"]["directory"],
                f"{network_type}_{predictor_type}_{distance_measure}_{betting_function}_{timestamp}",
            )

            os.makedirs(self.config["output"]["directory"], exist_ok=True)
            logger.info(f"Output directory: {self.config['output']['directory']}")

            # -------------------------------------------------- #
            # ----------- Step 1: Initialize components -------- #
            # -------------------------------------------------- #
            predictor = self._init_predictor()
            generator = self._init_generator()

            # -------------------------------------------------- #
            # ---------- Step 2: Generate graph sequence ------- #
            # -------------------------------------------------- #
            sequence_result = self._generate_sequence(generator)
            graphs = sequence_result["graphs"]
            true_change_points = sequence_result["change_points"]

            # -------------------------------------------------- #
            # ---------- Step 3: Extract features -------------- #
            # -------------------------------------------------- #
            features_numeric, features_raw = self._extract_features(graphs)
            logger.info(f"Extracted features shape: {features_numeric.shape}")

            # -------------------------------------------------- #
            # ---------- Step 4: Generate predictions ---------- #
            # -------------------------------------------------- #
            predicted_graphs = self._generate_predictions(graphs, predictor)
            predicted_features = (
                self._process_predictions(predicted_graphs)
                if predicted_graphs
                else None
            )

            if predicted_features is not None:
                logger.info(f"Generated predictions shape: {predicted_features.shape}")

            # -------------------------------------------------- #
            # ---------- Step 5: Run multiple trials ----------- #
            # -------------------------------------------------- #
            trial_results = self._run_multiple_trials(
                features_numeric, predicted_features, true_change_points
            )

            # -------------------------------------------------- #
            # ---------- Step 6: Create visualizations --------- #
            # -------------------------------------------------- #
            if self.config["output"]["visualization"]["enabled"]:
                self._create_visualizations(
                    trial_results["aggregated"], true_change_points, features_raw
                )

            # -------------------------------------------------- #
            # --------- Step 7: Compile and return results ----- #
            # -------------------------------------------------- #
            results = self._compile_results(
                sequence_result,
                features_numeric,
                features_raw,
                predicted_graphs,
                trial_results,
                predictor,
            )

            logger.info("Algorithm completed successfully")
            return results

        except Exception as e:
            logger.error(f"Algorithm failed: {str(e)}")
            raise

    def _run_multiple_trials(
        self,
        features_numeric: np.ndarray,
        predicted_features: Optional[np.ndarray],
        true_change_points: List[int],
    ) -> Dict[str, Any]:
        """Run multiple trials of the detector.

        Args:
            features_numeric: Extracted numeric features
            predicted_features: Predicted feature vectors
            true_change_points: Ground truth change points

        Returns:
            Dictionary containing individual trial results and aggregated statistics
        """
        trials_config = self.config["trials"]
        n_trials = trials_config["n_trials"]
        base_seed = trials_config["random_seeds"]

        logger.info(f"Running {n_trials} detection trials...")

        # Handle random seeds
        if base_seed is None:
            # Generate completely random seeds
            random_seeds = np.random.randint(0, 2**31 - 1, size=n_trials)
        elif isinstance(base_seed, (int, float)):
            # Generate deterministic sequence of seeds from base seed
            rng = np.random.RandomState(int(base_seed))
            random_seeds = rng.randint(0, 2**31 - 1, size=n_trials)
        else:
            # Use provided list of seeds
            random_seeds = np.array(base_seed)

        logger.info(f"Using seeds: {random_seeds.tolist()}")

        individual_results = []
        for trial_idx, seed in enumerate(random_seeds):
            logger.info(f"Running trial {trial_idx + 1}/{n_trials} with seed {seed}")

            # Initialize detector with current trial seed
            detector = self._init_detector(random_state=seed)

            # Run detection
            detection_result = detector.run(
                data=features_numeric, predicted_data=predicted_features
            )

            if detection_result is None:
                raise RuntimeError(f"Detection failed for trial {trial_idx + 1}")

            individual_results.append(detection_result)

        # Aggregate results across trials
        aggregated_results = self._aggregate_trial_results(individual_results)

        trial_results = {
            "individual_trials": (
                individual_results if trials_config["save_individual_trials"] else None
            ),
            "aggregated": aggregated_results,
            "random_seeds": random_seeds.tolist(),
        }

        return trial_results

    def _aggregate_trial_results(
        self, individual_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple trials.

        Args:
            individual_results: List of detection results from each trial

        Returns:
            Dictionary containing aggregated statistics
        """
        n_trials = len(individual_results)
        first_result = individual_results[0]

        # Initialize aggregated results
        aggregated = {
            "detection_frequencies": {
                "traditional": {},
                "horizon": {},
            }
        }

        # Define all possible martingale keys
        possible_martingale_keys = [
            # Single-view and common keys
            "traditional_martingales",
            "horizon_martingales",
            # Multiview specific keys
            "traditional_sum_martingales",
            "traditional_avg_martingales",
            "horizon_sum_martingales",
            "horizon_avg_martingales",
            "individual_traditional_martingales",
            "individual_horizon_martingales",
        ]

        # Initialize martingale arrays for keys that exist in results
        martingale_keys = [
            key for key in possible_martingale_keys if key in first_result
        ]
        logger.debug(f"Found martingale keys: {martingale_keys}")

        # Initialize arrays for each martingale type
        for key in martingale_keys:
            if key.startswith("individual_"):
                # For individual martingales (list of arrays), initialize list of zeros arrays
                n_features = len(first_result[key])
                aggregated[key] = [
                    np.zeros_like(first_result[key][i]) for i in range(n_features)
                ]
            else:
                # For regular martingales, initialize single zeros array
                aggregated[key] = np.zeros_like(first_result[key])

        # Initialize change point lists
        change_point_keys = []
        if "traditional_change_points" in first_result:
            change_point_keys.append("traditional_change_points")
            aggregated["traditional_change_points"] = []
        if "horizon_change_points" in first_result:
            change_point_keys.append("horizon_change_points")
            aggregated["horizon_change_points"] = []

        # Aggregate results across trials
        for result in individual_results:
            # Aggregate martingale values
            for key in martingale_keys:
                if key.startswith("individual_"):
                    # For individual martingales, sum each feature's martingales separately
                    for i in range(len(aggregated[key])):
                        aggregated[key][i] += result[key][i]
                else:
                    # For regular martingales, simple addition
                    aggregated[key] += result[key]

            # Count detection frequencies
            for cp_key in change_point_keys:
                detector_type = cp_key.split("_")[0]  # traditional or horizon
                for cp in result[cp_key]:
                    aggregated["detection_frequencies"][detector_type][cp] = (
                        aggregated["detection_frequencies"][detector_type].get(cp, 0)
                        + 1
                    )

        # Average martingale values
        for key in martingale_keys:
            if key.startswith("individual_"):
                # Average each feature's martingales separately
                for i in range(len(aggregated[key])):
                    aggregated[key][i] /= n_trials
            else:
                # Average regular martingales
                aggregated[key] /= n_trials

        # Convert detection frequencies to probabilities and get consensus change points
        threshold = n_trials / 2  # Majority vote threshold

        for detector_type in ["traditional", "horizon"]:
            cp_key = f"{detector_type}_change_points"
            if cp_key in change_point_keys:
                frequencies = aggregated["detection_frequencies"][detector_type]
                change_points = []

                for cp, freq in frequencies.items():
                    frequencies[cp] = freq / n_trials  # Convert to probability
                    if freq > threshold:
                        change_points.append(cp)

                aggregated[cp_key] = sorted(change_points)

        return aggregated

    def _init_predictor(self):
        """Initialize the graph predictor."""
        predictor_config = self.config["model"]["predictor"]
        logger.info(f"Initializing {predictor_config['type']} predictor")
        return PredictorFactory.create(
            predictor_config["type"], predictor_config["config"]
        )

    def _init_generator(self):
        """Initialize the graph sequence generator."""
        network_type = self.config["model"]["network"]
        logger.info(f"Initializing {network_type} graph generator")
        return GraphGenerator(network_type)

    def _init_detector(self, random_state: Optional[int] = None) -> ChangePointDetector:
        """Initialize the change point detector with optional random state."""
        det_config = self.config["detection"]
        logger.info(f"Initializing detector with {self.config['model']['type']} method")

        # Create DetectorConfig with all necessary parameters
        detector_config = DetectorConfig(
            method=self.config["model"]["type"],
            threshold=det_config["threshold"],
            history_size=self.config["model"]["predictor"]["config"]["n_history"],
            batch_size=det_config["batch_size"],
            reset=det_config["reset"],
            max_window=det_config["max_window"],
            betting_func_config={
                "name": det_config["betting_func_config"]["name"],
                "params": det_config["betting_func_config"].get(
                    det_config["betting_func_config"]["name"], {}
                ),
            },
            distance_measure=det_config["distance"]["measure"],
            distance_p=det_config["distance"]["p"],
            random_state=random_state,  # Use provided random state
        )

        logger.info(f"Using distance measure: {detector_config.distance_measure}")
        logger.info(
            f"Using betting function: {detector_config.betting_func_config['name']}"
        )

        return ChangePointDetector(detector_config)

    def _generate_sequence(self, generator):
        """Generate the graph sequence using model-specific configuration."""
        model_name = get_full_model_name(self.config["model"]["network"])
        logger.info(f"Generating {model_name} graph sequence")

        # Get model-specific configuration
        model_config = get_config(model_name)
        return generator.generate_sequence(model_config["params"].__dict__)

    def _extract_features(self, graphs):
        """Extract features from graph sequence."""
        logger.info("Extracting graph features")
        feature_extractor = NetworkFeatureExtractor()
        features_raw = []
        features_numeric = []

        for adj_matrix in graphs:
            graph = adjacency_to_graph(adj_matrix)
            raw_features = feature_extractor.get_features(graph)
            numeric_features = feature_extractor.get_numeric_features(graph)
            features_raw.append(raw_features)
            features_numeric.append(
                [numeric_features[name] for name in self.config["features"]]
            )

        return np.array(features_numeric), features_raw

    def _generate_predictions(self, graphs, predictor):
        """Generate predictions for the graph sequence."""
        logger.info("Generating graph predictions")
        predicted_graphs = []
        horizon = self.config["detection"]["prediction_horizon"]

        for t in range(len(graphs)):
            history_start = max(0, t - predictor.history_size)
            history = [{"adjacency": g} for g in graphs[history_start:t]]

            if t >= predictor.history_size:
                predictions = predictor.predict(history, horizon=horizon)
                predicted_graphs.append(predictions)

        return predicted_graphs

    def _process_predictions(self, predicted_graphs):
        """Process predictions into feature space."""
        logger.info("Processing predictions into feature space")
        feature_extractor = NetworkFeatureExtractor()
        predicted_features = []

        for predictions in predicted_graphs:
            timestep_features = []
            for pred_adj in predictions:
                graph = adjacency_to_graph(pred_adj)
                numeric_features = feature_extractor.get_numeric_features(graph)
                feature_vector = [
                    numeric_features[name] for name in self.config["features"]
                ]
                timestep_features.append(feature_vector)
            predicted_features.append(timestep_features)

        return np.array(predicted_features)

    def _create_visualizations(
        self, detection_result, true_change_points, features_raw
    ):
        """Create visualizations of the results."""
        logger.info("Creating visualizations")
        output_config = self.config["output"]
        det_config = self.config["detection"]

        # Get betting function configuration
        betting_config = {
            "function": det_config["betting_func_config"]["name"],
            "params": det_config["betting_func_config"].get(
                det_config["betting_func_config"]["name"], {}
            ),
        }

        visualizer = MartingaleVisualizer(
            martingales=detection_result,
            change_points=true_change_points,
            threshold=det_config["threshold"],
            betting_config=betting_config,
            output_dir=output_config["directory"],
            prefix=output_config["prefix"],
            skip_shap=output_config["visualization"]["skip_shap"],
            method=self.config["model"]["type"],
        )
        visualizer.create_visualization()

    def _compile_results(
        self,
        sequence_result,
        features_numeric,
        features_raw,
        predicted_graphs,
        trial_results,
        predictor,
    ):
        """Compile all results into a single dictionary."""
        logger.info("Compiling results")
        results = {
            "true_change_points": sequence_result["change_points"],
            "model_name": get_full_model_name(self.config["model"]["network"]),
            "params": self.config,
        }

        # Add data if configured to save
        if self.config["output"]["save_features"]:
            results.update(
                {
                    "features_raw": features_raw,
                    "features_numeric": features_numeric,
                }
            )

        if self.config["output"]["save_predictions"]:
            results.update(
                {
                    "predicted_graphs": predicted_graphs,
                    "predictor_states": predictor.get_state(),
                }
            )

        if self.config["output"]["save_martingales"]:
            results.update(trial_results["aggregated"])

        return results


def main(config_path: str) -> Dict[str, Any]:
    """Run the algorithm with the given configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing all results
    """
    algorithm = GraphChangeDetection(config_path)
    return algorithm.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python algorithm.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
