# src/algorithm.py

"""Forecast-based Graph Structural Change Detection Algorithm."""

from pathlib import Path
from typing import Dict, Any

import logging
import sys
import yaml
import numpy as np

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
        required_sections = ["model", "detection", "features", "output"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

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
        if det_config["betting_function"] not in [
            "power",
            "exponential",
            "mixture",
            "constant",
            "beta",
            "kernel",
        ]:
            raise ValueError("Invalid betting function specified")

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run(self) -> Dict[str, Any]:
        """Run the complete detection pipeline."""
        logger.info(f"Starting {self.config['name']} pipeline")
        logger.info(f"Description: {self.config['description']}")

        try:
            # -------------------------------------------------- #
            # ----------- Step 1: Initialize components -------- #
            # -------------------------------------------------- #
            predictor = self._init_predictor()
            generator = self._init_generator()
            detector = self._init_detector()

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
            # ---------- Step 5: Run detection ----------------- #
            # -------------------------------------------------- #
            detection_result = detector.run(
                data=features_numeric, predicted_data=predicted_features
            )

            if detection_result is None:
                raise RuntimeError("Detection failed to produce results")

            # -------------------------------------------------- #
            # ---------- Step 6: Create visualizations --------- #
            # -------------------------------------------------- #
            if self.config["output"]["visualization"]["enabled"]:
                self._create_visualizations(
                    detection_result, true_change_points, features_raw
                )

            # -------------------------------------------------- #
            # --------- Step 7: Compile and return results ----- #
            # -------------------------------------------------- #
            results = self._compile_results(
                sequence_result,
                features_numeric,
                features_raw,
                predicted_graphs,
                detection_result,
                predictor,
            )

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

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

    def _init_detector(self):
        """Initialize the change point detector."""
        det_config = self.config["detection"]
        logger.info(f"Initializing detector with {self.config['model']['type']} method")

        # Get distance configuration
        distance_config = det_config.get("distance", {"measure": "euclidean", "p": 2.0})
        logger.info(f"Using distance measure: {distance_config['measure']}")

        return ChangePointDetector(
            martingale_method=self.config["model"]["type"],
            threshold=det_config["threshold"],
            epsilon=det_config["epsilon"],
            batch_size=det_config["batch_size"],
            reset=det_config["reset"],
            max_window=det_config["max_window"],
            random_state=42,  # Fixed for reproducibility
            betting_func=det_config["betting_function"],
            distance_measure=distance_config["measure"],
            distance_p=distance_config["p"],
        )

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
        visualizer = MartingaleVisualizer(
            martingales=detection_result,
            change_points=true_change_points,
            threshold=self.config["detection"]["threshold"],
            epsilon=self.config["detection"]["epsilon"],
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
        detection_result,
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
            results.update(detection_result)

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
