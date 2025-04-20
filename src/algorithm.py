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

from src.changepoint.detector import DetectorConfig, ChangePointDetector
from src.changepoint.martingale import BettingFunctionConfig
from src.configs import get_config, get_full_model_name
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.plot.plot_changepoint import MartingaleVisualizer
from src.predictor import PredictorFactory
from src.output_manager import OutputManager


logger = logging.getLogger(__name__)


class GraphChangeDetection:
    """Main algorithm class for graph change point detection."""

    def __init__(self, config_path: str):
        """Initialize the algorithm with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run(
        self, prediction: bool = None, visualize: bool = None, save_csv: bool = None
    ) -> Dict[str, Any]:
        """Run the complete detection algorithm with multiple trials.

        Args:
            prediction: Whether to generate and use predictions for detection.
                       If None, uses config value.
            visualize: Whether to create visualizations of the results.
                       If None, uses config value.
            save_csv: Whether to save results to CSV files.
                      If None, uses config value.

        Returns:
            Dictionary containing all results
        """
        # Use provided parameters or fall back to config values
        enable_prediction = (
            prediction
            if prediction is not None
            else self.config["execution"].get("enable_prediction", True)
        )
        enable_visualization = (
            visualize
            if visualize is not None
            else self.config["execution"].get("enable_visualization", True)
        )
        enable_csv_export = (
            save_csv
            if save_csv is not None
            else self.config["execution"].get("save_csv", True)
        )

        logger.debug(
            f"Running with prediction={enable_prediction}, visualize={enable_visualization}, save_csv={enable_csv_export}"
        )

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
            generator = self._init_generator()

            # Only initialize predictor if prediction is enabled
            predictor = None
            if enable_prediction:
                predictor = self._init_predictor()

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
            # ---------- Step 4: Generate predictions (optional) #
            # -------------------------------------------------- #
            predicted_graphs = None
            predicted_features = None

            if enable_prediction:
                predicted_graphs = self._generate_predictions(graphs, predictor)
                predicted_features = self._process_predictions(predicted_graphs)
                logger.info(f"Generated predictions shape: {predicted_features.shape}")

            # -------------------------------------------------- #
            # ---------- Step 5: Run multiple trials ----------- #
            # -------------------------------------------------- #
            try:
                trial_results = self._run_multiple_trials(
                    features_numeric, predicted_features, true_change_points
                )
            except Exception as e:
                logger.error(f"Trial execution failed: {str(e)}")
                # Create a minimal trial result to continue processing
                trial_results = {
                    "individual_trials": [],
                    "aggregated": {},
                    "random_seeds": [],
                }

            # -------------------------------------------------- #
            # ------ Step 6: Create visualizations (optional) -- #
            # -------------------------------------------------- #
            if (
                enable_visualization
                and self.config["output"]["visualization"]["enabled"]
                and trial_results["aggregated"]
            ):
                try:
                    self._create_visualizations(
                        trial_results["aggregated"], true_change_points, features_raw
                    )
                except Exception as e:
                    logger.error(f"Visualization failed: {str(e)}")

            # -------------------------------------------------- #
            # ---------- Step 7: Export results to CSV (optional) #
            # -------------------------------------------------- #
            if enable_csv_export and trial_results["aggregated"]:
                try:
                    # Create output directory if it doesn't exist
                    csv_output_dir = os.path.join(self.config["output"]["directory"])

                    # Pass the entire config object to OutputManager for direct access
                    output_manager = OutputManager(csv_output_dir, self.config)

                    # Export both individual trials and aggregated results
                    output_manager.export_to_csv(
                        trial_results["aggregated"],
                        true_change_points,
                        individual_trials=trial_results["individual_trials"],
                    )
                except Exception as e:
                    logger.error(f"Failed to export results to CSV: {str(e)}")
                    # Don't raise the exception, continue with algorithm

            # -------------------------------------------------- #
            # --------- Step 8: Compile and return results ----- #
            # -------------------------------------------------- #
            results = self._compile_results(
                sequence_result,
                features_numeric,
                features_raw,
                predicted_graphs,
                trial_results,
                predictor,
            )

            logger.info("Successfully completed algorithm")
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
            predicted_features: Predicted feature vectors (can be None)
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

        # Normalize each feature - EXACTLY as in multiview_vis.py
        features_normalized = (
            features_numeric - np.mean(features_numeric, axis=0)
        ) / np.std(features_numeric, axis=0)

        logger.info(f"Normalized feature data shape: {features_normalized.shape}")

        # Also normalize predicted features if available using the SAME statistics
        predicted_normalized = None
        if predicted_features is not None:
            # For multiview detection with predictions
            feature_means = np.mean(features_numeric, axis=0)
            feature_stds = np.std(features_numeric, axis=0)

            # Initialize the normalized predictions array
            predicted_normalized = []

            # For each timestep's predictions
            for t in range(len(predicted_features)):
                normalized_horizons = []
                # For each horizon
                for h in range(len(predicted_features[t])):
                    # Normalize using the same stats as the main features
                    normalized_horizon = (
                        predicted_features[t][h] - feature_means
                    ) / feature_stds
                    normalized_horizons.append(normalized_horizon)
                predicted_normalized.append(normalized_horizons)

            # Convert to numpy array if needed
            if isinstance(predicted_features, np.ndarray):
                predicted_normalized = np.array(predicted_normalized)

        individual_results = []
        for trial_idx, seed in enumerate(random_seeds):
            logger.info(f"Running trial {trial_idx + 1}/{n_trials} with seed {seed}")

            # Make sure seed is an integer or None
            int_seed = int(seed) if seed is not None else None

            # Create a much simpler detector configuration like in multiview_vis.py
            from src.changepoint.detector import DetectorConfig

            detector_config = DetectorConfig(
                method=self.config["model"]["type"],
                threshold=self.config["detection"]["threshold"],
                history_size=self.config["model"]["predictor"]["config"]["n_history"],
                batch_size=self.config["detection"]["batch_size"],
                reset=self.config["detection"]["reset"],
                max_window=self.config["detection"]["max_window"],
                betting_func_config={
                    "name": self.config["detection"]["betting_func_config"]["name"],
                    "params": {
                        "epsilon": self.config["detection"]["betting_func_config"][
                            "power"
                        ]["epsilon"]
                    },
                },
                distance_measure=self.config["detection"]["distance"]["measure"],
                distance_p=self.config["detection"]["distance"]["p"],
                random_state=int_seed,  # Use integer seed
            )

            # Initialize detector with this config - EXACTLY as in multiview_vis.py
            from src.changepoint.detector import ChangePointDetector

            detector = ChangePointDetector(detector_config)

            # Log detector settings
            logger.info(f"Using threshold: {detector_config.threshold:.2f}")
            logger.info(
                f"Using betting function: {detector_config.betting_func_config['name']}"
            )
            logger.info(f"Using distance measure: {detector_config.distance_measure}")
            logger.info(f"Using random seed: {int_seed}")

            try:
                # Run detection with normalized features - EXACTLY as in multiview_vis.py
                detection_result = detector.run(
                    data=features_normalized,
                    predicted_data=predicted_normalized,
                    reset_state=True,  # Always start with fresh state for each trial
                )

                if detection_result is None:
                    logger.warning(f"Detection returned None for trial {trial_idx + 1}")
                    continue

                # Log detection results
                if "traditional_change_points" in detection_result:
                    cp = detection_result["traditional_change_points"]
                    logger.info(
                        f"Traditional change points detected: {len(cp)} at {cp}"
                    )

                if "horizon_change_points" in detection_result:
                    cp = detection_result["horizon_change_points"]
                    if cp:
                        logger.info(
                            f"Horizon change points detected: {len(cp)} at {cp}"
                        )

                individual_results.append(detection_result)
            except Exception as e:
                logger.error(f"Trial {trial_idx + 1} failed: {str(e)}")
                continue

        if not individual_results:
            raise RuntimeError("All detection trials failed")

        # Simplify the aggregation - just use the first trial's results for visualization
        # This is more like what multiview_vis.py does (it only runs one trial)
        aggregated_results = individual_results[0].copy()

        # Add all trials together for CSV export
        trial_results = {
            "individual_trials": individual_results,
            "aggregated": aggregated_results,
            "random_seeds": random_seeds.tolist(),
        }

        return trial_results

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

        # Ensure random_state is compatible type
        if random_state is not None:
            random_state = int(random_state)

        # Get betting function config parameters
        betting_func_name = det_config["betting_func_config"]["name"]
        betting_func_params = det_config["betting_func_config"].get(
            betting_func_name, {}
        )

        # Create a proper BettingFunctionConfig instance
        betting_func_config = BettingFunctionConfig(
            name=betting_func_name, params=betting_func_params
        )

        # Create DetectorConfig with proper configuration objects
        detector_config = DetectorConfig(
            method=self.config["model"]["type"],
            threshold=det_config["threshold"],
            history_size=self.config["model"]["predictor"]["config"]["n_history"],
            batch_size=det_config["batch_size"],
            reset=det_config["reset"],
            max_window=det_config["max_window"],
            betting_func_config=betting_func_config,
            distance_measure=det_config["distance"]["measure"],
            distance_p=det_config["distance"]["p"],
            random_state=random_state,  # Use provided random state
        )

        logger.info(f"Using distance measure: {detector_config.distance_measure}")
        logger.info(f"Using betting function: {betting_func_name}")

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
            history_size = (
                self.config["model"]["predictor"]["config"]["n_history"]
                or predictor.history_size
            )
            history_start = max(0, t - history_size)
            history = [{"adjacency": g} for g in graphs[history_start:t]]

            if t >= history_size:
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
        self,
        detection_result,
        true_change_points,
        features_raw,
    ):
        """Create visualizations of the results."""
        logger.info("Creating visualizations")
        output_config = self.config["output"]
        det_config = self.config["detection"]

        # Prepare betting config EXACTLY as in multiview_vis.py
        epsilon = det_config["betting_func_config"]["power"]["epsilon"]
        betting_config = {
            "function": "power",
            "params": {"power": {"epsilon": epsilon}},
        }

        try:
            # Initialize a copy of the detection result that we can modify
            complete_result = detection_result.copy()

            # Add missing keys EXACTLY as in multiview_vis.py
            # Since we're running without predictions, add empty horizon martingales
            if "horizon_martingales" not in complete_result:
                complete_result["horizon_martingales"] = np.zeros_like(
                    complete_result["traditional_sum_martingales"]
                )

            if "horizon_change_points" not in complete_result:
                complete_result["horizon_change_points"] = []

            if "horizon_sum_martingales" not in complete_result:
                complete_result["horizon_sum_martingales"] = np.zeros_like(
                    complete_result["traditional_sum_martingales"]
                )

            if "horizon_avg_martingales" not in complete_result:
                complete_result["horizon_avg_martingales"] = np.zeros_like(
                    complete_result["traditional_avg_martingales"]
                )

            if "individual_horizon_martingales" not in complete_result:
                n_features = len(complete_result["individual_traditional_martingales"])
                complete_result["individual_horizon_martingales"] = [
                    np.zeros_like(feat)
                    for feat in complete_result["individual_traditional_martingales"]
                ]

            # For multiview detection, traditional_martingales isn't returned, only sum and avg
            if "traditional_martingales" not in complete_result:
                # Create traditional_martingales from traditional_sum_martingales
                complete_result["traditional_martingales"] = complete_result[
                    "traditional_sum_martingales"
                ].copy()

            # Make sure all change points are plain Python lists, not NumPy arrays
            for key in [
                "traditional_change_points",
                "horizon_change_points",
                "early_warnings",
            ]:
                if key in complete_result:
                    if isinstance(complete_result[key], np.ndarray):
                        complete_result[key] = complete_result[key].tolist()

                    # Ensure they're actually lists, not other types
                    if not isinstance(complete_result[key], list):
                        complete_result[key] = []

            # For safety - replace any scalar martingale values with arrays
            for key in complete_result:
                if key.endswith("_martingales") and np.isscalar(complete_result[key]):
                    complete_result[key] = np.array([complete_result[key]])

            # Create martingale visualizer
            visualizer = MartingaleVisualizer(
                martingales=complete_result,
                change_points=true_change_points,
                threshold=det_config["threshold"],
                betting_config=betting_config,
                output_dir=output_config["directory"],
                prefix=output_config["prefix"],
                skip_shap=output_config["visualization"]["skip_shap"],
                method=self.config["model"]["type"],
            )
            visualizer.create_visualization()
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            logger.error("Continuing without visualizations")
            # Print full traceback for more details if debugging
            import traceback

            logger.debug(traceback.format_exc())

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

        if predicted_graphs is not None and self.config["output"]["save_predictions"]:
            results.update(
                {
                    "predicted_graphs": predicted_graphs,
                }
            )
            if predictor is not None:
                results.update(
                    {
                        "predictor_states": predictor.get_state(),
                    }
                )

        if self.config["output"]["save_martingales"]:
            results.update(trial_results["aggregated"])

        return results


def main(
    config_path: str,
    prediction: bool = None,
    visualize: bool = None,
    save_csv: bool = None,
) -> Dict[str, Any]:
    """Run the algorithm with the given configuration.

    Args:
        config_path: Path to YAML configuration file
        prediction: Whether to generate and use predictions for detection.
                   If None, uses the value from the config file.
        visualize: Whether to create visualizations of the results.
                   If None, uses the value from the config file.
        save_csv: Whether to save results to CSV files.
                 If None, uses the value from the config file.

    Returns:
        Dictionary containing all results
    """
    algorithm = GraphChangeDetection(config_path)
    return algorithm.run(prediction=prediction, visualize=visualize, save_csv=save_csv)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run graph change detection algorithm")
    parser.add_argument("config_path", help="Path to configuration YAML file")
    parser.add_argument(
        "--no-prediction", action="store_true", help="Disable prediction in detection"
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Disable result visualization"
    )
    parser.add_argument(
        "--no-csv", action="store_true", help="Disable CSV export of results"
    )

    args = parser.parse_args()

    result = main(
        args.config_path,
        prediction=not args.no_prediction if args.no_prediction else None,
        visualize=not args.no_visualization if args.no_visualization else None,
        save_csv=not args.no_csv if args.no_csv else None,
    )
