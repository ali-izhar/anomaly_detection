# src/algorithm.py

"""Core pipeline for forecast-based graph structural change detection."""

from pathlib import Path
import logging
import yaml
import numpy as np
import time
import os

from src.changepoint.detector import DetectorConfig, ChangePointDetector
from src.changepoint.betting import BettingFunctionConfig
from src.configs import get_config, get_full_model_name
from src.graph.generator import GraphGenerator
from src.graph.features import NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.plot.plot_changepoint import MartingaleVisualizer
from src.plot.visualization_utils import (
    prepare_martingale_visualization_data,
    create_betting_config_for_visualization,
)
from src.predictor import PredictorFactory
from src.utils.output_manager import OutputManager
from src.utils.data_utils import (
    normalize_features,
    normalize_predictions,
    prepare_result_data,
)


logger = logging.getLogger(__name__)


class GraphChangeDetection:
    """Pipeline for graph change point detection.

    This class implements a complete pipeline for detecting structural changes in graph
    sequences, with optional prediction-enhanced detection capabilities. The pipeline has
    the following stages:

    1. Graph sequence generation
    2. Feature extraction
    3. (Optional) Future state prediction
    4. Change point detection
    5. (Optional) Visualization
    6. (Optional) Data export

    Each step is implemented as a separate method, allowing for flexible execution
    and extension of the pipeline.
    """

    def __init__(self, config_path=None, config_dict=None):
        """Initialize the pipeline with configuration.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)

        Raises:
            ValueError: If neither config_path nor config_dict is provided
        """
        if config_path:
            self.config = self._load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self._setup_logging()

    def _load_config(self, config_path):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dict containing configuration
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def run(self, prediction=None, visualize=None, save_csv=None):
        """Run the complete detection pipeline.

        Args:
            prediction: Whether to generate and use predictions for detection.
                       If None, uses config value.
            visualize: Whether to create visualizations of the results.
                       If None, uses config value.
            save_csv: Whether to save results to CSV files.
                      If None, uses config value.

        Returns:
            Dictionary containing all results

        Raises:
            RuntimeError: If the pipeline execution fails
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
            # Create output directory with descriptive name
            self._setup_output_directory()

            # Run each pipeline stage
            generator = self._init_generator()
            sequence_result = self._generate_sequence(generator)
            graphs = sequence_result["graphs"]
            true_change_points = sequence_result["change_points"]

            features_numeric, features_raw = self._extract_features(graphs)
            logger.info(f"Extracted features shape: {features_numeric.shape}")

            # Prediction is optional
            predictor = None
            predicted_graphs = None
            predicted_features = None

            if enable_prediction:
                predictor = self._init_predictor()
                predicted_graphs = self._generate_predictions(graphs, predictor)
                predicted_features = self._process_predictions(predicted_graphs)
                logger.info(f"Generated predictions shape: {predicted_features.shape}")

            # Run detection trials
            trial_results = self._run_detection_trials(
                features_numeric, predicted_features, true_change_points
            )

            # Optional visualization
            if (
                enable_visualization
                and self.config["output"]["visualization"]["enabled"]
                and trial_results["aggregated"]
            ):
                self._create_visualizations(
                    trial_results["aggregated"], true_change_points, features_raw
                )

            # Optional CSV export
            if enable_csv_export and trial_results["aggregated"]:
                self._export_results_to_csv(trial_results, true_change_points)

            # Compile final results
            results = prepare_result_data(
                sequence_result,
                features_numeric,
                features_raw,
                predicted_graphs,
                trial_results,
                predictor,
                self.config,
            )

            logger.info("Successfully completed pipeline")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _setup_output_directory(self):
        """Create timestamped output directory with descriptive name."""
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

    def _init_generator(self):
        """Initialize the graph sequence generator.

        Returns:
            GraphGenerator instance
        """
        network_type = self.config["model"]["network"]
        logger.info(f"Initializing {network_type} graph generator")
        return GraphGenerator(network_type)

    def _init_predictor(self):
        """Initialize the graph predictor.

        Returns:
            Predictor instance from PredictorFactory
        """
        predictor_config = self.config["model"]["predictor"]
        logger.info(f"Initializing {predictor_config['type']} predictor")
        return PredictorFactory.create(
            predictor_config["type"], predictor_config["config"]
        )

    def _init_detector(self, random_state=None):
        """Initialize the change point detector.

        Args:
            random_state: Optional random seed for reproducibility

        Returns:
            ChangePointDetector instance
        """
        det_config = self.config["detection"]
        logger.info(f"Initializing detector with {self.config['model']['type']} method")

        # Ensure random_state is compatible type
        if random_state is not None:
            random_state = int(random_state)

        # Get betting function config
        betting_func_name = det_config["betting_func_config"]["name"]
        betting_func_params = det_config["betting_func_config"].get(
            betting_func_name, {}
        )

        # Create proper BettingFunctionConfig
        betting_func_config = BettingFunctionConfig(
            name=betting_func_name, params=betting_func_params
        )

        # Create detector config
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
            random_state=random_state,
        )

        logger.info(f"Using distance measure: {detector_config.distance_measure}")
        logger.info(f"Using betting function: {betting_func_name}")

        return ChangePointDetector(detector_config)

    def _generate_sequence(self, generator):
        """Generate the graph sequence using model-specific configuration.

        Args:
            generator: GraphGenerator instance

        Returns:
            Dict containing generated graphs and true change points
        """
        model_name = get_full_model_name(self.config["model"]["network"])
        logger.info(f"Generating {model_name} graph sequence")

        # Get model-specific configuration
        model_config = get_config(model_name)
        return generator.generate_sequence(model_config["params"].__dict__)

    def _extract_features(self, graphs):
        """Extract features from graph sequence.

        Args:
            graphs: List of adjacency matrices

        Returns:
            Tuple containing (numeric_features, raw_features)
        """
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
        """Generate predictions for the graph sequence.

        Args:
            graphs: List of adjacency matrices
            predictor: Initialized predictor instance

        Returns:
            List of predicted future adjacency matrices
        """
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
        """Process predictions into feature space.

        Args:
            predicted_graphs: List of predicted adjacency matrices

        Returns:
            numpy array of predicted features
        """
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

    def _run_detection_trials(
        self, features_numeric, predicted_features, true_change_points
    ):
        """Run multiple trials of the detector.

        Args:
            features_numeric: Extracted numeric features
            predicted_features: Predicted feature vectors (can be None)
            true_change_points: Ground truth change points

        Returns:
            Dict containing individual trial results and aggregated statistics

        Raises:
            RuntimeError: If all detection trials fail
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

        # Normalize features using the utility function
        features_normalized, feature_means, feature_stds = normalize_features(
            features_numeric
        )
        logger.info(f"Normalized feature data shape: {features_normalized.shape}")

        # Normalize predicted features if available
        predicted_normalized = None
        if predicted_features is not None:
            predicted_normalized = normalize_predictions(
                predicted_features, feature_means, feature_stds
            )

        # Run individual trials
        individual_results = []
        for trial_idx, seed in enumerate(random_seeds):
            logger.info(f"Running trial {trial_idx + 1}/{n_trials} with seed {seed}")
            int_seed = int(seed) if seed is not None else None

            # Create detector configuration
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
                random_state=int_seed,
            )

            # Initialize detector
            detector = ChangePointDetector(detector_config)

            # Log detector settings
            logger.info(f"Using threshold: {detector_config.threshold:.2f}")
            logger.info(
                f"Using betting function: {detector_config.betting_func_config['name']}"
            )
            logger.info(f"Using distance measure: {detector_config.distance_measure}")
            logger.info(f"Using random seed: {int_seed}")

            try:
                # Run detection
                detection_result = detector.run(
                    data=features_normalized,
                    predicted_data=predicted_normalized,
                    reset_state=True,
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

        # Use the first trial's results for visualization
        aggregated_results = individual_results[0].copy()

        # Return combined results
        return {
            "individual_trials": individual_results,
            "aggregated": aggregated_results,
            "random_seeds": random_seeds.tolist(),
        }

    def _create_visualizations(
        self, detection_result, true_change_points, features_raw
    ):
        """Create visualizations of the detection results.

        Args:
            detection_result: Dict containing detection results
            true_change_points: List of ground truth change points
            features_raw: Raw feature data
        """
        logger.info("Creating visualizations")
        output_config = self.config["output"]
        det_config = self.config["detection"]

        try:
            # Prepare the data for visualization using the utility function
            complete_result = prepare_martingale_visualization_data(detection_result)

            # Create betting config using the utility function
            betting_func_name = det_config["betting_func_config"]["name"]
            epsilon = det_config["betting_func_config"]["power"]["epsilon"]
            betting_config = create_betting_config_for_visualization(
                betting_func_name, epsilon
            )

            # Create visualization
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
            import traceback

            logger.debug(traceback.format_exc())

    def _export_results_to_csv(self, trial_results, true_change_points):
        """Export detection results to CSV files.

        Args:
            trial_results: Dict containing detection trial results
            true_change_points: List of ground truth change points
        """
        try:
            csv_output_dir = os.path.join(self.config["output"]["directory"])
            output_manager = OutputManager(csv_output_dir, self.config)
            output_manager.export_to_csv(
                trial_results["aggregated"],
                true_change_points,
                individual_trials=trial_results["individual_trials"],
            )
            logger.info(f"Results exported to CSV in {csv_output_dir}")
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {str(e)}")
