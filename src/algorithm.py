"""Core pipeline for graph structural change detection."""

import logging
import os
import time
import yaml
import numpy as np

from src.changepoint import ChangePointDetector, DetectorConfig, CUSUMDetector, CUSUMConfig, EWMADetector, EWMAConfig
from src.configs import get_config, get_full_model_name
from src.graph import GraphGenerator, NetworkFeatureExtractor
from src.graph.utils import adjacency_to_graph
from src.predictor.feature_predictor import FeaturePredictor
from src.utils import normalize_features, normalize_predictions, OutputManager, prepare_result_data

logger = logging.getLogger(__name__)


class GraphChangeDetection:
    """Pipeline for graph change point detection.

    Supports three detection methods:
    - martingale: Parallel horizon martingale (Algorithm 1 from ICDM 2025 paper)
    - cusum: CUSUM baseline
    - ewma: EWMA baseline

    For martingale method, predictions are required and both traditional and
    horizon martingales run in parallel.
    """

    def __init__(self, config_path=None, config_dict=None):
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("config_path or config_dict required")

    def _setup_output_dir(self):
        """Create output directory."""
        output_cfg = self.config.get("output", {})
        base_dir = output_cfg.get("directory", "results")

        # Only create subdirectory if base_dir doesn't have timestamp
        if not any(c.isdigit() for c in os.path.basename(base_dir)):
            ts = time.strftime("%Y%m%d_%H%M%S")
            net = self.config["model"]["network"]
            method = self.config["detection"]["method"]
            name = f"{net}_{method}_{ts}"
            self.config["output"]["directory"] = os.path.join(base_dir, name)

        os.makedirs(self.config["output"]["directory"], exist_ok=True)

    def _init_detector(self, seed=None):
        """Initialize detector based on method."""
        det = self.config["detection"]
        method = det.get("method", "martingale")
        seed = int(seed) if seed is not None else None

        if method == "martingale":
            betting_cfg = det.get("betting_func_config", {})
            betting_name = betting_cfg.get("name", "mixture")
            distance_metric = det.get("distance", {}).get("measure", "euclidean")
            mode = det.get("mode", "both")

            return ChangePointDetector(DetectorConfig(
                threshold=det.get("threshold", 30.0),
                history_size=self.config["model"]["predictor"]["config"].get("n_history", 10),
                window_size=det.get("max_window"),
                reset=det.get("reset", True),
                cooldown=det.get("cooldown", 30),
                betting_name=betting_name,
                betting_params=betting_cfg.get(betting_name, {"epsilons": [0.7, 0.8, 0.9]}),
                random_state=seed,
                distance_metric=distance_metric,
                mode=mode,
            ))

        elif method == "cusum":
            cusum_cfg = det.get("cusum", {})
            return CUSUMDetector(CUSUMConfig(
                threshold=det.get("threshold", 8.0),
                k=cusum_cfg.get("k", 0.5),
                startup_period=cusum_cfg.get("startup_period", 20),
                reset=det.get("reset", True),
            ))

        elif method == "ewma":
            ewma_cfg = det.get("ewma", {})
            return EWMADetector(EWMAConfig(
                threshold=det.get("threshold", 3.0),
                lambda_param=ewma_cfg.get("lambda", 0.1),
                L=ewma_cfg.get("L", 3.0),
                startup_period=ewma_cfg.get("startup_period", 20),
                reset=det.get("reset", True),
            ))

        raise ValueError(f"Unknown detection method: {method}")

    def _generate_sequence(self):
        """Generate graph sequence."""
        network = self.config["model"]["network"]
        generator = GraphGenerator(network)
        model_config = get_config(get_full_model_name(network))
        return generator.generate_sequence(model_config["params"].__dict__)

    def _extract_features(self, graphs):
        """Extract features from graphs."""
        extractor = NetworkFeatureExtractor()
        features = []
        for adj in graphs:
            graph = adjacency_to_graph(adj)
            numeric = extractor.get_numeric_features(graph)
            features.append([numeric[name] for name in self.config["features"]])
        return np.array(features)

    def _generate_feature_predictions(self, features):
        """Generate feature predictions using Holt's double exponential smoothing.

        Operates directly on extracted features rather than predicting adjacency
        matrices, matching the feature-space forecasting in Section IV-B (Eq. 21).

        Args:
            features: Raw feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_predictions, horizon, n_features)
        """
        horizon = self.config["detection"].get("prediction_horizon", 5)
        pred_cfg = self.config["model"]["predictor"].get("config", {})
        n_history = pred_cfg.get("n_history", 10)

        predictions = []
        for t in range(n_history, len(features)):
            fp = FeaturePredictor(
                alpha=pred_cfg.get("alpha", 0.3),
                beta=pred_cfg.get("beta", 0.1),
                n_history=n_history,
            )
            fp.fit(features[t - n_history:t])
            predictions.append(fp.predict(horizon))

        return np.array(predictions)

    def _run_trials(self, features, predictions, true_cps):
        """Run detection trials."""
        method = self.config["detection"].get("method", "martingale")
        n_trials = self.config["trials"]["n_trials"]
        seeds = self.config["trials"]["random_seeds"]

        if seeds is None:
            seeds = np.random.randint(0, 2**31 - 1, size=n_trials)
        elif isinstance(seeds, (int, float)):
            rng = np.random.RandomState(int(seeds))
            seeds = rng.randint(0, 2**31 - 1, size=n_trials)

        results = []
        for i, seed in enumerate(seeds[:n_trials]):
            detector = self._init_detector(int(seed) if method == "martingale" else None)
            try:
                if method == "martingale":
                    result = detector.run(data=features, predicted_data=predictions)
                else:
                    result = detector.run(data=features)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {e}")

        if not results:
            raise RuntimeError("All trials failed")

        return {"individual_trials": results, "aggregated": results[0]}

    def run(self, prediction=None, save_csv=None):
        """Run detection pipeline.

        Args:
            prediction: Override enable_prediction config. For martingale method,
                       predictions are required for parallel detection.
            save_csv: Override save_csv config.

        Returns:
            Dict with detection results including true and detected change points.
        """
        method = self.config["detection"].get("method", "martingale")
        mode = self.config["detection"].get("mode", "both")

        # Predictions needed for martingale horizon/both modes, not for traditional-only
        if method == "martingale":
            needs_predictions = mode in ("horizon", "both")
            enable_pred = needs_predictions if prediction is None else prediction
        else:
            enable_pred = prediction if prediction is not None else self.config["execution"].get("enable_prediction", False)

        enable_csv = save_csv if save_csv is not None else self.config["execution"].get("save_csv", True)

        self._setup_output_dir()

        # Generate sequence
        seq = self._generate_sequence()
        graphs, true_cps = seq["graphs"], seq["change_points"]
        logger.info(f"Generated {len(graphs)} graphs, CPs at {true_cps}")

        # Extract and normalize features
        features = self._extract_features(graphs)
        features_norm, means, stds = normalize_features(features)

        # Generate predictions using Holt's forecasting directly on features
        pred_norm = None
        if enable_pred:
            pred_features = self._generate_feature_predictions(features)
            pred_norm = normalize_predictions(pred_features, means, stds)

        # Run trials
        trial_results = self._run_trials(features_norm, pred_norm, true_cps)

        # Export CSV
        if enable_csv and trial_results["aggregated"]:
            manager = OutputManager(self.config["output"]["directory"], self.config)
            manager.export_to_csv(trial_results["aggregated"], true_cps, trial_results["individual_trials"])

        return prepare_result_data(seq, features, None, None, trial_results, None, self.config)
