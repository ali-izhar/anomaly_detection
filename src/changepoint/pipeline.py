# src/changepoint/pipeline.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
import networkx as nx
import logging

from .detector import ChangePointDetector
from ..graph.features import NetworkFeatureExtractor


class MartingalePipeline:
    """A pipeline for online change detection using martingale methods.

    Attributes:
        method (str): The martingale method to use ('single_view' or 'multiview').
        threshold (float): Detection threshold for martingale.
        epsilon (float): Sensitivity parameter for martingale updates.
        random_state (int): Seed for reproducibility.
        detector (ChangePointDetector): Instance of the change point detector.
        feature_extractor (NetworkFeatureExtractor): Instance of feature extractor.
        batch_size (int): Batch size for multiview processing.
        max_martingale (float): Early stopping threshold for multiview.
        reset (bool): Whether to reset after detection in single view.
        max_window (int): Maximum window size for strangeness computation.
    """

    def __init__(
        self,
        martingale_method: str = "multiview",
        # distance_metric: str = "euclidean",
        # bidding_metric: str = "exponential",
        threshold: float = 60.0,
        epsilon: float = 0.7,
        random_state: Optional[int] = 42,
        feature_set: str = "all",
        batch_size: int = 1000,
        max_martingale: Optional[float] = None,
        reset: bool = True,
        max_window: Optional[int] = None,
    ):
        """Initialize the pipeline with specified parameters.

        Args:
            martingale_method (str): Method to use ('single_view' or 'multiview').
            threshold (float): Threshold for change point detection.
            epsilon (float): Sensitivity parameter for martingale updates.
            random_state (int): Random seed for reproducibility.
            feature_set (str): Which feature set to use ('all', 'basic', 'spectral', 'centrality').
            batch_size (int): Batch size for multiview processing.
            max_martingale (float): Early stopping threshold for multiview.
            reset (bool): Whether to reset after detection in single view.
            max_window (int): Maximum window size for strangeness computation.
        """
        self.method = martingale_method
        self.threshold = threshold
        self.epsilon = epsilon
        self.random_state = random_state
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.max_martingale = max_martingale
        self.reset = reset
        self.max_window = max_window
        self.detector = ChangePointDetector()
        self.feature_extractor = NetworkFeatureExtractor()

    def _extract_numeric_features(
        self, feature_dict: dict, feature_set: str = "all"
    ) -> np.ndarray:
        """Extract numeric features from feature dictionary based on specified feature set.

        Args:
            feature_dict (dict): Dictionary of raw features.
            feature_set (str): Which feature set to use.

        Returns:
            np.ndarray: Array of numeric features.
        """
        features = []

        # Basic metrics
        if feature_set in ["all", "basic"]:
            degrees = feature_dict.get("degrees", [])
            features.append(np.mean(degrees) if degrees else 0.0)
            features.append(feature_dict.get("density", 0.0))
            clustering = feature_dict.get("clustering", [])
            features.append(np.mean(clustering) if clustering else 0.0)

        # Centrality metrics
        if feature_set in ["all", "centrality"]:
            betweenness = feature_dict.get("betweenness", [])
            features.append(np.mean(betweenness) if betweenness else 0.0)
            eigenvector = feature_dict.get("eigenvector", [])
            features.append(np.mean(eigenvector) if eigenvector else 0.0)
            closeness = feature_dict.get("closeness", [])
            features.append(np.mean(closeness) if closeness else 0.0)

        # Spectral metrics
        if feature_set in ["all", "spectral"]:
            singular_values = feature_dict.get("singular_values", [])
            features.append(max(singular_values) if singular_values else 0.0)
            laplacian_eigenvalues = feature_dict.get("laplacian_eigenvalues", [])
            features.append(
                min(x for x in laplacian_eigenvalues if x > 1e-10)
                if laplacian_eigenvalues
                else 0.0
            )

        return np.array(features)

    def process_raw_data(
        self,
        raw_data: Union[List[np.ndarray], List[nx.Graph]],
        data_type: str = "adjacency",
    ) -> Dict[str, Any]:
        """Process raw network data into features.

        Args:
            raw_data: List of either adjacency matrices or networkx graphs.
            data_type: Type of input data ('adjacency' or 'graph').

        Returns:
            Dict containing processed data and raw features.
        """
        features_raw = []
        features_numeric = []

        for item in raw_data:
            # Convert to networkx graph if needed
            if data_type == "adjacency":
                graph = nx.from_numpy_array(item)
            else:
                graph = item

            # Extract features
            feature_dict = self.feature_extractor.get_features(graph)
            features_raw.append(feature_dict)
            numeric_features = self._extract_numeric_features(
                feature_dict, self.feature_set
            )
            features_numeric.append(numeric_features)

        return {
            "features_raw": features_raw,
            "features_numeric": np.array(features_numeric),
        }

    def run(
        self,
        data: Union[np.ndarray, List[np.ndarray], List[nx.Graph]],
        data_type: str = "adjacency",
        predicted_data: Optional[List[List[np.ndarray]]] = None,
        history_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline from raw data to change detection.

        Args:
            data: Input data, can be:
                - np.ndarray of pre-extracted features
                - List of adjacency matrices
                - List of networkx graphs
            data_type: Type of input data ('features', 'adjacency', or 'graph')
            predicted_data: Optional list of predicted graphs. Structure:
                - First level list: predictions made at each timestep
                - Second level list: multiple predictions for future timesteps
                Each prediction is an adjacency matrix of same shape as input graphs
            history_size: Optional history size for horizon martingale computation

        Returns:
            Dict containing detection results and statistics
        """
        logger = logging.getLogger(__name__)
        
        # Debug predicted data
        if predicted_data is not None:
            logger.info(f"\nDEBUG: Predicted data received in pipeline:")
            logger.info(f"- Length of predicted_data: {len(predicted_data)}")
            if len(predicted_data) > 0:
                logger.info(f"- Shape of first prediction: {np.array(predicted_data[0]).shape}")
                logger.info(f"- Number of horizons: {len(predicted_data[0][0])}")

        # Process raw data if needed
        if data_type in ["adjacency", "graph"]:
            processed = self.process_raw_data(data, data_type)
            features = processed["features_numeric"]
            features_raw = processed["features_raw"]
        else:
            features = data
            features_raw = None

        # Debug features
        logger.info(f"\nDEBUG: Extracted features:")
        logger.info(f"- Number of features: {len(features)}")
        logger.info(f"- Length of each feature: {len(features[0])}")

        # Process predicted data if available
        predicted_features = None
        if predicted_data is not None:
            # First get features for all predictions
            predicted_features_raw = []
            for timestep_predictions in predicted_data:
                # Process each set of predictions for this timestep
                processed_predictions = self.process_raw_data(
                    timestep_predictions, data_type
                )
                predicted_features_raw.append(processed_predictions["features_numeric"])

            # Now restructure from [timestep][horizon][features] to [feature][timestep][horizon]
            if predicted_features_raw:
                num_features = predicted_features_raw[0].shape[1]
                num_timesteps = len(predicted_features_raw)
                num_horizons = predicted_features_raw[0].shape[0]

                predicted_features = []
                for feature_idx in range(num_features):
                    # Create a 2D array for each feature's predictions
                    feature_predictions = np.zeros((num_timesteps, num_horizons))
                    for timestep in range(num_timesteps):
                        for h in range(num_horizons):
                            feature_predictions[timestep, h] = predicted_features_raw[timestep][h][feature_idx]
                    predicted_features.append(feature_predictions)

                # Convert to list of numpy arrays
                predicted_features = [np.array(feature_pred) for feature_pred in predicted_features]

        # Run change detection
        if self.method == "single_view":
            result = self.detector.detect_changes(
                data=features,
                predicted_data=predicted_features,
                threshold=self.threshold,
                epsilon=self.epsilon,
                history_size=history_size,
                reset=self.reset,
                max_window=self.max_window,
                random_state=self.random_state,
            )
            # Add multiview fields as None for single view
            result["martingales_sum"] = None
            result["martingales_avg"] = None
            result["individual_martingales"] = None

        elif self.method == "multiview":
            # Split each feature into a separate view
            views = [features[:, i : i + 1] for i in range(features.shape[1])]

            # Split predicted features into views if available
            predicted_views = (
                predicted_features if predicted_features is not None else None
            )

            result = self.detector.detect_changes_multiview(
                data=views,
                predicted_data=predicted_views,
                threshold=self.threshold,
                epsilon=self.epsilon,
                history_size=history_size,
                max_window=self.max_window,
                max_martingale=self.max_martingale,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
            # Add single-view field as None for multiview
            result["martingales"] = result[
                "martingales_sum"
            ]  # Use sum as the main martingale

        else:
            raise ValueError(
                f"Unknown method: {self.method}. Choose 'single_view' or 'multiview'."
            )

        # Debug detection result
        logger.info(f"\nDEBUG: Detection result:")
        logger.info(f"- Has prediction_martingale_sum: {'prediction_martingale_sum' in result}")
        if 'prediction_martingale_sum' in result:
            logger.info(f"- Length of prediction_martingale_sum: {len(result['prediction_martingale_sum'])}")
            logger.info(f"- First few prediction martingale values: {result['prediction_martingale_sum'][:5]}")
            logger.info(f"- Max prediction martingale value: {np.max(result['prediction_martingale_sum'])}")

        # Add processed features to result if available
        if features_raw is not None:
            result["features_raw"] = features_raw
            result["features_numeric"] = features

        # Add predicted features to result if available
        if predicted_features is not None:
            result["predicted_features"] = predicted_features
            if "prediction_martingale_sum" in result:
                result["prediction_martingale_sum"] = result["prediction_martingale_sum"]
                if self.method == "multiview":
                    # Calculate average martingales for predictions
                    result["prediction_martingale_avg"] = result["prediction_martingale_avg"]

        return result
