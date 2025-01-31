# src/changepoint/pipeline.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
import networkx as nx

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

        Returns:
            Dict containing:
                - change_points: List of detected change points
                - features_raw: Raw feature dictionaries if processing from raw data
                - features_numeric: Numeric features used for detection
                - martingales: Martingale values (single view) or sum martingales (multiview)
                - martingales_sum: Sum of martingales (multiview only, None for single view)
                - martingales_avg: Average of martingales (multiview only, None for single view)
                - individual_martingales: Individual feature martingales (multiview only)
                - p_values: P-values for each point
                - strangeness: Strangeness values
                - predicted_features: Features extracted from predicted graphs
                - prediction_martingales: Martingales computed on predicted features

        Raises:
            ValueError: If an unknown method or data type is specified.
        """
        # Process raw data if needed
        if data_type in ["adjacency", "graph"]:
            processed = self.process_raw_data(data, data_type)
            features = processed["features_numeric"]
            features_raw = processed["features_raw"]
        else:
            features = data
            features_raw = None

        # Process predicted data if available
        predicted_features = None
        if predicted_data is not None:
            predicted_features = []
            for timestep_predictions in predicted_data:
                # Process each set of predictions for this timestep
                processed_predictions = self.process_raw_data(
                    timestep_predictions, data_type
                )
                predicted_features.append(processed_predictions["features_numeric"])

        # Run change detection
        if self.method == "single_view":
            result = self.detector.detect_changes(
                data=features,
                predicted_data=(
                    predicted_features if predicted_features is not None else None
                ),
                threshold=self.threshold,
                epsilon=self.epsilon,
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
            predicted_views = None
            if predicted_features is not None:
                predicted_views = []
                for timestep_features in predicted_features:
                    predicted_views.append(
                        [
                            timestep_features[:, i : i + 1]
                            for i in range(timestep_features.shape[1])
                        ]
                    )

            result = self.detector.detect_changes_multiview(
                data=views,
                predicted_data=predicted_views,
                threshold=self.threshold,
                epsilon=self.epsilon,
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

        # Add processed features to result if available
        if features_raw is not None:
            result["features_raw"] = features_raw
            result["features_numeric"] = features

        # Add predicted features to result if available
        if predicted_features is not None:
            result["predicted_features"] = predicted_features
            if "prediction_martingales" in result:
                result["prediction_martingales_sum"] = result.pop(
                    "prediction_martingales"
                )
                if self.method == "multiview":
                    # Calculate average martingales for predictions
                    result["prediction_martingales_avg"] = [
                        sum(m) / len(m) for m in result["prediction_martingales_sum"]
                    ]

        return result
