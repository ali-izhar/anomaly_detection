#!/usr/bin/env python
"""MIT Reality Dataset Processor

The dataset contains Bluetooth proximity data from 100 MIT students and faculty
collected over the 2007-2008 academic year.
"""

from datetime import datetime
from pathlib import Path

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint import ChangePointDetector, DetectorConfig, BettingFunctionConfig
from src.graph import NetworkFeatureExtractor
from src.utils.output_manager import OutputManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MITRealityProcessor:
    """Process the MIT Reality Mining dataset into graph snapshots and detect changes."""

    def __init__(
        self,
        data_path: str = "archive/data/Proximity.csv",
        proximity_threshold: float = 0.3,
        output_dir: str = "results/reality_mining",
    ):
        """Initialize the processor.

        Args:
            data_path: Path to Proximity.csv file
            proximity_threshold: Proximity threshold for edge creation
            output_dir: Directory to save output files
        """
        self.data_path = data_path
        self.proximity_threshold = proximity_threshold
        self.output_dir = output_dir
        self.data = None
        self.daily_graphs = None
        self.features = None
        self.timestamps = None
        self.features_normalized = None
        self.feature_means = None
        self.feature_stds = None
        self.predicted_features = None
        self.detection_results = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/graphs", exist_ok=True)
        os.makedirs(f"{output_dir}/matrices", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{output_dir}/processing.log"),
                logging.StreamHandler(),
            ],
        )

        # Define known events for ground truth evaluation
        self.events = {
            # Format: 'YYYY-MM-DD': 'Event Name'
            # 2007 holidays
            "2007-09-03": "Labor Day 2007",
            "2007-10-08": "Columbus Day 2007",
            "2007-11-22": "Thanksgiving 2007",
            "2007-12-24": "Christmas Eve 2007",
            "2007-12-25": "Christmas Day 2007",
            "2007-12-31": "New Year's Eve 2007",
            # 2008 holidays
            "2008-01-01": "New Year's Day 2008",
            "2008-01-21": "Martin Luther King Day 2008",
            "2008-02-18": "Presidents Day 2008",
            "2008-03-21": "Good Friday 2008",
            "2008-05-26": "Memorial Day 2008",
            "2008-07-04": "Independence Day 2008",
            "2008-09-01": "Labor Day 2008",
            "2008-10-13": "Columbus Day 2008",
            "2008-11-27": "Thanksgiving 2008",
            "2008-12-24": "Christmas Eve 2008",
            "2008-12-25": "Christmas Day 2008",
            "2008-12-31": "New Year's Eve 2008",
            # 2009 holidays
            "2009-01-01": "New Year's Day 2009",
            "2009-01-19": "Martin Luther King Day 2009",
            "2009-02-16": "Presidents Day 2009",
            "2009-04-10": "Good Friday 2009",
            "2009-05-25": "Memorial Day 2009",
            "2009-07-04": "Independence Day 2009",
        }

        # Components for analysis
        self.raw_data = None
        self.filtered_data = None
        self.filtered_dates = None
        self.adj_matrices = []
        self.dates = None

        # Evaluation metrics storage
        self.evaluation_metrics = None

    def load_data(self):
        """
        Load and preprocess the proximity data.

        Returns:
            DataFrame with preprocessed proximity data
        """
        logger.info(f"Loading data from {self.data_path}")

        try:
            # Load the data - correctly handling comma-delimited format
            df = pd.read_csv(
                self.data_path,
                delimiter=",",  # The file uses comma delimiter
                header=0,  # First row contains headers
            )

            # Convert timestamp to datetime
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

            # Remove records with invalid timestamps
            invalid_timestamps = df["time"].isna().sum()
            if invalid_timestamps > 0:
                logger.warning(
                    f"Removed {invalid_timestamps} records with invalid timestamps"
                )
                df = df.dropna(subset=["time"])

            # Rename columns to match our expected format
            df = df.rename(
                columns={
                    "user.id": "user_id",
                    "remote.user.id.if.known": "remote_user_id",
                    "time": "timestamp",
                    "prob2": "proximity_prob",
                }
            )

            # Extract date for grouping
            df["date"] = df["timestamp"].dt.date

            # Convert proximity_prob to float, handling empty values
            df["proximity_prob"] = pd.to_numeric(df["proximity_prob"], errors="coerce")

            # Apply filters - store both raw and filtered data
            self.raw_data = df.copy()
            self.filtered_data = df[
                df["proximity_prob"] >= self.proximity_threshold
            ].copy()

            # Extract unique dates in chronological order
            self.filtered_dates = (
                self.filtered_data["date"]
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

            # Log data stats
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"Unique users: {df['user_id'].nunique()}")
            logger.info(
                f"Loaded {len(df)} proximity records, kept {len(self.filtered_data)} after filtering"
            )

            # Save intermediate data if output directory is specified
            if self.output_dir:
                self.filtered_data.to_csv(
                    os.path.join(self.output_dir, "filtered_data.csv"), index=False
                )
                pd.DataFrame(self.filtered_dates, columns=["date"]).to_csv(
                    os.path.join(self.output_dir, "filtered_dates.csv"), index=False
                )

            return self.filtered_data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_daily_graphs(self):
        """
        Create daily graph snapshots from preprocessed data.

        Returns:
            Dictionary mapping dates to NetworkX graphs
        """
        if self.filtered_data is None:
            logger.error("No filtered data. Call load_data() first")
            raise ValueError("No filtered data. Call load_data() first")

        logger.info("Creating daily graph snapshots")

        daily_graphs = {}

        # Get all unique users to ensure consistent node sets
        all_users = set(self.filtered_data["user_id"].unique()) | set(
            self.filtered_data["remote_user_id"].unique()
        )

        # Group by date
        for date in self.filtered_dates:
            # Create a new graph for this day
            G = nx.Graph()

            # Add all unique users as nodes
            G.add_nodes_from(all_users)

            # Get data for this date
            day_df = self.filtered_data[self.filtered_data["date"] == date]

            # Skip days with too few interactions
            if len(day_df) <= 5:
                logger.warning(f"Skipping date {date} with insufficient interactions")
                continue

            # Add edges for valid interactions
            edge_count = 0
            for _, row in day_df.iterrows():
                G.add_edge(row["user_id"], row["remote_user_id"])
                edge_count += 1

            # Only include days with sufficient edges
            if G.number_of_edges() <= 1:
                logger.warning(f"Skipping date {date} with insufficient edges")
                continue

            # Store the graph with ISO format date string
            date_str = date.strftime("%Y-%m-%d")
            daily_graphs[date_str] = G

            # Check if this is an event date
            if date_str in self.events:
                logger.info(
                    f"Created graph for event day: {date_str} ({self.events[date_str]})"
                )

        self.daily_graphs = daily_graphs
        logger.info(f"Created {len(daily_graphs)} daily graph snapshots")

        # Create adjacency matrices for detection
        self.adj_matrices = []
        self.dates = []
        for date, G in sorted(daily_graphs.items()):
            self.adj_matrices.append(nx.to_numpy_array(G))
            self.dates.append(date)

        return daily_graphs

    def save_graphs(self):
        """
        Save the daily graphs to files.
        """
        if not self.output_dir or not self.daily_graphs:
            logger.warning("No output directory specified or no graphs to save")
            return

        logger.info(f"Saving graphs to {self.output_dir}")

        # Create adjacency matrix directory
        adj_dir = os.path.join(self.output_dir, "adjacency_matrices")
        os.makedirs(adj_dir, exist_ok=True)

        # Create graph visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Save each graph
        for date, G in self.daily_graphs.items():
            # Save adjacency matrix as CSV
            adj_matrix = nx.to_numpy_array(G)
            adj_path = os.path.join(adj_dir, f"{date}.csv")
            np.savetxt(adj_path, adj_matrix, delimiter=",")

            # Save basic graph visualization
            if len(G.nodes()) <= 100:  # Only visualize if not too large
                plt.figure(figsize=(10, 10))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=False, node_size=50, alpha=0.7)

                # Add title with event information if applicable
                title = f"Network on {date}"
                if date in self.events:
                    title += f" - {self.events[date]}"
                plt.title(title)

                viz_path = os.path.join(viz_dir, f"{date}.png")
                plt.savefig(viz_path)
                plt.close()

        # Save event annotations
        event_path = os.path.join(self.output_dir, "events.csv")
        with open(event_path, "w") as f:
            f.write("date,event\n")
            for date, event in self.events.items():
                f.write(f"{date},{event}\n")

        logger.info(f"Saved {len(self.daily_graphs)} graphs and event annotations")

    def generate_event_summary(self):
        """
        Generate a summary of graph properties around events.

        Returns:
            DataFrame with summary statistics for each date
        """
        if not self.daily_graphs:
            logger.error("No graphs created. Call create_daily_graphs() first")
            raise ValueError("No graphs created. Call create_daily_graphs() first")

        logger.info("Generating event summary")

        summary_data = []

        for date, G in sorted(self.daily_graphs.items()):
            # Calculate basic network metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)

            # Calculate mean degree and clustering coefficient if graph has edges
            if n_edges > 0:
                mean_degree = sum(dict(G.degree()).values()) / n_nodes
                clustering = nx.average_clustering(G)
            else:
                mean_degree = 0
                clustering = 0

            # Try to calculate other metrics, handling disconnected graphs
            try:
                components = list(nx.connected_components(G))
                largest_component_size = len(max(components, key=len))
                n_components = len(components)
            except:
                largest_component_size = 0
                n_components = 0

            # Add event information
            is_event = date in self.events
            event_name = self.events.get(date, "")

            # Create summary row
            summary_data.append(
                {
                    "date": date,
                    "nodes": n_nodes,
                    "edges": n_edges,
                    "density": density,
                    "mean_degree": mean_degree,
                    "clustering": clustering,
                    "components": n_components,
                    "largest_component": largest_component_size,
                    "is_event": is_event,
                    "event_name": event_name,
                }
            )

        # Create DataFrame and sort by date
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("date")

        # Save summary if output directory is specified
        if self.output_dir:
            summary_path = os.path.join(self.output_dir, "network_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved network summary to {summary_path}")

        return summary_df

    def extract_features(self):
        """
        Extract graph features for change point detection.

        Returns:
            numpy array of features for each day
        """
        if not self.adj_matrices:
            logger.error("No adjacency matrices. Call create_daily_graphs() first")
            raise ValueError("No adjacency matrices. Call create_daily_graphs() first")

        logger.info("Extracting features from daily graphs")

        feature_extractor = NetworkFeatureExtractor()
        features = []

        # Define which features to extract (match algorithm.yaml)
        feature_names = [
            "mean_degree",  # Average node degree
            "density",  # Graph density
            "mean_clustering",  # Average clustering coefficient
            "mean_betweenness",  # Average betweenness centrality
            "mean_eigenvector",  # Average eigenvector centrality
            "mean_closeness",  # Average closeness centrality
            "max_singular_value",  # Largest singular value
            "min_nonzero_laplacian",  # Smallest non-zero Laplacian eigenvalue
        ]

        for i, date in enumerate(self.dates):
            # Get graph from adjacency matrix
            adj_matrix = self.adj_matrices[i]
            G = nx.from_numpy_array(adj_matrix)

            # Skip small graphs
            if G.number_of_edges() <= 1:
                logger.warning(f"Skipping date {date} with insufficient edges")
                continue

            # Get graph features
            numeric_features = feature_extractor.get_numeric_features(G)

            # Extract selected features
            feature_vector = []
            for name in feature_names:
                if name in numeric_features:
                    feature_vector.append(numeric_features[name])
                else:
                    # Handle missing features with 0 instead of NaN
                    logger.warning(f"Feature {name} not available for date {date}")
                    feature_vector.append(0.0)

            features.append(feature_vector)

        # Convert to numpy array
        features_array = np.array(features)

        # Check for NaN values and replace them
        has_nans = np.isnan(features_array).any()
        if has_nans:
            logger.warning("Found NaN values in features, replacing with zeros")
            features_array = np.nan_to_num(features_array, nan=0.0)

        logger.info(
            f"Extracted {features_array.shape[1]} features from {features_array.shape[0]} days"
        )

        # Save features if output directory is specified
        if self.output_dir:
            features_df = pd.DataFrame(
                features_array, index=self.dates, columns=feature_names
            )
            features_path = os.path.join(self.output_dir, "graph_features.csv")
            features_df.to_csv(features_path)
            logger.info(f"Saved features to {features_path}")

        self.features = features_array
        return features_array

    def normalize_features(self):
        """
        Normalize features for change point detection.

        Returns:
            tuple: (normalized_features, feature_means, feature_stds)
        """
        if self.features is None:
            logger.error("No features extracted. Call extract_features() first")
            raise ValueError("No features extracted. Call extract_features() first")

        logger.info("Normalizing graph features")

        # Calculate means and standard deviations
        self.feature_means = np.mean(self.features, axis=0)
        self.feature_stds = np.std(self.features, axis=0)

        # Avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0

        # Normalize features
        self.features_normalized = (
            self.features - self.feature_means
        ) / self.feature_stds

        # Check for NaN values after normalization
        if np.isnan(self.features_normalized).any():
            logger.warning("Found NaN values after normalization, replacing with zeros")
            self.features_normalized = np.nan_to_num(self.features_normalized, nan=0.0)

        logger.info(
            f"Normalized {self.features_normalized.shape[1]} features across {self.features_normalized.shape[0]} timesteps"
        )

        # Save normalized features if output directory is specified
        if self.output_dir:
            # Save means and standard deviations for future use
            np.savetxt(
                os.path.join(self.output_dir, "feature_means.csv"),
                self.feature_means,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(self.output_dir, "feature_stds.csv"),
                self.feature_stds,
                delimiter=",",
            )

            # Save normalized features
            np.savetxt(
                os.path.join(self.output_dir, "features_normalized.csv"),
                self.features_normalized,
                delimiter=",",
            )

            logger.info(f"Saved normalization parameters to {self.output_dir}")

        return self.features_normalized, self.feature_means, self.feature_stds

    def generate_predictions(self, horizon=5):
        """
        Generate predictions for the graph sequence features.

        Args:
            horizon: Number of future steps to predict

        Returns:
            numpy array of predicted features
        """
        if self.features_normalized is None:
            logger.error(
                "Features must be normalized before prediction. Call normalize_features() first"
            )
            raise ValueError("Features must be normalized before prediction")

        logger.info(f"Generating predictions with horizon={horizon}")

        n_features = self.features_normalized.shape[1]
        n_timesteps = self.features_normalized.shape[0]

        # Initialize array with correct shape (timesteps, horizon, features)
        self.predicted_features = np.zeros((n_timesteps, horizon, n_features))

        # Config aligned with algorithm.yaml
        history_size = 10  # From algorithm.yaml predictor config

        # For each timestep, generate predictions for next 'horizon' steps
        for t in range(n_timesteps):
            # Define history window for this prediction
            history_start = max(0, t - history_size)
            history = self.features_normalized[history_start : t + 1]

            # Use adaptive prediction approach based on the algorithm.py
            # This is a simplified version of the Graph predictor configuration
            alpha = 0.8  # From algorithm.yaml
            gamma = 0.5  # From algorithm.yaml

            # For each horizon step
            for h in range(horizon):
                if len(history) > 0:
                    # Use weighted average favoring recent observations with exponential decay
                    weights = np.exp(np.linspace(0, 1, len(history)))

                    # Apply adaptive weighting
                    if h > 0 and t > 0:
                        # Add autoregressive component
                        ar_component = self.predicted_features[t, h - 1]
                        # Add trend component based on history
                        trend = np.zeros_like(ar_component)
                        if len(history) >= 2:
                            trend = np.mean([history[-1] - history[-2]], axis=0)

                        # Combine components
                        pred = alpha * np.average(history, axis=0, weights=weights) + (
                            1 - alpha
                        ) * (ar_component + gamma * trend)
                    else:
                        # Use weighted average for first step
                        pred = np.average(history, axis=0, weights=weights)
                else:
                    # If no history, use the current timestep
                    pred = self.features_normalized[t]

                # Store prediction
                self.predicted_features[t, h] = pred

                # Add prediction to history for next step (auto-regressive)
                history = np.vstack([history, pred.reshape(1, -1)])

        # Check for NaN values in predictions and replace them
        if np.isnan(self.predicted_features).any():
            logger.warning("Found NaN values in predictions, replacing with zeros")
            self.predicted_features = np.nan_to_num(self.predicted_features, nan=0.0)

        logger.info(
            f"Generated predictions with shape: {self.predicted_features.shape}"
        )

        # Save predictions if output directory is specified
        if self.output_dir:
            # Save predictions for each horizon step
            for h in range(horizon):
                np.savetxt(
                    os.path.join(self.output_dir, f"predictions_horizon_{h+1}.csv"),
                    self.predicted_features[:, h, :],
                    delimiter=",",
                )

            logger.info(f"Saved predictions to {self.output_dir}")

        return self.predicted_features

    def detect_change_points(self, use_prediction=True, horizon=5):
        """
        Detect change points in the graph sequence.

        Args:
            use_prediction: whether to use prediction-enhanced detection
            horizon: prediction horizon if prediction is enabled

        Returns:
            Dict containing detection results
        """
        if self.features is None:
            logger.error("No features extracted. Call extract_features() first")
            raise ValueError("No features extracted. Call extract_features() first")

        # Ensure features are normalized
        if self.features_normalized is None:
            logger.info("Features not normalized. Normalizing now.")
            self.normalize_features()

        logger.info(
            f"Detecting change points (prediction={use_prediction}, horizon={horizon if use_prediction else 'N/A'})"
        )

        # Generate predictions if needed and not already available
        predicted_data = None
        if use_prediction:
            if (
                self.predicted_features is None
                or self.predicted_features.shape[1] != horizon
            ):
                logger.info("Generating predictions for detection")
                self.generate_predictions(horizon=horizon)
            predicted_data = self.predicted_features

        # Configure detection parameters - aligned with algorithm.yaml
        betting_func_config = BettingFunctionConfig(
            name="mixture", params={"epsilons": [0.7, 0.8, 0.9]}, random_seed=42
        )

        detector_config = DetectorConfig(
            method="horizon" if use_prediction else "traditional",
            threshold=60.0,  # From algorithm.yaml
            history_size=10,  # From algorithm.yaml predictor config
            batch_size=1,
            reset=True,
            reset_on_traditional=True,  # From algorithm.yaml
            max_window=25,  # Adjusted for MIT Reality dataset
            betting_func_config=betting_func_config,
            distance_measure="mahalanobis",  # From algorithm.yaml
            distance_p=2.0,  # From algorithm.yaml
            random_state=42,
        )

        # Initialize and run detector
        detector = ChangePointDetector(detector_config)

        # Run detection
        try:
            detection_result = detector.run(
                data=self.features_normalized,
                predicted_data=predicted_data,
                reset_state=True,
            )
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            if use_prediction:
                logger.info("Falling back to traditional detection without prediction")
                # Create a new detector config instead of modifying the frozen one
                trad_config = DetectorConfig(
                    method="traditional",
                    threshold=60.0,
                    history_size=10,
                    batch_size=1,
                    reset=True,
                    max_window=25,
                    betting_func_config=betting_func_config,
                    distance_measure="mahalanobis",
                    distance_p=2.0,
                    random_state=42,
                )
                detector = ChangePointDetector(trad_config)
                detection_result = detector.run(
                    data=self.features_normalized, predicted_data=None, reset_state=True
                )
            else:
                raise

        # Map indices to dates
        if "traditional_change_points" in detection_result:
            trad_cp = detection_result["traditional_change_points"]
            detection_result["traditional_dates"] = [
                self.dates[i] for i in trad_cp if i < len(self.dates)
            ]

        if "horizon_change_points" in detection_result:
            horizon_cp = detection_result["horizon_change_points"]
            detection_result["horizon_dates"] = [
                self.dates[i] for i in horizon_cp if i < len(self.dates)
            ]

        # Evaluate detection against ground truth
        detection_result["metrics"] = self.evaluate_detection(
            detection_result, use_prediction
        )

        # Store results
        self.detection_results = detection_result

        # Export results
        self.export_detection_results(detection_result, use_prediction)

        return detection_result

    def evaluate_detection(self, detection_result, use_prediction=True):
        """
        Evaluate detection results against known events.

        Args:
            detection_result: Detection results dictionary
            use_prediction: Whether prediction was used

        Returns:
            Dict containing evaluation metrics
        """
        # Check against known events
        true_event_dates = list(self.events.keys())
        detected_dates = detection_result.get(
            "horizon_dates" if use_prediction else "traditional_dates", []
        )

        # Calculate detection metrics
        true_positives = [
            d
            for d in detected_dates
            if any(
                abs(
                    (
                        datetime.strptime(d, "%Y-%m-%d")
                        - datetime.strptime(e, "%Y-%m-%d")
                    ).days
                )
                <= 3
                for e in true_event_dates
            )
        ]

        false_positives = [d for d in detected_dates if d not in true_positives]
        false_negatives = [
            e
            for e in true_event_dates
            if not any(
                abs(
                    (
                        datetime.strptime(d, "%Y-%m-%d")
                        - datetime.strptime(e, "%Y-%m-%d")
                    ).days
                )
                <= 3
                for d in detected_dates
            )
        ]

        # Calculate evaluation metrics
        precision = len(true_positives) / len(detected_dates) if detected_dates else 0
        recall = len(true_positives) / len(true_event_dates) if true_event_dates else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        fpr = (
            len(false_positives) / (len(self.dates) - len(true_event_dates))
            if false_positives
            else 0
        )

        # Calculate detection delays
        delays = []
        for tp in true_positives:
            # Find closest true event date
            closest_event = min(
                true_event_dates,
                key=lambda e: abs(
                    (
                        datetime.strptime(tp, "%Y-%m-%d")
                        - datetime.strptime(e, "%Y-%m-%d")
                    ).days
                ),
            )
            # Calculate delay in days
            delay = (
                datetime.strptime(tp, "%Y-%m-%d")
                - datetime.strptime(closest_event, "%Y-%m-%d")
            ).days
            delays.append(abs(delay))

        mean_delay = np.mean(delays) if delays else 0

        # Store metrics
        metrics = {
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "fpr": fpr,
            "mean_delay": mean_delay,
            "tpr": recall,  # TPR is the same as recall
            "true_positive_dates": true_positives,
            "false_positive_dates": false_positives,
            "false_negative_dates": false_negatives,
        }

        logger.info(
            f"Evaluation metrics: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1_score:.2f}"
        )

        return metrics

    def export_detection_results(self, detection_result, use_prediction=True):
        """
        Export detection results to CSV files and Excel using OutputManager.

        Args:
            detection_result: Detection results dictionary
            use_prediction: Whether prediction was used
        """
        if not self.output_dir:
            return

        detector_type = "horizon" if use_prediction else "traditional"

        # Prepare results directory
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create simplified detection CSV files (basic output)
        # Export detected change points
        detected_dates = detection_result.get(
            "horizon_dates" if use_prediction else "traditional_dates", []
        )

        # Create binary detection result for each day
        results_df = pd.DataFrame(
            {
                "date": self.dates,
                "is_true_event": [d in self.events for d in self.dates],
                "detected_change": [d in detected_dates for d in self.dates],
            }
        )

        # Save detection results
        results_path = os.path.join(
            results_dir, f"detection_results_{detector_type}.csv"
        )
        results_df.to_csv(results_path, index=False)

        # Save metrics
        metrics_path = os.path.join(
            results_dir, f"detection_metrics_{detector_type}.csv"
        )
        pd.DataFrame([detection_result["metrics"]]).to_csv(metrics_path, index=False)

        # Save martingale scores if available
        if "martingale_scores" in detection_result:
            scores_path = os.path.join(
                results_dir, f"martingale_scores_{detector_type}.csv"
            )
            scores_df = pd.DataFrame(
                {
                    "date": self.dates[: len(detection_result["martingale_scores"])],
                    "martingale_score": detection_result["martingale_scores"],
                }
            )
            scores_df.to_csv(scores_path, index=False)

        # Save p-values if available
        if "p_values" in detection_result:
            pvals_path = os.path.join(results_dir, f"p_values_{detector_type}.csv")
            pvals_df = pd.DataFrame(
                {
                    "date": self.dates[: len(detection_result["p_values"])],
                    "p_value": detection_result["p_values"],
                }
            )
            pvals_df.to_csv(pvals_path, index=False)

        # Use OutputManager to create comprehensive Excel export (like algorithm.py)
        try:
            # Create config dictionary similar to algorithm.py for OutputManager
            config = {
                "output": {
                    "directory": self.output_dir,
                    "prefix": f"mit_reality_{detector_type}",
                    "save_martingales": True,
                    "save_predictions": use_prediction,
                    "save_features": True,
                },
                "model": {
                    "network": "reality_mining",
                    "type": "horizon" if use_prediction else "traditional",
                    "predictor": {
                        "type": "adaptive" if use_prediction else "none",
                        "config": {
                            "n_history": 10,
                        },
                    },
                },
                "detection": {
                    "threshold": 60.0,
                    "distance": {
                        "measure": "mahalanobis",
                        "p": 2.0,
                    },
                    "betting_func_config": {
                        "name": "mixture",
                        "mixture": {"epsilons": [0.7, 0.8, 0.9]},
                    },
                },
                "features": [
                    "mean_degree",
                    "density",
                    "mean_clustering",
                    "mean_betweenness",
                    "mean_eigenvector",
                    "mean_closeness",
                    "max_singular_value",
                    "min_nonzero_laplacian",
                ],
            }

            # Prepare data for OutputManager similar to algorithm.py's prepare_result_data
            # Create true change points from event dates
            true_change_points = []
            for i, date in enumerate(self.dates):
                if date in self.events:
                    true_change_points.append(i)

            # Create sequence result dict
            sequence_result = {
                "change_points": true_change_points,
                "features": self.features,
                "dates": self.dates,
                "events": self.events,
            }

            # Initialize OutputManager
            output_manager = OutputManager(self.output_dir, config)

            # Export to CSV/Excel with individual trials data
            output_manager.export_to_csv(
                detection_result,
                true_change_points,
                individual_trials=[
                    detection_result
                ],  # Use same result as a single trial
            )

            logger.info(
                f"Exported comprehensive detection results using OutputManager to {self.output_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to export comprehensive results: {str(e)}")
            logger.info("Basic CSV files were still created")

        logger.info(f"Exported detection results to {results_dir}")

    def compare_with_ground_truth(self, summary=None):
        """
        Compare detection results between traditional and horizon detection methods.

        Args:
            summary: Optional network summary DataFrame

        Returns:
            DataFrame with comparison metrics
        """
        if summary is None:
            summary = self.generate_event_summary()

        logger.info("Comparing detection methods with ground truth")

        # Run traditional detection if not already done
        if (
            self.detection_results is None
            or "traditional_dates" not in self.detection_results
        ):
            logger.info("Running traditional detection for comparison")
            trad_result = self.detect_change_points(use_prediction=False)
        else:
            trad_result = self.detection_results

        # Run horizon detection if not already done
        if (
            self.detection_results is None
            or "horizon_dates" not in self.detection_results
        ):
            logger.info("Running horizon detection for comparison")
            horizon_result = self.detect_change_points(use_prediction=True)
        else:
            horizon_result = self.detection_results

        # Extract metrics
        trad_metrics = trad_result.get("metrics", {})
        horizon_metrics = horizon_result.get("metrics", {})

        # Create comparison DataFrame
        comparison = pd.DataFrame(
            {
                "Metric": [
                    "True Positives",
                    "False Positives",
                    "False Negatives",
                    "Precision",
                    "Recall (TPR)",
                    "F1 Score",
                    "FPR",
                    "Mean Delay",
                ],
                "Traditional": [
                    trad_metrics.get("true_positives", 0),
                    trad_metrics.get("false_positives", 0),
                    trad_metrics.get("false_negatives", 0),
                    trad_metrics.get("precision", 0),
                    trad_metrics.get("recall", 0),
                    trad_metrics.get("f1_score", 0),
                    trad_metrics.get("fpr", 0),
                    trad_metrics.get("mean_delay", 0),
                ],
                "Horizon": [
                    horizon_metrics.get("true_positives", 0),
                    horizon_metrics.get("false_positives", 0),
                    horizon_metrics.get("false_negatives", 0),
                    horizon_metrics.get("precision", 0),
                    horizon_metrics.get("recall", 0),
                    horizon_metrics.get("f1_score", 0),
                    horizon_metrics.get("fpr", 0),
                    horizon_metrics.get("mean_delay", 0),
                ],
            }
        )

        # Save comparison
        if self.output_dir:
            comparison_path = os.path.join(self.output_dir, "detector_comparison.csv")
            comparison.to_csv(comparison_path, index=False)
            logger.info(f"Saved detector comparison to {comparison_path}")

            # Create visualization of the comparison
            self.visualize_comparison(comparison, summary)

        return comparison

    def visualize_comparison(self, comparison, summary):
        """
        Create visualization comparing traditional and horizon detection.

        Args:
            comparison: DataFrame with comparison metrics
            summary: Network summary DataFrame
        """
        if not self.output_dir:
            return

        # Create bar chart comparing key metrics
        plt.figure(figsize=(12, 10))

        # Plot precision, recall, F1
        plt.subplot(2, 2, 1)
        metrics = ["Precision", "Recall (TPR)", "F1 Score"]
        ind = np.arange(len(metrics))
        width = 0.35

        trad_values = [
            comparison[comparison["Metric"] == m]["Traditional"].values[0]
            for m in metrics
        ]
        horizon_values = [
            comparison[comparison["Metric"] == m]["Horizon"].values[0] for m in metrics
        ]

        plt.bar(ind - width / 2, trad_values, width, label="Traditional")
        plt.bar(ind + width / 2, horizon_values, width, label="Horizon")
        plt.ylabel("Value")
        plt.title("Detection Performance Metrics")
        plt.xticks(ind, metrics)
        plt.legend()

        # Plot FPR and delay
        plt.subplot(2, 2, 2)
        metrics = ["FPR", "Mean Delay"]
        ind = np.arange(len(metrics))

        trad_values = [
            comparison[comparison["Metric"] == m]["Traditional"].values[0]
            for m in metrics
        ]
        horizon_values = [
            comparison[comparison["Metric"] == m]["Horizon"].values[0] for m in metrics
        ]

        plt.bar(ind - width / 2, trad_values, width, label="Traditional")
        plt.bar(ind + width / 2, horizon_values, width, label="Horizon")
        plt.ylabel("Value")
        plt.title("Error Metrics")
        plt.xticks(ind, metrics)
        plt.legend()

        # Visualize detections on network density timeline
        plt.subplot(2, 1, 2)

        # Convert dates to datetime for better plotting
        date_objects = pd.to_datetime(summary["date"])

        # Plot density over time
        plt.plot(
            date_objects, summary["density"], "b-", alpha=0.5, label="Network Density"
        )

        # Mark known events
        for event_date, event_name in self.events.items():
            if event_date in summary["date"].values:
                plt.axvline(
                    x=pd.to_datetime(event_date),
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                    label=(
                        "Known Event"
                        if event_date == list(self.events.keys())[0]
                        else ""
                    ),
                )

        # Mark traditional detections
        if self.detection_results and "traditional_dates" in self.detection_results:
            for d in self.detection_results["traditional_dates"]:
                plt.axvline(
                    x=pd.to_datetime(d),
                    color="r",
                    linestyle="-",
                    alpha=0.5,
                    label=(
                        "Traditional Detection"
                        if d == self.detection_results["traditional_dates"][0]
                        else ""
                    ),
                )

        # Mark horizon detections
        if self.detection_results and "horizon_dates" in self.detection_results:
            for d in self.detection_results["horizon_dates"]:
                plt.axvline(
                    x=pd.to_datetime(d),
                    color="m",
                    linestyle="-",
                    alpha=0.7,
                    label=(
                        "Horizon Detection"
                        if d == self.detection_results["horizon_dates"][0]
                        else ""
                    ),
                )

        plt.xlabel("Date")
        plt.ylabel("Network Density")
        plt.title("Network Density with Detected Change Points")
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "detector_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Saved detector comparison visualization to {self.output_dir}")

    def process(self, run_detection=True, use_prediction=False):
        """Process the MIT Reality Mining dataset.

        This method orchestrates the complete processing pipeline, but allows
        for flexibility in which components are executed.

        Args:
            run_detection: Whether to run change point detection
            use_prediction: Whether to use prediction-enhanced detection

        Returns:
            tuple: (daily_graphs, summary, detection_result)
        """
        logger.info("Starting MIT Reality Mining dataset processing")

        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        self.data = self.load_data()

        # Step 2: Create daily graph snapshots
        logger.info("Step 2: Creating daily graph snapshots")
        self.daily_graphs = self.create_daily_graphs()
        self.save_graphs()

        # Step 3: Generate event summary
        logger.info("Step 3: Generating event summary")
        summary = self.generate_event_summary()

        # Step 4: Extract graph features
        logger.info("Step 4: Extracting graph features")
        self.features = self.extract_features()

        # Step 5: Normalize features
        logger.info("Step 5: Normalizing features")
        self.normalize_features()

        # Step 6: Generate predictions (optional)
        detection_result = None
        if run_detection:
            if use_prediction:
                logger.info("Step 6: Generating predictions")
                self.generate_predictions(
                    horizon=5
                )  # Match the horizon value in detect_change_points
            else:
                logger.info("Step 6: Skipping prediction (not enabled)")

            # Step 7: Detect change points
            logger.info(
                f"Step 7: Detecting change points (use_prediction={use_prediction})"
            )
            detection_result = self.detect_change_points(use_prediction=use_prediction)

            # Step 8: Visualize results - skip if methods aren't found
            try:
                logger.info("Step 8: Visualizing detection results")
                if hasattr(self, "visualize_detection_results"):
                    self.visualize_detection_results(
                        summary, use_prediction=use_prediction
                    )
                else:
                    logger.warning(
                        "Visualization method not found, skipping visualization"
                    )
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}")
                logger.info("Continuing without visualization")

        logger.info("MIT Reality Mining processing complete")
        return self.daily_graphs, summary, detection_result

    def visualize_detection_results(self, summary_df, use_prediction=True):
        """
        Visualize the detection results.

        Args:
            summary_df: DataFrame with network summary statistics
            use_prediction: Whether prediction-enhanced detection was used
        """
        if not self.output_dir or not self.detection_results:
            logger.warning("No output directory specified or no detection results")
            return

        logger.info("Visualizing detection results")

        # Get detected change points
        if use_prediction and "horizon_dates" in self.detection_results:
            detected_dates = self.detection_results["horizon_dates"]
            method_name = "Horizon Martingale"
        elif "traditional_dates" in self.detection_results:
            detected_dates = self.detection_results["traditional_dates"]
            method_name = "Traditional Martingale"
        else:
            logger.warning("No detection results to visualize")
            return

        # Create plot of network metrics over time with detected changes
        plt.figure(figsize=(15, 15))

        # Convert dates to datetime for better plotting
        date_objects = pd.to_datetime(summary_df["date"])

        # Plot density over time
        plt.subplot(3, 1, 1)
        plt.plot(date_objects, summary_df["density"], "b-")
        plt.title(f"Network Density Over Time with {method_name} Detection")

        # Mark known events
        for event_date in self.events:
            if event_date in summary_df["date"].values:
                plt.axvline(
                    x=pd.to_datetime(event_date),
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                    label=(
                        self.events[event_date]
                        if event_date == list(self.events.keys())[0]
                        else ""
                    ),
                )

        # Mark detected change points
        for cp_date in detected_dates:
            if cp_date in summary_df["date"].values:
                plt.axvline(
                    x=pd.to_datetime(cp_date),
                    color="r",
                    linestyle="-",
                    label="Detected Change" if cp_date == detected_dates[0] else "",
                )

        plt.ylabel("Density")
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")

        # Plot mean degree over time
        plt.subplot(3, 1, 2)
        plt.plot(date_objects, summary_df["mean_degree"], "b-")
        plt.title("Mean Degree Over Time")

        # Mark known events
        for event_date in self.events:
            if event_date in summary_df["date"].values:
                plt.axvline(
                    x=pd.to_datetime(event_date), color="g", linestyle="--", alpha=0.5
                )

        # Mark detected change points
        for cp_date in detected_dates:
            if cp_date in summary_df["date"].values:
                plt.axvline(x=pd.to_datetime(cp_date), color="r", linestyle="-")

        plt.ylabel("Mean Degree")
        plt.xticks(rotation=45)

        # Plot clustering coefficient over time
        plt.subplot(3, 1, 3)
        plt.plot(date_objects, summary_df["mean_clustering"], "b-")
        plt.title("Clustering Coefficient Over Time")

        # Mark known events
        for event_date in self.events:
            if event_date in summary_df["date"].values:
                plt.axvline(
                    x=pd.to_datetime(event_date), color="g", linestyle="--", alpha=0.5
                )

        # Mark detected change points
        for cp_date in detected_dates:
            if cp_date in summary_df["date"].values:
                plt.axvline(x=pd.to_datetime(cp_date), color="r", linestyle="-")

        plt.ylabel("Clustering")
        plt.xlabel("Date")
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save visualization
        viz_path = os.path.join(
            self.output_dir,
            f"detection_visualization_{'horizon' if use_prediction else 'traditional'}.png",
        )
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved detection visualization to {viz_path}")

        # Create a more detailed visualization of the martingale scores if available
        if "martingale_scores" in self.detection_results:
            self.visualize_martingale_scores(summary_df, use_prediction)

    def visualize_martingale_scores(self, summary_df, use_prediction=True):
        """
        Visualize the martingale scores over time.

        Args:
            summary_df: DataFrame with network summary statistics
            use_prediction: Whether prediction-enhanced detection was used
        """
        if "martingale_scores" not in self.detection_results:
            logger.warning("No martingale scores available to visualize")
            return

        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Get scores
        scores = self.detection_results["martingale_scores"]

        # Create plot
        plt.figure(figsize=(15, 10))

        # Convert dates to datetime for better plotting
        dates = self.dates[: len(scores)]
        date_objects = pd.to_datetime(dates)

        # Plot martingale scores
        plt.subplot(2, 1, 1)
        plt.plot(date_objects, scores, "b-", label="Martingale Score")
        plt.axhline(
            y=60.0, color="r", linestyle="--", label="Detection Threshold (60.0)"
        )
        plt.title(
            f"{'Horizon' if use_prediction else 'Traditional'} Martingale Scores Over Time"
        )

        # Mark known events
        for event_date in self.events:
            if event_date in dates:
                event_idx = dates.index(event_date)
                plt.axvline(
                    x=pd.to_datetime(event_date),
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                    label=(
                        "Known Event"
                        if event_date == list(self.events.keys())[0]
                        else ""
                    ),
                )

        # Mark detected change points
        if use_prediction and "horizon_change_points" in self.detection_results:
            cp_indices = self.detection_results["horizon_change_points"]
            for idx in cp_indices:
                if idx < len(dates):
                    plt.axvline(
                        x=pd.to_datetime(dates[idx]),
                        color="r",
                        linestyle="-",
                        label="Detected Change" if idx == cp_indices[0] else "",
                    )
        elif "traditional_change_points" in self.detection_results:
            cp_indices = self.detection_results["traditional_change_points"]
            for idx in cp_indices:
                if idx < len(dates):
                    plt.axvline(
                        x=pd.to_datetime(dates[idx]),
                        color="r",
                        linestyle="-",
                        label="Detected Change" if idx == cp_indices[0] else "",
                    )

        plt.ylabel("Martingale Score")
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")

        # Plot p-values if available
        if "p_values" in self.detection_results:
            plt.subplot(2, 1, 2)
            p_values = self.detection_results["p_values"]
            p_dates = self.dates[: len(p_values)]
            p_date_objects = pd.to_datetime(p_dates)

            plt.plot(p_date_objects, p_values, "g-", label="p-value")
            plt.title("p-values Over Time")
            plt.ylabel("p-value")
            plt.xlabel("Date")
            plt.xticks(rotation=45)
            plt.legend(loc="upper right")

        plt.tight_layout()

        # Save visualization
        viz_path = os.path.join(
            viz_dir,
            f"martingale_scores_{'horizon' if use_prediction else 'traditional'}.png",
        )
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved martingale scores visualization to {viz_path}")


def main():
    parser = argparse.ArgumentParser(description="Process MIT Reality Mining dataset")
    # Basic parameters
    parser.add_argument(
        "--data",
        type=str,
        default="archive/data/Proximity.csv",
        help="Path to Proximity.csv file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Proximity threshold for creating edges",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reality_mining",
        help="Directory to save processed data",
    )

    # Processing options
    parser.add_argument(
        "--detection", action="store_true", help="Run change point detection"
    )
    parser.add_argument(
        "--prediction", action="store_true", help="Use prediction-based detection"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run comparison with ground truth events"
    )

    # Modular operation support
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=[
            "all",
            "preprocess",
            "create_graphs",
            "extract_features",
            "detect_changes",
            "visualize",
            "analyze",
        ],
        help="Specific action to perform",
    )

    args = parser.parse_args()

    # Initialize the processor
    processor = MITRealityProcessor(
        data_path=args.data, proximity_threshold=args.threshold, output_dir=args.output
    )

    # Handle modular actions
    if args.action == "preprocess":
        logging.info("Action: Preprocessing data only")
        processor.data = processor.load_data()

    elif args.action == "create_graphs":
        logging.info("Action: Creating graph snapshots")
        processor.data = processor.load_data()
        processor.daily_graphs = processor.create_daily_graphs()
        processor.save_graphs()

    elif args.action == "extract_features":
        logging.info("Action: Extracting graph features")
        processor.data = processor.load_data()
        processor.daily_graphs = processor.create_daily_graphs()
        processor.features = processor.extract_features()

    elif args.action == "detect_changes":
        logging.info("Action: Detecting changes")
        processor.data = processor.load_data()
        processor.daily_graphs = processor.create_daily_graphs()
        processor.features = processor.extract_features()
        processor.detect_change_points(use_prediction=args.prediction)

    elif args.action == "visualize":
        logging.info("Action: Visualizing results")
        processor.data = processor.load_data()
        processor.daily_graphs = processor.create_daily_graphs()
        processor.features = processor.extract_features()
        summary = processor.generate_event_summary()
        processor.detect_change_points(use_prediction=args.prediction)
        processor.visualize_detection_results(summary, use_prediction=args.prediction)

    elif args.action == "analyze":
        logging.info("Action: Analyzing results")
        processor.data = processor.load_data()
        processor.daily_graphs = processor.create_daily_graphs()
        summary = processor.generate_event_summary()
        if args.compare:
            processor.compare_with_ground_truth(summary)

    else:  # "all" or any other value
        logging.info("Action: Running complete pipeline")
        # Run the complete processing pipeline
        daily_graphs, summary, detection_result = processor.process(
            run_detection=args.detection or args.compare, use_prediction=args.prediction
        )

        # Compare with ground truth if requested
        if args.compare:
            processor.compare_with_ground_truth(summary)


if __name__ == "__main__":
    main()
