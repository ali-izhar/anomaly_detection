#!/usr/bin/env python
"""MIT Reality Dataset Processor

The dataset contains Bluetooth proximity data from 100 MIT students and faculty
collected over the 2007-2008 academic year.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.changepoint import ChangePointDetector, DetectorConfig, BettingFunctionConfig
from src.graph import NetworkFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MITRealityProcessor:
    """Process the MIT Reality Mining dataset into graph snapshots and detect changes."""

    def __init__(self, data_path, proximity_threshold=0.3, output_dir=None):
        """
        Initialize the processor.

        Args:
            data_path: Path to the Proximity.csv file
            proximity_threshold: Threshold for creating edges (default: 0.3)
            output_dir: Directory to save processed data (default: None)
        """
        self.data_path = data_path
        self.proximity_threshold = proximity_threshold
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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
        self.daily_graphs = {}
        self.adj_matrices = []
        self.features = None
        self.dates = None

        # Evaluation metrics storage
        self.detection_results = None
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

        # Define which features to extract
        feature_names = [
            "n_nodes",
            "n_edges",
            "density",
            "diameter",
            "avg_clustering",
            "avg_degree",
            "avg_shortest_path",
            "degree_centrality",
            "eigenvector_centrality",
            "betweenness_centrality",
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

    def detect_change_points(self, use_prediction=True):
        """
        Detect change points in the graph sequence.

        Args:
            use_prediction: whether to use prediction-enhanced detection

        Returns:
            Dict containing detection results
        """
        if self.features is None:
            logger.error("No features extracted. Call extract_features() first")
            raise ValueError("No features extracted. Call extract_features() first")

        logger.info("Detecting change points in graph sequence")

        # Normalize features
        features_mean = np.mean(self.features, axis=0)
        features_std = np.std(self.features, axis=0)
        # Avoid division by zero
        features_std[features_std == 0] = 1.0
        features_normalized = (self.features - features_mean) / features_std

        # Check for NaN values after normalization
        if np.isnan(features_normalized).any():
            logger.warning("Found NaN values after normalization, replacing with zeros")
            features_normalized = np.nan_to_num(features_normalized, nan=0.0)

        # Configure detection parameters - use more sensitive parameters
        betting_func_config = BettingFunctionConfig(
            name="mixture", params={"epsilons": [0.5, 0.7, 0.9]}, random_seed=42
        )

        # Make detector extremely sensitive for the MIT Reality dataset
        detector_config = DetectorConfig(
            method="horizon" if use_prediction else "traditional",
            threshold=10,  # Very low threshold for maximum sensitivity
            history_size=5,  # Smaller history for quicker response to changes
            batch_size=1,
            reset=True,
            max_window=25,  # Smaller window for more sensitivity
            betting_func_config=betting_func_config,
            distance_measure="manhattan",  # Try manhattan distance which can be more sensitive
            random_state=42,
        )

        # Initialize and run detector
        detector = ChangePointDetector(detector_config)

        # If using prediction, prepare predicted data with correct shape
        predicted_data = None
        if use_prediction:
            # The framework expects predictions with shape (n_pred, horizon, n_features)
            # where n_pred is the number of timesteps, horizon is the prediction horizon
            # and n_features is the number of features

            horizon = 10  # Using horizon=10 as recommended in the paper
            n_features = features_normalized.shape[1]
            n_pred = features_normalized.shape[0]

            # Initialize array with correct shape
            predicted_data = np.zeros((n_pred, horizon, n_features))

            # For each timestep, generate predictions for next 'horizon' steps
            # Use ARIMA-style forecasting approach based on reality_mining_data.py
            for t in range(n_pred):
                # Start with last known value
                last_known = features_normalized[t : t + 1]

                # For each horizon step
                for h in range(horizon):
                    # Simple prediction using autoregressive approach
                    history_start = max(0, t - 4 + h)
                    history_end = t + 1 + h
                    if history_start < history_end:
                        # Use weighted average favoring recent observations
                        history_values = features_normalized[history_start:history_end]
                        # Create non-zero weights that increase exponentially for more recent values
                        weights = np.exp(np.linspace(0, 1, len(history_values)))
                        # Ensure weights aren't all zeros
                        if np.sum(weights) == 0:
                            # If weights sum to zero, use uniform weights
                            pred = np.mean(history_values, axis=0)
                        else:
                            pred = np.average(history_values, axis=0, weights=weights)
                    else:
                        # Fallback if window is invalid
                        pred = last_known[0]

                    # Store prediction
                    predicted_data[t, h] = pred

            # Check for NaN values in predictions and replace them
            if np.isnan(predicted_data).any():
                logger.warning("Found NaN values in predictions, replacing with zeros")
                predicted_data = np.nan_to_num(predicted_data, nan=0.0)

            logger.info(f"Generated predictions with shape: {predicted_data.shape}")

        # Run detection
        try:
            detection_result = detector.run(
                data=features_normalized,
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
                    threshold=10,  # Very low threshold for maximum sensitivity
                    history_size=5,  # Smaller history for quicker response to changes
                    batch_size=1,
                    reset=True,
                    max_window=25,  # Smaller window for more sensitivity
                    betting_func_config=betting_func_config,
                    distance_measure="manhattan",  # Try manhattan distance which can be more sensitive
                    random_state=42,
                )
                detector = ChangePointDetector(trad_config)
                detection_result = detector.run(
                    data=features_normalized, predicted_data=None, reset_state=True
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

        # Calculate evaluation metrics similar to rm_evaluator.py
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
        detection_result["metrics"] = {
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

        # Save detection results if output directory is specified
        if self.output_dir:
            results_df = pd.DataFrame(
                {
                    "date": self.dates,
                    "is_true_event": [d in true_event_dates for d in self.dates],
                    "detected_change": [d in detected_dates for d in self.dates],
                }
            )

            results_path = os.path.join(
                self.output_dir,
                f"detection_results_{'horizon' if use_prediction else 'traditional'}.csv",
            )
            results_df.to_csv(results_path, index=False)

            # Save metrics
            metrics_path = os.path.join(
                self.output_dir,
                f"detection_metrics_{'horizon' if use_prediction else 'traditional'}.csv",
            )
            pd.DataFrame([detection_result["metrics"]]).to_csv(
                metrics_path, index=False
            )

            logger.info(f"Saved detection results to {self.output_dir}")

        self.detection_results = detection_result
        self.evaluation_metrics = detection_result["metrics"]
        return detection_result

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
        plt.plot(date_objects, summary_df["clustering"], "b-")
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

        # Create a more detailed visualization focused on specific event periods
        self.visualize_event_periods(summary_df, detected_dates, use_prediction)

    def visualize_event_periods(self, summary_df, detected_dates, use_prediction=True):
        """
        Create focused visualizations around specific event periods.

        Args:
            summary_df: DataFrame with network summary statistics
            detected_dates: List of detected change dates
            use_prediction: Whether prediction-enhanced detection was used
        """
        if not self.output_dir:
            return

        # Define key event periods to analyze
        event_periods = {
            "NewYear_2008": {
                "start": "2007-12-15",
                "end": "2008-01-15",
                "title": "New Year 2008 Period",
            },
            "NewYear_2009": {
                "start": "2008-12-15",
                "end": "2009-01-15",
                "title": "New Year 2009 Period",
            },
            "Thanksgiving_2008": {
                "start": "2008-11-20",
                "end": "2008-12-04",
                "title": "Thanksgiving 2008 Period",
            },
        }

        for period_name, period_info in event_periods.items():
            # Filter data for this period
            period_start = pd.to_datetime(period_info["start"])
            period_end = pd.to_datetime(period_info["end"])

            # Convert summary_df dates to datetime
            summary_df["datetime"] = pd.to_datetime(summary_df["date"])

            # Filter summary for this period
            period_df = summary_df[
                (summary_df["datetime"] >= period_start)
                & (summary_df["datetime"] <= period_end)
            ]

            if len(period_df) == 0:
                logger.warning(
                    f"No data for period {period_name}, skipping visualization"
                )
                continue

            # Create visualization for this period
            plt.figure(figsize=(12, 10))

            # Plot network metrics
            metrics = ["density", "mean_degree", "clustering"]
            for i, metric in enumerate(metrics):
                plt.subplot(3, 1, i + 1)
                plt.plot(period_df["datetime"], period_df[metric], "b-o")
                plt.title(f"{metric.capitalize()} during {period_info['title']}")

                # Mark known events in this period
                for event_date, event_name in self.events.items():
                    event_dt = pd.to_datetime(event_date)
                    if period_start <= event_dt <= period_end:
                        if event_date in period_df["date"].values:
                            plt.axvline(
                                x=event_dt,
                                color="g",
                                linestyle="--",
                                alpha=0.7,
                                label=event_name,
                            )
                            # Add annotation
                            plt.annotate(
                                event_name,
                                xy=(event_dt, period_df[metric].max() * 0.9),
                                xytext=(event_dt, period_df[metric].max()),
                                rotation=90,
                                fontsize=9,
                                ha="right",
                            )

                # Mark detected changes in this period
                for cp_date in detected_dates:
                    cp_dt = pd.to_datetime(cp_date)
                    if period_start <= cp_dt <= period_end:
                        if cp_date in period_df["date"].values:
                            plt.axvline(
                                x=cp_dt,
                                color="r",
                                linestyle="-",
                                label="Detected Change",
                            )

                plt.ylabel(metric.capitalize())
                plt.grid(True, alpha=0.3)
                if i == 0:
                    plt.legend()

            plt.xlabel("Date")
            plt.tight_layout()

            # Save this period visualization
            period_path = os.path.join(
                self.output_dir,
                f"{period_name}_{'horizon' if use_prediction else 'traditional'}.png",
            )
            plt.savefig(period_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved {period_name} visualization to {period_path}")

    def process(self, run_detection=True, use_prediction=True):
        """
        Run the complete processing pipeline.

        Args:
            run_detection: Whether to run change point detection
            use_prediction: Whether to use prediction-enhanced detection

        Returns:
            Tuple of (daily_graphs, summary_df, detection_result)
        """
        # Load and preprocess data
        df = self.load_data()

        # Create daily graph snapshots
        daily_graphs = self.create_daily_graphs()

        # Save graphs if output directory is specified
        self.save_graphs()

        # Generate event summary
        summary_df = self.generate_event_summary()

        detection_result = None
        if run_detection:
            # Extract features for change point detection
            self.extract_features()

            # Run change point detection
            detection_result = self.detect_change_points(use_prediction=use_prediction)

            # Visualize results
            self.visualize_detection_results(summary_df, use_prediction=use_prediction)

        return daily_graphs, summary_df, detection_result


def main():
    """Main function to run the MIT Reality Mining dataset processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Process MIT Reality Mining dataset")
    parser.add_argument(
        "--data-path",
        "-f",
        type=str,
        default="archive/data/Proximity.csv",
        help="Path to Proximity.csv file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.3,
        help="Proximity threshold for edge creation (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results/reality_mining",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--detection",
        "-d",
        action="store_true",
        help="Run change point detection",
    )
    parser.add_argument(
        "--prediction",
        "-p",
        action="store_true",
        help="Use prediction-enhanced detection",
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare traditional and horizon detectors",
    )

    args = parser.parse_args()

    # Create processor and run
    processor = MITRealityProcessor(
        data_path=args.data_path,
        proximity_threshold=args.threshold,
        output_dir=args.output_dir,
    )

    daily_graphs, summary, detection_result = processor.process(
        run_detection=args.detection, use_prediction=args.prediction
    )

    # Compare traditional and horizon detectors if requested
    if args.compare and args.detection:
        # Run traditional detection
        processor.extract_features()
        trad_result = processor.detect_change_points(use_prediction=False)
        trad_metrics = trad_result["metrics"]

        # Run horizon detection
        horizon_result = processor.detect_change_points(use_prediction=True)
        horizon_metrics = horizon_result["metrics"]

        # Create comparison table
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
                    trad_metrics["true_positives"],
                    trad_metrics["false_positives"],
                    trad_metrics["false_negatives"],
                    trad_metrics["precision"],
                    trad_metrics["recall"],
                    trad_metrics["f1_score"],
                    trad_metrics["fpr"],
                    trad_metrics["mean_delay"],
                ],
                "Horizon": [
                    horizon_metrics["true_positives"],
                    horizon_metrics["false_positives"],
                    horizon_metrics["false_negatives"],
                    horizon_metrics["precision"],
                    horizon_metrics["recall"],
                    horizon_metrics["f1_score"],
                    horizon_metrics["fpr"],
                    horizon_metrics["mean_delay"],
                ],
            }
        )

        # Save comparison
        if args.output_dir:
            comparison.to_csv(
                os.path.join(args.output_dir, "detector_comparison.csv"), index=False
            )

        # Print comparison table
        print("\nComparison of Traditional vs Horizon Detection:")
        print(comparison)

        # Calculate delay reduction if applicable
        if trad_metrics["mean_delay"] > 0 and horizon_metrics["mean_delay"] > 0:
            delay_reduction = (
                (trad_metrics["mean_delay"] - horizon_metrics["mean_delay"])
                / trad_metrics["mean_delay"]
                * 100
            )
            print(f"\nDelay Reduction: {delay_reduction:.2f}%")

    # Print basic summary
    print(f"Processed {len(daily_graphs)} daily snapshots")
    print(f"Date range: {summary['date'].min()} to {summary['date'].max()}")
    print(f"Found {sum(summary['is_event'])} event days")

    # Print network stats
    print("\nNetwork Statistics:")
    print(f"Average nodes per day: {summary['nodes'].mean():.1f}")
    print(f"Average edges per day: {summary['edges'].mean():.1f}")
    print(f"Average density: {summary['density'].mean():.4f}")

    # Print detection results if available
    if detection_result:
        print("\nChange Point Detection Results:")
        method = "Horizon" if args.prediction else "Traditional"
        detected_dates = detection_result.get(
            f"{'horizon' if args.prediction else 'traditional'}_dates", []
        )
        metrics = detection_result.get("metrics", {})

        print(f"Detection method: {method} Martingale")
        print(f"Detected change points: {len(detected_dates)}")
        for date in detected_dates:
            event_match = ""
            for event_date, event_name in processor.events.items():
                days_diff = abs(
                    (
                        datetime.strptime(date, "%Y-%m-%d")
                        - datetime.strptime(event_date, "%Y-%m-%d")
                    ).days
                )
                if days_diff <= 3:
                    event_match = f" (near {event_name} on {event_date})"
                    break
            print(f"  - {date}{event_match}")

        print("\nDetection Metrics:")
        print(f"True Positives: {metrics.get('true_positives', 0)}")
        print(f"False Positives: {metrics.get('false_positives', 0)}")
        print(f"False Negatives: {metrics.get('false_negatives', 0)}")
        print(f"Precision: {metrics.get('precision', 0):.2f}")
        print(f"Recall (TPR): {metrics.get('recall', 0):.2f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.2f}")
        print(f"False Positive Rate: {metrics.get('fpr', 0):.4f}")
        print(f"Mean Detection Delay: {metrics.get('mean_delay', 0):.2f} days")

    return 0


if __name__ == "__main__":
    main()
