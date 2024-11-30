# tests/predict_changepoints.py

import sys
import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.graph_generator import GraphGenerator
from src.graph.features import extract_centralities, compute_embeddings
from src.models.spatiotemporal import SpatioTemporalPredictor, STModelConfig
from src.changepoint.detector import ChangePointDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangePointPredictor:
    """Predicts change points before they are detected by martingale framework."""

    def __init__(
        self,
        window_size: int = 10,
        forecast_horizon: int = 5,
        threshold: float = 30.0,
        epsilon: float = 0.8,
        hidden_dim: int = 64,
    ):
        self.config = STModelConfig(
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
        )
        self.threshold = threshold
        self.epsilon = epsilon
        self.detector = ChangePointDetector()
        self.model = None
        self.num_features = None  # Will be set during training
        self.num_nodes = None  # Will be set during training

    def prepare_training_data(
        self, graphs: List[np.ndarray], martingales: Dict[str, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from historical graphs and martingales."""

        # Extract features
        centralities = extract_centralities(graphs)
        embeddings = compute_embeddings(graphs, method="svd", n_components=2)

        # Combine features
        features = []
        n_nodes = graphs[0].shape[0]

        for i in range(len(graphs)):
            # Collect features for each node
            node_features = []
            for cent_type in centralities.values():
                # Reshape centrality to [n_nodes, 1]
                cent_values = np.array(cent_type[i]).reshape(n_nodes, 1)
                node_features.append(cent_values)

            # Add embeddings [n_nodes, embedding_dim]
            node_features.append(embeddings[i].reshape(n_nodes, -1))

            # Concatenate all features for this timestep
            timestep_features = np.hstack(node_features)  # [n_nodes, total_features]
            features.append(timestep_features)

        # Create sliding windows
        X, y = [], []
        if martingales:
            # Convert martingales to numpy arrays and stack them
            martingale_arrays = []
            for m in martingales.values():
                # Ensure each martingale is a 1D array
                m_array = np.array(
                    [
                        float(x.item()) if isinstance(x, np.ndarray) else float(x)
                        for x in m
                    ]
                )
                martingale_arrays.append(m_array)

            # Stack martingales into [n_timesteps, n_features]
            stacked_martingales = np.stack(martingale_arrays, axis=1)

            for i in range(
                len(features)
                - self.config.window_size
                - self.config.forecast_horizon
                + 1
            ):
                # Input window [window_size, n_nodes, features]
                feature_window = features[i : i + self.config.window_size]
                X.append(feature_window)

                # Target: future martingale values [forecast_horizon, n_features]
                target_window = stacked_martingales[
                    i
                    + self.config.window_size : i
                    + self.config.window_size
                    + self.config.forecast_horizon
                ]
                # Expand target to match model output dimensions
                target_expanded = np.tile(
                    target_window[:, np.newaxis, :], (1, n_nodes, 1)
                )
                y.append(target_expanded)

        # Convert to tensors with proper dimensions
        X = torch.FloatTensor(np.array(X))  # [batch, window_size, n_nodes, features]

        if y:
            y = torch.FloatTensor(
                np.array(y)
            )  # [batch, forecast_horizon, n_nodes, n_features]
        else:
            y = torch.FloatTensor()

        return X, y

    def train(
        self, graphs: List[np.ndarray], true_change_point: int, n_epochs: int = 100
    ):
        """Train the prediction model on historical data."""

        # Initialize detector and compute martingales
        self.detector.initialize(graphs[:true_change_point])
        centralities = self.detector.extract_features()

        martingales = {}
        for name, values in centralities.items():
            result = self.detector.martingale_test(
                data=values, threshold=self.threshold, epsilon=self.epsilon, reset=False
            )
            martingales[name] = np.array(result["martingales"])

        # Prepare training data
        X, y = self.prepare_training_data(graphs[:true_change_point], martingales)

        # Set dimensions
        self.num_nodes = graphs[0].shape[0]
        self.num_features = X.shape[-1]  # Last dimension contains features per node

        # Initialize model
        self.model = SpatioTemporalPredictor(
            config=self.config, num_nodes=self.num_nodes, num_features=self.num_features
        )

        # Normalize adjacency matrix
        adj_matrix = torch.FloatTensor(graphs[0])
        adj_matrix = adj_matrix + torch.eye(self.num_nodes)  # Add self-loops
        d = adj_matrix.sum(1)
        d_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(d_inv_sqrt, adj_matrix), d_inv_sqrt)

        # Train model
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = self.model(X, adj_norm)

            # Average predictions across nodes before computing loss
            pred_avg = pred.mean(dim=2)  # [batch, forecast_horizon, n_features]
            y_avg = y.mean(dim=2)  # [batch, forecast_horizon, n_features]

            loss = criterion(pred_avg, y_avg)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

    def predict(
        self, current_graphs: List[np.ndarray], current_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Predict future martingale values and potential change points."""

        self.model.eval()
        with torch.no_grad():
            # Normalize adjacency matrix
            adj_matrix = torch.FloatTensor(current_graphs[0])
            adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
            d = adj_matrix.sum(1)
            d_inv_sqrt = torch.diag(torch.pow(d, -0.5))
            adj_norm = torch.mm(torch.mm(d_inv_sqrt, adj_matrix), d_inv_sqrt)

            # Ensure correct input dimensions
            if current_features.size(0) == 0:
                # If empty tensor, create new input
                current_features = torch.zeros(
                    1,  # batch_size
                    self.config.window_size,  # seq_len
                    self.num_nodes,  # num_nodes
                    self.num_features,  # num_features
                )
            else:
                # Reshape existing tensor if needed
                if len(current_features.shape) < 4:
                    current_features = current_features.reshape(
                        1,  # batch_size
                        self.config.window_size,  # seq_len
                        self.num_nodes,  # num_nodes
                        self.num_features,  # num_features
                    )

            # Make predictions
            predictions = self.model(current_features, adj_norm)

            # Reshape predictions if needed
            pred_martingales = predictions.numpy()
            if len(pred_martingales.shape) == 4:  # [batch, steps, nodes, features]
                # Average over nodes if needed
                pred_martingales = pred_martingales.mean(axis=2)

            # Check when predictions exceed threshold
            threshold_exceeded = pred_martingales > self.threshold

            # Find earliest prediction of change
            detection_times = []
            confidence_scores = []

            for t in range(pred_martingales.shape[1]):
                # Check if any feature's martingale exceeds threshold
                if np.any(threshold_exceeded[:, t]):
                    detection_times.append(t)
                    # Confidence based on how many features predict change
                    confidence = np.mean(threshold_exceeded[:, t])
                    confidence_scores.append(confidence)

            result = {
                "predicted_martingales": pred_martingales,
                "detection_times": detection_times,
                "confidence_scores": confidence_scores,
                "earliest_detection": min(detection_times) if detection_times else None,
                "max_confidence": max(confidence_scores) if confidence_scores else 0.0,
            }

            return result


def main():
    # Generate synthetic data
    generator = GraphGenerator()
    graphs = generator.barabasi_albert(
        n=20,  # 20 nodes
        m1=2,  # Initial parameter
        m2=4,  # Parameter after change
        set1=50,  # Change point at t=50
        set2=50,  # 50 more graphs after change
    )

    # Initialize predictor
    predictor = ChangePointPredictor(
        window_size=10,
        forecast_horizon=5,
        threshold=20.0,  # Lower threshold for more sensitive detection
        epsilon=0.8,
    )

    # Train on first 40 graphs (before change)
    predictor.train(graphs, true_change_point=40)

    # Track predictions and actual changes
    predictions_log = []

    # Make predictions using sliding window
    for t in range(40, len(graphs) - predictor.config.window_size):
        # Get current window of graphs
        current_window = graphs[t : t + predictor.config.window_size]

        # Extract features directly
        centralities = extract_centralities(current_window)
        embeddings = compute_embeddings(current_window, method="svd", n_components=2)

        # Combine features
        features = []
        n_nodes = current_window[0].shape[0]

        for i in range(len(current_window)):
            node_features = []
            for cent_type in centralities.values():
                cent_values = np.array(cent_type[i]).reshape(n_nodes, 1)
                node_features.append(cent_values)
            node_features.append(embeddings[i].reshape(n_nodes, -1))
            timestep_features = np.hstack(node_features)
            features.append(timestep_features)

        # Convert to tensor with correct shape
        features = torch.FloatTensor(np.array(features))
        features = features.unsqueeze(0)

        # Make predictions
        predictions = predictor.predict(current_window, features)

        # Log prediction details
        pred_info = {
            "timestep": t,
            "martingale_values": predictions["predicted_martingales"].mean(axis=-1),
            "max_martingale": predictions["predicted_martingales"].max(),
            "detection_times": predictions["detection_times"],
            "confidence": predictions["max_confidence"],
        }
        predictions_log.append(pred_info)

        # Print prediction details
        logger.info(f"\nTimestep t={t}:")
        logger.info(f"Current window: {t} to {t + predictor.config.window_size}")
        logger.info(f"Max predicted martingale: {pred_info['max_martingale']:.4f}")

        if predictions["earliest_detection"] is not None:
            logger.info(
                f"Change predicted in {predictions['earliest_detection']} steps"
            )
            logger.info(f"Confidence: {predictions['max_confidence']:.2f}")
            logger.info(
                f"Predicted martingale sequence: {pred_info['martingale_values'].flatten()}"
            )

        # Early warning if martingale values are increasing
        if len(predictions_log) >= 2:
            prev_max = predictions_log[-2]["max_martingale"]
            curr_max = pred_info["max_martingale"]
            if curr_max > prev_max * 1.5:  # 50% increase
                logger.warning(f"Significant increase in martingale values detected!")
                logger.warning(
                    f"Previous max: {prev_max:.4f}, Current max: {curr_max:.4f}"
                )

    # Analyze prediction performance
    true_change_point = 50
    earliest_warning = None
    detection_delay = None

    for pred in predictions_log:
        if pred["max_martingale"] > predictor.threshold:
            detection_time = pred["timestep"]
            if earliest_warning is None:
                earliest_warning = detection_time - true_change_point
                detection_delay = max(0, detection_time - true_change_point)
                break

    logger.info("\nPrediction Performance Summary:")
    logger.info(f"True change point: t={true_change_point}")
    if earliest_warning is not None:
        if earliest_warning < 0:
            logger.info(f"Early warning: {-earliest_warning} steps before change")
        else:
            logger.info(f"Detection delay: {detection_delay} steps after change")
    else:
        logger.info("No change detected")


if __name__ == "__main__":
    main()
