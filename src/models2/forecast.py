# src/models2/forecast.py

import torch
import torch.nn as nn
from src.models2.gnn import GNNEncoder
from src.models2.temporal import TemporalModel


class ForecastingModel(nn.Module):
    def __init__(self, config, num_nodes=50, node_feat_dim=16):
        super(ForecastingModel, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.m_horizon = config["data"]["m_horizon"]

        # GNN Encoder
        gnn_config = config["model"]["gnn"]
        self.gnn = GNNEncoder(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            hidden_dim=gnn_config["hidden_dim"],
            num_layers=gnn_config["num_layers"],
            activation=gnn_config["activation"],
        )

        # Temporal Model
        temporal_config = config["model"]["temporal"]
        # Input dim: GNN hidden_dim + 6 global features
        temporal_input_dim = gnn_config["hidden_dim"] + config["data"]["num_features"]
        self.temporal = TemporalModel(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_config["hidden_dim"],
            num_layers=temporal_config["num_layers"],
            dropout=temporal_config["dropout"],
            model_type=temporal_config["type"],
        )

        # Prediction Head
        self.fc = nn.Sequential(
            nn.Linear(temporal_config["hidden_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, self.m_horizon * config["data"]["num_features"]),
        )

    def forward(self, adj_matrices, features):
        """
        Args:
            adj_matrices: (batch, T, N, N)
            features: (batch, T, 6)
        Returns:
            predictions: (batch, m_horizon, 6)
        """
        batch_size, T, N, _ = adj_matrices.size()
        # Initialize node features (e.g., learnable embeddings)
        # For simplicity, use fixed initial node features
        device = adj_matrices.device
        node_features = torch.ones(batch_size, N, self.node_feat_dim).to(device)

        graph_embeddings = []
        for t in range(T):
            A_t = adj_matrices[:, t, :, :]  # (batch, N, N)
            H_t = self.gnn(A_t, node_features)  # (batch, hidden_dim)
            graph_embeddings.append(H_t)

        graph_embeddings = torch.stack(
            graph_embeddings, dim=1
        )  # (batch, T, hidden_dim)

        # Concatenate with global features
        combined = torch.cat(
            [graph_embeddings, features], dim=-1
        )  # (batch, T, hidden_dim + 6)

        # Temporal Modeling
        temporal_out = self.temporal(combined)  # (batch, temporal_hidden_dim)

        # Prediction
        predictions = self.fc(temporal_out)  # (batch, m_horizon * 6)
        predictions = predictions.view(
            batch_size, self.m_horizon, -1
        )  # (batch, m_horizon, 6)

        return predictions
