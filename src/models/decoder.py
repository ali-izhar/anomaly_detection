# src/models/decoder.py

"""Decoder for multi-step prediction."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemporalDecoder(nn.Module):
    """LSTM-based decoder for multi-step prediction.

    Given encoded temporal features, this decoder predicts future timesteps in an autoregressive fashion.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_features: int = 22,  # 4 centrality + 2 SVD + 16 LSVD
        num_nodes: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            hidden_dim: LSTM hidden dimension
            num_features: Total number of features to predict per node:
                - 4 centrality features
                - 2 SVD dimensions
                - 16 LSVD dimensions
            num_nodes: Number of nodes in graph (100 for this dataset)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_nodes = num_nodes

        # LSTM for temporal processing
        self.node_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Separate prediction heads for different feature types
        self.centrality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),  # 4 centrality features
        )

        self.svd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # 2D SVD
        )

        self.lsvd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),  # 16D LSVD
        )

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 10,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded features [batch_size, seq_len, num_nodes, hidden_dim]
            steps: Number of future steps to predict
            hidden: Optional LSTM hidden state

        Returns:
            Predictions [batch_size, steps, num_nodes, num_features]
            where num_features = 22 (4 centrality + 2 SVD + 16 LSVD)
        """
        batch_size = x.size(0)
        device = x.device
        predictions = []

        # Process each node independently
        for node_idx in range(self.num_nodes):
            node_feats = x[:, :, node_idx, :]  # [batch_size, seq_len, hidden_dim]

            # Initial LSTM processing
            lstm_out, node_hidden = self.node_lstm(node_feats, hidden)
            lstm_out = self.layernorm(lstm_out)
            lstm_out = self.dropout(lstm_out)

            # Start prediction from last state
            h_t = lstm_out[:, -1:]  # [batch_size, 1, hidden_dim]
            curr_hidden = node_hidden

            node_preds = []
            for _ in range(steps):
                # Generate predictions for each feature type
                centrality_pred = self.centrality_head(h_t)
                svd_pred = self.svd_head(h_t)
                lsvd_pred = self.lsvd_head(h_t)

                # Combine predictions
                combined_pred = torch.cat(
                    [centrality_pred, svd_pred, lsvd_pred], dim=-1
                )  # [batch_size, 1, 22]
                node_preds.append(combined_pred)

                # Update hidden state
                h_t, curr_hidden = self.node_lstm(h_t, curr_hidden)
                h_t = self.layernorm(h_t)
                h_t = self.dropout(h_t)

            # Stack time steps: [batch_size, steps, num_features]
            node_preds = torch.cat(node_preds, dim=1)
            predictions.append(node_preds)

        # Stack nodes: [batch_size, steps, num_nodes, num_features]
        predictions = torch.stack(predictions, dim=2)
        return predictions
