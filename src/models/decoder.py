# src/models/decoder.py

"""Decoder for multi-step prediction."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemporalDecoder(nn.Module):
    """LSTM-based decoder for multi-step prediction."""

    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        num_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: LSTM hidden dimension
            num_features: Number of features per node
            num_nodes: Number of nodes in the graph
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_nodes = num_nodes

        # LSTM processes the entire graph state
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Project LSTM output to node features
        self.fc_out = nn.Linear(hidden_dim, num_nodes * num_features)

    def forward(
        self,
        x: torch.Tensor,
        steps: int,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded features [batch_size, seq_len, hidden_dim]
            steps: Number of future steps to predict
            hidden: Optional initial hidden state

        Returns:
            Predictions [batch_size, steps, num_nodes, num_features]
        """
        batch_size = x.size(0)

        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)

        # Generate predictions
        predictions = []
        h_t = lstm_out[:, -1:]  # Use last hidden state [batch_size, 1, hidden_dim]

        # Autoregressive prediction
        for _ in range(steps):
            # Project to node features
            pred = self.fc_out(h_t)  # [batch_size, 1, num_nodes * num_features]
            pred = pred.reshape(batch_size, 1, self.num_nodes, self.num_features)
            predictions.append(pred)

            # Feed prediction back as input
            h_t, hidden = self.lstm(h_t, hidden)

        # Stack predictions [batch_size, steps, num_nodes, num_features]
        return torch.cat(predictions, dim=1)
