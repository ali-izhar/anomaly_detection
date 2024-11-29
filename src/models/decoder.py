# src/models/decoder.py

"""Decoder for multi-step prediction."""

import torch
import torch.nn as nn
from typing import List, Optional


class TemporalDecoder(nn.Module):
    """LSTM-based decoder for multi-step prediction.

    Takes encoded spatio-temporal features and generates
    predictions for future timesteps using:

    h_t = LSTM(h_t-1, x_t)
    y_t = W_out * h_t

    where h_t is the hidden state and y_t is the prediction.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: LSTM hidden dimension
            num_features: Number of features to predict
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_dim, num_features)

    def forward(
        self, x: torch.Tensor, steps: int, hidden: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded features [batch_size, seq_len, num_nodes * hidden_dim]
               Features include centralities and embeddings from features.py
            steps: Number of future steps to predict
            hidden: Optional initial hidden state

        Returns:
            Predictions [batch_size, steps, num_nodes, num_features]
        """
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)

        # Generate predictions
        predictions = []
        h_t = lstm_out[:, -1:]  # Use last hidden state

        # Autoregressive prediction
        for _ in range(steps):
            pred = self.fc_out(h_t)
            predictions.append(pred)
            h_t, hidden = self.lstm(pred, hidden)

        return torch.stack(predictions, dim=1)
