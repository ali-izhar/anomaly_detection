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
        num_features: int = 22,
        num_nodes: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            hidden_dim: LSTM hidden dimension
            num_features: Number of features per node
            num_nodes: Number of nodes in the graph
            num_layers: Number of LSTM layers
            dropout: Dropout rate for LSTM and intermediate layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_nodes = num_nodes
        
        # Process nodes independently to reduce memory
        self.node_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        
        # Smaller prediction head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )

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
        """
        batch_size = x.size(0)
        predictions = []
        
        # Process each node independently
        for node_idx in range(self.num_nodes):
            # Get node features: [B, T, H]
            node_feats = x[:, :, node_idx, :]
            
            # Process sequence
            lstm_out, node_hidden = self.node_lstm(node_feats, hidden)
            lstm_out = self.layernorm(lstm_out)
            lstm_out = self.dropout(lstm_out)
            
            # Start prediction from last state
            h_t = lstm_out[:, -1:]  # [B, 1, H]
            
            node_preds = []
            curr_hidden = node_hidden
            
            # Generate predictions for this node
            for _ in range(steps):
                # Project to features
                out = self.fc_out(h_t)  # [B, 1, F]
                node_preds.append(out)
                
                # Update hidden state
                h_t, curr_hidden = self.node_lstm(h_t, curr_hidden)
                h_t = self.layernorm(h_t)
                h_t = self.dropout(h_t)
            
            # Stack time steps: [B, steps, F]
            node_preds = torch.cat(node_preds, dim=1)
            predictions.append(node_preds)
        
        # Stack nodes: [B, steps, N, F]
        predictions = torch.stack(predictions, dim=2)
        return predictions
