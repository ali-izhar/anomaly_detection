# src/models2/temporal.py

import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers=2, dropout=0.2, model_type="LSTM"
    ):
        super(TemporalModel, self).__init__()
        self.model_type = model_type
        if model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == "GRU":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == "Transformer":
            # For Transformer, you might need positional encoding and different handling
            self.transformer = nn.Transformer(
                d_model=hidden_dim, num_encoder_layers=num_layers, dropout=dropout
            )
            self.fc = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError("Unsupported temporal model type")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x (batch, seq_len, input_dim)
        Returns:
            (batch, hidden_dim)
        """
        if self.model_type in ["LSTM", "GRU"]:
            out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_dim)
            # Take the last time step
            out = out[:, -1, :]  # (batch, hidden_dim)
        elif self.model_type == "Transformer":
            # Transpose for transformer: (seq_len, batch, input_dim)
            x = self.fc(x).transpose(0, 1)
            out = self.transformer(x, x)  # Self-attention
            # Take the last time step
            out = out[-1, :, :]  # (batch, hidden_dim)
        return out
