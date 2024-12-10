# src/models2/temporal.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict
from torch import Tensor


class TemporalEncoder(nn.Module):
    """Temporal encoder for processing sequential data.

    Supports multiple architectures (LSTM, GRU, Transformer) for encoding
    temporal sequences into fixed-length representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        model_type: str = "LSTM",
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize temporal encoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNNs
            model_type: One of ['LSTM', 'GRU', 'Transformer']
        """
        super().__init__()
        if config is None:
            config = {}
        temporal_config = config.get("model", {}).get("temporal", {})
        hw_config = config.get("hardware", {})

        # Use config values
        self.hidden_dim = temporal_config.get("hidden_dim", hidden_dim)
        self.num_layers = temporal_config.get("num_layers", num_layers)
        self.dropout_rate = temporal_config.get("dropout", dropout)
        self.model_type = temporal_config.get("type", model_type)

        # Memory optimization
        self.use_checkpoint = hw_config.get("gradient_checkpointing", True)

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Add residual connections
        self.use_residual = True if num_layers > 1 else False

        if model_type == "LSTM":
            # Use variational dropout for better regularization
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )

            # Add attention mechanism
            if bidirectional:
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim * 2,  # *2 for bidirectional
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True,
                )
            else:
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
                )
        elif model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif model_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,  # Number of attention heads
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through the temporal encoder.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional mask for variable length sequences [batch_size, seq_len]

        Returns:
            If model_type is Transformer:
                encoded: Encoded sequence [batch_size, seq_len, hidden_dim]
            If model_type is LSTM/GRU:
                encoded: Encoded sequence [batch_size, seq_len, num_directions * hidden_dim]
                hidden: Final hidden state [num_layers * num_directions, batch_size, hidden_dim]
        """
        if self.model_type == "Transformer":
            # Project input to hidden dimension
            x = self.input_proj(x)

            # Create attention mask from padding mask
            if mask is not None:
                # Transformer expects mask to be float with -inf for padding
                attention_mask = (
                    mask.float()
                    .masked_fill(mask == 0, float("-inf"))
                    .masked_fill(mask == 1, float(0.0))
                )
                encoded = self.transformer(x, src_key_padding_mask=attention_mask)
            else:
                encoded = self.transformer(x)

            return encoded

        else:  # LSTM or GRU
            if mask is not None:
                # Pack padded sequence
                lengths = mask.sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )

            # Process with RNN
            output, hidden = self.rnn(x)

            if mask is not None:
                # Unpack sequence
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            return output, hidden


class TemporalDecoder(nn.Module):
    """Temporal decoder for generating sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "LSTM",
        bidirectional: bool = True,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize temporal decoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output features
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            model_type: One of ['LSTM', 'GRU']
        """
        super().__init__()

        if config is None:
            config = {}
        temporal_config = config.get("model", {}).get("temporal", {})
        hw_config = config.get("hardware", {})

        # Use config values with defaults
        self.model_type = temporal_config.get("type", model_type)
        self.hidden_dim = temporal_config.get("hidden_dim", hidden_dim)
        self.num_layers = temporal_config.get("num_layers", num_layers)
        self.dropout_rate = temporal_config.get("dropout", dropout)
        self.bidirectional = bidirectional

        # Memory optimization settings
        self.use_checkpoint = hw_config.get("gradient_checkpointing", True)

        rnn_hidden = self.hidden_dim // 2 if bidirectional else self.hidden_dim

        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=rnn_hidden,
                num_layers=self.num_layers,
                dropout=self.dropout_rate if self.num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=rnn_hidden,
                num_layers=self.num_layers,
                dropout=self.dropout_rate if self.num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Adjust output projection for bidirectional
        proj_input_dim = self.hidden_dim if bidirectional else self.hidden_dim
        self.output_proj = nn.Linear(proj_input_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Forward pass through the temporal decoder.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            hidden: Optional initial hidden state

        Returns:
            output: Decoded sequence [batch_size, seq_len, output_dim]
            hidden: Final hidden state
        """
        # Process with RNN
        output, hidden = self.rnn(x, hidden)

        # Project to output dimension
        output = self.output_proj(output)

        return output, hidden


class TemporalPredictor(nn.Module):
    """End-to-end model for temporal prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        forecast_horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "LSTM",
        bidirectional: bool = True,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output features
            forecast_horizon: Number of future steps to predict
            num_layers: Number of layers in encoder/decoder
            dropout: Dropout probability
            model_type: Type of temporal model to use
            bidirectional: Whether to use bidirectional RNNs
            config: Configuration dictionary
        """
        super().__init__()

        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            model_type=model_type,
            config=config,
        )

        self.decoder = TemporalDecoder(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type,
            bidirectional=bidirectional,
            config=config,
        )

        self.forecast_horizon = forecast_horizon

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass for prediction.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            mask: Optional mask for variable length sequences

        Returns:
            predictions: Predicted future values [batch_size, forecast_horizon, output_dim]
        """
        batch_size = x.size(0)

        # Encode input sequence
        if self.encoder.model_type == "Transformer":
            memory = self.encoder(x, mask)
            hidden = None
        else:
            _, hidden = self.encoder(x, mask)

        # Initialize decoder input
        decoder_input = torch.zeros(
            batch_size, 1, self.decoder.output_proj.out_features, device=x.device
        )

        # Generate predictions
        predictions = []
        for _ in range(self.forecast_horizon):
            output, hidden = self.decoder(decoder_input, hidden)
            predictions.append(output)
            decoder_input = output

        return torch.cat(predictions, dim=1)
