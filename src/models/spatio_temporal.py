# src/models/spatio_temporal.py

"""Spatio-temporal model for soccer player movement data."""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import yaml

from src.models.graph_conv import CustomGCNLayer


class BaseTemporalGNN(nn.Module, ABC):
    """Abstract base class for temporal graph neural networks."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize base temporal GNN.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def _process_spatial(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process spatial relationships in the data."""
        pass

    @abstractmethod
    def _process_temporal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process temporal dependencies in the data."""
        pass

    @abstractmethod
    def _generate_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Generate final predictions from processed features."""
        pass


class SpatialProcessor(nn.Module):
    """Processes spatial relationships using GCN."""

    def __init__(
        self,
        num_node_features: int,
        gcn_out_features: int,
        model_name: str,
        config_path: Optional[Path] = None,
    ):
        """Initialize spatial processor.

        Args:
            num_node_features: Number of input features per node
            gcn_out_features: Output dimension of GCN
            model_name: GCN variant to use
            config_path: Path to config file
        """
        super().__init__()
        self.gcn = CustomGCNLayer(
            num_node_features,
            gcn_out_features,
            version=model_name,
            config_path=config_path,
        )
        self.num_node_features = num_node_features

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """Process single timestep through GCN."""
        return self.gcn(
            x=x.view(-1, self.num_node_features),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )


class TemporalProcessor(nn.Module):
    """Processes temporal dependencies using LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize temporal processor.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate (applied between LSTM layers)
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Process sequence through LSTM."""
        outputs, _ = self.lstm(x, hidden)
        return outputs[:, -1]  # Return final timestep

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )


class PredictionHead(nn.Module):
    """Generates predictions from processed features."""

    def __init__(
        self, input_size: int, num_nodes: int, num_node_features: int, n_steps_out: int
    ):
        """Initialize prediction head.

        Args:
            input_size: Input feature dimension
            num_nodes: Number of nodes in graph
            num_node_features: Number of features per node
            n_steps_out: Number of timesteps to predict
        """
        super().__init__()
        self.fc = nn.Linear(input_size, n_steps_out * num_nodes * num_node_features)
        self.n_steps_out = n_steps_out
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions from features."""
        predictions = self.fc(x)
        return predictions.view(
            -1, self.n_steps_out, self.num_nodes, self.num_node_features  # batch size
        )


class GCNLSTMModel(BaseTemporalGNN):
    """Combined GCN-LSTM model for spatio-temporal graph prediction.

    Architecture:
    1. GCN layer processes spatial relationships at each timestep
    2. LSTM processes the temporal evolution of GCN outputs
    3. Fully connected layer maps to multi-step predictions

    Mathematical formulation:
    - Z_t = GCN(X_t, A_t) for each timestep t
    - H_t = LSTM(Z_1:t)
    - Y_t+1:t+k = FC(H_t)

    where X_t is node features, A_t is adjacency, Z_t is GCN output,
    H_t is LSTM hidden state, and Y_t+1:t+k are k-step predictions.
    """

    def __init__(
        self,
        num_node_features: int,
        gcn_out_features: int,
        lstm_hidden_size: int,
        num_nodes: int,
        n_steps_out: int,
        model_name: str = "base",
        config_path: Optional[str] = None,
        num_lstm_layers: int = 2,
    ):
        """Initialize the GCN-LSTM model."""
        # Load config
        if config_path is None:
            raise ValueError("config_path must be provided")

        with open(config_path) as f:
            config = yaml.safe_load(f)["model_params"][model_name]

        super().__init__(config)

        # Model components
        self.spatial_processor = SpatialProcessor(
            num_node_features, gcn_out_features, model_name, config_path
        )

        self.temporal_processor = TemporalProcessor(
            input_size=gcn_out_features * num_nodes,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
        )

        self.prediction_head = PredictionHead(
            input_size=lstm_hidden_size,
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            n_steps_out=n_steps_out,
        )

        # Activation functions
        self.tanh = nn.Tanh()

        # Model parameters
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.n_steps_out = n_steps_out

    def _process_spatial(
        self,
        x_seq: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """Process a single sequence through GCN."""
        gcn_outputs = []

        for x_t, edge_idx_t, edge_weight_t in zip(x_seq, edge_indices, edge_weights):
            gcn_out = self.spatial_processor(x_t, edge_idx_t, edge_weight_t)
            gcn_outputs.append(gcn_out.view(1, -1))

        return torch.cat(gcn_outputs, dim=0)

    def _process_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through LSTM."""
        return self.temporal_processor(self.tanh(x))

    def _generate_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Generate final predictions."""
        return self.prediction_head(self.tanh(x)).squeeze(-1)

    def forward(
        self,
        x_sequences: torch.Tensor,
        edge_indices_sequences: List[List[torch.Tensor]],
        edge_weights_sequences: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass of the GCN-LSTM model."""
        batch_size = x_sequences.size(0)

        # Process each sequence through GCN
        gcn_outputs = []
        for batch_idx in range(batch_size):
            sequence_out = self._process_spatial(
                x_sequences[batch_idx],
                edge_indices_sequences[batch_idx],
                edge_weights_sequences[batch_idx],
            )
            gcn_outputs.append(sequence_out.unsqueeze(0))

        # Combine batch dimension and process through LSTM
        gcn_outputs = torch.cat(gcn_outputs, dim=0)
        lstm_out = self._process_temporal(gcn_outputs)

        # Generate predictions
        return self._generate_prediction(lstm_out)

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}("
            f"num_node_features={self.num_node_features}, "
            f"num_nodes={self.num_nodes}, "
            f"n_steps_out={self.n_steps_out})"
        )
