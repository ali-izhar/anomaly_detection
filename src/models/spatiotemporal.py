# src/models/spatiotemporal.py

"""Spatio-temporal model combining GNN and LSTM components."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoder import STEncoder
from .decoder import TemporalDecoder


@dataclass
class STModelConfig:
    """Configuration for the spatio-temporal model."""
    
    # Model dimensions
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    
    # GNN parameters
    gnn_type: str = "gcn"
    attention_heads: int = 4
    
    # LSTM parameters
    lstm_layers: int = 2
    bidirectional: bool = True
    
    # Constants
    num_nodes: int = 100
    window_size: int = 20
    forecast_horizon: int = 10
    num_features: int = 22  # 4 centrality + 2 SVD + 16 LSVD


class SpatioTemporalPredictor(nn.Module):
    """Combined model for spatio-temporal graph prediction."""

    def __init__(self, config: STModelConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Encoder for processing input sequences
        self.encoder = STEncoder(
            in_channels=config.num_features,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_nodes=config.num_nodes,
            window_size=config.window_size
        )
        
        # Decoder for generating predictions
        self.decoder = TemporalDecoder(
            hidden_dim=config.hidden_dim * (2 if config.bidirectional else 1),
            num_features=config.num_features,
            num_nodes=config.num_nodes,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        # Optional: Feature-specific projection layers
        self.feature_projections = nn.ModuleDict({
            'centrality': nn.Linear(config.hidden_dim, 4),  # 4 centrality features
            'svd': nn.Linear(config.hidden_dim, 2),        # 2-dim SVD
            'lsvd': nn.Linear(config.hidden_dim, 16)       # 16-dim LSVD
        })

    def forward(
        self, 
        x: Dict[str, torch.Tensor],
        adj: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Dictionary of input features
               Centrality features: [batch_size, window_size, num_nodes]
               Spectral features: [batch_size, window_size, num_nodes, embedding_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            hidden: Optional initial hidden state for decoder
        """
        batch_size = next(iter(x.values())).size(0)
        
        # Initialize list for combined features
        combined_features = []
        
        # Process centrality features
        if any(feat in x for feat in ['degree', 'betweenness', 'closeness', 'eigenvector']):
            centrality_tensors = []
            for feat in ['degree', 'betweenness', 'closeness', 'eigenvector']:
                if feat in x:
                    # Add feature dimension: [B, T, N] -> [B, T, N, 1]
                    feat_tensor = x[feat].unsqueeze(-1)
                    centrality_tensors.append(feat_tensor)
            if centrality_tensors:
                centrality_features = torch.cat(centrality_tensors, dim=-1)  # [B, T, N, 4]
                combined_features.append(centrality_features)
        
        # Process spectral features
        if 'svd' in x:
            # SVD already has shape [B, T, N, 2]
            combined_features.append(x['svd'])
        if 'lsvd' in x:
            # LSVD already has shape [B, T, N, 16]
            combined_features.append(x['lsvd'])
        
        # Concatenate all features along the last dimension
        # Final shape: [B, T, N, total_features]
        input_features = torch.cat(combined_features, dim=-1)
        
        # Encode the sequence
        encoded, _ = self.encoder(input_features, adj)
        
        # Generate predictions
        decoded = self.decoder(
            encoded,
            steps=self.config.forecast_horizon,
            hidden=hidden
        )
        
        # Split predictions back into feature types
        predictions = {}
        start_idx = 0
        
        # Centrality features (4)
        if any(feat in x for feat in ['degree', 'betweenness', 'closeness', 'eigenvector']):
            centrality_preds = decoded[..., start_idx:start_idx + 4]
            # Split back into individual features
            predictions['degree'] = centrality_preds[..., 0]
            predictions['betweenness'] = centrality_preds[..., 1]
            predictions['closeness'] = centrality_preds[..., 2]
            predictions['eigenvector'] = centrality_preds[..., 3]
            start_idx += 4
        
        # SVD features (2)
        if 'svd' in x:
            predictions['svd'] = decoded[..., start_idx:start_idx + 2]
            start_idx += 2
        
        # LSVD features (16)
        if 'lsvd' in x:
            predictions['lsvd'] = decoded[..., start_idx:start_idx + 16]
        
        return predictions
