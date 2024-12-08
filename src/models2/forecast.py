# src/models2/forecast.py

import torch
import torch.nn as nn
from typing import Dict, Optional
from torch import Tensor

from src.models2.gnn import TemporalGNN
from src.models2.temporal import TemporalPredictor


class GraphTemporalForecaster(nn.Module):
    """Graph-based temporal forecasting model.

    Combines graph neural networks for spatial dependencies with
    temporal models for sequence modeling to generate multi-step forecasts.
    """

    def __init__(
        self,
        config: Dict,
        num_nodes: int,
        node_feat_dim: int,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """Initialize the forecaster.

        Args:
            config: Model configuration dictionary
            num_nodes: Number of nodes in the graph
            node_feat_dim: Dimension of node features
            device: Device to run the model on
        """
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.forecast_horizon = config["data"]["m_horizon"]
        self.device = device

        # Get configs
        gnn_config = config["model"]["gnn"]
        temporal_config = config["model"]["temporal"]
        hw_config = config.get("hardware", {})
        
        # Memory optimization settings
        self.use_checkpoint = hw_config.get("gradient_checkpointing", True)
        self.memory_efficient = hw_config.get("memory_efficient_attention", True)
        
        # Define dimensions
        self.gnn_hidden_dim = gnn_config["hidden_dim"]
        self.temporal_hidden_dim = temporal_config["hidden_dim"]
        
        # Initialize GNN
        self.gnn = TemporalGNN(
            in_channels=node_feat_dim,
            hidden_channels=self.gnn_hidden_dim,
            out_channels=self.gnn_hidden_dim,  # GNN output dimension
            num_nodes=num_nodes,
            num_layers=gnn_config["num_layers"],
            dropout=gnn_config.get("dropout", 0.3),
            config=config
        )

        # Add projections for dimension matching
        self.gnn_to_temporal_proj = nn.Linear(self.gnn_hidden_dim, self.temporal_hidden_dim)
        self.feature_to_temporal_proj = nn.Linear(node_feat_dim, self.temporal_hidden_dim)
        self.temporal_to_output_proj = nn.Linear(self.temporal_hidden_dim, node_feat_dim)

        # Initialize temporal model with correct dimensions
        self.temporal = TemporalPredictor(
            input_dim=self.temporal_hidden_dim,  # Input dimension matches projection
            hidden_dim=self.temporal_hidden_dim,
            output_dim=self.temporal_hidden_dim,  # Output will be projected to final dim
            forecast_horizon=self.forecast_horizon,
            num_layers=temporal_config["num_layers"],
            dropout=temporal_config["dropout"],
            model_type=temporal_config["type"],
            config=config
        )

        # Add batch normalization
        self.input_norm = nn.BatchNorm1d(node_feat_dim)
        self.gnn_norm = nn.LayerNorm(self.temporal_hidden_dim)
        self.feature_norm = nn.LayerNorm(self.temporal_hidden_dim)
        self.output_norm = nn.LayerNorm(node_feat_dim)
        
        # Initialize weights with smaller values
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Reduced gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)

    def forward(
        self,
        batch_data: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass through the model.

        Args:
            batch_data: Dictionary containing:
                - adj_matrices: [batch_size, seq_len, N, N]
                - features: [batch_size, seq_len, num_features]
            mask: Optional mask for variable length sequences [batch_size, seq_len]

        Returns:
            Dictionary containing:
                - predictions: [batch_size, forecast_horizon, num_features]
                - node_embeddings: Node-level embeddings if needed for visualization/analysis
                - graph_embeddings: Graph-level embeddings if needed for visualization/analysis
        """
        # Unpack and normalize input
        adj_matrices = batch_data["adj_matrices"]
        features = batch_data["features"]
        
        batch_size, seq_len, num_features = features.shape
        features_flat = features.view(-1, num_features)
        features_norm = self.input_norm(features_flat).view(batch_size, seq_len, num_features)
        
        # Process with GNN
        node_embeddings, graph_embeddings = self.gnn(x=features_norm, adj=adj_matrices, mask=mask)
        
        # Project GNN output to temporal dimension
        graph_features = self.gnn_norm(self.gnn_to_temporal_proj(graph_embeddings))
        features_proj = self.feature_norm(self.feature_to_temporal_proj(features_norm))
        
        # Scaled combination
        combined_features = graph_features + 0.1 * features_proj
        
        # Generate predictions with normalization
        temporal_output = self.temporal(combined_features, mask)
        predictions = self.output_norm(self.temporal_to_output_proj(temporal_output))
        
        # Smaller residual connection
        last_input = features_norm[:, -1:].expand(-1, predictions.shape[1], -1)
        predictions = predictions + 0.01 * last_input
        
        return {
            "predictions": predictions,
            "node_embeddings": node_embeddings,
            "graph_embeddings": graph_embeddings,
        }

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> Dict:
        """Configure optimizers and learning rate schedulers."""
        # Get optimizer settings from config and convert to proper types
        opt_config = self.config["training"]["optimizer"]
        betas = tuple(float(b) for b in opt_config["betas"])  # Convert to float tuple
        eps = float(opt_config["eps"])  # Convert eps to float
        
        # Create optimizers with configured parameters
        gnn_optimizer = torch.optim.AdamW(
            self.gnn.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=betas,
            eps=eps
        )
        
        temporal_optimizer = torch.optim.AdamW(
            self.temporal.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=betas,
            eps=eps
        )

        # Get scheduler settings from config and convert to proper types
        sched_config = self.config["training"]["scheduler"]
        
        # Create schedulers
        gnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gnn_optimizer,
            mode=sched_config["mode"],
            factor=float(sched_config["factor"]),
            patience=int(sched_config["patience"]),
            min_lr=float(sched_config["min_lr"]),
            cooldown=int(sched_config["cooldown"])
        )
        
        temporal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            temporal_optimizer,
            mode=sched_config["mode"],
            factor=float(sched_config["factor"]),
            patience=int(sched_config["patience"]),
            min_lr=float(sched_config["min_lr"]),
            cooldown=int(sched_config["cooldown"])
        )

        return {
            "optimizers": {
                "gnn": gnn_optimizer,
                "temporal": temporal_optimizer,
            },
            "schedulers": {
                "gnn": gnn_scheduler,
                "temporal": temporal_scheduler,
            },
        }

    @torch.no_grad()
    def forecast(
        self,
        batch_data: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate forecasts for inference.

        Args:
            batch_data: Input data dictionary
            mask: Optional mask for variable length sequences

        Returns:
            Forecasted values [batch_size, forecast_horizon, num_features]
        """
        self.eval()
        outputs = self.forward(batch_data, mask)
        return outputs["predictions"]
