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

        # GNN for spatial modeling
        gnn_config = config["model"]["gnn"]
        self.gnn = TemporalGNN(
            in_channels=node_feat_dim,
            hidden_channels=gnn_config["hidden_dim"],
            out_channels=gnn_config["hidden_dim"],
            num_nodes=num_nodes,
            num_layers=gnn_config["num_layers"],
            dropout=gnn_config.get("dropout", 0.1),
        )

        # Temporal predictor for sequence modeling
        temporal_config = config["model"]["temporal"]
        # Input dim: GNN output + global features
        temporal_input_dim = gnn_config["hidden_dim"]
        self.temporal = TemporalPredictor(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_config["hidden_dim"],
            output_dim=node_feat_dim,
            forecast_horizon=self.forecast_horizon,
            num_layers=temporal_config["num_layers"],
            dropout=temporal_config["dropout"],
            model_type=temporal_config["type"],
        )

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
        # Unpack input data
        adj_matrices = batch_data["adj_matrices"]
        features = batch_data["features"]

        batch_size, seq_len = adj_matrices.shape[:2]

        # Process spatial dependencies with GNN
        node_embeddings, graph_embeddings = self.gnn(
            x=features, adj=adj_matrices, mask=mask
        )

        # Combine GNN embeddings with global features
        if "global_features" in batch_data:
            # If we have additional global features, concatenate them
            combined_features = torch.cat(
                [graph_embeddings, batch_data["global_features"]], dim=-1
            )
        else:
            combined_features = graph_embeddings

        # Generate predictions with temporal model
        predictions = self.temporal(combined_features, mask)

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
        """Configure optimizers and learning rate schedulers.

        Args:
            learning_rate: Learning rate for optimizers
            weight_decay: Weight decay for regularization

        Returns:
            Dictionary containing optimizers and schedulers
        """
        # Convert to float in case they come as strings
        lr = float(learning_rate)
        wd = float(weight_decay)
        
        # Separate optimizers for GNN and temporal components
        gnn_optimizer = torch.optim.AdamW(
            self.gnn.parameters(),
            lr=lr,
            weight_decay=wd,
        )
        temporal_optimizer = torch.optim.AdamW(
            self.temporal.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        # Calculate total steps for schedulers based on dataset size
        batch_size = self.config["data"]["batch_size"]
        train_size = int(900 * self.config["data"]["train_split"])  # Approximate dataset size
        steps_per_epoch = train_size // batch_size
        total_steps = steps_per_epoch * self.config["training"]["epochs"]

        # Learning rate schedulers
        gnn_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            gnn_optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
        )
        temporal_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            temporal_optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
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
