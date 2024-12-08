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
        
        # Initialize components with full config
        self.gnn = TemporalGNN(
            in_channels=node_feat_dim,
            hidden_channels=gnn_config["hidden_dim"],
            out_channels=gnn_config["hidden_dim"],
            num_nodes=num_nodes,
            num_layers=gnn_config["num_layers"],
            dropout=gnn_config.get("dropout", 0.3),
            config=config
        )

        # Temporal predictor for sequence modeling
        temporal_input_dim = gnn_config["hidden_dim"]
        self.temporal = TemporalPredictor(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_config["hidden_dim"],
            output_dim=node_feat_dim,
            forecast_horizon=self.forecast_horizon,
            num_layers=temporal_config["num_layers"],
            dropout=temporal_config["dropout"],
            model_type=temporal_config["type"],
            config=config
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
