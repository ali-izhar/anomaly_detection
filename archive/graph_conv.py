# archive/graph_conv.py

"""Graph Convolutional Network (GCN) layers for soccer player movement data."""

import yaml
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops


class BaseGCNLayer(MessagePassing, ABC):
    """Abstract base class for GCN layers with common functionality."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Dict[str, Any],
    ):
        """
        Initialize base GCN layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            config: Configuration dictionary
        """
        super().__init__(aggr=config.get("aggregation", "add"))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = config.get("epsilon", 1e-8)

    def _compute_normalization(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute normalized edge weights using degree matrix.

        Args:
            edge_index: Edge connectivity [2, E]
            edge_weight: Optional edge weights [E]
            num_nodes: Number of nodes in graph
            dtype: Data type for computation

        Returns:
            Normalized edge weights [E, 1]
        """
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=dtype)
        norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)

        if edge_weight is not None:
            norm = norm * edge_weight.view(-1).clamp(min=self.epsilon)

        return norm.view(-1, 1)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass implementation to be defined by subclasses."""
        pass


class StandardGCN(BaseGCNLayer):
    """Standard GCN implementation: h_i = sigma(sum_j (1/sqrt(d_i*d_j)) * W*h_j)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Dict[str, Any],
    ):
        super().__init__(in_channels, out_channels, config)
        self.lin = Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.bn(self.lin(x))
        norm = self._compute_normalization(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm * x_j


class MultiplicativeGCN(BaseGCNLayer):
    """Multiplicative interaction GCN variants."""

    def __init__(
        self, in_channels: int, out_channels: int, config: Dict[str, Any], version: str
    ):
        super().__init__(in_channels, out_channels, config)
        self.version = version

        # Common layers
        self.lin1 = Linear(in_channels, out_channels)
        self.lin2 = Linear(in_channels, out_channels)

        # Version-specific layers
        if version == "MIv2":
            self.W2 = Parameter(torch.Tensor(out_channels, out_channels))
            self.W3 = Parameter(torch.Tensor(out_channels, out_channels))
            nn.init.xavier_uniform_(self.W2)
            nn.init.xavier_uniform_(self.W3)
        elif version == "MIv3":
            self.lin2 = Linear(out_channels, out_channels)
            self.W2 = Parameter(torch.Tensor(out_channels, out_channels))
            nn.init.xavier_uniform_(self.W2)
        elif version == "MIv4":
            self.relu = nn.ReLU()

        if version != "MIv3":
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.version == "MIv3":
            h = self.lin1(x)
            x = self.lin2(h)
            norm = self._compute_normalization(
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=x.size(0),
                dtype=x.dtype,
            )
            return self.propagate(
                edge_index, x=x, h=h, edge_weight=edge_weight, norm=norm
            )

        h = self.lin2(x)
        x = self.bn(self.lin1(x))

        if self.version == "MIv4":
            h = self.relu(h)

        norm = self._compute_normalization(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )
        return self.propagate(edge_index, x=x, h=h, edge_weight=edge_weight, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm * x_j

    def update(
        self,
        aggr_out: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.version == "MIv1":
            return aggr_out * x
        elif self.version == "MIv2":
            return (
                h
                + torch.matmul(aggr_out, self.W2)
                + torch.matmul(x * aggr_out, self.W3)
            )
        elif self.version == "MIv3":
            return h * aggr_out
        elif self.version == "MIv4":
            return h * self.lin2(aggr_out)


class TeamAwareGCN(BaseGCNLayer):
    """Team-aware GCN: h_i = sum_j a_ij * h_j - alpha * sum_k a_ik * h_k"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: Dict[str, Any],
    ):
        super().__init__(in_channels, out_channels, config)
        self.alpha = config.get("alpha", 0.5)
        self.lin = Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        team_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = self.bn(self.lin(x))
        norm = self._compute_normalization(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )
        return self.propagate(
            edge_index, x=x, edge_weight=edge_weight, team_labels=team_labels, norm=norm
        )

    def message(
        self,
        x_j: torch.Tensor,
        edge_index: torch.Tensor,
        size: Tuple[int, int],
        norm: torch.Tensor,
        team_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = edge_index[0]
        same_team = team_labels[row] == team_labels[edge_index[1]]
        return (
            torch.where(same_team.view(-1, 1), norm * x_j, torch.zeros_like(x_j)),
            torch.where(~same_team.view(-1, 1), norm * x_j, torch.zeros_like(x_j)),
        )


class CustomGCNLayer(MessagePassing):
    """Factory class for creating specific GCN variants."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        version: str = "base",
        config_path: Optional[str] = None,
    ):
        """
        Initialize appropriate GCN variant.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            version: GCN variant to use
            config_path: Path to config file
        """
        # Load config
        if config_path is None:
            raise ValueError("config_path must be provided")

        with open(config_path) as f:
            config = yaml.safe_load(f)["model_params"][version]

        super().__init__(aggr=config.get("aggregation", "add"))

        # Create appropriate GCN variant
        if version == "base":
            self.gcn = StandardGCN(in_channels, out_channels, config)
        elif version in ["MIv1", "MIv2", "MIv3", "MIv4"]:
            self.gcn = MultiplicativeGCN(in_channels, out_channels, config, version)
        elif version == "same-0.5opp":
            self.gcn = TeamAwareGCN(in_channels, out_channels, config)
        else:
            raise ValueError(f"Unknown GCN version: {version}")

        self.version = version

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        team_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass wrapper for specific GCN variant."""
        # Remove padding and self-loops
        mask = edge_index[0] != -1
        edge_index = edge_index[:, mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return self.gcn(
            x=x, edge_index=edge_index, edge_weight=edge_weight, team_labels=team_labels
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.gcn})"
