# src/models/layers.py

"""Graph neural network layer implementations for spatio-temporal prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer implementing the propagation rule:

    H^(l+1) = sigma(D^(-1/2) * A * D^(-1/2) * H^(l) * W^(l))

    where:
    - H^(l) is in R^(N x d) (per sequence element) and is the node feature matrix at layer l
    - A is in R^(N x N) and is the adjacency matrix with self-loops
    - D is the degree matrix where D_ii = sum_j A_ij
    - W^(l) is the trainable weight matrix
    - sigma is a nonlinear activation function
    """

    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module = None):
        """
        Args:
            in_dim: Input feature dimension d
            out_dim: Output feature dimension d'
            activation: Optional nonlinear activation function (e.g., nn.ReLU())
        """
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        self.weights.requires_grad_(True)
        self.bias.requires_grad_(True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features matrix H in R^(batch_size x num_nodes x in_dim)
               - batch_size: typically 32-128
               - num_nodes: 100 (fixed for this dataset)
               - in_dim: input feature dimension
                 * First layer: 22 (4 centrality + 2 SVD + 16 LSVD)
                 * Later layers: hidden_dim
            adj: Normalized adjacency matrix D^(-1/2) * A * D^(-1/2)
                Shape: [num_nodes x num_nodes] or [batch_size x num_nodes x num_nodes]

        Returns:
            Output features of shape [batch_size x num_nodes x out_dim]
        """
        # Input validation
        batch_size, num_nodes, in_features = x.size()
        assert num_nodes == 100, f"Expected 100 nodes, got {num_nodes}"
        assert in_features == self.weights.size(0), (
            f"Input feature dim {in_features} doesn't match "
            f"weight matrix dim {self.weights.size(0)}"
        )

        # Ensure input tensors require grad
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # Linear transformation
        support = torch.matmul(x, self.weights)  # [batch_size x num_nodes x out_dim]

        # Handle batched or single adjacency matrix
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # Validate adjacency matrix
        assert adj.size(1) == adj.size(2) == num_nodes, (
            f"Adjacency matrix should be square with size {num_nodes}, "
            f"got shape {adj.size()}"
        )

        # Graph convolution
        output = torch.matmul(adj, support)  # [batch_size x num_nodes x out_dim]
        output = output + self.bias

        return output if self.activation is None else self.activation(output)


class TemporalAttention(nn.Module):
    """Temporal self-attention mechanism for capturing dependencies across timesteps."""

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of hidden representations
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Optional: Multi-head attention parameters
        self.num_heads = 4
        assert (
            hidden_dim % self.num_heads == 0
        ), "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // self.num_heads

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size x seq_len x num_nodes x hidden_dim]
               - batch_size: typically 32-128
               - seq_len: window_size (20)
               - num_nodes: 100 (fixed)
               - hidden_dim: model hidden dimension

        Returns:
            Attended features [batch_size x seq_len x num_nodes x hidden_dim]
        """
        batch_size, seq_len, num_nodes, _ = x.size()

        # Input validation
        assert num_nodes == 100, f"Expected 100 nodes, got {num_nodes}"
        assert seq_len <= 200, f"Sequence length {seq_len} exceeds maximum 200"

        # Apply layer normalization first
        x = self.layer_norm(x)

        # Linear projections
        q = self.query(x)  # [batch_size x seq_len x num_nodes x hidden_dim]
        k = self.key(x)
        v = self.value(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_nodes, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, num_nodes, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, num_nodes, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(
            2, 3
        )  # [batch_size x seq_len x num_heads x num_nodes x head_dim]
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(
            attn, v
        )  # [batch_size x seq_len x num_heads x num_nodes x head_dim]

        # Reshape and project output
        out = out.transpose(
            2, 3
        ).contiguous()  # [batch_size x seq_len x num_nodes x num_heads x head_dim]
        out = out.view(batch_size, seq_len, num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        return out
