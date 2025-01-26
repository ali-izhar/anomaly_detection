# src/configs/params.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class BAParams:
    """Parameters for Barabasi-Albert network."""

    n: int  # Number of nodes in graph
    seq_len: int  # Total sequence length
    min_segment: int  # Minimum time steps between changes
    min_changes: int  # Minimum number of parameter jumps
    max_changes: int  # Maximum number of parameter jumps
    m: int  # Edges per new node
    min_m: int  # Min edges for anomaly injection
    max_m: int  # Max edges for anomaly injection
    n_std: Optional[float] = None  # Node count evolution noise
    m_std: Optional[float] = None  # Edge count evolution noise


@dataclass
class WSParams:
    """Parameters for Watts-Strogatz network."""

    n: int  # Number of nodes in graph
    seq_len: int  # Total sequence length
    min_segment: int  # Minimum time steps between changes
    min_changes: int  # Minimum number of parameter jumps
    max_changes: int  # Maximum number of parameter jumps
    k_nearest: int  # Number of nearest neighbors
    min_k: int  # Min neighbors for anomaly
    max_k: int  # Max neighbors for anomaly
    rewire_prob: float  # Edge rewiring probability
    min_prob: float  # Min rewiring prob for anomaly
    max_prob: float  # Max rewiring prob for anomaly
    n_std: Optional[float] = None  # Node count evolution noise
    k_std: Optional[float] = None  # Neighbor count evolution noise
    prob_std: Optional[float] = None  # Rewiring prob evolution noise


@dataclass
class ERParams:
    """Parameters for Erdos-Renyi network."""

    n: int  # Number of nodes in graph
    seq_len: int  # Total sequence length
    min_segment: int  # Minimum time steps between changes
    min_changes: int  # Minimum number of parameter jumps
    max_changes: int  # Maximum number of parameter jumps
    prob: float  # Edge probability
    min_prob: float  # Min prob for anomaly
    max_prob: float  # Max prob for anomaly
    n_std: Optional[float] = None  # Node count evolution noise
    prob_std: Optional[float] = None  # Probability evolution noise


@dataclass
class SBMParams:
    """Parameters for Stochastic Block Model."""

    n: int  # Number of nodes in graph
    seq_len: int  # Total sequence length
    min_segment: int  # Minimum time steps between changes
    min_changes: int  # Minimum number of parameter jumps
    max_changes: int  # Maximum number of parameter jumps
    num_blocks: int  # Number of communities
    min_block_size: int  # Min nodes per community
    max_block_size: int  # Max nodes per community
    intra_prob: float  # Within-community edge prob
    inter_prob: float  # Between-community edge prob
    min_intra_prob: float  # Min within-community prob
    max_intra_prob: float  # Max within-community prob
    min_inter_prob: float  # Min between-community prob
    max_inter_prob: float  # Max between-community prob
    n_std: Optional[float] = None  # Node count evolution noise
    blocks_std: Optional[float] = None  # Community count noise
    intra_prob_std: Optional[float] = None  # Within-community prob noise
    inter_prob_std: Optional[float] = None  # Between-community prob noise
