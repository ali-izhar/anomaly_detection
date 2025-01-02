# src/graph/params.py

"""Parameter classes for different graph types.

This module defines parameter classes for various NetworkX graph models, organized by their
density characteristics:

Sparse Models:
- Barabasi-Albert (BA): Scale-free networks with preferential attachment
- Watts-Strogatz (WS): Small-world networks with high clustering
- Random Regular (RR): Regular graphs with fixed degree
- Random Geometric (RG): Spatial networks with distance-based connections
- Random Trees (RT): Tree structures with no cycles

Dense Models:
- Erdos-Renyi (ER): Random graphs with uniform edge probability
- Stochastic Block Model (SBM): Community-structured networks
- Random Core-Periphery (RCP): Networks with dense core and sparse periphery
- Complete Graph (CG): Fully connected networks
- Dense Random Geometric (DRG): High-density spatial networks

Mixed Density Models:
- Newman-Watts (NW): Modified WS model with added random edges
- Holme-Kim (HK): BA variant with added clustering
- LFR Benchmark (LFR): Realistic community networks
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseParams:
    """Base parameters for all graph types.
    
    Standard deviation parameters (with _std suffix) control the evolution of the network.
    Min/max parameters control the range for anomaly injection during change points.
    """
    # Required parameters
    n: int  # Number of nodes
    seq_len: int  # Sequence length
    min_segment: int  # Minimum segment length
    min_changes: int  # Minimum number of changes
    max_changes: int  # Maximum number of changes
    # Optional parameters
    n_std: Optional[float] = None  # Standard deviation for evolving node count


@dataclass
class BAParams:
    """Parameters for Barabási-Albert network generation."""
    # Required base parameters
    n: int  # Number of nodes
    seq_len: int  # Sequence length
    min_segment: int  # Minimum segment length
    min_changes: int  # Minimum number of changes
    max_changes: int  # Maximum number of changes
    # Required model-specific parameters
    m: int  # Number of edges to attach from a new node
    min_m: int  # Minimum m for anomaly injection
    max_m: int  # Maximum m for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None  # Standard deviation for evolving node count
    m_std: Optional[float] = None  # Standard deviation for evolving m parameter


@dataclass
class WSParams:
    """Parameters for Watts-Strogatz small-world graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    k_nearest: int  # Number of nearest neighbors (k)
    min_k: int  # Minimum k for anomaly injection
    max_k: int  # Maximum k for anomaly injection
    rewire_prob: float  # Rewiring probability (p)
    min_prob: float  # Minimum rewiring probability for anomaly injection
    max_prob: float  # Maximum rewiring probability for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    k_std: Optional[float] = None
    prob_std: Optional[float] = None


@dataclass
class RRParams:
    """Parameters for Random Regular graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    degree: int  # Degree of each node (d)
    min_degree: int  # Minimum degree for anomaly injection
    max_degree: int  # Maximum degree for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    degree_std: Optional[float] = None


@dataclass
class RGParams:
    """Parameters for Random Geometric graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    radius: float  # Connection radius (r)
    min_radius: float  # Minimum radius for anomaly injection
    max_radius: float  # Maximum radius for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    radius_std: Optional[float] = None
    dim: int = 2  # Dimension of geometric space


@dataclass
class ERParams:
    """Parameters for Erdos-Renyi graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    prob: float  # Edge probability (p)
    min_prob: float  # Minimum probability for anomaly injection
    max_prob: float  # Maximum probability for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    prob_std: Optional[float] = None


@dataclass
class SBMParams:
    """Parameters for Stochastic Block Model graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    num_blocks: int  # Number of communities
    min_block_size: int  # Minimum community size
    max_block_size: int  # Maximum community size
    intra_prob: float  # Within-community probability
    inter_prob: float  # Between-community probability
    min_intra_prob: float  # Minimum intra-community probability
    max_intra_prob: float  # Maximum intra-community probability
    min_inter_prob: float  # Minimum inter-community probability
    max_inter_prob: float  # Maximum inter-community probability
    # Optional parameters
    n_std: Optional[float] = None
    blocks_std: Optional[float] = None
    intra_prob_std: Optional[float] = None
    inter_prob_std: Optional[float] = None


@dataclass
class RCPParams:
    """Parameters for Random Core-Periphery graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    core_size: int  # Size of the core
    core_prob: float  # Edge probability within core
    periph_prob: float  # Edge probability in periphery
    min_core_size: int  # Minimum core size for anomaly injection
    max_core_size: int  # Maximum core size for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    core_size_std: Optional[float] = None
    core_prob_std: Optional[float] = None
    periph_prob_std: Optional[float] = None
    core_periph_prob: float = 0.1
    core_periph_prob_std: Optional[float] = None


@dataclass
class CGParams:
    """Parameters for Complete Graph sequences."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    min_removal_prob: float  # Minimum removal probability for anomaly injection
    max_removal_prob: float  # Maximum removal probability for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    edge_removal_prob: float = 0.0
    removal_prob_std: Optional[float] = None


@dataclass
class DRGParams:
    """Parameters for Dense Random Geometric graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    radius: float  # Connection radius (must be large)
    min_radius: float  # Minimum radius for anomaly injection
    max_radius: float  # Maximum radius for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    radius_std: Optional[float] = None
    dim: int = 2


@dataclass
class NWParams:
    """Parameters for Newman-Watts graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    k_nearest: int  # Number of nearest neighbors
    prob: float  # Shortcut probability
    min_prob: float  # Minimum probability for anomaly injection
    max_prob: float  # Maximum probability for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    k_std: Optional[float] = None
    prob_std: Optional[float] = None


@dataclass
class HKParams:
    """Parameters for Holme-Kim graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    edges: int  # Edges per node
    min_edges: int  # Minimum edges for anomaly injection
    max_edges: int  # Maximum edges for anomaly injection
    min_triad_prob: float  # Minimum triad probability for anomaly injection
    max_triad_prob: float  # Maximum triad probability for anomaly injection
    # Optional parameters
    n_std: Optional[float] = None
    edges_std: Optional[float] = None
    triad_prob: float = 0.1
    triad_prob_std: Optional[float] = None


@dataclass
class LFRParams:
    """Parameters for LFR Benchmark graph."""
    # Required base parameters
    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int
    # Required model-specific parameters
    avg_degree: int  # Average node degree
    max_degree: int  # Maximum node degree
    mu: float  # Mixing parameter
    min_mu: float  # Minimum mixing parameter for anomaly injection
    max_mu: float  # Maximum mixing parameter for anomaly injection
    min_community: int  # Minimum community size
    max_community: int  # Maximum community size
    # Optional parameters
    n_std: Optional[float] = None
    degree_std: Optional[float] = None
    mu_std: Optional[float] = None
    tau1: float = 2.5
    tau1_std: Optional[float] = None
    tau2: float = 1.5
    tau2_std: Optional[float] = None
