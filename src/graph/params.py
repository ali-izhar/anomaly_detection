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


@dataclass
class BaseParams:
    """Base parameters for all graph types."""

    n: int  # Number of nodes
    seq_len: int  # Sequence length
    min_segment: int  # Minimum segment length
    min_changes: int  # Minimum number of changes
    max_changes: int  # Maximum number of changes


# ============= Sparse Models =============


@dataclass
class BAParams(BaseParams):
    """Parameters for Barabasi-Albert graph.

    Generates scale-free networks using preferential attachment.
    Typically produces sparse networks with power-law degree distribution.
    Average degree: 2 * initial_edges
    """

    initial_edges: int  # Initial number of edges per new node (m)
    min_edges: int  # Minimum edges after change
    max_edges: int  # Maximum edges after change
    pref_exp: float = 1.0  # Preferential attachment exponent


@dataclass
class WSParams(BaseParams):
    """Parameters for Watts-Strogatz small-world graph.

    Generates small-world networks with high clustering.
    Typically sparse with regular structure plus random rewiring.
    Average degree: 2 * k_nearest
    """

    k_nearest: int  # Number of nearest neighbors (k)
    min_k: int  # Minimum k after change
    max_k: int  # Maximum k after change
    rewire_prob: float = 0.1  # Rewiring probability (p)
    min_prob: float = 0.0  # Minimum rewiring probability after change
    max_prob: float = 0.3  # Maximum rewiring probability after change


@dataclass
class RRParams(BaseParams):
    """Parameters for Random Regular graph.

    Generates graphs where each node has exactly d neighbors.
    Maintains constant sparsity with regular structure.
    Average degree: degree
    """

    degree: int  # Degree of each node (d)
    min_degree: int  # Minimum degree after change
    max_degree: int  # Maximum degree after change


@dataclass
class RGParams(BaseParams):
    """Parameters for Random Geometric graph.

    Generates spatial networks with distance-based connections.
    Typically sparse with geometric constraints.
    Average degree varies with radius.
    """

    radius: float  # Connection radius (r)
    min_radius: float  # Minimum radius after change
    max_radius: float  # Maximum radius after change
    dim: int = 2  # Dimension of geometric space


@dataclass
class RTParams(BaseParams):
    """Parameters for Random Tree.

    Generates tree structures with no cycles.
    Always sparse (|E| = |V| - 1).
    Average degree: 2 - 2/n
    """

    branching_factor: float  # Average number of children
    min_branching: float  # Minimum branching factor after change
    max_branching: float  # Maximum branching factor after change


# ============= Dense Models =============


@dataclass
class ERParams(BaseParams):
    """Parameters for Erdos-Renyi graph.

    Generates random graphs with uniform edge probability.
    Can be dense or sparse depending on probability.
    Average degree: (n-1) * prob
    """

    initial_prob: float  # Initial edge probability (p)
    min_prob: float  # Minimum probability after change
    max_prob: float  # Maximum probability after change


@dataclass
class SBMParams(BaseParams):
    """Parameters for Stochastic Block Model graph.

    Generates networks with community structure.
    Can be dense within communities and sparse between them.
    Average degree depends on probabilities and block sizes.
    """

    num_blocks: int  # Number of communities
    min_block_size: int  # Minimum community size
    max_block_size: int  # Maximum community size
    initial_intra_prob: float  # Initial within-community probability
    initial_inter_prob: float  # Initial between-community probability
    min_intra_prob: float  # Minimum intra-community probability
    max_intra_prob: float  # Maximum intra-community probability
    min_inter_prob: float  # Minimum inter-community probability
    max_inter_prob: float  # Maximum inter-community probability


@dataclass
class RCPParams(BaseParams):
    """Parameters for Random Core-Periphery graph.

    Generates networks with dense core and sparse periphery.
    Mixed density with controlled core-periphery structure.
    Average degree varies between core and periphery.
    """

    core_size: int  # Size of the core
    core_prob: float  # Edge probability within core
    periph_prob: float  # Edge probability in periphery
    min_core_size: int  # Minimum core size after change
    max_core_size: int  # Maximum core size after change
    core_periph_prob: float = 0.1  # Edge probability between core and periphery


@dataclass
class CGParams(BaseParams):
    """Parameters for Complete Graph sequences.

    Generates fully connected graphs or near-complete graphs.
    Always dense (|E| = n(n-1)/2).
    Average degree: n-1
    """

    edge_removal_prob: float = 0.0  # Probability of edge removal
    min_removal_prob: float = 0.0  # Minimum removal probability after change
    max_removal_prob: float = 0.1  # Maximum removal probability after change


@dataclass
class DRGParams(BaseParams):
    """Parameters for Dense Random Geometric graph.

    Generates dense spatial networks with large radius.
    Typically dense with geometric constraints.
    Average degree increases with radius.
    """

    radius: float  # Connection radius (must be large)
    min_radius: float  # Minimum radius after change
    max_radius: float  # Maximum radius after change
    dim: int = 2  # Dimension of geometric space


# ============= Mixed Density Models =============


@dataclass
class NWParams(BaseParams):
    """Parameters for Newman-Watts graph.

    Modified WS model with added random edges.
    Can have varying density based on parameters.
    Average degree: k_nearest + n * prob
    """

    k_nearest: int  # Number of nearest neighbors
    initial_prob: float  # Initial shortcut probability
    min_prob: float  # Minimum probability after change
    max_prob: float  # Maximum probability after change


@dataclass
class HKParams(BaseParams):
    """Parameters for Holme-Kim graph.

    BA variant with tunable clustering.
    Typically sparse with controlled clustering.
    Average degree: 2 * initial_edges * (1 + triad_prob)
    """

    initial_edges: int  # Initial edges per node
    min_edges: int  # Minimum edges after change
    max_edges: int  # Maximum edges after change
    triad_prob: float = 0.1  # Probability of triad formation
    min_triad_prob: float = 0.0  # Minimum triad probability after change
    max_triad_prob: float = 0.3  # Maximum triad probability after change


@dataclass
class LFRParams(BaseParams):
    """Parameters for LFR Benchmark graph.

    Generates realistic networks with communities.
    Can have varying density based on parameters.
    Average degree specified directly.
    """

    avg_degree: int  # Average node degree
    max_degree: int  # Maximum node degree
    mu: float  # Mixing parameter
    min_mu: float  # Minimum mixing parameter after change
    max_mu: float  # Maximum mixing parameter after change
    tau1: float = 2.5  # Node degree exponent
    tau2: float = 1.5  # Community size exponent
    min_community: int = 20  # Minimum community size
    max_community: int = 100  # Maximum community size
