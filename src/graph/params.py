# src/graph/params.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseParams:
    """
    Base parameters for all graph types.

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
        Total length of the generated sequence (number of graphs).
    min_segment : int
        Minimum segment size between change points (ensures each segment
        has at least this many time steps).
    min_changes : int
        Minimum number of parameter changes (jumps).
    max_changes : int
        Maximum number of parameter changes (jumps).

    Optional:
    ---------
    n_std : float, optional
        Standard deviation for evolving 'n' each step (if desired).
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    n_std: Optional[float] = None


@dataclass
class BAParams:
    """
    Parameters for Barabási-Albert network generation (and dynamic extensions).

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
        Number of time steps (graphs) to generate.
    min_segment : int
        Minimum length for each segment before a change point.
    min_changes : int
        Minimum number of jumps in parameters.
    max_changes : int
        Maximum number of jumps.
    m : int
        Number of edges to attach from each new node (BA parameter).
    min_m : int
        Minimum 'm' for anomaly injection.
    max_m : int
        Maximum 'm' for anomaly injection.

    Optional:
    ---------
    n_std : float, optional
        Standard deviation for evolving 'n' each step.
    m_std : float, optional
        Standard deviation for evolving 'm' each step (Gaussian updates).
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    m: int
    min_m: int
    max_m: int

    n_std: Optional[float] = None
    m_std: Optional[float] = None


@dataclass
class WSParams:
    """
    Parameters for Watts-Strogatz small-world network generation.

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
        Number of time steps (graphs).
    min_segment : int
        Minimum segment length.
    min_changes : int
        Minimum number of jump changes.
    max_changes : int
        Maximum number of jump changes.
    k_nearest : int
        Each node is connected to k neighbors on each side in the ring.
    min_k : int
        Minimum k for anomaly injection.
    max_k : int
        Maximum k for anomaly injection.
    rewire_prob : float
        Probability p of rewiring each edge.
    min_prob : float
        Minimum rewiring probability for anomaly injection.
    max_prob : float
        Maximum rewiring probability for anomaly injection.

    Optional:
    ---------
    n_std : float
        Standard deviation for evolving 'n'.
    k_std : float
        Std. dev. for evolving 'k_nearest'.
    prob_std : float
        Std. dev. for evolving 'rewire_prob'.
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    k_nearest: int
    min_k: int
    max_k: int
    rewire_prob: float
    min_prob: float
    max_prob: float

    n_std: Optional[float] = None
    k_std: Optional[float] = None
    prob_std: Optional[float] = None


@dataclass
class ERParams:
    """
    Parameters for Erdős-Rényi G(n,p) random graph generation.

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
        Number of graphs to produce.
    min_segment : int
        Minimum segment length.
    min_changes : int
        Min number of changes.
    max_changes : int
        Max number of changes.
    prob : float
        Initial edge probability p.
    min_prob : float
        Min anomaly injection for p.
    max_prob : float
        Max anomaly injection for p.

    Optional:
    ---------
    n_std : float, optional
        Evolve n.
    prob_std : float, optional
        Evolve p.
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    prob: float
    min_prob: float
    max_prob: float

    n_std: Optional[float] = None
    prob_std: Optional[float] = None


@dataclass
class SBMParams:
    """
    Parameters for Stochastic Block Model (SBM) graph generation.

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
        Sequence length.
    min_segment : int
        Minimum segment size.
    min_changes : int
        Minimum number of parameter changes.
    max_changes : int
        Maximum number of parameter changes.
    num_blocks : int
        How many communities.
    min_block_size : int
        Minimum block size (unused here, but can be relevant for anomaly).
    max_block_size : int
        Maximum block size.
    intra_prob : float
        p(intra-block).
    inter_prob : float
        p(inter-block).
    min_intra_prob : float
        Minimum for anomaly injection.
    max_intra_prob : float
        Maximum for anomaly injection.
    min_inter_prob : float
        Minimum for anomaly injection.
    max_inter_prob : float
        Maximum for anomaly injection.

    Optional:
    ---------
    n_std : float
    blocks_std : float
    intra_prob_std : float
    inter_prob_std : float
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    num_blocks: int
    min_block_size: int
    max_block_size: int
    intra_prob: float
    inter_prob: float
    min_intra_prob: float
    max_intra_prob: float
    min_inter_prob: float
    max_inter_prob: float

    n_std: Optional[float] = None
    blocks_std: Optional[float] = None
    intra_prob_std: Optional[float] = None
    inter_prob_std: Optional[float] = None


@dataclass
class RCPParams:
    """
    Parameters for Random Core-Periphery (RCP) graph generation.

    Required:
    ---------
    n : int
    seq_len : int
    min_segment : int
    min_changes : int
    max_changes : int
    core_size : int
    core_prob : float
    periph_prob : float
    min_core_size : int
    max_core_size : int

    Optional:
    ---------
    n_std : float
    core_size_std : float
    core_prob_std : float
    periph_prob_std : float
    core_periph_prob : float = 0.1
    core_periph_prob_std : float
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    core_size: int
    core_prob: float
    periph_prob: float
    min_core_size: int
    max_core_size: int

    n_std: Optional[float] = None
    core_size_std: Optional[float] = None
    core_prob_std: Optional[float] = None
    periph_prob_std: Optional[float] = None
    core_periph_prob: float = 0.1
    core_periph_prob_std: Optional[float] = None


@dataclass
class LFRParams:
    """
    Parameters for LFR (Lancichinetti-Fortunato-Radicchi) benchmark networks.

    Required:
    ---------
    n : int
        Number of nodes.
    seq_len : int
    min_segment : int
    min_changes : int
    max_changes : int
    avg_degree : int
    max_degree : int
    mu : float
        Mixing parameter (fraction of edges that go outside the node's community).
    min_mu : float
    max_mu : float
    min_community : int
    max_community : int

    Optional:
    ---------
    n_std : float
    degree_std : float
    mu_std : float
    tau1 : float = 2.5
    tau1_std : float
    tau2 : float = 1.5
    tau2_std : float
    """

    n: int
    seq_len: int
    min_segment: int
    min_changes: int
    max_changes: int

    avg_degree: int
    max_degree: int
    mu: float
    min_mu: float
    max_mu: float
    min_community: int
    max_community: int

    n_std: Optional[float] = None
    degree_std: Optional[float] = None
    mu_std: Optional[float] = None
    tau1: float = 2.5
    tau1_std: Optional[float] = None
    tau2: float = 1.5
    tau2_std: Optional[float] = None
