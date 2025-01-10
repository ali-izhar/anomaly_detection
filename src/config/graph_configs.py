# src/config/graph_configs.py

"""Predefined configurations including standard and evolving parameters for graph models."""

from typing import Dict, Any
from graph.params import BAParams, WSParams, ERParams, SBMParams, RCPParams, LFRParams


def get_ba_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Barabási-Albert network."""
    return {
        "model": "barabasi_albert",
        "params": BAParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Highly structured growth (constrained)
            m=3,  # Fixed number of edges per new node
            min_m=2,  # Minimum edges per new node
            max_m=4,  # Maximum edges per new node
            # Evolution parameters - Minimal variability
            n_std=None,  # Fixed number of nodes
            m_std=0.2,  # Very small variation in edge addition
        ),
    }


def get_ws_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Watts-Strogatz network."""
    return {
        "model": "watts_strogatz",
        "params": WSParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Clear small-world structure
            k_nearest=6,  # Each node connected to 6 nearest neighbors
            min_k=4,  # Minimum nearest neighbors
            max_k=8,  # Maximum nearest neighbors
            rewire_prob=0.1,  # Low rewiring probability for clear structure
            min_prob=0.05,  # Minimum rewiring probability
            max_prob=0.15,  # Maximum rewiring probability
            # Evolution parameters - Minimal variability
            n_std=None,  # Fixed number of nodes
            k_std=0.2,  # Small variation in degree
            prob_std=0.01,  # Very small rewiring variations
        ),
    }


def get_er_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Erdős-Rényi network."""
    return {
        "model": "erdos_renyi",
        "params": ERParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Stable density
            prob=0.15,  # Base probability
            min_prob=0.1,  # Minimum probability
            max_prob=0.2,  # Maximum probability
            # Evolution parameters - Minimal variability
            n_std=None,  # Fixed number of nodes
            prob_std=0.01,  # Very small probability variations
        ),
    }


def get_sbm_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Stochastic Block Model network with extremely distinct states
    and drastic changes at change points."""
    return {
        "model": "stochastic_block_model",
        "params": SBMParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Two extreme states
            num_blocks=2,  # Two clear communities
            min_block_size=n // 2 - 5,  # Almost equal sized communities
            max_block_size=n // 2 + 5,  # Small variation allowed
            # State 1: Extremely separated communities
            intra_prob=0.95,  # Almost complete connection within communities
            inter_prob=0.01,  # Almost no connections between communities
            # State transitions between two extreme configurations:
            # State 1: Extremely separated communities (0.95, 0.01)
            # State 2: Almost uniform connectivity (0.4, 0.4)
            min_intra_prob=0.4,  # Lower bound - uniform state
            max_intra_prob=0.95,  # Upper bound - separated state
            min_inter_prob=0.01,  # Lower bound - separated state
            max_inter_prob=0.4,  # Upper bound - uniform state
            # Evolution parameters - Absolutely no gradual evolution
            n_std=None,  # Fixed number of nodes
            blocks_std=None,  # Fixed number of blocks
            intra_prob_std=None,  # No probability variations
            inter_prob_std=None,  # No probability variations
        ),
    }


def get_rcp_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Random Core-Periphery network."""
    return {
        "model": "random_core_periphery",
        "params": RCPParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Clear core-periphery structure
            core_size=n // 5,  # Core is 20% of network
            min_core_size=n // 6,  # Minimum core size
            max_core_size=n // 4,  # Maximum core size
            # High contrast between core and periphery
            core_prob=0.8,  # Dense core
            periph_prob=0.05,  # Sparse periphery
            core_periph_prob=0.2,  # Moderate core-periphery connectivity
            # Evolution parameters - Minimal variability
            n_std=None,  # Fixed number of nodes
            core_size_std=1.0,  # Very small core size variations
            core_prob_std=0.02,  # Small probability variations
            periph_prob_std=0.005,  # Minimal periphery variations
            core_periph_prob_std=0.01,  # Small core-periphery variations
        ),
    }


def get_lfr_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for LFR Benchmark network."""
    return {
        "model": "lfr_benchmark",
        "params": LFRParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Clear community structure
            avg_degree=8,  # Higher average degree for stability
            max_degree=20,  # Reasonable maximum degree
            mu=0.1,  # Strong communities (low mixing)
            min_mu=0.05,  # Very strong communities
            max_mu=0.15,  # Still maintain clear structure
            min_community=n // 6,  # Reasonable community sizes
            max_community=n // 3,
            # Power law parameters - Realistic but stable
            tau1=2.5,  # Degree distribution exponent
            tau2=1.5,  # Community size distribution exponent
            # Evolution parameters - Minimal variability
            n_std=None,  # Fixed number of nodes
            degree_std=0.2,  # Very small degree variations
            mu_std=0.01,  # Minimal mixing parameter variations
        ),
    }


# Dictionary mapping model names to their config functions
GRAPH_CONFIGS = {
    # Full model names
    "barabasi_albert": get_ba_config,
    "watts_strogatz": get_ws_config,
    "erdos_renyi": get_er_config,
    "stochastic_block_model": get_sbm_config,
    "random_core_periphery": get_rcp_config,
    "lfr_benchmark": get_lfr_config,
    # Short aliases
    "ba": get_ba_config,
    "ws": get_ws_config,
    "er": get_er_config,
    "sbm": get_sbm_config,
    "rcp": get_rcp_config,
    "lfr": get_lfr_config,
}
