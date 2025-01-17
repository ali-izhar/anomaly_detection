# src/config/graph_configs.py

"""Predefined configurations including standard and evolving parameters for graph models."""

from typing import Dict, Any
from graph.params import BAParams, WSParams, ERParams, SBMParams


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
    """Get configuration for Stochastic Block Model network with extremely stable
    community structure and minimal variations at change points."""
    return {
        "model": "stochastic_block_model",
        "params": SBMParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters - Two perfectly balanced communities
            num_blocks=2,  # Two communities
            min_block_size=n // 2,  # Exactly equal sized communities
            max_block_size=n // 2,  # No size variation allowed
            # Base state: Extremely separated communities
            intra_prob=0.95,  # Almost complete connection within communities
            inter_prob=0.01,  # Almost no connections between communities
            # State transitions between two extreme configurations:
            # State 1: Extremely separated communities (0.95, 0.01)
            # State 2: Almost merged communities (0.3, 0.3)
            min_intra_prob=0.3,  # Lower bound for second state
            max_intra_prob=0.95,  # Upper bound for first state
            min_inter_prob=0.01,  # Lower bound for first state
            max_inter_prob=0.3,  # Upper bound for second state
            # Evolution parameters - No gradual evolution
            n_std=None,  # Fixed number of nodes
            blocks_std=None,  # Fixed number of blocks
            intra_prob_std=None,  # No probability variations
            inter_prob_std=None,  # No probability variations
        ),
    }


# Dictionary mapping model names to their config functions
GRAPH_CONFIGS = {
    # Full model names
    "barabasi_albert": get_ba_config,
    "watts_strogatz": get_ws_config,
    "erdos_renyi": get_er_config,
    "stochastic_block_model": get_sbm_config,
    # Short aliases
    "ba": get_ba_config,
    "ws": get_ws_config,
    "er": get_er_config,
    "sbm": get_sbm_config,
}
