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
    """Get configuration for Barabási-Albert network.

    Returns evolving preferential attachment network with occasional
    structural changes through m parameter.
    """
    return {
        "model": "barabasi_albert",
        "params": BAParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            m=3,
            min_m=1,
            max_m=6,
            # Evolution parameters
            n_std=None,  # Fixed node count
            m_std=0.5,  # Evolving edge count
        ),
    }


def get_ws_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Watts-Strogatz network.

    Returns evolving small-world network with changes in both
    connectivity (k) and rewiring probability (p).
    """
    return {
        "model": "watts_strogatz",
        "params": WSParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            k_nearest=4,
            min_k=2,
            max_k=8,
            rewire_prob=0.1,
            min_prob=0.05,
            max_prob=0.3,
            # Evolution parameters
            n_std=None,
            k_std=0.3,
            prob_std=0.02,
        ),
    }


def get_er_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Erdős-Rényi network.

    Returns evolving random network with changes in edge probability.
    """
    return {
        "model": "erdos_renyi",
        "params": ERParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            prob=0.1,
            min_prob=0.05,
            max_prob=0.2,
            # Evolution parameters
            n_std=None,
            prob_std=0.01,
        ),
    }


def get_sbm_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Stochastic Block Model network.

    Returns evolving community structure with changes in both
    intra and inter-community connection probabilities.
    """
    return {
        "model": "stochastic_block_model",
        "params": SBMParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            num_blocks=4,
            min_block_size=20,
            max_block_size=35,
            intra_prob=0.3,
            inter_prob=0.05,
            min_intra_prob=0.2,
            max_intra_prob=0.4,
            min_inter_prob=0.02,
            max_inter_prob=0.1,
            # Evolution parameters
            n_std=None,
            blocks_std=None,
            intra_prob_std=0.02,
            inter_prob_std=0.01,
        ),
    }


def get_rcp_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for Random Core-Periphery network.

    Returns evolving core-periphery structure with changes in
    core size and connection probabilities.
    """
    return {
        "model": "random_core_periphery",
        "params": RCPParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            core_size=20,
            core_prob=0.7,
            periph_prob=0.1,
            min_core_size=15,
            max_core_size=30,
            # Evolution parameters
            n_std=None,
            core_size_std=2.0,
            core_prob_std=0.05,
            periph_prob_std=0.02,
            core_periph_prob=0.2,
            core_periph_prob_std=0.03,
        ),
    }


def get_lfr_config(
    n: int = 100,
    seq_len: int = 200,
    min_segment: int = 40,
    min_changes: int = 1,
    max_changes: int = 3,
) -> Dict[str, Any]:
    """Get configuration for LFR Benchmark network.

    Returns evolving community structure with power-law degree
    distribution and community sizes.
    """
    return {
        "model": "lfr_benchmark",
        "params": LFRParams(
            # Base parameters
            n=n,
            seq_len=seq_len,
            min_segment=min_segment,
            min_changes=min_changes,
            max_changes=max_changes,
            # Model parameters
            avg_degree=6,
            max_degree=20,
            mu=0.2,
            min_mu=0.1,
            max_mu=0.4,
            min_community=20,
            max_community=50,
            # Evolution parameters
            n_std=None,
            degree_std=0.5,
            mu_std=0.02,
            tau1=2.5,
            tau2=1.5,
        ),
    }


# Dictionary mapping model names to their config functions
GRAPH_CONFIGS = {
    # Full names
    "barabasi_albert": get_ba_config,
    "watts_strogatz": get_ws_config,
    "erdos_renyi": get_er_config,
    "stochastic_block_model": get_sbm_config,
    "random_core_periphery": get_rcp_config,
    "lfr_benchmark": get_lfr_config,
    # Short names
    "ba": get_ba_config,
    "ws": get_ws_config,
    "er": get_er_config,
    "sbm": get_sbm_config,
    "rcp": get_rcp_config,
    "lfr": get_lfr_config,
}
