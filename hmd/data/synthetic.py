"""Synthetic dynamic-network generators — Table I (Ali & Ho, ICDM 2025).

Nine scenarios used in §V: three SBM, two ER, two BA, two NWS. All share
T=200, N=50, Δ_min=40, 1-2 change points per sequence.

Critical design choices
-----------------------
- **I.i.d. per-snapshot (fresh sample every t), not temporal evolution.** The
  martingale property (Def 6, Thm 1) requires p-value uniformity under H₀ =
  exchangeability of feature sequences within a regime. Persistent edges
  break exchangeability.
- **Δ_min = 0.2T = 40.** Gives the detector w_cal ≈ 20 recalibration window
  + cooldown between CPs — so a missed CP2 is a detection failure, not a
  timing artifact.
- **Extreme parameter ranges** (Table I): large pre/post contrast forces
  differences between detectors to reflect detection *speed*, not *ability*.
- **2-block SBM** (not 3+): minimal nontrivial community structure;
  "community merge" is unambiguous with two blocks.
- **NWS, not WS**: `nx.newman_watts_strogatz_graph` adds shortcuts without
  rewiring, avoiding disconnected-graph pathologies at low p_rewire.
- **ER p = 0.05 baseline**: above the giant-component threshold (1/N = 0.02
  for N=50) so features stay well-conditioned.
- **One function per scenario**: nine paper rows ↔ nine named functions.
  `change_points[i]` is the first timestep of the new regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np

# -----------------------------------------------------------------------------
# Dataclass
# -----------------------------------------------------------------------------

T_DEFAULT = 200
N_DEFAULT = 50
DELTA_MIN = 40  # 0.2 * T_DEFAULT; see module docstring for rationale.


@dataclass(frozen=True)
class Sequence:
    """A synthetic graph sequence with ground-truth change points."""

    graphs: list[nx.Graph]
    change_points: list[int]
    scenario: str
    params: dict = field(default_factory=dict)
    seed: int = 0


# -----------------------------------------------------------------------------
# Change-point placement
# -----------------------------------------------------------------------------


def _sample_change_points(
    T: int, n_cp: int, delta_min: int, rng: np.random.Generator
) -> list[int]:
    """Sample `n_cp` change points in [delta_min, T-delta_min] with pairwise
    separation ≥ delta_min. Uses rejection sampling; feasibility is checked up
    front so we never infinite-loop.

    Feasibility: we need n_cp points in a window of width (T - 2*delta_min)
    separated by ≥ delta_min, i.e. (T - 2*delta_min) ≥ (n_cp - 1) * delta_min.
    """
    lo, hi = delta_min, T - delta_min
    if hi <= lo:
        raise ValueError(f"No valid range for change points: T={T}, Δ_min={delta_min}")
    if (hi - lo) < (n_cp - 1) * delta_min:
        raise ValueError(
            f"Cannot fit {n_cp} CPs with Δ_min={delta_min} in T={T} "
            f"(need span ≥ {(n_cp - 1) * delta_min}, have {hi - lo})"
        )

    for _ in range(10_000):  # rejection; practically terminates in <10 tries.
        cps = sorted(int(x) for x in rng.integers(lo, hi + 1, size=n_cp))
        if n_cp == 1 or all(cps[i + 1] - cps[i] >= delta_min for i in range(n_cp - 1)):
            return cps
    raise RuntimeError("Change-point rejection sampling failed (should be impossible).")


# -----------------------------------------------------------------------------
# Per-model snapshot samplers (each draws a FRESH graph with given params)
# -----------------------------------------------------------------------------


def _sample_sbm(
    n: int, p_intra: float, p_inter: float, rng: np.random.Generator
) -> nx.Graph:
    """2-block SBM with equal-sized communities.

    Uses nx.stochastic_block_model with a seed derived from the shared rng so
    reproducibility threads cleanly across snapshots.
    """
    sizes = [n // 2, n - n // 2]
    P = [[p_intra, p_inter], [p_inter, p_intra]]
    # nx.stochastic_block_model accepts an int seed. Derive from rng.
    seed = int(rng.integers(0, 2**31 - 1))
    G = nx.stochastic_block_model(sizes, P, seed=seed)
    # Drop the "block" node attribute — detector features are permutation-
    # invariant and we don't want to leak labels downstream.
    return nx.Graph(G)


def _sample_er(n: int, p: float, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _sample_ba(n: int, m: int, rng: np.random.Generator) -> nx.Graph:
    # m must be ≥1 and < n. BA requires an initial m-node connected core.
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _sample_nws(
    n: int, k: int, p_rewire: float, rng: np.random.Generator
) -> nx.Graph:
    # k must be even (neighbors on each side). NetworkX enforces this.
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.newman_watts_strogatz_graph(n, k, p_rewire, seed=seed)


# -----------------------------------------------------------------------------
# Regime assembly
# -----------------------------------------------------------------------------


def _assemble(
    T: int,
    change_points: list[int],
    samplers: list[Callable[[np.random.Generator], nx.Graph]],
    rng: np.random.Generator,
) -> list[nx.Graph]:
    """Build T snapshots. `samplers[k](rng)` is invoked once per timestep in
    regime k (regime 0 before the first CP, 1 after it, etc.).

    i.i.d. per snapshot is the contract: each sampler must draw a fresh graph
    each call from its regime's distribution. See module docstring.
    """
    graphs: list[nx.Graph] = []
    bounds = [0, *change_points, T]
    regime_idx = 0
    for t in range(T):
        while t >= bounds[regime_idx + 1]:
            regime_idx += 1
        graphs.append(samplers[regime_idx](rng))
    return graphs


def _build(
    T: int,
    n_cp: int,
    regime_params: list[dict],
    sampler_factory: Callable[[dict], Callable[[np.random.Generator], nx.Graph]],
    scenario: str,
    seed: int,
    delta_min: int = DELTA_MIN,
) -> Sequence:
    """Generic builder: place CPs, assemble snapshots, package a Sequence."""
    assert len(regime_params) == n_cp + 1, "need (n_cp + 1) regimes"
    rng = np.random.default_rng(seed)
    cps = _sample_change_points(T, n_cp, delta_min, rng)
    samplers = [sampler_factory(p) for p in regime_params]
    graphs = _assemble(T, cps, samplers, rng)
    return Sequence(
        graphs=graphs,
        change_points=cps,
        scenario=scenario,
        params={"regimes": regime_params, "delta_min": delta_min, "T": T},
        seed=seed,
    )


# -----------------------------------------------------------------------------
# SBM scenarios
# -----------------------------------------------------------------------------


def sbm_community_merge(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Strong 2-block → weakened blocks (communities merging).

    Pre :  p_intra=0.95, p_inter=0.01  (near-clique blocks, clean cut)
    Post:  p_intra=0.70, p_inter=0.15  (fuzzy blocks, cut weakened)
    """
    regimes = [
        {"p_intra": 0.95, "p_inter": 0.01},
        {"p_intra": 0.70, "p_inter": 0.15},
    ]

    def factory(p: dict):
        return lambda rng: _sample_sbm(n, p["p_intra"], p["p_inter"], rng)

    return _build(T, 1, regimes, factory, "sbm_community_merge", seed)


def sbm_density_change(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Both intra- and inter-block probabilities drop (overall densification
    reversal while preserving community structure)."""
    regimes = [
        {"p_intra": 0.95, "p_inter": 0.01},
        {"p_intra": 0.50, "p_inter": 0.05},
    ]

    def factory(p: dict):
        return lambda rng: _sample_sbm(n, p["p_intra"], p["p_inter"], rng)

    return _build(T, 1, regimes, factory, "sbm_density_change", seed)


def sbm_mixed_changes(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Two change points: strong 2-block → merge → density drop.

    Tests the detector's ability to reset between consecutive CPs (Δ_min=40
    is enforced globally).
    """
    regimes = [
        {"p_intra": 0.95, "p_inter": 0.01},  # pre-CP1: strong 2-block
        {"p_intra": 0.70, "p_inter": 0.15},  # between CP1 and CP2: merging
        {"p_intra": 0.40, "p_inter": 0.05},  # post-CP2: density drop
    ]

    def factory(p: dict):
        return lambda rng: _sample_sbm(n, p["p_intra"], p["p_inter"], rng)

    return _build(T, 2, regimes, factory, "sbm_mixed_changes", seed)


# -----------------------------------------------------------------------------
# ER scenarios
# -----------------------------------------------------------------------------


def er_density_increase(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Sparse → denser: p=0.05 (sparse, connected on average) → p=0.30 (dense).

    Matches Table I: pre p=0.05, post p ∈ [0.05, 0.40].
    """
    regimes = [{"p": 0.05}, {"p": 0.30}]

    def factory(params: dict):
        return lambda rng: _sample_er(n, params["p"], rng)

    return _build(T, 1, regimes, factory, "er_density_increase", seed)


def er_density_decrease(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Dense → sparse. Same magnitudes as the increase case, reversed."""
    regimes = [{"p": 0.30}, {"p": 0.05}]

    def factory(params: dict):
        return lambda rng: _sample_er(n, params["p"], rng)

    return _build(T, 1, regimes, factory, "er_density_decrease", seed)


# -----------------------------------------------------------------------------
# BA scenarios
# -----------------------------------------------------------------------------


def ba_parameter_shift(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Moderate hub formation: m=1 (tree) → m=4 (mild hub structure)."""
    regimes = [{"m": 1}, {"m": 4}]

    def factory(params: dict):
        return lambda rng: _sample_ba(n, params["m"], rng)

    return _build(T, 1, regimes, factory, "ba_parameter_shift", seed)


def ba_hub_addition(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Dramatic hub formation: m=1 → m=6. Upper end of Table I's range."""
    regimes = [{"m": 1}, {"m": 6}]

    def factory(params: dict):
        return lambda rng: _sample_ba(n, params["m"], rng)

    return _build(T, 1, regimes, factory, "ba_hub_addition", seed)


# -----------------------------------------------------------------------------
# NWS scenarios
# -----------------------------------------------------------------------------


def nws_rewiring_increase(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Rewiring (shortcut-addition in NWS) increases: p_rewire 0.05 → 0.15.

    k=6 held constant; only the shortcut density changes.
    """
    regimes = [
        {"k": 6, "p_rewire": 0.05},
        {"k": 6, "p_rewire": 0.15},
    ]

    def factory(params: dict):
        return lambda rng: _sample_nws(n, params["k"], params["p_rewire"], rng)

    return _build(T, 1, regimes, factory, "nws_rewiring_increase", seed)


def nws_k_parameter_shift(
    T: int = T_DEFAULT, n: int = N_DEFAULT, seed: int = 0
) -> Sequence:
    """Neighborhood size doubles: k=4 → k=8 (p_rewire held at 0.1)."""
    regimes = [
        {"k": 4, "p_rewire": 0.1},
        {"k": 8, "p_rewire": 0.1},
    ]

    def factory(params: dict):
        return lambda rng: _sample_nws(n, params["k"], params["p_rewire"], rng)

    return _build(T, 1, regimes, factory, "nws_k_parameter_shift", seed)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

ALL_SCENARIOS: dict[str, Callable[..., Sequence]] = {
    "sbm_community_merge": sbm_community_merge,
    "sbm_density_change": sbm_density_change,
    "sbm_mixed_changes": sbm_mixed_changes,
    "er_density_increase": er_density_increase,
    "er_density_decrease": er_density_decrease,
    "ba_parameter_shift": ba_parameter_shift,
    "ba_hub_addition": ba_hub_addition,
    "nws_rewiring_increase": nws_rewiring_increase,
    "nws_k_parameter_shift": nws_k_parameter_shift,
}
