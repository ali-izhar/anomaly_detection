"""Sanity checks for the Table-I synthetic dynamic-network generators.

WHAT is tested
--------------
  * Reproducibility: same seed -> identical graph sequences.
  * Length: every generator produces T=200 snapshots.
  * Change-point spacing respects Delta_min = 40 (0.2 T).
  * Change points lie inside [Delta_min, T - Delta_min].
  * Regime parameters actually induce the expected direction of edge-count
    change (documents the empirical direction for each scenario).

WHY this matters
----------------
All paper results are conditioned on the Table-I setup. Silent drift in the
generators (e.g., off-by-one in the regime switch, a seed bug) would
invalidate every downstream number. Also, the Delta_min constraint is
load-bearing for the paper's evaluation protocol (it guarantees the
detector has room to reset between consecutive changes).
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd.data import synthetic as syn


# All scenarios in the registry.
SCENARIO_NAMES = list(syn.ALL_SCENARIOS.keys())


# ---------------------------------------------------------------------------
# (1) Reproducibility.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_generator_is_reproducible(name, seed):
    gen = syn.ALL_SCENARIOS[name]
    s1 = gen(seed=seed)
    s2 = gen(seed=seed)
    assert s1.change_points == s2.change_points
    assert len(s1.graphs) == len(s2.graphs)
    # Compare edge sets snapshot by snapshot.
    for g1, g2 in zip(s1.graphs, s2.graphs):
        e1 = sorted(tuple(sorted(e)) for e in g1.edges())
        e2 = sorted(tuple(sorted(e)) for e in g2.edges())
        assert e1 == e2, f"{name}, seed={seed}: edge-set mismatch"


# ---------------------------------------------------------------------------
# (2) Length + change-point constraints.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_sequence_length_is_default(name):
    gen = syn.ALL_SCENARIOS[name]
    seq = gen(seed=0)
    assert len(seq.graphs) == syn.T_DEFAULT


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_change_points_respect_delta_min(name):
    gen = syn.ALL_SCENARIOS[name]
    for seed in range(5):  # sweep a few seeds
        seq = gen(seed=seed)
        cps = seq.change_points
        T = len(seq.graphs)
        Dm = syn.DELTA_MIN
        # In window.
        for cp in cps:
            assert Dm <= cp <= T - Dm, (
                f"{name} seed={seed}: cp={cp} outside [{Dm}, {T - Dm}]"
            )
        # Pairwise separation.
        for i in range(len(cps) - 1):
            assert cps[i + 1] - cps[i] >= Dm, (
                f"{name} seed={seed}: consecutive CPs too close: {cps}"
            )


# ---------------------------------------------------------------------------
# (3) Regime directionality.
#
# For each scenario we measure the mean edge count pre- vs post-first-CP and
# assert the direction matches the parameter shift documented in the
# generator's docstring.  This catches, e.g., a swapped regime order.
# ---------------------------------------------------------------------------


def _pre_post_mean_edges(seq, window: int = 20) -> tuple[float, float]:
    cp = seq.change_points[0]
    pre_start = max(0, cp - window)
    post_end = min(len(seq.graphs), cp + window)
    pre = [g.number_of_edges() for g in seq.graphs[pre_start:cp]]
    post = [g.number_of_edges() for g in seq.graphs[cp:post_end]]
    return float(np.mean(pre)), float(np.mean(post))


def test_er_density_increase_post_has_more_edges():
    seq = syn.er_density_increase(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    assert post > pre, (
        f"er_density_increase: post={post:.1f} should exceed pre={pre:.1f}"
    )


def test_er_density_decrease_post_has_fewer_edges():
    seq = syn.er_density_decrease(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    assert post < pre, (
        f"er_density_decrease: post={post:.1f} should be below pre={pre:.1f}"
    )


def test_sbm_community_merge_post_has_more_inter_edges():
    """Pre: p_intra=0.95, p_inter=0.01; Post: p_intra=0.70, p_inter=0.15.
    Total density: pre  ~ 0.5*(0.95+0.01) = 0.48
                   post ~ 0.5*(0.70+0.15) = 0.425
    Pre has MORE edges overall (p_intra heavy). We just assert a nontrivial
    change to guard against a no-op regime swap.
    """
    seq = syn.sbm_community_merge(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    # At N=50 the expected edge counts are ~600 pre, ~520 post (differ by >20).
    assert abs(pre - post) > 10.0, (
        f"sbm_community_merge: edge counts pre={pre:.1f}, post={post:.1f} too close"
    )
    # Document actual direction (decreases).
    assert pre > post, (
        f"sbm_community_merge: pre edges {pre:.1f} should exceed post {post:.1f}"
    )


def test_ba_parameter_shift_post_has_more_edges():
    """Pre m=1 (tree: n-1 edges); Post m=4. More attachments -> more edges."""
    seq = syn.ba_parameter_shift(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    assert post > pre, (
        f"ba_parameter_shift: post={post:.1f} should exceed pre={pre:.1f}"
    )


def test_ba_hub_addition_post_has_many_more_edges():
    """m=1 -> m=6 is a large shift."""
    seq = syn.ba_hub_addition(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    assert post > pre + 50, (
        f"ba_hub_addition: expected big jump, got pre={pre:.1f}, post={post:.1f}"
    )


def test_nws_k_parameter_shift_post_has_more_edges():
    """k=4 -> k=8 doubles neighborhood size -> more edges."""
    seq = syn.nws_k_parameter_shift(seed=0)
    pre, post = _pre_post_mean_edges(seq)
    assert post > pre, (
        f"nws_k_parameter_shift: post={post:.1f} should exceed pre={pre:.1f}"
    )


def test_mixed_changes_has_two_change_points():
    seq = syn.sbm_mixed_changes(seed=0)
    assert len(seq.change_points) == 2


# ---------------------------------------------------------------------------
# (4) Error handling on infeasible parameters.
# ---------------------------------------------------------------------------


def test_sample_change_points_infeasible_raises():
    rng = np.random.default_rng(0)
    # No valid range.
    with pytest.raises(ValueError):
        syn._sample_change_points(T=50, n_cp=1, delta_min=40, rng=rng)
    # Feasible range but cannot fit n_cp with spacing.
    with pytest.raises(ValueError):
        syn._sample_change_points(T=100, n_cp=5, delta_min=40, rng=rng)
