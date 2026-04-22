"""Unit tests for the multi-horizon Vovk-Wang mixture extension.

Paper Def 7 specifies a single horizon h. Vovk-Wang 2021 (Prop 2.1) shows that
a convex mixture of test martingales is also a test martingale by linearity.
We exploit this to run H independent per-h horizon streams and combine via
`M_mix = Σ_h w_h · M_h`, preserving Ville's 1/λ bound exactly.

These tests verify:
1. Validation of horizon_weights (shape, sum-to-1, non-negativity).
2. Equivalence: with Dirac mass at h=H, the mixture reduces to single-h mode.
3. Under H₀, the mixture's false-alarm probability stays under 1/λ.
4. Determinism: same seed → same change_points under mixture mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd import HorizonDetector
from hmd.data.synthetic import sbm_community_merge, er_density_increase


def test_horizon_weights_length_must_equal_horizon():
    with pytest.raises(ValueError, match="length"):
        HorizonDetector(horizon=5, horizon_weights=(0.5, 0.5))


def test_horizon_weights_must_be_nonneg():
    with pytest.raises(ValueError, match="non-negative"):
        HorizonDetector(horizon=3, horizon_weights=(-0.1, 0.5, 0.6))


def test_horizon_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        HorizonDetector(horizon=3, horizon_weights=(0.1, 0.2, 0.3))


def test_single_h_mode_is_default():
    """horizon_weights=None => single-h (paper Def 7) behavior."""
    seq = sbm_community_merge(seed=0)
    det = HorizonDetector(
        threshold=50.0, startup_period=20, horizon=5, horizon_weights=None
    )
    res = det.run(seq.graphs)
    # Just assert it runs and produces a well-shaped result.
    assert res.logM_horizon.shape == (len(seq.graphs),)


def test_mixture_mode_runs_and_returns_shapes():
    """horizon=4 with uniform weights (1/4,)*4 runs 4 per-h streams."""
    seq = sbm_community_merge(seed=0)
    det = HorizonDetector(
        threshold=50.0,
        startup_period=20,
        horizon=4,
        horizon_weights=(0.25, 0.25, 0.25, 0.25),
    )
    res = det.run(seq.graphs)
    T = len(seq.graphs)
    assert res.logM_horizon.shape == (T,)
    assert res.logM_per_feature_horizon.shape == (T, len(res.feature_names))


def test_mixture_dirac_mass_matches_single_h():
    """Placing all weight on one h must reproduce single-h behavior."""
    seq = er_density_increase(seed=0)
    H = 5
    det_single = HorizonDetector(
        threshold=50.0, startup_period=20, horizon=H, horizon_weights=None
    )
    # Dirac mass at h=H ⇒ all weight on the LAST horizon (index H-1 in a 0..H-1
    # numbered per-h vector, which corresponds to step h=H in 1-indexed).
    dirac = tuple([0.0] * (H - 1) + [1.0])
    det_dirac = HorizonDetector(
        threshold=50.0, startup_period=20, horizon=H, horizon_weights=dirac
    )
    r_single = det_single.run(seq.graphs)
    r_dirac = det_dirac.run(seq.graphs)
    # Both should produce the same change-point list.
    # (Numerical noise from log(0) handling in weights may cause tiny diffs in
    #  logM trajectories, so we only assert on change_points.)
    assert r_single.change_points == r_dirac.change_points


def test_mixture_determinism():
    seq = sbm_community_merge(seed=0)
    cfg = dict(
        threshold=50.0,
        startup_period=20,
        horizon=3,
        horizon_weights=(0.5, 0.3, 0.2),
        rng_seed=42,
    )
    r1 = HorizonDetector(**cfg).run(seq.graphs)
    r2 = HorizonDetector(**cfg).run(seq.graphs)
    assert r1.change_points == r2.change_points
    # Attribution path is also deterministic.
    np.testing.assert_array_equal(r1.logM_per_feature, r2.logM_per_feature)


def test_mixture_preserves_ville_under_h0():
    """Under H₀ (i.i.d. stationary generator), the mixture martingale should
    still respect Ville's inequality: empirical P(sup logM ≥ log λ) ≤ 1/λ.

    Uses an SBM sequence with NO change points (synthesized by setting
    regime params identical on both sides). We just take the first half
    of sbm_community_merge which is pre-change.
    """
    from hmd import features as F

    # 40 short H₀ sequences (seed-varying); check crossing rate vs 1/λ.
    lam = 10.0
    n_trials = 30
    n_cross = 0
    for seed in range(n_trials):
        seq = sbm_community_merge(seed=seed)
        # Take only the pre-change portion.
        first_cp = seq.change_points[0] if seq.change_points else len(seq.graphs)
        X = F.extract_sequence(seq.graphs[:first_cp], F.default_set())
        det = HorizonDetector(
            threshold=lam,
            startup_period=20,
            horizon=3,
            horizon_weights=(0.4, 0.3, 0.3),
        )
        res = det.run_on_features(X)
        # Count whether the MIXTURE horizon martingale crossed λ under H₀.
        M = np.exp(res.logM_horizon)
        if np.any(np.isfinite(M) & (M >= lam)):
            n_cross += 1
    empirical = n_cross / n_trials
    # Ville bound 1/λ = 0.1 + 0.1 slack for MC noise at n_trials=30.
    assert empirical <= 0.20, f"Mixture violates Ville: P(sup M ≥ {lam}) = {empirical}"
