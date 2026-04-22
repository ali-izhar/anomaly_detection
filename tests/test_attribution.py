"""Unit tests for hmd.attribution (Martingale-Shapley equivalence, §III-D).

Tests the four Shapley axioms at the level of our implementation:
- Efficiency: Σ_k ψ_k(t) = 100%
- Symmetry: identical logM columns → identical ψ
- Dummy: a feature with logM_k = 0 while others grow → ψ_k → 0
- Degenerate cases: all logM = 0 (startup) → uniform shares; -inf → uniform.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd import attribution as A


NAMES = [f"f{i}" for i in range(4)]


def test_efficiency_axiom_sums_to_100():
    logM = np.array([[0.1, 0.2, 0.5, 1.0]])
    pct = A.shapley_values(logM, 0, NAMES)
    assert abs(sum(pct.values()) - 100.0) < 1e-9


def test_symmetry_axiom_identical_columns_equal_shares():
    logM = np.array([[2.3, 2.3, 2.3, 2.3]])
    pct = A.shapley_values(logM, 0, NAMES)
    vals = list(pct.values())
    assert all(abs(v - 25.0) < 1e-9 for v in vals), pct


def test_dummy_axiom_zero_contribution_gets_small_share():
    # One feature at logM=10, others at 0. Exp gives e^10 vs 1 → hot feature ≈ 100%.
    logM = np.array([[10.0, 0.0, 0.0, 0.0]])
    pct = A.shapley_values(logM, 0, NAMES)
    # Feature 0 dominates.
    assert pct["f0"] > 99.9
    # Dummies all equal and tiny.
    assert abs(pct["f1"] - pct["f2"]) < 1e-9
    assert pct["f1"] < 0.05


def test_all_zero_logm_returns_uniform_shares():
    """At startup (or right after reset) all logM=0 → equal shares 100/K."""
    logM = np.array([[0.0, 0.0, 0.0, 0.0]])
    pct = A.shapley_values(logM, 0, NAMES)
    for v in pct.values():
        assert abs(v - 25.0) < 1e-9


def test_all_neg_infinite_logm_returns_uniform():
    """Degenerate case: pathologically underflowed streams."""
    logM = np.full((1, 4), -np.inf)
    pct = A.shapley_values(logM, 0, NAMES)
    for v in pct.values():
        assert abs(v - 25.0) < 1e-9


def test_dominant_driver_returns_argmax():
    logM = np.array([[-1.0, 5.0, 2.0, 4.0]])
    name, pct = A.dominant_driver(logM, 0, NAMES)
    assert name == "f1"
    assert pct > 50.0


def test_attribution_trajectory_shape_and_rowsum():
    rng = np.random.default_rng(0)
    T, K = 30, 4
    logM = rng.standard_normal((T, K)) * 2.0
    traj = A.attribution_trajectory(logM, NAMES)
    assert traj.shape == (T, K)
    np.testing.assert_allclose(traj.sum(axis=1), 100.0, atol=1e-8)


def test_log_space_avoids_overflow():
    """ψ_k is normalized in log-space; logM=700 would overflow if exp'd naively."""
    logM = np.array([[700.0, 500.0, 300.0, 100.0]])
    pct = A.shapley_values(logM, 0, NAMES)
    # Should not be NaN/inf and dominated by f0.
    assert all(np.isfinite(v) for v in pct.values())
    assert pct["f0"] > 99.0
