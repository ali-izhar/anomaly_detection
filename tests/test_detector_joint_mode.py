"""Unit tests for HorizonDetector(detection_mode='joint').

Joint mode collapses the K per-feature nonconformity scores into a single
scalar via a distance metric in ℝ^K (Mahalanobis / Euclidean / Cosine /
Chebyshev), and runs ONE martingale per stream instead of K. Paper Table III
varies the distance metric as a sensitivity sweep — this test path exists to
support reproduction of that table.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd import HorizonDetector
from hmd.data.synthetic import sbm_community_merge


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis", "cosine", "chebyshev"])
def test_joint_mode_runs_and_returns_shapes(metric):
    """Smoke test: each distance metric runs and produces expected shapes."""
    seq = sbm_community_merge(seed=0)
    T = len(seq.graphs)
    det = HorizonDetector(
        threshold=50.0,
        horizon=5,
        history_size=20,
        startup_period=20,
        detection_mode="joint",
        joint_distance=metric,
    )
    res = det.run(seq.graphs)
    assert res.logM_traditional.shape == (T,)
    assert res.logM_horizon.shape == (T,)
    # Attribution shadow stream still produces per-feature shape.
    K = len(res.feature_names)
    assert res.logM_per_feature.shape == (T, K)
    # Martingale starts at log 1 = 0 during startup.
    np.testing.assert_equal(res.logM_per_feature[:20], 0.0)


def test_joint_mode_attribution_still_works():
    """Even though primary detection is joint, the shadow per-feature martingale
    provides ψ_k attribution. Shares should sum to 100%."""
    seq = sbm_community_merge(seed=0)
    det = HorizonDetector(
        threshold=50.0,
        startup_period=20,
        detection_mode="joint",
        joint_distance="mahalanobis",
    )
    res = det.run(seq.graphs)
    # Pick a mid-run t where logM has had time to accumulate.
    t_mid = len(seq.graphs) // 2
    shares = res.attribution_at(t_mid)
    assert abs(sum(shares.values()) - 100.0) < 1e-6


def test_joint_mode_determinism():
    seq = sbm_community_merge(seed=0)
    cfg = dict(
        threshold=50.0,
        startup_period=20,
        detection_mode="joint",
        joint_distance="mahalanobis",
        rng_seed=42,
    )
    r1 = HorizonDetector(**cfg).run(seq.graphs)
    r2 = HorizonDetector(**cfg).run(seq.graphs)
    assert r1.change_points == r2.change_points
    np.testing.assert_array_equal(r1.logM_traditional, r2.logM_traditional)


def test_invalid_joint_distance_rejected():
    with pytest.raises(ValueError, match="joint_distance"):
        HorizonDetector(detection_mode="joint", joint_distance="manhattan")


def test_invalid_detection_mode_rejected():
    with pytest.raises(ValueError, match="detection_mode"):
        HorizonDetector(detection_mode="unknown")
