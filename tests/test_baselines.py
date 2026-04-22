"""Unit tests for hmd.baselines (CUSUM, EWMA control charts).

Covers:
- No false alarms on constant / stationary signal.
- Detection of a clear step change.
- Return types and shapes.
- Per-feature breach aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd.baselines import BaselineResult, CUSUMConfig, EWMAConfig, cusum, ewma


def _step_sequence(T: int = 100, K: int = 3, change_t: int = 50, shift: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (T, K))
    X[change_t:] += shift
    return X


# ---------- Constant signal → no alarms ----------

def test_cusum_no_alarms_on_constant_signal():
    X = np.full((60, 3), 1.7)
    r = cusum(X)
    assert r.change_points == []
    assert r.score.shape == (60, 3)


def test_ewma_no_alarms_on_constant_signal():
    X = np.full((60, 3), -0.5)
    r = ewma(X)
    assert r.change_points == []
    assert r.score.shape == (60, 3)


# ---------- Step change → detection ----------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cusum_detects_step_change(seed):
    X = _step_sequence(T=100, change_t=50, shift=3.0, seed=seed)
    r = cusum(X, CUSUMConfig(threshold=5.0, startup_period=20))
    assert any(45 <= cp <= 80 for cp in r.change_points), (
        f"CUSUM missed step change at t=50 (seed={seed}); detections: {r.change_points}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ewma_detects_step_change(seed):
    X = _step_sequence(T=100, change_t=50, shift=3.0, seed=seed)
    r = ewma(X, EWMAConfig(lambda_=0.3, L=3.0, startup_period=20))
    assert any(45 <= cp <= 80 for cp in r.change_points), (
        f"EWMA missed step change at t=50 (seed={seed}); detections: {r.change_points}"
    )


# ---------- Return types ----------

def test_cusum_returns_correct_types():
    X = _step_sequence(seed=0)
    r = cusum(X)
    assert isinstance(r, BaselineResult)
    assert isinstance(r.change_points, list)
    assert isinstance(r.score, np.ndarray)
    assert isinstance(r.score_sum, np.ndarray)
    assert r.score_sum.shape == (X.shape[0],)
    assert isinstance(r.breach_feature, list)


def test_breach_feature_length_matches_detections():
    X = _step_sequence(seed=0)
    r = cusum(X, CUSUMConfig(threshold=5.0))
    assert len(r.breach_feature) == len(r.change_points)


# ---------- Reset behavior ----------

def test_cusum_detects_multiple_changes_after_reset():
    """CUSUM should re-fire on a second change after resetting."""
    rng = np.random.default_rng(0)
    T = 200
    X = rng.normal(0, 1, (T, 2))
    X[50:100] += 3.0
    X[150:] += 4.0
    r = cusum(X, CUSUMConfig(threshold=5.0, startup_period=20))
    # At least one detection before t=100 and one after t=150.
    early = [cp for cp in r.change_points if cp < 100]
    late = [cp for cp in r.change_points if cp >= 150]
    assert len(early) >= 1, r.change_points
    assert len(late) >= 1, r.change_points
