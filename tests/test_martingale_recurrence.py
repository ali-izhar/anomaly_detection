"""Unit tests for the martingale recurrence.

Theorems touched
----------------
These tests do NOT test a theorem per se; they verify that the *implementation*
of Algorithm 1's recurrence

    M_t = M_{t-1} * g(p_t)

in hmd.martingale.run_single_feature (and the K-wide analogue
run_multi_feature) is equivalent to a naive direct-product reference,
and is numerically stable for T up to 1000.

WHY this matters
----------------
The code operates in LOG space to avoid float64 overflow (M_t easily blows
past 10**308 for long sequences with g > 1). If log-space accumulation
disagrees with direct-space cumulative product on short sequences, the
log-space path has a bug that cannot be caught by downstream statistical
tests -- it would manifest only as mysterious off-by-one differences in
plots. We pin it here.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd import betting as B
from hmd import martingale as M


# ---------------------------------------------------------------------------
# (1) Short-sequence exact agreement with np.cumprod reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_run_single_feature_matches_cumprod(seed):
    rng = np.random.default_rng(seed)
    T = 20
    # Draw p in [0.01, 0.99] to avoid the eps_clip boundary region.
    p = rng.uniform(0.01, 0.99, size=T)
    # Keep p[0] non-nan to make the reference trivial (NaN handling tested separately).
    g = B.power(0.7)

    # Reference: naive direct-space cumulative product with M_0=1 and NO update
    # at t=0 (the module convention: logM[0] = 0).
    gp = g(p)
    ref = np.empty(T)
    ref[0] = 1.0
    for t in range(1, T):
        ref[t] = ref[t - 1] * gp[t]

    logM = M.run_single_feature(p, g)
    got = np.exp(logM)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_run_multi_feature_matches_cumprod(seed):
    rng = np.random.default_rng(seed)
    T, K = 20, 5
    P = rng.uniform(0.01, 0.99, size=(T, K))
    g = B.default()

    gp = g(P)  # (T, K)
    ref = np.empty((T, K))
    ref[0, :] = 1.0
    for t in range(1, T):
        ref[t] = ref[t - 1] * gp[t]

    logM = M.run_multi_feature(P, g)
    got = np.exp(logM)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# (2) NaN p[0] handled as g(p)=1 (no update).
# ---------------------------------------------------------------------------


def test_nan_pvalue_at_t0_no_update():
    p = np.asarray([np.nan, 0.3, 0.7, 0.2])
    g = B.power(0.5)
    logM = M.run_single_feature(p, g)
    # logM[0] = 0 (M=1); step 1: * g(0.3); step 2: * g(0.7); step 3: * g(0.2).
    gp = g(p[1:])
    expected = np.concatenate([[1.0], np.cumprod(gp)])
    np.testing.assert_allclose(np.exp(logM), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# (3) Log-space stability: no overflow / NaN for T = 1000.
# ---------------------------------------------------------------------------


def test_log_space_stable_for_long_sequence():
    rng = np.random.default_rng(12345)
    T = 1000
    # Bias p towards small values so g(p) is frequently large (>> 1).
    # This is the worst case for overflow.
    p = rng.beta(a=0.2, b=1.0, size=T)
    g = B.power(0.3)

    logM = M.run_single_feature(p, g)
    assert logM.shape == (T,)
    assert np.all(np.isfinite(logM))
    # logM must obey the underflow clamp.
    assert np.all(logM >= M.LOG_UNDERFLOW_CLAMP - 1e-9)


# ---------------------------------------------------------------------------
# (4) update_traditional agrees with the recurrence element by element.
# ---------------------------------------------------------------------------


def test_update_traditional_matches_single_step():
    rng = np.random.default_rng(0)
    K = 6
    p = rng.uniform(0.1, 0.9, size=K)
    g = B.power(0.5)
    logM0 = np.zeros(K)
    logM1 = M.update_traditional(logM0, p, g)
    expected = np.log(g(p))
    np.testing.assert_allclose(logM1, expected, rtol=1e-12)


def test_update_traditional_shape_mismatch_raises():
    logM = np.zeros(3)
    p = np.zeros(4) + 0.5
    with pytest.raises(ValueError, match="shape mismatch"):
        M.update_traditional(logM, p, B.power(0.5))


# ---------------------------------------------------------------------------
# (5) sum_of_martingales is logsumexp of logM (Corollary 1).
# ---------------------------------------------------------------------------


def test_sum_of_martingales_is_logsumexp():
    rng = np.random.default_rng(0)
    T, K = 15, 4
    logM = rng.standard_normal((T, K))  # arbitrary values
    sum_log = M.sum_of_martingales(logM)
    # Compare to direct logsumexp-by-row.
    ref = np.log(np.exp(logM).sum(axis=1))
    np.testing.assert_allclose(sum_log, ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Reset-on-detection logic lives inline in HorizonDetector.run_on_features
# (see tests/test_detector_smoke.py for coverage).
# ---------------------------------------------------------------------------
