"""Conformal p-value uniformity under H_0 (Ali & Ho, ICDM 2025, Theorem 1 proof).

WHAT is tested
--------------
The smoothed conformal p-value (Vovk 2005, used in hmd.conformal for Def 3)
is constructed so that

    p_t | F_{t-1}  ~  Unif(0, 1)   under exchangeability.

Exchangeability holds when {X_s} is i.i.d., which is the regime we simulate.
We feed i.i.d. N(0,1) samples through

    nonconformity (Def 2) -> conformal_pvalue (Def 3, Vovk-smoothed)

and test H_0: "the output is Unif(0,1)" via a Kolmogorov-Smirnov test.

WHY this matters
----------------
Uniformity is THE load-bearing assumption for Theorem 1 (martingale property
of M_t = prod g(p_s)). Any systematic deviation from uniformity would bias
E[g(p_t)] away from 1, making M_t not a martingale and Ville's inequality
vacuous for the false-alarm bound.

A single KS test is noisy; we sweep seeds and assert both
  (a) no seed rejects uniformity at alpha = 0.05
  (b) the MEAN KS p-value over seeds exceeds 0.1 (very conservative; a truly
      uniform distribution gives mean p ~ 0.5).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as ss

from hmd import conformal as C


SEEDS = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19]  # 10 seeds
T_PER_SEED = 200


def _ks_pvalue_scalar(seed: int) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T_PER_SEED)
    S = C.nonconformity(x)
    p = C.conformal_pvalue(S, rng=rng)
    # Drop t=0 which is NaN by construction (Def 3 convention).
    # Also drop t=1 where with a single prior sample the sampling
    # distribution is discrete and KS is conservative.
    p_clean = p[2:]
    assert np.all(np.isfinite(p_clean))
    assert np.all((p_clean > 0.0) & (p_clean <= 1.0))
    return float(ss.kstest(p_clean, "uniform").pvalue)


def _ks_pvalue_multi(seed: int, K: int = 4) -> list[float]:
    """Return one KS p-value per column of conformal_pvalue_multi."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T_PER_SEED, K))
    S = C.nonconformity_multi(X)
    P = C.conformal_pvalue_multi(S, rng=rng)
    out: list[float] = []
    for k in range(K):
        col = P[2:, k]
        assert np.all(np.isfinite(col))
        out.append(float(ss.kstest(col, "uniform").pvalue))
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_scalar_pvalue_is_uniform_per_seed(seed):
    """For each seed, KS does not reject Unif(0,1) at alpha=0.05."""
    p_ks = _ks_pvalue_scalar(seed)
    assert p_ks > 0.05, (
        f"seed={seed}: KS rejected uniformity (KS p = {p_ks:.4f} <= 0.05)"
    )


def test_scalar_pvalue_mean_ks_pvalue_exceeds_threshold():
    """Across seeds, the mean KS p-value should be well above 0.1.

    For truly uniform samples KS p-values are themselves Unif(0,1), so the
    population mean is 0.5. Anything substantially below 0.1 indicates a
    systematic miscalibration, not statistical fluctuation.
    """
    ks_p = np.asarray([_ks_pvalue_scalar(s) for s in SEEDS])
    assert ks_p.mean() > 0.1, (
        f"mean KS p-value = {ks_p.mean():.4f} <= 0.1; p-values look non-uniform. "
        f"Per-seed: {ks_p.tolist()}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_multi_pvalue_is_uniform_per_seed(seed):
    per_col = _ks_pvalue_multi(seed, K=4)
    # At least the majority of columns (3/4) must pass at alpha=0.05.
    # Individual columns can fail by chance since we run 10 seeds * 4 cols = 40
    # tests; at alpha=0.05 we expect ~2 failures. The majority-pass check is
    # a robust per-seed gate; the aggregate MEAN check below is the
    # authoritative assertion.
    n_pass = sum(p > 0.05 for p in per_col)
    assert n_pass >= 3, (
        f"seed={seed}: only {n_pass}/4 columns passed uniform KS; p-values={per_col}"
    )


def test_multi_pvalue_mean_ks_pvalue_exceeds_threshold():
    all_ks: list[float] = []
    for s in SEEDS:
        all_ks.extend(_ks_pvalue_multi(s, K=4))
    arr = np.asarray(all_ks)
    assert arr.mean() > 0.1, (
        f"pooled mean KS p-value = {arr.mean():.4f} <= 0.1 across 40 columns"
    )


def test_pvalue_bounds():
    """Smoothed form guarantees p in (0, 1] strictly -- g(p) stays finite."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    S = C.nonconformity(x)
    p = C.conformal_pvalue(S, rng=rng)
    # Entries for t>=1 must be strictly positive.
    assert np.all(p[1:] > 0.0), "smoothed conformal p-value should never be 0"
    assert np.all(p[1:] <= 1.0)


def test_predictive_pvalue_bounded_below_by_inverse_t_plus_one():
    """Eq 10: p_{t,h} in [1/(t+1), 1].

    The +1 Laplace smoothing guarantees a strictly positive floor; this is
    why g(p) can never return infinity when fed a predictive p-value.
    """
    rng = np.random.default_rng(0)
    T = 100
    S_hist = rng.standard_normal(T)
    S_pred = rng.standard_normal(T)
    p = C.predictive_pvalue(S_hist, S_pred)
    assert p.shape == (T,)
    assert np.all(p > 0.0)
    assert np.all(p <= 1.0)
    # At t=0 pool is empty; p[0] == 1 by convention (denominator = 1).
    assert p[0] == 1.0
    # Lower bound: 1/(t+1)
    for t in range(T):
        assert p[t] >= 1.0 / (t + 1.0) - 1e-12
