"""Test-martingale recurrences (§III-B, §IV-A, Defs 6, 7, Cor 1, Algorithm 1).

Critical design choices
-----------------------
- **Log-space accumulation** (mandatory). M_t = ∏ g(p_i) overflows float64
  after ~700 steps; logM comparisons against log(λ) are exact.
- **Algorithm 1 interpretation of Def 7** (the paper has two inconsistent
  forms). Def 7 as printed reads M_{t,h}^(k) = M_{t-1}^(k) · g(p_{t,h}^(k))
  (horizon multiplies the *traditional* stream's previous value), but
  Algorithm 1 line 13 uses M_{t-1}^(k,h) (horizon's own previous). Only the
  latter gives the parallel-stream structure required by Thm 4's application
  of Ville's inequality to the horizon stream in isolation. We follow
  Algorithm 1.
- **Reset on detection** zeros both streams (Alg 1 lines 20-21) but KEEPS the
  S and X history. Keeping history gives more stable post-reset p-values at
  the cost of slower re-detection because new-regime scores compete against
  a pool that still contains pre-change values. Paper does this implicitly.
- **Sum-combination across features** (Corollary 1): M^A = Σ_k M^(k). Only
  linear combination that preserves the martingale property without feature
  independence. In log-space this is `logsumexp` via scipy.
- **Horizon stream starts at t ≥ w** (Alg 1 line 9). Before the forecaster
  has w observations, no prediction is possible; the detector skips updates.
- **Underflow clamp at log(1e-300) ≈ -690** on logM so a pathological
  g(p) = 0 doesn't permanently kill a stream via log(0) = -inf.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "update_traditional",
    "run_single_feature",
    "run_multi_feature",
    "sum_of_martingales",
    "LOG_UNDERFLOW_CLAMP",
]


# Why this value: log(1e-300) ≈ -690.776. Lowest safe logM we allow. float64
# can represent exp down to ~-745, but we leave headroom so intermediate
# arithmetic doesn't underflow to -inf.
LOG_UNDERFLOW_CLAMP: float = float(np.log(1e-300))


def _log_g(p: np.ndarray, g: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """log(g(p)) with an underflow floor.

    Why: if g(p) is 0 exactly (pathological), np.log returns -inf and poisons
    the stream. Clamp at a finite, very negative floor before logging.
    """
    gp = np.asarray(g(p), dtype=np.float64)
    # Clamp AT LEAST 1e-300 before log; preserves ordering almost everywhere.
    gp = np.maximum(gp, 1e-300)
    return np.log(gp)


def update_traditional(
    logM: np.ndarray,
    p: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Elementwise log-martingale update: logM ← max(logM + log g(p), clamp).

    Parameters
    ----------
    logM : np.ndarray, shape (K,)
        Current log-martingale values (per feature). Init to zeros (M=1).
    p : np.ndarray, shape (K,)
        p-values at the current timestep.
    g : callable
        Vectorized betting function p ↦ g(p).

    Returns
    -------
    np.ndarray, shape (K,) — updated log-martingale.
    """
    logM = np.asarray(logM, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if logM.shape != p.shape:
        raise ValueError(
            f"shape mismatch: logM {logM.shape} vs p {p.shape}"
        )
    out = logM + _log_g(p, g)
    return np.maximum(out, LOG_UNDERFLOW_CLAMP)


def run_single_feature(
    p: np.ndarray, g: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply recurrence M_t = M_{t-1} · g(p_t) over a 1-D p-sequence.

    Useful for unit-testing the recurrence without the detector around it.

    Parameters
    ----------
    p : np.ndarray, shape (T,)
    g : callable

    Returns
    -------
    logM : np.ndarray, shape (T,)
        logM[0] = 0 (M=1 initial). logM[t] = logM[t-1] + log g(p[t]) clamped.
        p[0] may be NaN (Def 3 convention); we treat that as g(p)=1 ⇒ no
        update at t=0, consistent with the M_0=1 initial condition.
    """
    p = np.asarray(p, dtype=np.float64)
    T = p.shape[0]
    logM = np.zeros(T, dtype=np.float64)
    if T == 0:
        return logM
    # t=0: leave at 0 (M=1). For t ≥ 1, cumulate log g(p).
    # Handle NaN p[0] gracefully — log-contribution is 0 (no update).
    lg = _log_g(np.where(np.isnan(p), 1.0, p), g)
    lg[np.isnan(p)] = 0.0  # explicit: NaN input → no update
    # Cumulative sum starting from 0 (the M_0=1 log-value).
    logM[0] = 0.0
    for t in range(1, T):
        logM[t] = max(logM[t - 1] + lg[t], LOG_UNDERFLOW_CLAMP)
    return logM


def run_multi_feature(
    p: np.ndarray, g: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply the per-feature recurrence across K features.

    No coupling between features — each column is its own martingale.

    Parameters
    ----------
    p : np.ndarray, shape (T, K)
    g : callable

    Returns
    -------
    logM : np.ndarray, shape (T, K).
    """
    p = np.asarray(p, dtype=np.float64)
    T, K = p.shape
    logM = np.zeros((T, K), dtype=np.float64)
    if T == 0:
        return logM
    # Compute log g(p) elementwise, treating NaN inputs as 1.0 (no update).
    p_safe = np.where(np.isnan(p), 1.0, p)
    lg = _log_g(p_safe, g)
    lg[np.isnan(p)] = 0.0
    # Row-wise running cumulative sum starting from zero; clamp per row.
    for t in range(1, T):
        row = logM[t - 1] + lg[t]
        logM[t] = np.maximum(row, LOG_UNDERFLOW_CLAMP)
    return logM


def sum_of_martingales(logM: np.ndarray) -> np.ndarray:
    """Corollary 1: log-space sum across features.

        M^A_t = Σ_k M^(k)_t   ⟹   logM^A_t = logsumexp_k(logM^(k)_t)

    Uses scipy.special.logsumexp for numerical stability (never materializes
    exp(logM) — which could overflow).

    Parameters
    ----------
    logM : np.ndarray, shape (T, K)

    Returns
    -------
    logM_sum : np.ndarray, shape (T,)
    """
    logM = np.asarray(logM, dtype=np.float64)
    if logM.ndim != 2:
        raise ValueError(f"expected shape (T, K); got {logM.shape}")
    return logsumexp(logM, axis=1)


