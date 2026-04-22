"""Non-conformity scores and conformal p-values (Paper §III-B, Defs 2-3, Eq 10).

Critical design choices
-----------------------
- **Per-feature scalar path (default).** Algorithm 1 loops k = 1..K and keeps
  separate S^(k), p^(k), M^(k) per feature. `nonconformity_joint` provides the
  joint-vector variant (one scalar via Mahalanobis / Euclidean / Cosine /
  Chebyshev distance) for Table III's distance-metric sweep.
- **K=1 K-means = running mean.** Only hyperparameter-free choice for the
  "robust location estimator" in Def 2.
- **Vovk-smoothed Def 3 (NOT as printed).** The paper's printed form

      p_t = (#{s<t : S_s > S_t} + θ_t · #{s<t : S_s = S_t}) / t

  gives p_t = 0 when S_t is a unique max (prob 1/t under H₀), driving
  g(p_t) → ∞ and breaking Thm 1. The canonical Vovk (2005) form, used here,
  includes the test sample in both numerator (ties) and denominator (pool):

      p_t = (greater + θ_t · (1 + equal)) / (t + 1)

  Monte-Carlo: the as-printed form gives P(sup M ≥ 50) ≈ 0.977 vs the 1/λ =
  0.02 Ville bound; the Vovk form gives ≈ 0.002 ≪ 0.02. Our implementation
  follows Vovk; paper should be revised.
- **Eq 10 (predictive p-value) differs deliberately from Def 3.** It uses '≥'
  and '+1/(t+1)' Laplace smoothing because a forecast S_{t,h} need not be
  exchangeable with the historical pool; smoothing bounds p ≥ 1/(t+1) > 0.
- **Leak-free semantics.** C_t uses only X_{<t}; the empirical CDF uses only
  S_{<t}. Leak-proof by construction: p[0] = NaN, prefix-slice inside each step.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from hmd._backend import xp  # noqa: F401  (kept for backend symmetry)

__all__ = [
    "running_centroid",
    "nonconformity",
    "conformal_pvalue",
    "predictive_pvalue",
    "nonconformity_multi",
    "conformal_pvalue_multi",
    "predictive_pvalue_multi",
    "nonconformity_joint",
    "smoothed_pvalue_step",
    "predictive_pvalue_step",
]


# ---------------------------------------------------------------------------
# Single-step primitives — shared by batch functions below and the streaming
# inner loop in hmd/detector.py so there is exactly one source of truth for
# the p-value formula.
# ---------------------------------------------------------------------------


def smoothed_pvalue_step(
    S_t: float, prior: np.ndarray, theta: float
) -> float:
    """Canonical Vovk-smoothed conformal p-value for a single step.

        p = (greater + θ · (1 + equal)) / (n + 1)

    where ``prior`` is the length-n array of prior non-conformity scores,
    ``S_t`` is the current one, and ``θ ∈ (0, 1)`` is the tie-break draw.
    Returns 1.0 if prior is empty (undefined; return max conservative value).
    """
    n = prior.size
    if n == 0:
        return 1.0
    greater = int(np.sum(prior > S_t))
    equal = int(np.sum(prior == S_t))
    return (greater + theta * (1 + equal)) / (n + 1)


def predictive_pvalue_step(S_pred: float, prior: np.ndarray) -> float:
    """Eq 10 predictive p-value for a single step.

        p = (|{s : prior_s ≥ S_pred}| + 1) / (n + 1)

    Laplace-smoothed (+1 / (n+1)): required because the forecast S_pred need
    not be exchangeable with prior and can exceed every historical score.
    """
    n = prior.size
    ge = int(np.sum(prior >= S_pred))
    return (ge + 1) / (n + 1)


# ---------------------------------------------------------------------------
# Per-feature scalar path (default — matches Algorithm 1 lines 4-6)
# ---------------------------------------------------------------------------


def running_centroid(x: np.ndarray) -> np.ndarray:
    """Running K=1 K-means centroid on a scalar sequence.

    C[t] = mean(x[0], ..., x[t-1]) for t ≥ 1; C[0] = NaN (undefined).

    Why leak-free: Def 2 requires C_t fitted on {X_s}_{s=1}^{t-1}. We build the
    prefix-mean via cumulative sum so no index t can ever see x[t].

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float64)
    C = np.empty(T, dtype=np.float64)
    C[0] = np.nan  # no history at t=0
    if T > 1:
        # cumulative sum of x[0..t-1], divided by t for t=1..T-1
        csum = np.cumsum(x[:-1])  # csum[i] = sum x[0..i], length T-1
        denom = np.arange(1, T, dtype=np.float64)  # 1..T-1
        C[1:] = csum / denom
    return C


def nonconformity(x: np.ndarray) -> np.ndarray:
    """Def 2: S_t = |x_t - C_t| on a scalar sequence.

    S[0] = 0 by convention (no history ⇒ deviation undefined; 0 is the natural
    identity for subsequent cumulative stats and yields a uniform p-value).

    Parameters
    ----------
    x : np.ndarray, shape (T,)

    Returns
    -------
    S : np.ndarray, shape (T,)
    """
    x = np.asarray(x, dtype=np.float64)
    C = running_centroid(x)
    S = np.empty_like(x)
    S[0] = 0.0
    if x.shape[0] > 1:
        S[1:] = np.abs(x[1:] - C[1:])
    return S


def conformal_pvalue(
    S: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Def 3 — Vovk-smoothed conformal p-value, batch form.

        p_t = (#{s < t : S_s > S_t} + θ · (1 + #{s < t : S_s = S_t})) / (t + 1)

    Both the `(1 + equal)` and `(t + 1)` include the test sample; see module
    header for why this differs from Def 3 as printed. p[0] = NaN.

    Parameters
    ----------
    S : np.ndarray of shape (T,) — the non-conformity sequence.
    rng : np.random.Generator, optional — seeded for tie-break reproducibility.

    Returns
    -------
    p : np.ndarray of shape (T,) — p-values in (0, 1].
    """
    S = np.asarray(S, dtype=np.float64)
    T = S.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    p = np.empty(T, dtype=np.float64)
    if T == 0:
        return p
    p[0] = np.nan
    # Naive O(T^2): T is ~200; this runs once per feature per experiment.
    for t in range(1, T):
        p[t] = smoothed_pvalue_step(S[t], S[:t], rng.uniform(0.0, 1.0))
    return p


def predictive_pvalue(S_hist: np.ndarray, S_pred: np.ndarray) -> np.ndarray:
    """Eq 10: predictive p-value with Laplace smoothing.

        p_{t,h} = (|{s ∈ [1..t] : S_hist_s ≥ S_pred_t}| + 1) / (t + 1)

    Uses '≥' (not '>') and +1 / (t+1) smoothing — required so that p never
    hits exactly 0 (which would kill the martingale via g(0) = ∞).

    Parameters
    ----------
    S_hist : np.ndarray, shape (T,)
        Historical non-conformity scores (scalar sequence).
    S_pred : np.ndarray, shape (T,)
        Predictive non-conformity scores. S_pred[t] = ||X̂_{t+h} - C_t||.

    Returns
    -------
    p : np.ndarray, shape (T,)
        At t=0 the denominator is 1 ⇒ p[0] = 1 (uninformative).
    """
    S_hist = np.asarray(S_hist, dtype=np.float64)
    S_pred = np.asarray(S_pred, dtype=np.float64)
    T = S_hist.shape[0]
    if T != S_pred.shape[0]:
        raise ValueError("S_hist and S_pred must have the same length")
    p = np.empty(T, dtype=np.float64)
    # Paper index set {s ∈ [1..t]} (1-indexed, t elements) ⇒ Python S_hist[:t]
    # (0-indexed, t elements). Denominator is (t + 1). Max of count is t, so
    # p_{t,h} ∈ [1/(t+1), 1]. At t=0 the pool is empty, count=0, p[0] = 1.
    for t in range(T):
        prior = S_hist[:t]  # length t
        count = np.sum(prior >= S_pred[t])
        p[t] = (count + 1.0) / (t + 1.0)
    return p


# ---------------------------------------------------------------------------
# Per-feature vectorized helpers (run all K features in one call)
# ---------------------------------------------------------------------------


def nonconformity_multi(X: np.ndarray) -> np.ndarray:
    """Apply Def 2 column-wise to a multivariate sequence.

    Parameters
    ----------
    X : np.ndarray, shape (T, K)

    Returns
    -------
    S : np.ndarray, shape (T, K)
    """
    X = np.asarray(X, dtype=np.float64)
    T, K = X.shape
    S = np.empty_like(X)
    S[0, :] = 0.0
    if T <= 1:
        return S
    # Running mean per column, vectorized:
    #   C[t, k] = mean(X[:t, k]) for t ≥ 1.
    csum = np.cumsum(X[:-1, :], axis=0)  # (T-1, K)
    denom = np.arange(1, T, dtype=np.float64)[:, None]  # (T-1, 1)
    C_tail = csum / denom  # (T-1, K), corresponds to t = 1..T-1
    S[1:, :] = np.abs(X[1:, :] - C_tail)
    return S


def conformal_pvalue_multi(
    S: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply Def 3 column-wise.

    Parameters
    ----------
    S : np.ndarray, shape (T, K)
    rng : np.random.Generator, optional

    Returns
    -------
    p : np.ndarray, shape (T, K)
    """
    S = np.asarray(S, dtype=np.float64)
    T, K = S.shape
    if rng is None:
        rng = np.random.default_rng()
    p = np.empty_like(S)
    if T == 0:
        return p
    p[0, :] = np.nan
    # Vectorized form of smoothed_pvalue_step across the K feature columns.
    for t in range(1, T):
        prior = S[:t, :]  # (t, K)
        curr = S[t, :]  # (K,)
        greater = np.sum(prior > curr[None, :], axis=0)  # (K,)
        equal = np.sum(prior == curr[None, :], axis=0)  # (K,)
        theta = rng.uniform(0.0, 1.0, size=K)
        p[t, :] = (greater + theta * (1 + equal)) / (t + 1)
    return p


def predictive_pvalue_multi(S_hist: np.ndarray, S_pred: np.ndarray) -> np.ndarray:
    """Apply Eq 10 column-wise.

    Parameters
    ----------
    S_hist : np.ndarray, shape (T, K)
    S_pred : np.ndarray, shape (T, K)

    Returns
    -------
    p : np.ndarray, shape (T, K)
    """
    S_hist = np.asarray(S_hist, dtype=np.float64)
    S_pred = np.asarray(S_pred, dtype=np.float64)
    if S_hist.shape != S_pred.shape:
        raise ValueError("S_hist and S_pred must share shape (T, K)")
    T, K = S_hist.shape
    p = np.empty_like(S_hist)
    # See predictive_pvalue() for index-set rationale: S_hist[:t], length t.
    for t in range(T):
        prior = S_hist[:t, :]  # (t, K)
        count = np.sum(prior >= S_pred[t, :][None, :], axis=0)  # (K,)
        p[t, :] = (count + 1.0) / (t + 1.0)
    return p


# ---------------------------------------------------------------------------
# Joint-vector path (Table III distance sweep)
# ---------------------------------------------------------------------------


def nonconformity_joint(X: np.ndarray, metric: str = "mahalanobis") -> np.ndarray:
    """S_t = distance(X_t, C_t) with C_t the running K=1 centroid in ℝ^K.

    The four metrics in Table III have distinct inductive biases:

    - euclidean     : Σ_k (x_k - c_k)^2, unit-agnostic equal weighting. Over-
                      weights features with large natural scale (e.g., mean
                      degree dominates clustering coeff). Pre-standardize X
                      upstream if this matters.
    - mahalanobis   : (x - c)^T Σ^{-1} (x - c), de-correlates and re-scales
                      using the empirical covariance Σ from {X_s}_{s<t}.
                      Captures coordinated moves across features. Paper's best
                      metric for structural change. Regularized with λ_reg·I
                      for small t where Σ is singular.
    - cosine        : 1 − <x - c, 0> / (||x||·||c||): angular distance. Scale-
                      invariant — useful when absolute magnitudes drift but
                      *directions* are stable. Weaker in practice because most
                      graph changes ARE magnitude shifts.
    - chebyshev     : max_k |x_k - c_k|. Sensitive to the single most-deviated
                      feature. Noisy: one flaky feature dominates; does not
                      pool evidence across features.

    Parameters
    ----------
    X : np.ndarray, shape (T, K)
    metric : {"euclidean", "mahalanobis", "cosine", "chebyshev"}

    Returns
    -------
    S : np.ndarray, shape (T,)
    """
    X = np.asarray(X, dtype=np.float64)
    T, K = X.shape
    S = np.empty(T, dtype=np.float64)
    S[0] = 0.0
    if T <= 1:
        return S
    lam_reg = 1e-6  # Mahalanobis ridge

    # Prefix sums so centroid at time t is csum[t-1]/t.
    csum = np.cumsum(X, axis=0)

    for t in range(1, T):
        C_t = csum[t - 1] / t  # mean of X[:t]
        diff = X[t] - C_t

        if metric == "euclidean":
            S[t] = float(np.linalg.norm(diff))

        elif metric == "chebyshev":
            S[t] = float(np.max(np.abs(diff)))

        elif metric == "cosine":
            # cosine distance = 1 - <a, b>/(||a|| ||b||). Compare X_t to C_t;
            # if either is zero-vector, fall back to Euclidean (cosine undefined).
            nx_ = float(np.linalg.norm(X[t]))
            nc_ = float(np.linalg.norm(C_t))
            if nx_ == 0.0 or nc_ == 0.0:
                S[t] = float(np.linalg.norm(diff))
            else:
                S[t] = 1.0 - float(X[t] @ C_t) / (nx_ * nc_)

        elif metric == "mahalanobis":
            if t < 2:
                # Covariance undefined with 1 sample; fall back to Euclidean.
                S[t] = float(np.linalg.norm(diff))
            else:
                # Sample covariance of X[:t] with ridge regularization.
                # NOTE: use rowvar=False so Σ is (K, K), not (t, t).
                Sigma = np.cov(X[:t], rowvar=False)
                # np.cov on K=1 returns a 0-d array; guard that.
                if Sigma.ndim == 0:
                    Sigma = Sigma.reshape(1, 1)
                Sigma = Sigma + lam_reg * np.eye(K)
                try:
                    # scipy.linalg.solve with assume_a='pos' uses Cholesky —
                    # more stable than inv() @ diff. Higham §10.
                    y = sla.solve(Sigma, diff, assume_a="pos")
                    S[t] = float(np.sqrt(max(0.0, float(diff @ y))))
                except sla.LinAlgError:
                    S[t] = float(np.linalg.norm(diff))
        else:
            raise ValueError(
                f"Unknown metric {metric!r}; expected one of "
                "'euclidean', 'mahalanobis', 'cosine', 'chebyshev'."
            )
    return S
