"""Feature attribution via the Martingale–Shapley equivalence (Paper §III-D).

Ho et al. [12] prove that each feature's martingale value at detection time
IS its Shapley value for the additive game v(S) = Σ_{k∈S} M_t^(k), so the
2^K coalition enumeration collapses to O(K). The four Shapley axioms
(Efficiency, Symmetry, Dummy, Additivity) hold "for free" via:

    ψ_k(t) = M_t^(k) / M_t^A × 100%        (Eq 7)

Computed in log-space (`exp(logM^(k) - logsumexp(logM))`) to avoid overflow
at high scores. At t=0 or just after a reset, all logM = 0 and we return
equal shares 100/K — the correct limit.
"""

from __future__ import annotations

import numpy as np


def shapley_values(logM_per_feature: np.ndarray, t: int, names: list[str]) -> dict[str, float]:
    """ψ_k(t) as a dict {feature_name: percent}.

    logM_per_feature: (T, K) array of per-feature log-martingale values.
    t: time index.
    names: (K,) feature names, same ordering as logM columns.

    Returns: {name: percent} with Σ = 100.0 (up to fp).
    """
    lm = logM_per_feature[t]
    m = lm.max()
    if not np.isfinite(m):
        return {n: 100.0 / len(names) for n in names}
    w = np.exp(lm - m)
    pct = 100.0 * w / w.sum()
    return dict(zip(names, pct.tolist()))


def dominant_driver(logM_per_feature: np.ndarray, t: int, names: list[str]) -> tuple[str, float]:
    """Return the single feature with maximum ψ_k(t) and its percent.

    Matches the "Driver" column in paper Table IV.
    """
    attr = shapley_values(logM_per_feature, t, names)
    name = max(attr, key=attr.get)  # type: ignore[arg-type]
    return name, attr[name]


def attribution_trajectory(logM_per_feature: np.ndarray, names: list[str]) -> np.ndarray:
    """(T, K) array of ψ_k(t) values over all t. Rows sum to 100.

    Useful for plots showing how the dominant-feature attribution evolves
    through time (e.g., for Figure 2 supplementary panels).
    """
    T, K = logM_per_feature.shape
    out = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        lm = logM_per_feature[t]
        m = lm.max()
        if not np.isfinite(m):
            out[t] = 100.0 / K
            continue
        w = np.exp(lm - m)
        out[t] = 100.0 * w / w.sum()
    return out
