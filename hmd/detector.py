"""Horizon Martingale Detector — Algorithm 1 glue.

Orchestrates features → conformal → betting → martingale → forecaster per
Algorithm 1, page 6 of Ali & Ho (ICDM 2025), and returns a structured result
ready for plotting, attribution, and metric evaluation.

Critical design choices
-----------------------
- **One class, config-driven.** Every variant is a `DetectorConfig` knob
  (`enable_horizon`, `detection_mode`) or a swapped dependency
  (`betting=`, `forecaster=`). No detector hierarchy.
- **Online simulation on offline arrays.** At time t we slice `X[:t+1]` only —
  no lookahead. Full logM trajectories are returned for post-hoc plotting
  without violating the online guarantee.
- **Both streams reset on any detection** (Alg 1 lines 20-21): after a confirmed
  shift, both streams' calibration is stale.
- **History NOT reset.** Running centroid and empirical CDF keep growing.
  Trade-off: stable p-values at the cost of slower re-detection.
- **Startup period** (default 20) — NOT in paper. The running-centroid
  non-conformity S_t is not exchangeable across t for small t (centroid SE
  differs), which breaks rank-uniformity of p-values and causes false-alarm
  inflation on high-variance feature generators (BA trees). Matches standard
  SPC burn-in practice; aligns with paper's `w_cal = 0.1 T = 20`.
- **Detection mode**: `per_feature` (default, Algorithm 1 strict) sums K
  martingales via Corollary 1; `joint` (Table III's distance-metric sweep,
  paper-implicit for Table IV) runs one martingale on a scalar Mahalanobis
  distance. Per-feature shadow streams still run under joint mode so
  attribution keeps working.
- **Cooldown = 0 (paper-strict).** Eval window Δ=20 deduplicates consecutive
  detections naturally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp
from tqdm import tqdm

from hmd import betting as _betting
from hmd import conformal as _conformal
from hmd import features as _features
from hmd import forecaster as _forecaster
from hmd import martingale as _martingale
from hmd.martingale import LOG_UNDERFLOW_CLAMP as _LOG_M_FLOOR

if TYPE_CHECKING:
    import networkx as nx
    from hmd.features import FeatureSpec
    from hmd.forecaster import Predictor


@dataclass
class DetectorConfig:
    """Configuration for HorizonDetector. Every default justified below.

    threshold (λ): 50.0
        Paper's optimal for synthetic networks (Table III). Ville's inequality
        gives P(false alarm) ≤ 1/λ = 0.02. For MIT Reality use 20.0 (Table III).

    horizon (h): 5
        Paper's optimal (Section VI). h=1 gives minimal predictive lead, h>5
        accumulates forecast error. h=5 is the sweet spot on their scenarios.

    history_size (w): 20
        Width of the EWMA sliding window. Paper uses w_cal = 0.1T for
        calibration windows; with T=200 this is 20. Also sets the minimum t
        before the HORIZON stream begins (Algorithm 1 line 9).

    betting: default mixture power betting
        Paper Table III: mixture power + Mahalanobis + λ=50 is optimal.

    forecaster: default EWMA(α=0.5)
        Paper §IV-B(a). Stable under H₀ (preserves calibration). Swap here to
        plug in ARIMA / GNN / any Predictor protocol implementation.

    features: default 8-feature set (§III-A)
        Permutation-invariant per-graph scalars: mean_degree, mean_clustering,
        mean_betweenness, mean_closeness, mean_eigenvector, algebraic_connectivity,
        spectral_gap, density.

    enable_traditional / enable_horizon: both True
        Ablation knobs. Paper's "Martingale" baseline = {traditional=True, horizon=False}.
        Pure "Horizon" ablation = {traditional=False, horizon=True}.

    cooldown: 0
        Steps to ignore further detections after a trigger. 0 = paper-strict.

    rng_seed: 0
        For p-value tie-breaking (Def 3 θ_t ∼ U(0,1)). Fixed seed = reproducible.
    """

    threshold: float = 50.0
    horizon: int = 5
    history_size: int = 20
    betting: _betting.Betting | None = None
    forecaster: "Predictor | None" = None
    features: "Sequence[FeatureSpec] | None" = None
    enable_traditional: bool = True
    enable_horizon: bool = True
    cooldown: int = 20
    """Steps to suppress further detections after a trigger. Default 20 matches
    the evaluation window Δ=20, so consecutive same-change detections are not
    double-counted as FPs. Paper Alg 1 specifies no cooldown (cooldown=0) but
    reports results with Δ=20 eval window that naturally deduplicates. For
    strict Algorithm 1 replication set cooldown=0; for MIT Reality tune
    lower/higher based on event frequency."""
    rng_seed: int = 0
    show_progress: bool = False
    startup_period: int = 20
    """Burn-in length. M accumulates nothing and detections are blocked for
    t < startup_period. At t = startup_period both streams start fresh
    (logM=0) using S[:startup_period] as the reference for subsequent p-values."""

    detection_mode: str = "per_feature"
    """'per_feature' (Algorithm 1, default) runs K per-feature martingales
    summed via Corollary 1. 'joint' uses hmd.conformal.nonconformity_joint
    (with joint_distance) for a single scalar per step; martingale dim K=1."""

    joint_distance: str = "mahalanobis"
    """Only used when detection_mode='joint'. One of
    {'euclidean', 'mahalanobis', 'cosine', 'chebyshev'}. Paper's best overall
    config is 'mahalanobis' (Table III)."""

    normalize_features: bool = False
    """If True, z-score each feature column using running mean/std computed
    over X[:startup_period] (the calibration window). Under H₀ this makes
    per-feature nonconformity ~ N(0,1) across features, preventing
    high-variance features (e.g., BA trees' spectral gap) from dominating
    the sum martingale. Rank-based p-values remain valid either way, but
    normalization improves finite-sample calibration uniformity. Paper
    does not explicitly specify; set True for better empirical numbers."""

    horizon_weights: "tuple[float, ...] | None" = None
    """Vovk-Wang mixture weights over horizons h=1..H, where H = horizon.

    None (default): single-horizon mode — use only h = self.horizon. Matches
    paper Def 7 exactly.

    A tuple of length `horizon`: run H independent per-feature horizon
    streams (one per h ∈ {1,..,H}) and combine via the convex mixture
    M_mix_t^(k) = Σ_h w_h · M_t^(k,h). By linearity this mixture is still a
    test martingale (Vovk-Wang 2021 Prop 2.1) — Ville's 1/λ bound is
    preserved EXACTLY. This is the mathematically correct generalization of
    Def 7 to multi-horizon evidence; the legacy geometric-mean-with-decay
    combining `∏_h g^{decay^h}` is NOT a valid martingale (SymPy-verified
    by our theorist; correlated p-values across h violate the independence
    assumption needed for the product form).

    Typical weights: decay-weighted `w_h ∝ 0.7^h` (legacy heuristic lifted
    into a mixture), uniform `(1/H,...,1/H)`, or peaked `(.1, .2, .4, .2, .1)`.
    Must sum to 1; enforced in __post_init__.

    Example: ``horizon_weights=(0.4, 0.3, 0.2, 0.1)`` with ``horizon=4``.
    """

    def __post_init__(self):
        if self.betting is None:
            object.__setattr__(self, "betting", _betting.default())
        if self.forecaster is None:
            object.__setattr__(self, "forecaster", _forecaster.default())
        if self.features is None:
            object.__setattr__(self, "features", _features.default_set())
        if not (self.enable_traditional or self.enable_horizon):
            raise ValueError("At least one of enable_traditional / enable_horizon must be True.")
        if self.horizon < 1:
            raise ValueError("horizon must be ≥ 1 (one step ahead).")
        if self.history_size < 2:
            raise ValueError("history_size must be ≥ 2 for EWMA.")
        if self.startup_period < 0:
            raise ValueError("startup_period must be ≥ 0.")
        if self.detection_mode not in ("per_feature", "joint"):
            raise ValueError("detection_mode must be 'per_feature' or 'joint'.")
        if self.joint_distance not in ("euclidean", "mahalanobis", "cosine", "chebyshev"):
            raise ValueError("joint_distance must be one of euclidean/mahalanobis/cosine/chebyshev.")
        if self.horizon_weights is not None:
            w = np.asarray(self.horizon_weights, dtype=np.float64)
            if w.ndim != 1 or w.size != self.horizon:
                raise ValueError(
                    f"horizon_weights length {len(self.horizon_weights)} must equal horizon={self.horizon}"
                )
            if (w < 0).any():
                raise ValueError("horizon_weights must all be non-negative.")
            if not np.isclose(float(w.sum()), 1.0, atol=1e-8):
                raise ValueError(
                    f"horizon_weights must sum to 1 (Vovk-Wang mixture); got {float(w.sum())}"
                )


@dataclass
class DetectionResult:
    """Structured detector output. Everything needed for metrics + plotting + attribution."""

    change_points: list[int]
    """Time indices τ at which |detection condition| first crossed after the
    previous reset. May contain 0, 1, or many entries."""

    logM_traditional: np.ndarray
    """(T,) summed log-martingale log Σ_k exp(logM^(k)_t). NaN before t=1."""

    logM_horizon: np.ndarray
    """(T,) summed log-horizon-martingale. NaN for t < history_size."""

    logM_per_feature: np.ndarray
    """(T, K) per-feature log-martingale for attribution."""

    logM_per_feature_horizon: np.ndarray
    """(T, K) per-feature log-horizon-martingale."""

    features: np.ndarray
    """(T, K) extracted feature sequence — useful for debugging and plots."""

    feature_names: list[str]

    pvalues: np.ndarray
    """(T, K) conformal p-values (Def 3)."""

    pvalues_horizon: np.ndarray
    """(T, K) predictive p-values (Eq 10). NaN for t < history_size."""

    config: DetectorConfig = field(repr=False)

    scenario: str | None = None
    true_change_points: list[int] | None = None

    # ---- helpers ----

    @property
    def M_traditional(self) -> np.ndarray:
        """M_t^A (not log). Use for threshold comparison on linear scale."""
        return np.exp(self.logM_traditional)

    @property
    def M_horizon(self) -> np.ndarray:
        return np.exp(self.logM_horizon)

    def attribution_at(self, t: int, stream: str = "traditional") -> dict[str, float]:
        """ψ_k(t) = M_t^(k) / M_t^A × 100% (paper §III-D, Eq 7).

        stream: 'traditional' or 'horizon'.
        Returns dict {feature_name: percent}. Percentages sum to ~100 (exact up to fp).
        """
        logM_per = self.logM_per_feature if stream == "traditional" else self.logM_per_feature_horizon
        lm = logM_per[t]
        # Normalize in log space to avoid overflow: pct = exp(logM^(k) - logsumexp(logM))
        m = lm.max()
        if not np.isfinite(m):
            return {name: 0.0 for name in self.feature_names}
        shifted = np.exp(lm - m)
        pct = 100.0 * shifted / shifted.sum()
        return dict(zip(self.feature_names, pct.tolist()))


class HorizonDetector:
    """Algorithm 1 (Ali & Ho, ICDM 2025, page 6).

    Usage
    -----
        >>> from hmd import HorizonDetector
        >>> from hmd.data.synthetic import sbm_community_merge
        >>> seq = sbm_community_merge(seed=0)
        >>> det = HorizonDetector()   # paper defaults
        >>> result = det.run(seq.graphs)
        >>> result.change_points
        [...]
        >>> result.attribution_at(result.change_points[0])
        {'mean_degree': 12.4, 'algebraic_connectivity': 34.1, ...}

    Ablations
    ---------
        Pure horizon:       HorizonDetector(enable_traditional=False)
        Pure traditional:   HorizonDetector(enable_horizon=False)
        Swap forecaster:    HorizonDetector(forecaster=MyGNNPredictor())
        Swap betting:       HorizonDetector(betting=betting.power(0.5))
    """

    def __init__(self, config: DetectorConfig | None = None, **kwargs):
        if config is None:
            config = DetectorConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either `config` OR keyword args, not both.")
        self.config = config

    # ---- Public API ----

    def run(
        self,
        graphs: "Sequence[nx.Graph]",
        *,
        true_change_points: list[int] | None = None,
        scenario: str | None = None,
    ) -> DetectionResult:
        """Extract features from graphs, then run Algorithm 1."""
        X = _features.extract_sequence(graphs, self.config.features, show_progress=self.config.show_progress)
        return self.run_on_features(X, true_change_points=true_change_points, scenario=scenario)

    def run_on_features(
        self,
        X: np.ndarray,
        *,
        true_change_points: list[int] | None = None,
        scenario: str | None = None,
    ) -> DetectionResult:
        """Run Algorithm 1 on a precomputed (T, K) feature matrix.

        Why this entry point exists: experiments (Table III/IV sweeps) precompute
        features once and then rerun the detector with many configs. Feature
        extraction dominates wallclock (O(n³) eigendecomp), so caching here
        gives ~K× speedup on sweeps.
        """
        cfg = self.config
        T, K = X.shape
        rng = np.random.default_rng(cfg.rng_seed)
        # Optional z-scoring using the calibration window's statistics.
        # We freeze μ, σ from X[:startup] so the normalization is constant
        # over the detection horizon (avoids self-adaptive wash-out of
        # post-change signal). If startup is too small for a stable σ,
        # we silently skip normalization.
        if cfg.normalize_features and cfg.startup_period >= 5:
            mu = X[: cfg.startup_period].mean(axis=0)
            sigma = X[: cfg.startup_period].std(axis=0, ddof=1)
            sigma = np.where(sigma < 1e-10, 1.0, sigma)  # avoid /0 on constant features
            X = (X - mu) / sigma
        g = cfg.betting
        w = cfg.history_size
        h = cfg.horizon
        lam = cfg.threshold
        log_lam = np.log(lam)
        startup = cfg.startup_period
        # In per-feature mode the martingale dimension D = K (one per feature).
        # In joint mode D = 1 (single scalar nonconformity per step); we keep
        # the K-wide per-feature SHADOW streams running in parallel for
        # attribution (Table IV "Driver" column).
        D = 1 if cfg.detection_mode == "joint" else K

        # Storage ------------------------------------------------------------
        logM_per_feature = np.full((T, K), np.nan)      # shadow per-feature (for attribution)
        logM_per_feature_h = np.full((T, K), np.nan)
        logM_sum_trad = np.full(T, np.nan)               # primary detection stream
        logM_sum_hrzn = np.full(T, np.nan)
        pvals_trad = np.full((T, K), np.nan)
        pvals_hrzn = np.full((T, K), np.nan)

        # Running state — per-feature accumulators always kept (for attribution).
        logM_trad_pf = np.zeros(K, dtype=np.float64)
        logM_hrzn_pf = np.zeros(K, dtype=np.float64)
        # Joint-mode scalar accumulators (D=1).
        logM_trad_joint = 0.0
        logM_hrzn_joint = 0.0

        # Multi-horizon (Vovk-Wang mixture) accumulators. When horizon_weights
        # is set, we maintain H independent per-feature horizon streams (one per
        # h=1..H) and aggregate via the VW mixture: logM_hrzn_pf[k] =
        # logsumexp_h(logM_hrzn_per_h[h, k] + log w_h). Each per-h stream is a
        # valid test martingale by Thm 3, and the convex mixture preserves the
        # martingale property by linearity — Ville's 1/λ bound holds exactly.
        using_mixture = cfg.horizon_weights is not None
        if using_mixture:
            H_horizons = int(cfg.horizon)
            # log(0) = -inf is mathematically correct for zero-weight horizons
            # (that stream contributes nothing to the mixture); suppress the
            # "divide by zero" warning since it's intentional.
            with np.errstate(divide="ignore"):
                log_hw = np.log(np.asarray(cfg.horizon_weights, dtype=np.float64))
            logM_hrzn_per_h = np.zeros((H_horizons, K), dtype=np.float64)
        else:
            H_horizons = 0
            log_hw = None
            logM_hrzn_per_h = None

        # Nonconformity history buffers.
        # S_hist_pf[k][:t] → scalar S values for feature k (per-feature path).
        # S_hist_joint[:t]  → scalar joint nonconformity (joint path).
        S_hist_pf: list[list[float]] = [[] for _ in range(K)]
        S_hist_joint: list[float] = []

        change_points: list[int] = []
        last_detection = -cfg.cooldown - 1

        iter_range = range(T)
        if cfg.show_progress:
            iter_range = tqdm(iter_range, desc="Algorithm 1", total=T, leave=False)

        for t in iter_range:
            x_t = X[t]  # (K,)

            # Startup gate: the running centroid C_t = mean(X[:t]) needs a
            # stabilization window before |x_t - C_t| is a trustworthy
            # non-conformity score. For Mahalanobis joint mode the sample
            # covariance requires at least K+1 samples to be invertible. We
            # skip ALL stream work for t < startup and begin accumulating
            # the p-value reference history only from t ≥ startup so the
            # empirical CDF is not poisoned by cold-start outliers.
            if t < startup:
                logM_per_feature[t] = logM_trad_pf
                logM_per_feature_h[t] = logM_hrzn_pf
                continue

            # Non-conformity for this step.
            C_t = X[:t].mean(axis=0)
            S_t_pf = np.abs(x_t - C_t)
            if cfg.detection_mode == "joint":
                S_joint_full = _conformal.nonconformity_joint(X[: t + 1], metric=cfg.joint_distance)
                S_t_joint = float(S_joint_full[-1])

            # --- Traditional p-values + martingale updates ---
            # Per-feature (always run — needed for attribution even in joint mode).
            p_t_pf = _pvalues_vectorized(S_t_pf, S_hist_pf, rng)  # (K,)
            pvals_trad[t] = p_t_pf
            if cfg.enable_traditional:
                logM_trad_pf = _martingale.update_traditional(logM_trad_pf, p_t_pf, g)

            # Joint (scalar) — only if joint mode is active.
            if cfg.detection_mode == "joint":
                p_t_joint = _scalar_conformal_pvalue(S_t_joint, S_hist_joint, rng)
                if cfg.enable_traditional:
                    gp = g(np.asarray([p_t_joint]))[0]
                    logM_trad_joint = max(_LOG_M_FLOOR, logM_trad_joint + np.log(max(gp, 1e-300)))

            # --- Horizon stream ---
            if cfg.enable_horizon and t >= w:
                hist_window = X[t - w + 1 : t + 1]
                C_t_full = X[: t + 1].mean(axis=0)

                if using_mixture:
                    # Run H independent per-h horizon streams, then VW-mix.
                    # Each h gets its own forecast → own p-value → own martingale.
                    last_p_h_pf = np.empty(K)
                    for h_idx in range(H_horizons):
                        h_step = h_idx + 1  # horizons are 1-indexed (h=1..H)
                        X_hat_h = cfg.forecaster.predict_multi(hist_window, horizon=h_step)
                        S_pred_h = np.abs(X_hat_h - C_t_full)
                        p_h_pf = np.empty(K)
                        for k in range(K):
                            S_prev = np.asarray(S_hist_pf[k], dtype=np.float64)
                            p_h_pf[k] = _conformal.predictive_pvalue_step(float(S_pred_h[k]), S_prev)
                        logM_hrzn_per_h[h_idx] = _martingale.update_traditional(
                            logM_hrzn_per_h[h_idx], p_h_pf, g
                        )
                        last_p_h_pf = p_h_pf
                    # VW mixture over h axis → per-feature logM.
                    weighted = logM_hrzn_per_h + log_hw[:, None]  # (H, K)
                    max_per_k = weighted.max(axis=0)
                    logM_hrzn_pf = max_per_k + np.log(
                        np.exp(weighted - max_per_k[None, :]).sum(axis=0)
                    )
                    pvals_hrzn[t] = last_p_h_pf  # trace last-h p-values (informational)
                else:
                    # Single-horizon (paper Def 7, default).
                    X_hat = cfg.forecaster.predict_multi(hist_window, horizon=h)
                    S_pred_pf = np.abs(X_hat - C_t_full)
                    p_h_pf = np.empty(K)
                    for k in range(K):
                        S_prev = np.asarray(S_hist_pf[k], dtype=np.float64)
                        p_h_pf[k] = _conformal.predictive_pvalue_step(float(S_pred_pf[k]), S_prev)
                    pvals_hrzn[t] = p_h_pf
                    logM_hrzn_pf = _martingale.update_traditional(logM_hrzn_pf, p_h_pf, g)

                # Joint predictive score.
                if cfg.detection_mode == "joint":
                    X_aug = np.vstack([X[: t + 1], X_hat[None, :]])
                    S_pred_joint_full = _conformal.nonconformity_joint(X_aug, metric=cfg.joint_distance)
                    S_pred_joint = float(S_pred_joint_full[-1])
                    p_h_joint = _conformal.predictive_pvalue_step(
                        S_pred_joint, np.asarray(S_hist_joint, dtype=np.float64)
                    )
                    gp = g(np.asarray([p_h_joint]))[0]
                    logM_hrzn_joint = max(_LOG_M_FLOOR, logM_hrzn_joint + np.log(max(gp, 1e-300)))

            # Record per-feature shadow state for attribution.
            logM_per_feature[t] = logM_trad_pf
            logM_per_feature_h[t] = logM_hrzn_pf

            # --- Primary detection stream (aggregate) ---
            if cfg.detection_mode == "joint":
                logM_sum_trad[t] = logM_trad_joint if cfg.enable_traditional else np.nan
                logM_sum_hrzn[t] = logM_hrzn_joint if (cfg.enable_horizon and t >= w) else np.nan
            else:
                if cfg.enable_traditional:
                    logM_sum_trad[t] = _logsumexp(logM_trad_pf)
                if cfg.enable_horizon and t >= w:
                    logM_sum_hrzn[t] = _logsumexp(logM_hrzn_pf)

            # Detection rule (Algorithm 1 line 18).
            triggered = False
            if t - last_detection > cfg.cooldown:
                if cfg.enable_traditional and logM_sum_trad[t] >= log_lam:
                    triggered = True
                elif cfg.enable_horizon and t >= w and logM_sum_hrzn[t] >= log_lam:
                    triggered = True

            if triggered:
                change_points.append(t)
                last_detection = t
                # Reset ALL streams (lines 20–21). Attribution streams reset too.
                logM_trad_pf[:] = 0.0
                logM_hrzn_pf[:] = 0.0
                logM_trad_joint = 0.0
                logM_hrzn_joint = 0.0
                if using_mixture:
                    logM_hrzn_per_h[:] = 0.0

            # Floor against -inf accumulation on per-feature (primary case).
            np.maximum(logM_trad_pf, _LOG_M_FLOOR, out=logM_trad_pf)
            np.maximum(logM_hrzn_pf, _LOG_M_FLOOR, out=logM_hrzn_pf)

            # Append THIS step's S_t to the reference history AFTER all p-values
            # for this step were computed. This enforces leak-free semantics:
            # p_t uses S_1..S_{t-1} as reference, not S_t itself.
            for k in range(K):
                S_hist_pf[k].append(float(S_t_pf[k]))
            if cfg.detection_mode == "joint":
                S_hist_joint.append(S_t_joint)

        return DetectionResult(
            change_points=change_points,
            logM_traditional=logM_sum_trad,
            logM_horizon=logM_sum_hrzn,
            logM_per_feature=logM_per_feature,
            logM_per_feature_horizon=logM_per_feature_h,
            features=X,
            feature_names=_features.feature_names(cfg.features),
            pvalues=pvals_trad,
            pvalues_horizon=pvals_hrzn,
            config=cfg,
            scenario=scenario,
            true_change_points=true_change_points,
        )


def _logsumexp(x: np.ndarray) -> float:
    """Stable log-sum-exp. Wraps scipy; guards all-(-inf) inputs."""
    m = float(np.max(x))
    if not np.isfinite(m):
        return m
    return float(_scipy_logsumexp(x))


def _pvalues_vectorized(S_t: np.ndarray, S_hist_pf: list[list[float]], rng) -> np.ndarray:
    """Per-step Vovk-smoothed p-values across K features. Wraps
    `conformal.smoothed_pvalue_step` so detector and conformal module share
    exactly one formula implementation.
    """
    K = S_t.shape[0]
    p = np.empty(K)
    for k in range(K):
        prev = np.asarray(S_hist_pf[k], dtype=np.float64)
        p[k] = _conformal.smoothed_pvalue_step(float(S_t[k]), prev, rng.uniform())
    return p


def _scalar_conformal_pvalue(S_t: float, S_hist: list[float], rng) -> float:
    """Single-scalar Vovk-smoothed p-value. Wraps smoothed_pvalue_step."""
    prev = np.asarray(S_hist, dtype=np.float64)
    return _conformal.smoothed_pvalue_step(S_t, prev, rng.uniform())
