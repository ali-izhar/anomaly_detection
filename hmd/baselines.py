"""Classical SPC baselines — Page's CUSUM and Roberts' EWMA chart (§VI-A).

Paper uses these as the "no network assumptions" comparison:
- **CUSUM** (Page 1954): h=5, k=0.5. Detects shifts of size ~ kσ.
- **EWMA chart** (Roberts 1959): λ=0.3, L=3.

Critical design choices
-----------------------
- **Per-feature + any-breach aggregation.** Joint CUSUM would need the joint
  pre/post distribution — exactly what we don't have. Per-feature with
  any-breach is standard multi-channel SPC (Montgomery §9) and is the fair
  comparison to the paper's per-feature sum martingale.
- **Leak-free running standardization** via Welford's algorithm (Knuth TAOCP
  §4.2.2): μ̂_t, σ̂_t use X[:t] only, and Welford avoids the
  catastrophic-cancellation failure of naive `(Σx² − nμ²)/(n−1)` on constant
  signals (verified: 0 FPs on constant input, any magnitude).
- **Exact finite-t EWMA variance** Var(Z_t) = σ² · λ/(2−λ) · (1 − (1−λ)^(2t))
  — the asymptotic σ²·λ/(2−λ) is too loose at small t and delays detection.
- **Two-sided CUSUM** (both C⁺ and C⁻). One-sided would miss
  decrease-direction changes (e.g. community merge → drops in modularity).
- **Reset after detection** (Montgomery §9-1.2). Without it the statistic
  stays saturated indefinitely after the first crossing.
- **Startup period = 20** matches `hmd.DetectorConfig.history_size` so the
  baselines and HMD don't have asymmetric burn-in advantages.

NOT implemented (paper doesn't use): FIR headstart, adaptive EWMA, robust
scale estimation (MAD).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Running moments (leak-free μ̂_t, σ̂_t from X[:t])
# ---------------------------------------------------------------------------


def _running_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (μ̂, σ̂) with shape (T, K) where row t uses X[:t] only.

    σ̂ uses ddof=1 (unbiased sample std) once t ≥ 2; rows with t < 2 are NaN.

    Why Welford's algorithm, not naive (Σx² - n·μ²)/(n-1):
    the naive form suffers catastrophic cancellation when Var(x) << (E[x])²,
    giving σ̂ > 0 on inputs that are actually constant (loses ~7 digits in
    float64 when x = 7.2 and n ≈ 20 → spurious σ̂ ≈ 1e-7). Welford preserves
    σ̂ = 0 exactly on constants. Reference: Knuth TAOCP vol 2 §4.2.2,
    Welford 1962.
    """
    T, K = X.shape
    mean = np.full((T, K), np.nan)
    std = np.full((T, K), np.nan)
    m = np.zeros(K, dtype=np.float64)  # running mean
    m2 = np.zeros(K, dtype=np.float64)  # running sum of squared deviations
    for t in range(T):
        if t >= 1:
            mean[t] = m
        if t >= 2:
            std[t] = np.sqrt(m2 / (t - 1.0))
        # Welford update: incorporate X[t] for use at steps > t.
        x = X[t]
        delta = x - m
        m = m + delta / (t + 1.0)
        delta2 = x - m
        m2 = m2 + delta * delta2
    return mean, std


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Structured output matching `hmd.detector.DetectionResult`'s spirit.

    score:        (T, K) running per-feature statistic. For CUSUM this is the
                  element-wise max(C⁺, C⁻); for EWMA it is the standardized
                  deviation |Z_t - μ̂_t| / σ̂_{Z,t}.
    score_sum:    (T,) per-step summary for plotting. max over features — the
                  quantity actually compared to the threshold under the
                  any-breach rule. NaN before startup_period.
    breach_feature: index (0..K-1) of the feature that triggered each detection,
                    or None if the detection cannot be attributed (should not
                    happen under correct semantics).
    """

    change_points: list[int]
    score: np.ndarray
    score_sum: np.ndarray
    breach_feature: list[int | None]


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


@dataclass
class CUSUMConfig:
    threshold: float = 5.0  # h, control limit (paper §VI-A)
    k: float = 0.5  # reference value, half the targeted shift in σ units (paper §VI-A)
    startup_period: int = 20  # matches hmd.DetectorConfig.history_size
    cooldown: int = 0


def cusum(X: np.ndarray, config: CUSUMConfig | None = None) -> BaselineResult:
    """Page's two-sided CUSUM run per feature with any-breach aggregation.

    For each feature k, letting z_t = (x_t - μ̂_t) / σ̂_t with μ̂_t, σ̂_t
    estimated from X[:t] only:

        C⁺_t = max(0, C⁺_{t-1} + (z_t - k))
        C⁻_t = max(0, C⁻_{t-1} + (-z_t - k))

    Alarm when max_k max(C⁺_t[k], C⁻_t[k]) ≥ threshold, after startup.
    """
    if config is None:
        config = CUSUMConfig()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, K); got shape {X.shape}.")
    T, K = X.shape

    mu, sd = _running_mean_std(X)

    C_plus = np.zeros(K)
    C_minus = np.zeros(K)
    score = np.zeros((T, K))
    score_sum = np.full(T, np.nan)
    change_points: list[int] = []
    breach_feature: list[int | None] = []
    last_detection = -config.cooldown - 1

    for t in range(T):
        # Leak-free z-score. Require σ̂_t > 0; fall back to 0 (no contribution).
        # NOTE: if σ̂_t == 0 (constant history) then z is undefined; setting
        # (z - k) → 0 keeps C⁺, C⁻ flat at 0, so constant signals give no alarms.
        if t < config.startup_period or np.any(np.isnan(sd[t])):
            # Not enough data to standardize; hold accumulators at 0.
            score[t] = np.maximum(C_plus, C_minus)
            continue

        sd_safe = sd[t].copy()
        zero_mask = sd_safe <= 0.0
        sd_safe[zero_mask] = 1.0  # divisor; numerator will be 0 for constant features
        z = (X[t] - mu[t]) / sd_safe
        z[zero_mask] = 0.0  # force no contribution

        C_plus = np.maximum(0.0, C_plus + (z - config.k))
        C_minus = np.maximum(0.0, C_minus + (-z - config.k))
        score[t] = np.maximum(C_plus, C_minus)
        score_sum[t] = score[t].max()

        # Any-breach detection.
        if t - last_detection > config.cooldown and score_sum[t] >= config.threshold:
            # Which feature fired first (tie → lowest index, deterministic).
            k_fired = int(np.argmax(score[t]))
            change_points.append(t)
            breach_feature.append(k_fired)
            last_detection = t
            # Reset both accumulators after a detection (standard CUSUM practice:
            # Montgomery §9-1.2 "restarting" — the post-change mean is unknown,
            # so the old accumulator is stale. Analogous to the martingale's
            # reset on detection in hmd/detector.py).
            C_plus = np.zeros(K)
            C_minus = np.zeros(K)

    return BaselineResult(
        change_points=change_points,
        score=score,
        score_sum=score_sum,
        breach_feature=breach_feature,
    )


# ---------------------------------------------------------------------------
# EWMA (Roberts 1959)
# ---------------------------------------------------------------------------


@dataclass
class EWMAConfig:
    lambda_: float = 0.3  # smoothing coefficient (paper §VI-A)
    L: float = 3.0  # control limit in σ units (paper §VI-A)
    startup_period: int = 20
    cooldown: int = 0


def ewma(X: np.ndarray, config: EWMAConfig | None = None) -> BaselineResult:
    """Roberts (1959) EWMA chart, per feature with any-breach aggregation.

        Z_t = λ·x_t + (1-λ)·Z_{t-1},  Z_0 := μ̂_{startup_period}.

    Alarm when |Z_t - μ̂_t| > L · σ̂_{Z,t}, where σ̂_{Z,t} uses the exact
    finite-t variance:

        σ̂²_{Z,t} = σ̂²_t · (λ / (2-λ)) · (1 - (1-λ)^(2·(t - t0 + 1)))

    where t0 = startup_period is the index at which EWMA tracking begins.
    Using the step count since tracking started (not absolute t) gives the
    correct small-sample variance for Z.
    """
    if config is None:
        config = EWMAConfig()
    if not (0.0 < config.lambda_ <= 1.0):
        raise ValueError(f"lambda_ must be in (0, 1]; got {config.lambda_}.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, K); got shape {X.shape}.")
    T, K = X.shape
    lam = config.lambda_
    L = config.L

    mu, sd = _running_mean_std(X)

    Z = np.zeros(K)  # will be overwritten at startup
    score = np.zeros((T, K))
    score_sum = np.full(T, np.nan)
    change_points: list[int] = []
    breach_feature: list[int | None] = []
    last_detection = -config.cooldown - 1

    tracking_started = False
    t0 = config.startup_period  # first time step at which EWMA updates

    for t in range(T):
        if t < t0 or np.any(np.isnan(sd[t])):
            continue

        if not tracking_started:
            Z = mu[t].copy()  # Z_0 = μ̂_{t0}
            tracking_started = True

        # Update Z (exponential smoothing).
        Z = lam * X[t] + (1.0 - lam) * Z

        # Exact finite-t EWMA variance — step count since tracking began.
        # At the first tracked step (t == t0) we have done one update, so n=1.
        n_updates = t - t0 + 1
        var_factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * n_updates))
        sigma_Z = sd[t] * np.sqrt(var_factor)
        # Guard constant features (σ̂_t = 0 → σ_Z = 0 → no deviation possible).
        sigma_Z_safe = np.where(sigma_Z > 0.0, sigma_Z, 1.0)
        z_score = np.abs(Z - mu[t]) / sigma_Z_safe
        z_score = np.where(sigma_Z > 0.0, z_score, 0.0)

        score[t] = z_score
        score_sum[t] = z_score.max()

        if t - last_detection > config.cooldown and score_sum[t] > L:
            k_fired = int(np.argmax(z_score))
            change_points.append(t)
            breach_feature.append(k_fired)
            last_detection = t
            # Reset Z to current running mean (post-change re-centering).
            Z = mu[t].copy()

    return BaselineResult(
        change_points=change_points,
        score=score,
        score_sum=score_sum,
        breach_feature=breach_feature,
    )
