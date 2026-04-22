"""Smoke + structural-invariant tests for the HorizonDetector pipeline.

WHAT is tested
--------------
End-to-end correctness of the Algorithm 1 glue in hmd.detector:

  * Runs without raising on a real (small) SBM sequence.
  * DetectionResult fields have the advertised shapes.
  * Attribution percentages sum to 100% (Eq 7).
  * Ablation: disabling a stream leaves its logM all NaN.
  * Determinism: same seed -> identical change_points.

WHY this matters
----------------
The detector glue is where features, conformal p-values, betting, the
martingale recurrence, and the forecaster all meet. The theorems cited
in the paper are proved for the individual components; the glue must
not violate them by (e.g.) feeding stale predictions, leaking future
data into p-values, or silently disabling a stream that the user
intended to use.

We build a 50-snapshot SBM sequence (not the default T=200 to keep the
test fast) with delta_min=10 so one change point is placed.
"""

from __future__ import annotations

import numpy as np
import pytest

from hmd import HorizonDetector, DetectorConfig
from hmd.data import synthetic as syn


def _build_small_sbm(seed: int = 0, T: int = 50, N: int = 20):
    """Short SBM sequence (T=50, N=20) with a single mid-sequence change point.

    The default delta_min=40 in synthetic.py requires T>=80 to place any CP.
    We dip into the internals to build a short test fixture without modifying
    the library. This is a test-only helper.
    """
    rng = np.random.default_rng(seed)
    cps = syn._sample_change_points(T, 1, 10, rng)
    regimes = [
        {"p_intra": 0.95, "p_inter": 0.01},
        {"p_intra": 0.70, "p_inter": 0.15},
    ]
    samplers = [
        lambda r, p=p: syn._sample_sbm(N, p["p_intra"], p["p_inter"], r)
        for p in regimes
    ]
    graphs = syn._assemble(T, cps, samplers, rng)
    return graphs, cps


def _small_cfg(**overrides):
    base = dict(
        startup_period=10,
        history_size=5,
        threshold=20.0,
        show_progress=False,
        rng_seed=0,
    )
    base.update(overrides)
    return DetectorConfig(**base)


def test_runs_without_error():
    graphs, _ = _build_small_sbm(seed=0)
    det = HorizonDetector(_small_cfg())
    res = det.run(graphs)
    assert isinstance(res.change_points, list)


def test_result_field_shapes():
    graphs, cps = _build_small_sbm(seed=0)
    T = len(graphs)
    det = HorizonDetector(_small_cfg())
    res = det.run(graphs)
    K = len(res.feature_names)
    assert K == 8, f"expected 8 default features, got {K}"
    assert res.logM_traditional.shape == (T,)
    assert res.logM_horizon.shape == (T,)
    assert res.logM_per_feature.shape == (T, K)
    assert res.logM_per_feature_horizon.shape == (T, K)
    assert res.features.shape == (T, K)
    assert res.pvalues.shape == (T, K)
    assert res.pvalues_horizon.shape == (T, K)


def test_attribution_sums_to_one_hundred():
    graphs, _ = _build_small_sbm(seed=1)
    det = HorizonDetector(_small_cfg())
    res = det.run(graphs)
    # Pick several post-startup timesteps for attribution.
    for t in [15, 25, 40, len(graphs) - 1]:
        pct = res.attribution_at(t, stream="traditional")
        s = sum(pct.values())
        assert abs(s - 100.0) < 1e-6, (
            f"traditional attribution at t={t} sums to {s}, not 100%"
        )
        pct_h = res.attribution_at(t, stream="horizon")
        s_h = sum(pct_h.values())
        assert abs(s_h - 100.0) < 1e-6, (
            f"horizon attribution at t={t} sums to {s_h}, not 100%"
        )


def test_ablation_disables_horizon():
    graphs, _ = _build_small_sbm(seed=2)
    det = HorizonDetector(_small_cfg(enable_horizon=False))
    res = det.run(graphs)
    # Horizon stream must be all NaN when disabled.
    assert np.all(np.isnan(res.logM_horizon)), (
        "enable_horizon=False should leave logM_horizon all NaN"
    )
    # Traditional stream must have SOME finite entries.
    assert np.any(np.isfinite(res.logM_traditional))


def test_ablation_disables_traditional():
    graphs, _ = _build_small_sbm(seed=3)
    det = HorizonDetector(_small_cfg(enable_traditional=False))
    res = det.run(graphs)
    assert np.all(np.isnan(res.logM_traditional)), (
        "enable_traditional=False should leave logM_traditional all NaN"
    )
    assert np.any(np.isfinite(res.logM_horizon))


def test_cannot_disable_both_streams():
    """DetectorConfig must reject a fully-disabled run."""
    with pytest.raises(ValueError):
        DetectorConfig(enable_traditional=False, enable_horizon=False)


def test_determinism_same_seed_same_outputs():
    """Two identical runs with matching rng_seed produce identical results."""
    graphs, _ = _build_small_sbm(seed=4)
    cfg = _small_cfg(rng_seed=7)
    r1 = HorizonDetector(cfg).run(graphs)
    r2 = HorizonDetector(cfg).run(graphs)
    assert r1.change_points == r2.change_points
    # logM paths must be bit-exact too -- the only randomness is the tie-break
    # theta, seeded by rng_seed.
    np.testing.assert_array_equal(
        np.nan_to_num(r1.logM_traditional, nan=-1.0),
        np.nan_to_num(r2.logM_traditional, nan=-1.0),
    )
    np.testing.assert_array_equal(
        np.nan_to_num(r1.logM_horizon, nan=-1.0),
        np.nan_to_num(r2.logM_horizon, nan=-1.0),
    )


def test_startup_period_gates_detection():
    """No change point can be reported before t = startup_period."""
    graphs, _ = _build_small_sbm(seed=5)
    startup = 15
    det = HorizonDetector(_small_cfg(startup_period=startup))
    res = det.run(graphs)
    for t in res.change_points:
        assert t >= startup, (
            f"detection at t={t} precedes startup_period={startup}"
        )
