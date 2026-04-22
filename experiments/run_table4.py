"""Reproduce Table IV of Ali & Ho (ICDM 2025): main method comparison.

Four detectors x nine synthetic scenarios x ``n_trials`` seeds. Outputs
per-trial and per-(scenario, detector) aggregates as CSV, plus a
tabulate-formatted stdout table matching the paper's Table IV layout.

Usage
-----

    .venv/bin/python experiments/run_table4.py --n-trials 10 --out results/table4.csv
    .venv/bin/python experiments/run_table4.py --n-trials 5 --workers 8

Outputs
-------
    results/table4_raw.csv   one row per (detector, scenario, seed) trial.
    results/table4.csv       pivot: one row per scenario, columns per
                             detector-metric pair (mirrors Table IV).

Why this design
===============

**Parallelize at the seed level, cache features within a seed.** Feature
extraction (N=50 graphs x T=200 snapshots) dominates wallclock — ~3.5s per
sequence. A naive loop over (detector, scenario, seed) would re-extract
features four times per sequence. We instead:

  1. Dispatch ONE worker per (scenario, seed) pair.
  2. Inside the worker: generate the sequence once, extract features once,
     then run all four detectors on the cached feature matrix via
     ``HorizonDetector.run_on_features(X)`` and ``cusum(X)``, ``ewma(X)``.
  3. Workers return a list of four rows (one per detector) which the parent
     aggregates.

This gives a ~4x speedup over the naive scheme and a ~N-worker speedup over
serial. Total work: 9 scenarios x 10 seeds = 90 units, ~3.7s each on cold
feature extract plus ~0.2s for four detector passes => ~5 minutes serial,
<1 minute on 8 cores.

**Scenario-level parallelism would be worse.** Scenarios are unbalanced in
runtime (ba_hub_addition is denser, slower); seed-level parallelism gives
finer granularity for load balancing.

**No cross-process shared state.** Each worker is fully self-contained,
reproducibility only requires that the RNG seed flows through. Using
``multiprocessing.Pool.imap_unordered`` lets us stream results into a tqdm
progress bar as they finish.

**Error handling.** If a single (scenario, seed) worker raises, we log a
warning and continue — one buggy trial shouldn't kill a 5-minute sweep.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Ensure the repo root is importable when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hmd import HorizonDetector  # noqa: E402
from hmd.baselines import CUSUMConfig, EWMAConfig, cusum, ewma  # noqa: E402
from hmd.data.synthetic import ALL_SCENARIOS  # noqa: E402
from hmd import features as _features  # noqa: E402

from experiments.metrics import aggregate, tpr_fpr_add  # noqa: E402


# ----------------------------------------------------------------------------
# Detectors — each returns a list of detected change points given (X, T).
# ----------------------------------------------------------------------------

DETECTOR_ORDER = ["Martingale", "Horizon", "CUSUM", "EWMA"]

# Global detector config — populated by main() before spawning workers.
# On Linux fork, child workers inherit this module-level state.
_CFG: dict = {
    "threshold": 50.0,
    "startup_period": 20,
    "detection_mode": "per_feature",
    "joint_distance": "mahalanobis",
    "normalize_features": False,
}


def _hmd_kwargs() -> dict:
    return {
        "threshold": _CFG["threshold"],
        "startup_period": _CFG["startup_period"],
        "detection_mode": _CFG["detection_mode"],
        "joint_distance": _CFG["joint_distance"],
        "normalize_features": _CFG["normalize_features"],
    }


def _run_martingale(X: np.ndarray, true_cps: list[int], T: int) -> dict:
    det = HorizonDetector(enable_traditional=True, enable_horizon=False, **_hmd_kwargs())
    r = det.run_on_features(X)
    m = tpr_fpr_add(r.change_points, true_cps, T=T)
    m["n_detections"] = len(r.change_points)
    return m


def _run_horizon(X: np.ndarray, true_cps: list[int], T: int) -> dict:
    det = HorizonDetector(enable_traditional=True, enable_horizon=True, **_hmd_kwargs())
    r = det.run_on_features(X)
    m = tpr_fpr_add(r.change_points, true_cps, T=T)
    m["n_detections"] = len(r.change_points)
    return m


def _run_cusum(X: np.ndarray, true_cps: list[int], T: int) -> dict:
    r = cusum(X, CUSUMConfig(threshold=5.0, k=0.5, startup_period=_CFG["startup_period"]))
    m = tpr_fpr_add(r.change_points, true_cps, T=T)
    m["n_detections"] = len(r.change_points)
    return m


def _run_ewma(X: np.ndarray, true_cps: list[int], T: int) -> dict:
    r = ewma(X, EWMAConfig(lambda_=0.3, L=3.0, startup_period=_CFG["startup_period"]))
    m = tpr_fpr_add(r.change_points, true_cps, T=T)
    m["n_detections"] = len(r.change_points)
    return m


_DETECTOR_FNS = {
    "Martingale": _run_martingale,
    "Horizon": _run_horizon,
    "CUSUM": _run_cusum,
    "EWMA": _run_ewma,
}


# ----------------------------------------------------------------------------
# Per-worker entry point
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    scenario: str
    seed: int


def _process_trial(trial: Trial) -> list[dict]:
    """Run all four detectors on a single (scenario, seed). Returns 4 rows."""
    try:
        scenario_fn = ALL_SCENARIOS[trial.scenario]
        seq = scenario_fn(seed=trial.seed)
        T = len(seq.graphs)

        # Feature extraction — expensive, done once per (scenario, seed).
        X = _features.extract_sequence(seq.graphs, _features.default_set())

        rows = []
        for name in DETECTOR_ORDER:
            m = _DETECTOR_FNS[name](X, seq.change_points, T)
            rows.append(
                {
                    "scenario": trial.scenario,
                    "detector": name,
                    "seed": trial.seed,
                    "tpr": m["tpr"],
                    "fpr": m["fpr"],
                    "add": m["add"],
                    "n_tp": m["n_tp"],
                    "n_fp": m["n_fp"],
                    "n_detections": m["n_detections"],
                }
            )
        return rows
    except Exception as e:
        warnings.warn(f"Trial ({trial.scenario}, seed={trial.seed}) failed: {e!r}")
        return []


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = ["scenario", "detector", "seed", "tpr", "fpr", "add", "n_tp", "n_fp", "n_detections"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_pivot_csv(agg: list[dict], path: Path) -> None:
    """Pivot: one row per scenario, columns per detector-metric pair.

    Matches paper Table IV's layout: scenario down the left, each detector's
    TPR / FPR / ADD grouped together.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build lookup: scenario -> detector -> row
    by_key: dict[tuple, dict] = {(r["scenario"], r["detector"]): r for r in agg}
    scenarios = sorted({r["scenario"] for r in agg})
    detectors = DETECTOR_ORDER

    fields = ["scenario"]
    for det in detectors:
        for metric in ("tpr", "fpr", "add"):
            fields.append(f"{det}_{metric}")

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in scenarios:
            row = {"scenario": sc}
            for det in detectors:
                r = by_key.get((sc, det), {})
                for metric in ("tpr", "fpr", "add"):
                    val = r.get(f"{metric}_mean")
                    row[f"{det}_{metric}"] = "" if val is None else f"{val:.3f}"
            w.writerow(row)


def _print_table(agg: list[dict]) -> None:
    """Print a Table IV-like stdout summary using `tabulate`."""
    try:
        from tabulate import tabulate
    except ImportError:
        warnings.warn("tabulate not installed; skipping pretty-print.")
        return

    by_key: dict[tuple, dict] = {(r["scenario"], r["detector"]): r for r in agg}
    scenarios = sorted({r["scenario"] for r in agg})

    headers = ["Scenario"] + [f"{d} TPR/FPR/ADD" for d in DETECTOR_ORDER]
    table: list[list] = []
    for sc in scenarios:
        row = [sc]
        for det in DETECTOR_ORDER:
            r = by_key.get((sc, det), {})
            tpr = r.get("tpr_mean")
            fpr = r.get("fpr_mean")
            add = r.get("add_mean")
            parts = []
            parts.append("-" if tpr is None else f"{tpr:.2f}")
            parts.append("-" if fpr is None else f"{fpr:.3f}")
            parts.append("-" if add is None else f"{add:.1f}")
            row.append(" / ".join(parts))
        table.append(row)

    # Overall averages across scenarios per detector.
    avg_row = ["AVERAGE"]
    for det in DETECTOR_ORDER:
        rows = [by_key[(sc, det)] for sc in scenarios if (sc, det) in by_key]
        parts = []
        for metric in ("tpr", "fpr", "add"):
            vals = [r[f"{metric}_mean"] for r in rows if r[f"{metric}_mean"] is not None]
            parts.append("-" if not vals else f"{np.mean(vals):.3f}" if metric == "fpr" else f"{np.mean(vals):.2f}" if metric == "tpr" else f"{np.mean(vals):.1f}")
        avg_row.append(" / ".join(parts))
    table.append(avg_row)

    print("\nTable IV (reproduction) — TPR / FPR / ADD per scenario")
    print(tabulate(table, headers=headers, tablefmt="github"))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Reproduce Ali & Ho ICDM 2025 Table IV.")
    ap.add_argument("--n-trials", type=int, default=10, help="seeds per scenario")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/table4.csv"),
        help="pivoted CSV output path; a sibling *_raw.csv is also written",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="multiprocessing workers (default: CPU count - 1)",
    )
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="subset of scenarios to run (default: all 9)",
    )
    ap.add_argument("--threshold", type=float, default=50.0, help="detection threshold λ")
    ap.add_argument("--startup", type=int, default=20, help="startup_period")
    ap.add_argument(
        "--mode",
        choices=["per_feature", "joint"],
        default="per_feature",
        help="detection_mode: per_feature (Algorithm 1) or joint (Mahalanobis, Table IV)",
    )
    ap.add_argument(
        "--joint-distance",
        choices=["euclidean", "mahalanobis", "cosine", "chebyshev"],
        default="mahalanobis",
    )
    ap.add_argument("--normalize", action="store_true", help="z-score features pre-detection")
    args = ap.parse_args()

    # Push CLI into module-level config so worker processes inherit it on fork.
    _CFG["threshold"] = args.threshold
    _CFG["startup_period"] = args.startup
    _CFG["detection_mode"] = args.mode
    _CFG["joint_distance"] = args.joint_distance
    _CFG["normalize_features"] = args.normalize

    scenarios = args.scenarios or list(ALL_SCENARIOS.keys())
    trials = [Trial(sc, seed) for sc in scenarios for seed in range(args.n_trials)]

    print(f"Config: λ={args.threshold}, startup={args.startup}, mode={args.mode}"
          + (f"/{args.joint_distance}" if args.mode == "joint" else "")
          + f", normalize={args.normalize}")
    print(f"Running {len(trials)} trials across {args.workers} workers "
          f"({len(scenarios)} scenarios x {args.n_trials} seeds x 4 detectors each).")

    t0 = time.time()
    all_rows: list[dict] = []
    with mp.Pool(args.workers) as pool:
        iterator = pool.imap_unordered(_process_trial, trials, chunksize=1)
        for batch in tqdm(iterator, total=len(trials), desc="trials"):
            all_rows.extend(batch)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / max(1, len(trials)):.2f}s/trial).")

    raw_path = args.out.with_name(args.out.stem + "_raw.csv")
    _write_raw_csv(all_rows, raw_path)
    print(f"  raw trials -> {raw_path}")

    agg = aggregate(all_rows, group_keys=["scenario", "detector"])
    _write_pivot_csv(agg, args.out)
    print(f"  aggregated -> {args.out}")

    _print_table(agg)

    # Overall averages per detector, printed compactly for report.
    print("\nPer-detector averages across all scenarios:")
    for det in DETECTOR_ORDER:
        rows = [r for r in agg if r["detector"] == det]
        tprs = [r["tpr_mean"] for r in rows if r["tpr_mean"] is not None]
        fprs = [r["fpr_mean"] for r in rows if r["fpr_mean"] is not None]
        adds = [r["add_mean"] for r in rows if r["add_mean"] is not None]
        print(
            f"  {det:<12s} TPR={np.mean(tprs):.3f} "
            f"FPR={np.mean(fprs):.4f} "
            f"ADD={np.mean(adds):.2f}"
        )


if __name__ == "__main__":
    main()
