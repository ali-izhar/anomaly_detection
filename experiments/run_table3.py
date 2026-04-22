"""Reproduce Table III of Ali & Ho (ICDM 2025): parameter sensitivity sweep.

Sweeps the Horizon detector over:
    * betting: power(eps) for eps in {0.2, 0.5, 0.7, 0.9}, plus mixture (default),
               plus beta with a handful of (alpha, beta) tunings
    * detection mode + joint distance: {euclidean, mahalanobis, cosine, chebyshev}
      in JOINT mode (the paper's "Distance Metric" column only makes sense here)
    * threshold lambda: {20, 50, 100}

For each (config, scenario, seed) we record TPR, FPR, ADD per
``experiments.metrics.tpr_fpr_add``.

Usage
-----

    .venv/bin/python experiments/run_table3.py --n-trials 10 --workers 12
    .venv/bin/python experiments/run_table3.py --out results/table3.csv

Outputs
-------
    results/table3_raw.csv   one row per (config, scenario, seed) trial.
    results/table3.csv       pivoted by network / metric / configuration.

Why this design
===============

**Each work unit = one (config, scenario, seed).** The sweep space is
| configs | x | scenarios | x | seeds | which for defaults ~15 x 9 x 10 = 1350
units. Parallelizing at this granularity (rather than per config or per
seed) gives the best load balance across workers.

**Feature caching across configs.** Inside a worker, once features are
extracted for a (scenario, seed), we reuse them across the ~15 Horizon
configs via ``HorizonDetector.run_on_features(X)``. This dwarfs the detector
pass cost (~0.05s vs 3.7s).

**Scheduling.** We group trials by (scenario, seed) so that one worker's
unit of work is "run all configs on one feature matrix". ``imap_unordered``
streams finished work units as they complete; a tqdm bar reports progress
in config-scenario-seed units.

**Error isolation.** A config that raises (e.g., an invalid beta tuning)
logs a warning and continues for the remaining configs of that (scenario,
seed). We never abort the sweep.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hmd import HorizonDetector  # noqa: E402
from hmd import betting as _betting  # noqa: E402
from hmd import features as _features  # noqa: E402
from hmd.data.synthetic import ALL_SCENARIOS  # noqa: E402

from experiments.metrics import aggregate, tpr_fpr_add  # noqa: E402


# ----------------------------------------------------------------------------
# Configuration grid
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    betting_kind: str  # 'power' | 'mixture' | 'beta'
    betting_params: tuple  # depends on betting_kind
    threshold: float
    detection_mode: str   # 'per_feature' | 'joint'
    joint_distance: str   # used when detection_mode == 'joint'

    def to_row_key(self) -> dict:
        return {
            "config_name": self.name,
            "betting_kind": self.betting_kind,
            "betting_params": str(self.betting_params),
            "threshold": self.threshold,
            "detection_mode": self.detection_mode,
            "joint_distance": self.joint_distance,
        }


def _make_betting(kind: str, params: tuple):
    if kind == "power":
        (eps,) = params
        return _betting.power(eps)
    if kind == "mixture":
        return _betting.default()
    if kind == "beta":
        a, b = params
        return _betting.beta(a, b)
    raise ValueError(f"unknown betting kind {kind}")


def default_config_grid() -> list[Config]:
    """Build the Table III sweep.

    Each axis held at the paper's default while the target axis is varied.
    This mirrors Table III's "one-factor-at-a-time" layout rather than a
    full-factorial Cartesian product (which would be 4 x 4 x 4 x 3 = 192
    configs, unnecessary to reproduce the paper's claims).
    """
    configs: list[Config] = []

    # --- Betting function axis ---
    # Paper: Table III varies ε ∈ {0.2, 0.5, 0.7, 0.9} and reports mixture + beta.
    for eps in (0.2, 0.5, 0.7, 0.9):
        configs.append(Config(
            name=f"power(eps={eps})",
            betting_kind="power", betting_params=(eps,),
            threshold=50.0, detection_mode="per_feature", joint_distance="mahalanobis",
        ))
    configs.append(Config(
        name="mixture_default",
        betting_kind="mixture", betting_params=(),
        threshold=50.0, detection_mode="per_feature", joint_distance="mahalanobis",
    ))
    for a, b in [(0.2, 2.5), (0.4, 1.8), (0.6, 1.2)]:
        configs.append(Config(
            name=f"beta(a={a},b={b})",
            betting_kind="beta", betting_params=(a, b),
            threshold=50.0, detection_mode="per_feature", joint_distance="mahalanobis",
        ))

    # --- Distance metric axis (joint mode) ---
    for dist in ("euclidean", "mahalanobis", "cosine", "chebyshev"):
        configs.append(Config(
            name=f"joint_{dist}",
            betting_kind="mixture", betting_params=(),
            threshold=50.0, detection_mode="joint", joint_distance=dist,
        ))

    # --- Threshold axis ---
    for lam in (20.0, 50.0, 100.0):
        configs.append(Config(
            name=f"threshold={lam:g}",
            betting_kind="mixture", betting_params=(),
            threshold=lam, detection_mode="per_feature", joint_distance="mahalanobis",
        ))

    return configs


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkUnit:
    scenario: str
    seed: int


def _process_unit(unit: WorkUnit, configs: list[Config]) -> list[dict]:
    """Extract features once for (scenario, seed), then sweep over configs."""
    rows: list[dict] = []
    try:
        seq = ALL_SCENARIOS[unit.scenario](seed=unit.seed)
        T = len(seq.graphs)
        X = _features.extract_sequence(seq.graphs, _features.default_set())
    except Exception as e:
        warnings.warn(f"feature extraction failed ({unit.scenario}, seed={unit.seed}): {e!r}")
        return rows

    for cfg in configs:
        try:
            bet = _make_betting(cfg.betting_kind, cfg.betting_params)
            det = HorizonDetector(
                threshold=cfg.threshold,
                startup_period=20,
                enable_traditional=True,
                enable_horizon=True,
                betting=bet,
                detection_mode=cfg.detection_mode,
                joint_distance=cfg.joint_distance,
            )
            r = det.run_on_features(X)
            m = tpr_fpr_add(r.change_points, seq.change_points, T=T)
            row = {
                "scenario": unit.scenario,
                "seed": unit.seed,
                **cfg.to_row_key(),
                "tpr": m["tpr"],
                "fpr": m["fpr"],
                "add": m["add"],
                "n_tp": m["n_tp"],
                "n_fp": m["n_fp"],
                "n_detections": m["n_detections"],
            }
            rows.append(row)
        except Exception as e:
            warnings.warn(
                f"config {cfg.name} failed on ({unit.scenario}, seed={unit.seed}): {e!r}"
            )
    return rows


# Pool workers can't pickle closures; use a module-level wrapper.
_GLOBAL_CONFIGS: list[Config] = []


def _worker(unit: WorkUnit) -> list[dict]:
    return _process_unit(unit, _GLOBAL_CONFIGS)


def _init_pool(configs: list[Config]) -> None:
    global _GLOBAL_CONFIGS
    _GLOBAL_CONFIGS = configs


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = [
        "scenario", "seed", "config_name", "betting_kind", "betting_params",
        "threshold", "detection_mode", "joint_distance",
        "tpr", "fpr", "add", "n_tp", "n_fp", "n_detections",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_pivot_csv(agg: list[dict], path: Path) -> None:
    """Pivot: rows = (scenario, metric); columns = config_name.

    This matches Table III's "network x metric x configuration" layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = sorted({r["config_name"] for r in agg})
    scenarios = sorted({r["scenario"] for r in agg})
    metrics = ["tpr", "fpr", "add"]

    # lookup
    by_key = {(r["scenario"], r["config_name"]): r for r in agg}

    fields = ["scenario", "metric"] + configs
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in scenarios:
            for metric in metrics:
                row = {"scenario": sc, "metric": metric.upper()}
                for cfg in configs:
                    r = by_key.get((sc, cfg), {})
                    v = r.get(f"{metric}_mean")
                    row[cfg] = "" if v is None else f"{v:.3f}"
                w.writerow(row)


def _print_summary(agg: list[dict]) -> None:
    """Print per-config averages (across scenarios) sorted by F1-like score."""
    try:
        from tabulate import tabulate
    except ImportError:
        return

    configs = sorted({r["config_name"] for r in agg})
    rows = []
    for cfg in configs:
        grp = [r for r in agg if r["config_name"] == cfg]
        tprs = [r["tpr_mean"] for r in grp if r["tpr_mean"] is not None]
        fprs = [r["fpr_mean"] for r in grp if r["fpr_mean"] is not None]
        adds = [r["add_mean"] for r in grp if r["add_mean"] is not None]
        rows.append([
            cfg,
            f"{np.mean(tprs):.3f}" if tprs else "-",
            f"{np.mean(fprs):.4f}" if fprs else "-",
            f"{np.mean(adds):.1f}" if adds else "-",
        ])
    # sort by TPR desc, then ADD asc
    rows.sort(key=lambda r: (-(float(r[1]) if r[1] != "-" else 0),
                             float(r[3]) if r[3] != "-" else 1e9))
    print("\nTable III summary (averaged across scenarios):")
    print(tabulate(rows, headers=["Config", "TPR", "FPR", "ADD"], tablefmt="github"))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Reproduce Ali & Ho ICDM 2025 Table III.")
    ap.add_argument("--n-trials", type=int, default=10, help="seeds per scenario")
    ap.add_argument("--out", type=Path, default=Path("results/table3.csv"))
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--scenarios", nargs="*", default=None)
    args = ap.parse_args()

    scenarios = args.scenarios or list(ALL_SCENARIOS.keys())
    configs = default_config_grid()
    work = [WorkUnit(sc, seed) for sc in scenarios for seed in range(args.n_trials)]

    print(f"Running {len(work)} (scenario, seed) units across {args.workers} workers.")
    print(f"Each unit sweeps {len(configs)} configs -> "
          f"{len(work) * len(configs)} total detector runs.")

    t0 = time.time()
    all_rows: list[dict] = []
    with mp.Pool(args.workers, initializer=_init_pool, initargs=(configs,)) as pool:
        iterator = pool.imap_unordered(_worker, work, chunksize=1)
        for batch in tqdm(iterator, total=len(work), desc="(scenario,seed) units"):
            all_rows.extend(batch)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/max(1,len(work)):.2f}s/unit).")

    raw = args.out.with_name(args.out.stem + "_raw.csv")
    _write_raw_csv(all_rows, raw)
    print(f"  raw -> {raw}")
    agg = aggregate(all_rows, group_keys=["scenario", "config_name"])
    _write_pivot_csv(agg, args.out)
    print(f"  pivoted -> {args.out}")
    _print_summary(agg)


if __name__ == "__main__":
    main()
