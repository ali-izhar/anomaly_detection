"""Reproduce Figure 2 of Ali & Ho (ICDM 2025): martingale traces.

Figure 2 shows log M_t^A (traditional) and log M_t^H (horizon) on a single
seed=0 trial for four representative scenarios, with threshold and ground-
truth change points annotated. This is the canonical qualitative figure that
illustrates WHY horizon leads traditional in early detection.

Usage
-----

    .venv/bin/python experiments/figure2.py --out results/figure2.png

Layout
------
2 x 2 grid:
    top-left:     SBM community merge
    top-right:    ER density increase
    bottom-left:  BA parameter shift
    bottom-right: NWS rewiring increase

Per panel:
    * logM_traditional in blue, logM_horizon in orange
    * horizontal red dashed line at log(threshold)
    * vertical gray dashed line(s) at ground-truth change points
    * diamond markers at detected change points (colored by which stream fired)

Why this design
===============

**One seed, not an average.** Figure 2 in the paper is a trajectory plot; we
pick seed=0 deterministically for reproducibility. A seed-averaged logM plot
would smear the change points and obscure the early-detection lead.

**Log scale on Y, linear on X.** The paper plots in log-M (same axis as
log(threshold)), which is the natural scale for a multiplicative process.

**seaborn whitegrid.** Clean, minimal style; 300 DPI for paper-quality
reproduction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hmd import HorizonDetector  # noqa: E402
from hmd.data.synthetic import ALL_SCENARIOS  # noqa: E402


PANELS = [
    ("sbm_community_merge", "SBM: community merge"),
    ("er_density_increase", "ER: density increase"),
    ("ba_parameter_shift", "BA: parameter shift"),
    ("nws_rewiring_increase", "NWS: rewiring increase"),
]


def _run_one(scenario_key: str, seed: int = 0, threshold: float = 50.0):
    seq = ALL_SCENARIOS[scenario_key](seed=seed)
    det = HorizonDetector(
        threshold=threshold,
        startup_period=20,
        enable_traditional=True,
        enable_horizon=True,
    )
    r = det.run(seq.graphs)
    return seq, r


def main():
    ap = argparse.ArgumentParser(description="Reproduce Figure 2 martingale traces.")
    ap.add_argument("--out", type=Path, default=Path("results/figure2.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    log_thr = np.log(args.threshold)
    color_trad = "#1f77b4"   # blue
    color_hrzn = "#ff7f0e"   # orange
    color_thr = "#d62728"    # red
    color_truth = "#555555"  # gray

    for ax, (key, title) in zip(axes, PANELS):
        seq, res = _run_one(key, seed=args.seed, threshold=args.threshold)
        T = len(seq.graphs)
        t_axis = np.arange(T)

        # Plot log-martingale curves (NaNs inside the startup period are
        # skipped automatically by matplotlib).
        ax.plot(t_axis, res.logM_traditional, color=color_trad, lw=1.8, label="Traditional $\\log M^A_t$")
        ax.plot(t_axis, res.logM_horizon, color=color_hrzn, lw=1.8, label="Horizon $\\log M^H_t$")
        ax.axhline(log_thr, color=color_thr, lw=1.2, ls="--", label=f"$\\log\\lambda={log_thr:.2f}$")

        # Ground truth.
        for cp in seq.change_points:
            ax.axvline(cp, color=color_truth, lw=1.0, ls=":", alpha=0.8)

        # Mark detections: diamond marker, colored by which stream first crossed.
        for tau in res.change_points:
            trad_crossed = (
                not np.isnan(res.logM_traditional[tau])
                and res.logM_traditional[tau] >= log_thr
            )
            hrzn_crossed = (
                not np.isnan(res.logM_horizon[tau])
                and res.logM_horizon[tau] >= log_thr
            )
            if hrzn_crossed and not trad_crossed:
                ax.plot(tau, res.logM_horizon[tau], marker="D",
                        markersize=8, markerfacecolor=color_hrzn,
                        markeredgecolor="black", lw=0)
            elif trad_crossed:
                ax.plot(tau, res.logM_traditional[tau], marker="D",
                        markersize=8, markerfacecolor=color_trad,
                        markeredgecolor="black", lw=0)

        ax.set_title(f"{title}  (true CPs: {seq.change_points})", fontsize=11)
        ax.set_ylabel("$\\log M_t$")
        ax.set_xlim(0, T)
        # Give the plot a reasonable y-floor so pre-startup NaNs don't distort.
        finite_vals = np.concatenate([
            res.logM_traditional[np.isfinite(res.logM_traditional)],
            res.logM_horizon[np.isfinite(res.logM_horizon)],
        ])
        if finite_vals.size > 0:
            ymin = min(float(finite_vals.min()), -5.0)
            ymax = max(float(finite_vals.max()), log_thr + 2.0)
            ax.set_ylim(ymin, ymax * 1.05 if ymax > 0 else ymax)

    for ax in axes[-2:]:
        ax.set_xlabel("Time step $t$")

    # Shared legend in the top-left panel only.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize=9, framealpha=0.92)

    fig.suptitle(
        f"Figure 2: Martingale traces (seed={args.seed}, $\\lambda$={args.threshold:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {args.out} ({args.dpi} DPI).")


if __name__ == "__main__":
    main()
