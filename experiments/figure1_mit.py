"""Reproduce paper Figure 1 — MIT Reality Martingale vs Horizon traces.

Paper reference: Fig 1 (Ali & Ho, ICDM 2025, page 7).

Usage
-----
    .venv/bin/python experiments/figure1_mit.py \\
        --threshold 20 --startup 20 --out results/figure1.png

Why this script is independent from Figure 2
---------------------------------------------
MIT Reality has genuine calendar-aligned events; Figure 2 uses i.i.d. synthetic
generators with injected change points. The plotting aesthetic differs (MIT
annotates holidays; synthetic annotates "change point t=X"). Keeping them as
separate scripts prevents cross-contamination of styling and argument defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hmd import HorizonDetector
from hmd.data.mit_reality import MIT_EVENTS, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=20.0, help="Detection λ (paper: 20 for MIT)")
    ap.add_argument("--startup", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--history", type=int, default=20)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--edge-threshold", type=float, default=0.0, help="prob2 cutoff for MIT edges")
    ap.add_argument("--out", type=Path, default=Path("results/figure1.png"))
    args = ap.parse_args()

    seq, meta = load(threshold=args.edge_threshold)
    print(f"Loaded MIT: {meta.n_days} days, {meta.n_users} users, edge-threshold={meta.threshold}")

    det = HorizonDetector(
        threshold=args.threshold,
        startup_period=args.startup,
        horizon=args.horizon,
        history_size=args.history,
        detection_mode="per_feature",
        normalize_features=args.normalize,
    )
    result = det.run(seq.graphs, true_change_points=seq.change_points, scenario="mit_reality")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    t = np.arange(meta.n_days)
    M_t = result.M_traditional
    M_h = result.M_horizon
    log_thr = np.log(args.threshold)

    # Plot in log scale so spikes are visible alongside baseline.
    ax.semilogy(t, np.where(np.isfinite(M_t) & (M_t > 0), M_t, np.nan),
                color="#1f77b4", linewidth=1.5, label="Traditional Martingale")
    ax.semilogy(t, np.where(np.isfinite(M_h) & (M_h > 0), M_h, np.nan),
                color="#ff7f0e", linewidth=1.5, label="Horizon Martingale")

    # Threshold line.
    ax.axhline(args.threshold, color="red", linestyle="--", linewidth=1.0, label=f"λ = {args.threshold:g}")

    # True event markers (vertical dashed).
    for day, label in MIT_EVENTS.items():
        ax.axvline(day, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.annotate(label.split("/")[0].strip(), (day, 0.5), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(2, 5), fontsize=7, rotation=90,
                    color="gray", ha="left", va="bottom")

    # Detections as diamond markers on whichever stream triggered.
    for cp in result.change_points:
        y_val = M_t[cp] if (np.isfinite(M_t[cp]) and M_t[cp] >= args.threshold) else M_h[cp]
        if np.isfinite(y_val) and y_val > 0:
            ax.plot(cp, y_val, marker="D", markersize=7, markerfacecolor="none",
                    markeredgecolor="black", markeredgewidth=1.2, zorder=5)

    ax.set_xlabel("Day (since 2008-09-19)")
    ax.set_ylabel("Martingale (log scale)")
    ax.set_title("MIT Reality — Martingale vs Horizon (log $M_t$)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xlim(0, meta.n_days - 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.out}")
    print(f"Detections: {result.change_points}")
    print(f"True events: {sorted(MIT_EVENTS.keys())}")


if __name__ == "__main__":
    main()
