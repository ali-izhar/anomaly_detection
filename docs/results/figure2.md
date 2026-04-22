# Figure 2 — synthetic network traces

```bash
.venv/bin/python experiments/figure2.py --out results/figure2.png
```

2×2 grid of martingale traces: SBM community-merge, ER density-increase, BA parameter-shift, NWS rewiring-increase. Each panel overlays Traditional (blue) and Horizon (orange) on a single seed=0 run with threshold, true CPs, and detections marked.

The output is at `results/figure2.png`. Copy to `docs/assets/` to include in the built site.
