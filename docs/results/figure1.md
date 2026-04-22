# Figure 1 — MIT Reality

```bash
.venv/bin/python experiments/figure1_mit.py --threshold 20 --out results/figure1.png
cp results/figure1.png docs/assets/figure1.png  # if you want it in the built site
```

Loads `hmd/data/mit_reality/Proximity.csv` (bundled with the package), aggregates to 24-h snapshots aligned to day-0 = 2008-09-19 (paper Table II).

## Events (paper Table II)

| Day | Date | Event |
|---:|---|---|
| 23 | 2008-10-12 | Columbus Day / Fall Break |
| 68 | 2008-11-26 | Thanksgiving Holiday |
| 97 | 2008-12-25 — 2009-01-01 | Christmas / New Year |
| 173 | 2009-03-15 | Spring Break |
| 234 | 2009-05-15 | End of Spring Semester |

## Caveats

- Our loader finds 74 active users in the paper's 289-day window; paper reports 94 (inactive users dropped).
- Full numerical reproduction of the paper's MIT Fig 1 requires a trend-aware forecaster (e.g. `hmd.forecaster.HoltForecaster`).
