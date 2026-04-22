# Table III — parameter sensitivity

```bash
.venv/bin/python experiments/run_table3.py --out results/table3.csv
```

Grid sweep over betting function, distance metric, and threshold λ. ~1350 runs. Use `HMD_BACKEND=cupy` + `pip install cupy-cuda13x` for the batched GPU path.

The output CSV mirrors Table III's layout. Paper's "globally optimal" row (mixture + Mahalanobis + λ=50) is reproducible with our default betting `mixture([1/3]*3, [0.7, 0.8, 0.9])`.
