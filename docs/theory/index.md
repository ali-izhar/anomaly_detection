# Theory overview

The paper has two layers. Each page walks through the math alongside the code that implements it.

| Section | What's proved | Implementation |
|---|---|---|
| [§III — Martingale framework](martingale.md) | Def 2 non-conformity, Def 3 p-value, Def 5 betting, Def 6 martingale recurrence, Thm 1 validity, Thm 2 Ville, Cor 1 sum | `hmd/{conformal,betting,martingale}.py` |
| [§IV — Horizon extension](horizon.md) | Def 7 horizon recurrence, Def 8 calibrated forecaster, Thm 3, Thm 4 | `hmd/detector.py` |
| [§IV-B — Forecasting](forecasting.md) | Eq 20, 21 EWMA | `hmd/forecaster.py` |
| [§III-D — Attribution](attribution.md) | Martingale-Shapley equivalence | `hmd/attribution.py` |

## Two-layer claim

**Layer 1 (§III, existing framework)** — conformal p-values → betting-function martingale → Ville's inequality bounds false-alarm probability by 1/λ.

**Layer 2 (§IV, paper's contribution)** — a properly calibrated forecaster lets us run a *second* martingale on predictive p-values, so evidence accumulates from forecasted future states. Under Def 8, the horizon stream is also a test martingale (Thm 3) with the same 1/λ guarantee (Thm 4).
