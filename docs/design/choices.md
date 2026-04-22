# Design choices

Every non-obvious implementational decision in `hmd/`, justified briefly.

## Features

**Why mean-aggregate node-level features?** Permutation invariance — nodes have no canonical ordering across time. Mean is 1-Lipschitz smoother than max under single-node perturbation.

**Why 8 features (including `density`)?** Density is a natural global scalar, easy to compute, complements the centrality means. Drop with `features=default_set()[:7]` for strict-paper runs.

**Why unnormalized Laplacian `L = D − A`?** Paper §III-A says so. Eigenvalues correlate with density; non-conformity scoring re-ranks, so it's tolerable.

## Conformal / betting

**Vovk-smoothed p-value.** We use the canonical Vovk-Gammerman-Shafer 2005 form: `(greater + θ·(1+equal)) / (t+1)`. Test sample in both numerator and denominator; `p_t | F_{t-1} ∼ Unif(0, 1]` exactly under exchangeability.

**Default betting `mixture([1/3, 1/3, 1/3], [0.7, 0.8, 0.9])`.** Three ε values in the conservative half of (0, 1); chosen so g(p) stays near 1 under H₀ and avoids martingale-bankroll depletion. Reproduces Table IV's TPR≈1.0.

**Eps clip `1e-10`.** Prevents `g(0) = ∞` if a p-value hits 0 exactly. Calibration bias is `ε · eps_clip^ε` — negligible.

## Martingale

**Log-space accumulation.** `M_t = ∏ g(p_i)` overflows float64 around T=700; `logM` stays in range at any practical T.

**Horizon stream uses its own accumulator** per Algorithm 1 line 13: `M_t^(k,h) = M_{t-1,h}^(k) · g(p_{t,h}^(k))`. This is the form Thm 4's Ville application requires.

**Reset on detection zeros M but keeps feature/S history.** Matches paper Alg 1 lines 20-21. Historical reference pool keeps growing for stable post-reset p-values.

**Sum across features (not product).** Corollary 1's sum requires only per-feature martingale property; product would need feature independence.

## Detector glue

**`startup_period = 20` default.** Running-centroid SE is O(1/√t); for small t the non-conformity distribution varies → p-values fail rank-uniformity. 20 matches paper's own `w_cal = 0.1T`.

**`cooldown = 20` default.** Matches eval window Δ=20; avoids consecutive-step duplicate detections from the same change. Set 0 for paper-strict replication.

**Horizon detection rule triggers on either stream** (Alg 1 line 18). Union bound gives `P(FP) ≤ 2/λ`.

**`horizon_weights`: None (single-h) or tuple (Vovk-Wang mixture).** Mixture is the valid multi-horizon form — linearity of expectation preserves Ville exactly.

## Synthetic data

**I.i.d. per snapshot within a regime.** Exchangeability under H₀ is what Thm 1 requires. We tested Gaussian drift; no meaningful benefit.

## Backend

See [backend.md](backend.md). NumPy on CPU is faster than cupy on GPU for single-run detection at N=50. GPU only helps for batched experiments.
