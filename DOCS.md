# Horizon Martingale Detection Framework

> Reference [paper](early_detection_and_attribution_of_structural_changes_in_dynamic_networks_ICDM.pdf) for complete understanding.

### Problem Formulation

Dynamic networks evolve through structural changes modeled as parametric shifts in the generative process:

```math
\boldsymbol{\theta}_t = \begin{cases}
\boldsymbol{\theta}_0 & \text{for } t < \tau \\
\boldsymbol{\theta}_1 \neq \boldsymbol{\theta}_0 & \text{for } t \geq \tau
\end{cases}
```
where $\boldsymbol{\theta}_t$ controls network generation through parameters such as connection probabilities, community structures, and growth dynamics. These parametric shifts induce distributional changes in observable graph-theoretic features, including degree distributions, clustering coefficients, and spectral characteristics, enabling detection through statistical monitoring of feature sequences.

The change detection problem minimizes expected detection delay while controlling false alarms:

```math
\min_{\tau^*} \mathbb{E}[(\tau^* - \tau)^+ \mid \tau^* \geq \tau] \quad \text{subject to} \quad \mathbb{P}(\tau^* < \tau) \leq \alpha
```
where $\tau^\*$ is the stopping time for the detection rule, $(\tau^* - \tau)^+ = \max\{0, \tau^* - \tau\}$ represents detection delay, and $\alpha \in (0,1)$ is the acceptable false alarm probability.


### Martingale Framework

#### Traditional Martingale Construction

1. **Non-conformity Score**: $S_t = ||X_t - C_t||$ where $C_t$ is the cluster center from historical observations
2. **Conformal P-value**: $p_t = \frac{\\#\\{s : S_s > S_t\\} + \theta_t\\#\\{s : S_s = S_t\\}}{t}$
3. **Martingale Update**: $M_t = M_{t-1} \cdot g(p_t)$ where $g$ is a valid betting function

### Horizon Martingale Extension

**Key Innovation**: Horizon martingales accumulate evidence from predicted future states:

```math
M_{t,h}^{(k)} = M_{t-1}^{(k)} \cdot g(p_{t,h}^{(k)})
```

where $p_{t,h}^{(k)}$ are predictive p-values computed from forecasted features $\hat{X}_{t+h}^{(k)}$.

**Theoretical Guarantees**:
- **Martingale Property**: Preserved under proper forecasting calibration
- **False Alarm Control**: $\mathbb{P}(\tau_h^{(1)} < \infty \mid \mathcal{H}_0) \leq \frac{1}{\lambda}$ via Ville's inequality

### Feature Attribution

**Martingale-Shapley Equivalence**: Each feature's martingale value equals its Shapley value, providing exact attribution with $O(K)$ complexity instead of $O(2^K)$.

Relative contribution: $\psi_k(t) = \frac{M^{(k)}_t}{M^A_t} \times 100\\%$

## Implementation Architecture

### Core Modules

```
src/
├── algorithm.py                    # Main detection pipeline
├── changepoint/                    # Change detection algorithms
│   ├── detector.py                 # Unified detector interface
│   ├── martingale_traditional.py   # Standard martingale
│   ├── martingale_horizon.py       # Horizon martingale
│   ├── betting.py                  # Betting function implementations
│   ├── strangeness.py              # Non-conformity score computation
│   ├── distance.py                 # Distance metrics (6 types)
│   ├── cusum.py                    # CUSUM baseline
│   └── ewma.py                     # EWMA baseline
├── forecast/                       # Forecasting models
│   ├── arima.py                    # ARIMA-based forecasting
│   ├── grnn.py                     # Neural network forecasting
│   └── hybrid.py                   # Hybrid ARIMA-GRNN
├── graph/                          # Graph generation and features
│   ├── generator.py                # Dynamic graph sequences
│   ├── features.py                 # Network feature extraction
│   └── evolution.py                # Parameter evolution
├── predictor/                      # Graph prediction algorithms
│   └── graph.py                    # Hybrid temporal-structural prediction
├── utils/                          # Analysis and visualization
│   ├── data_utils.py               # Feature normalization
│   └── analysis_utils.py           # Performance metrics
└── scripts/                        # Research analysis tools
    ├── run_table_iv_experiments.py # Comparative evaluation
    └── parameter_sweep.py          # Sensitivity analysis
```

### Detection Pipeline

The main pipeline in [`src/algorithm.py`](src/algorithm.py) implements an 8-step process:

1. **Graph Generation** ([`src/graph/generator.py`](src/graph/generator.py)): Creates temporal sequences with controlled change points
2. **Feature Extraction** ([`src/graph/features.py`](src/graph/features.py)): Computes 8 network features:
   ```python
   # From src/graph/features.py
   features = [
       "mean_degree", "density", "mean_clustering", # Local connectivity
       "mean_betweenness", "mean_eigenvector", "mean_closeness", # Centrality
       "max_singular_value", "min_nonzero_laplacian" # Spectral
   ]
   ```
3. **Forecasting** ([`src/forecast/`](src/forecast/)): Optional future state prediction
4. **Normalization** ([`src/utils/data_utils.py`](src/utils/data_utils.py)): Feature standardization
5. **Detection** ([`src/changepoint/detector.py`](src/changepoint/detector.py)): Parallel traditional and horizon martingale computation
6. **Attribution**: Feature importance via Martingale-Shapley equivalence

### Betting Functions

Implemented in [`src/changepoint/betting.py`](src/changepoint/betting.py) with rigorous validation:

| Function | Formula | Implementation | Parameters |
|----------|---------|----------------|------------|
| Power | $g(p) = \epsilon p^{\epsilon-1}$ | `PowerBetting` | $\epsilon \in (0,1)$ |
| Mixture | $g(p) = \frac{1}{K}\sum_{i=1}^K \epsilon_i p^{\epsilon_i-1}$ | `MixtureBetting` | $\{\epsilon_i\}$ |
| Beta | $g(p) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$ | `BetaBetting` | $\alpha, \beta > 0$ |

```python
# From src/changepoint/betting.py - Power betting example
def __call__(self, prev_m: float, pvalue: float) -> float:
    if pvalue == 0: return float("inf")
    if pvalue == 1: return 0.0
    return prev_m * self.params.epsilon * (pvalue ** (self.params.epsilon - 1))
```

### Network Models

Configured in [`src/configs/models.yaml`](src/configs/models.yaml) with precise parameter ranges:

| Model | Parameters | Implementation | Change Scenarios |
|-------|------------|----------------|------------------|
| **SBM** | $p_{\text{intra}} \in [0.3,0.95]$, $p_{\text{inter}} \in [0.01,0.3]$ | `nx.stochastic_block_model` | Community merge, density change |
| **BA** | $m \in [1,6]$ (edges per node) | `nx.barabasi_albert_graph` | Hub formation, parameter shift |
| **ER** | $p \in [0.05,0.4]$ (edge probability) | `nx.erdos_renyi_graph` | Density increase/decrease |
| **NWS** | $k \in [4,8]$, $p_{\text{rewire}} \in [0.05,0.15]$ | `nx.watts_strogatz_graph` | Rewiring, topology shift |

```python
# From src/graph/generator.py - SBM example
def sbm_generator(n, num_blocks, intra_prob, inter_prob, **kwargs):
    sizes = [n // num_blocks] * (num_blocks - 1)
    sizes.append(n - sum(sizes))
    p = np.full((num_blocks, num_blocks), inter_prob)
    np.fill_diagonal(p, intra_prob)
    return nx.stochastic_block_model(sizes=sizes, p=p, seed=self.rng)
```

## Implementation Details

### Non-conformity Score Computation

From [`src/changepoint/strangeness.py`](src/changepoint/strangeness.py):

```python
def strangeness_point(data, config=None, random_state=None):
    """Compute strangeness as minimum distance to cluster centers.
    
    S_t = min_j d(X_t, c_j) where c_j are cluster centers
    """
    # K-means clustering with K=1 (robust location estimator)
    model = KMeans(n_clusters=config.n_clusters, random_state=random_state)
    model.fit(data_array)
    
    # Compute distances to all cluster centers
    distances = compute_cluster_distances(data_array, model, config.distance_config)
    return distances.min(axis=1)  # Minimum distance = strangeness
```

### Distance Metrics

Six distance metrics implemented in [`src/changepoint/distance.py`](src/changepoint/distance.py):

```python
# Euclidean: d(x,y) = √(Σᵢ(xᵢ-yᵢ)²)
def _compute_euclidean(x, y, eps):
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    xy = np.dot(x, y.T)
    return np.sqrt(np.maximum(x2 + y2 - 2*xy, 0))

# Mahalanobis: d(x,y) = √((x-y)ᵀΣ⁻¹(x-y))
def _compute_mahalanobis(x, y, config):
    combined = np.vstack([x, y])
    cov = np.cov(combined, rowvar=False) + np.eye(cov.shape[0]) * config.cov_reg
    inv_cov = np.linalg.pinv(cov)
    # ... (see full implementation)
```

### Horizon Martingale Implementation

From [`src/changepoint/martingale_horizon.py`](src/changepoint/martingale_horizon.py):

```python
def compute_horizon_martingale(data, predicted_data, config):
    """Implement M_{t,h}^{(k)} = M_{t-1}^{(k)} * g(p_{t,h}^{(k)})"""
    
    # Exponential decay weighting for multiple horizons
    decay_rate = 0.8  # Emphasizes near-term predictions
    horizon_weights = [np.exp(-decay_rate * h) for h in range(num_horizons)]
    
    # Combine horizons using weighted sum
    horizon_val = sum(w * m for w, m in zip(horizon_weights, horizon_martingales_at_t))
    
    # Cooldown mechanism to prevent rapid successive detections
    if i - state.last_detection_time < config.cooldown_period:
        detect_change = False
```

## Configuration

### Basic Usage

```bash
# Run with default configuration
python src/run.py -c src/configs/algorithm.yaml

# Override key parameters  
python src/run.py -c src/configs/algorithm.yaml \
  --network sbm \
  --threshold 50 \
  --betting-func mixture \
  --distance mahalanobis
```

### Parameter Sensitivity Analysis

Comprehensive evaluation in [`src/run_parameter_sweep.py`](src/run_parameter_sweep.py):

```bash
# Run 320 experiments with parallel processing
python src/run_parameter_sweep.py --workers 8

# Experiments generated:
# - 4 networks × 4 distances × 4 epsilons = 64 power betting
# - 4 networks × 4 distances × 4 alphas × 3 betas = 192 beta betting  
# - 4 networks × 4 distances = 16 mixture betting
# - 4 networks × 4 distances × 3 thresholds = 48 threshold analysis
```

### Key Configuration Parameters

From [`src/configs/algorithm.yaml`](src/configs/algorithm.yaml):

```yaml
detection:
  method: "martingale"           # martingale, cusum, ewma
  threshold: 60.0                # Detection threshold λ
  prediction_horizon: 5          # Forecast horizon h
  cooldown_period: 30            # Minimum timesteps between detections
  betting_func_config:
    name: "mixture"              # power, mixture, beta
    mixture:
      epsilons: [0.7, 0.8, 0.9]  # Sensitivity parameters
  distance:
    measure: "mahalanobis"       # euclidean, mahalanobis, cosine, chebyshev
    p: 2.0                       # Order for Minkowski distance

model:
  type: "multiview"              # multiview or single_view
  predictor:
    type: "graph"                # graph, arima, grnn, hybrid
    config:
      alpha: 0.8                 # EWMA decay parameter
      n_history: 10              # Historical window size
```

## Experimental Results

### Performance Summary

Results from [`src/scripts/run_table_iv_experiments.py`](src/scripts/run_table_iv_experiments.py):

| Network | Delay Reduction | TPR Improvement | Dominant Feature | Implementation |
|---------|----------------|-----------------|------------------|----------------|
| SBM | 13.1-23.8% | 0-2% | Spectral (20.7-34.8%) | Community structure changes |
| ER | 16.9% | 2.9-4.0% | Spectral (20.2%) | Global density shifts |
| BA | 14.5% | 2.9-5.9% | Spectral (35.4%) | Hub formation dynamics |
| NWS | 21.2-24.8% | 0% | Closeness/Laplacian (37.6-55.7%) | Small-world transitions |
| MIT Reality | 22.2% | 7.9% | Betweenness (21.6%) | Academic event detection |

### Computational Complexity

From [`src/changepoint/detector.py`](src/changepoint/detector.py) and [`src/graph/features.py`](src/graph/features.py):

- **Martingale Updates**: $O(K \cdot T)$ where $K$ = features, $T$ = timesteps
- **Feature Extraction**: $O(n^3 \cdot T)$ (centrality computation dominates)
- **Horizon Overhead**: $<5\%$ additional cost over traditional martingale
- **Memory**: $O(K \cdot w)$ for rolling windows, $w$ = history size

### Testing and Validation

Comprehensive test suite in [`tests/`](tests/):

```python
# From tests/test_changepoint/test_detector.py
def test_multiview_detection():
    """Test multiview change point detection with synthetic data"""
    detector = ChangePointDetector(DetectorConfig(
        method="multiview", threshold=20.0, history_size=5
    ))
    
    # Generate data with change point
    data = np.concatenate([
        np.random.normal(0, 1, (25, 3)),  # Pre-change
        np.random.normal(3, 1, (25, 3))   # Post-change
    ])
    
    results = detector.run(data)
    assert "traditional_change_points" in results
    assert "horizon_change_points" in results
```

## Reproducibility

From [`src/run_parameter_sweep.py`](src/run_parameter_sweep.py) - all experiments use deterministic seeding:

```python
# Deterministic seeding based on experiment parameters
param_string = str(sorted(experiment.get("parameters", {}).items()))
seed_hash = hashlib.md5(param_string.encode()).hexdigest()
deterministic_seed = int(seed_hash[:8], 16) % (2**31 - 1)
```

**Parameter Sensitivity Analysis**: 320 experiments with parallel processing
- **Power betting**: 64 experiments across network types and distance metrics
- **Beta betting**: 192 experiments with comprehensive α,β parameter grid
- **Mixture betting**: 16 experiments with optimal epsilon combinations
- **Threshold analysis**: 48 experiments across λ ∈ {20, 50, 100}
