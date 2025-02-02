# Change Point Detection via Martingale Framework

## Overview

Given a data stream $X_1, X_2, \ldots, X_n$, a **change point** at time $k$ indicates that the distribution of $X_{1:k}$ differs from $X_{k+1:n}$. We perform **online** detection, processing observations sequentially and raising alarms for suspected distribution shifts.

## The Martingale Framework

### Core Components

1. **Strangeness Measure**: $\alpha(x)$ quantifies how "unusual" an observation $x$ is
2. **P-value Computation**: Converts strangeness to a probability via conformal prediction
3. **Betting Function**: Updates martingale value based on p-value
4. **Threshold Detection**: Raises alarm when martingale exceeds threshold

### Betting Functions

We support multiple betting functions $\epsilon(p)$ that convert p-values to martingale updates:

1. **Power Martingale**:
   $$
   M_n = M_{n-1} \times \epsilon \times p_n^{\epsilon-1}
   $$
   - Most sensitive to very small p-values
   - Parameter $\epsilon \in (0,1)$ controls sensitivity

2. **Simple Mixture**:
   $$
   M_n = M_{n-1} \times \frac{\epsilon}{p_n}
   $$
   - Linear relationship with inverse p-value
   - More stable than power martingale

3. **Beta Martingale**:
   $$
   M_n = M_{n-1} \times \text{Beta}(p_n; \alpha, \beta)
   $$
   - Uses Beta distribution density
   - Parameters $\alpha, \beta$ control shape of betting

4. **Kernel Martingale**:
   $$
   M_n = M_{n-1} \times K(p_n)
   $$
   - Uses kernel function (e.g., Gaussian)
   - Smooth betting across p-value range

### Strangeness Measures

For graph/network data, we compute strangeness using:

1. **Structural Features**:
   - Node degree distribution
   - Clustering coefficients
   - Betweenness centrality
   - Eigenvector centrality
   - Graph density
   - Average path length
   - Connected components

2. **Distance Metrics**:
   - Euclidean distance in feature space
   - Graph edit distance
   - Spectral distance
   - NetSimile distance

### P-value Computation

Given strangeness scores $\{\alpha_1, \ldots, \alpha_n\}$:

$$
p_n = \frac{|\{i: \alpha_i > \alpha_n\}| + \theta|\{i: \alpha_i = \alpha_n\}|}{n}
$$

where $\theta \sim U(0,1)$ breaks ties randomly.

Properties:
- Under null (no change): $p_n \sim U(0,1)$
- Under alternative: $p_n$ tends to be small
- Exchangeability is key assumption

### Horizon Martingale

Extension using predicted future states:

$$
\hat{M}_t = M_{t-1} \times \prod_{j=1}^h \epsilon(p_{t+j})^{\epsilon-1}
$$

where:
- $h$ is prediction horizon length
- $\hat{p}_{t+j}$ is p-value for predicted state at $t+j$
- Uses same previous value $M_{t-1}$ as traditional martingale

## Implementation Details

### Key Files

- `detector.py`: Main change point detector class
- `martingale.py`: Martingale computation and betting functions
- `strangeness.py`: Strangeness measures and p-value computation
- `predictor/`: Graph prediction models for horizon martingale

### Usage

```python
detector = ChangePointDetector(
    betting_function="power",  # or "mixture", "beta", "kernel"
    epsilon=0.1,              # sensitivity parameter
    threshold=20.0,           # detection threshold
    window_size=100,          # rolling window size
    reset=True               # reset after detection
)

# Single-view detection
results = detector.detect(data_stream)

# Multi-view detection
results = detector.detect_multiview(feature_streams)
```

### Reset Strategy

- **Traditional Martingale**: Resets to 1.0 immediately after detection
- **Horizon Martingale**: Only resets when traditional martingale confirms change
- **Window**: Cleared on traditional martingale detection

## Mathematical Properties

1. **Martingale Property**:
   Under null hypothesis (no change):
   $$
   \mathbb{E}[M_n | M_1,\ldots,M_{n-1}] = M_{n-1}
   $$

2. **Growth Rate**:
   Under alternative (after change):
   $$
   \mathbb{E}[\log M_n] \approx n \times \text{KL}(P_1\|P_0)
   $$
   where KL is Kullback-Leibler divergence

3. **False Alarm Rate**:
   By Ville's inequality:
   $$
   \mathbb{P}(\sup_n M_n \geq \lambda) \leq \frac{1}{\lambda}
   $$
   for threshold $\lambda$

## References

1. Ho, S. S., & Wechsler, H. (2005). A martingale framework for detecting changes in data streams by testing exchangeability.
2. Vovk, V. et al. (2005+). Conformal Prediction frameworks.
3. Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction.
4. Doob, J. L. (1953). Stochastic Processes.
5. Newman, M. E. J. (2010). Networks: An Introduction.
