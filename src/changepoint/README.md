# Change Point Detection

Detect significant changes in sequential data, especially in dynamic networks, using martingale-based statistical methods.

## Components

- **`martingale.py`**: Implements martingale-based change point detection algorithms.
- **`detector.py`**: Integrates graph feature extraction with change point detection.

## Theory

### Change Points

A change point in a sequence $X_1, X_2, \ldots, X_n$ exists at time $k$ if the distributions of $X_{1:k}$ and $X_{k+1:n}$ differ significantly.

### Martingale Theory

A martingale $\\{M_n\\}_{n \geq 0}$ satisfies:

$$
E[M_{n+1} \mid M_1, \ldots, M_n] = M_n
$$

**Comparison with Markov Processes:**

- **Markov Process**: Future state depends only on the present state. *Example*: In the PageRank algorithm, the next page visited depends solely on the current page.
- **Martingale**: Expected future value equals the present value. *Example*: In fair betting games, the expected winnings tomorrow equal the current winnings. In financial modeling, stock prices are modeled as martingales assuming that future price movements are independent of past trends. Or in a random walk, the expected position equals the current position.

### Application to Change Detection

Under stable conditions, the martingale remains fair. A deviation indicates a potential change point.

#### Martingale Update

$$
M_n = M_{n-1} \cdot \epsilon \cdot \frac{p_n}{\epsilon - 1}
$$

- $p_n$: p-value at time $n$. If the current data is normal, $p_n$ is high, and $M_n$ decreases or remains stable. However, if the data is unusual, $p_n$ is low, and $M_n$ increases, signaling a change.
- $\epsilon$: Sensitivity parameter (0.5 to 0.99). $\epsilon$ controls the influence of the p-value on the martingale update. For instance, if $\epsilon = 0$, then $\frac{\epsilon}{1 - \epsilon} = 0$, and the martingale update diminishes regardless of $p_n$, making detection ineffective. On the other hand, if $\epsilon = 1$, then $\frac{\epsilon}{1 - \epsilon} = \infty$, and the martingale update increases indefinitely, leading to frequent false alarms.

    - $\epsilon \approx 0.95$: Conservative, fewer false alarms.
    - $\epsilon \approx 0.5$: Sensitive, detects subtle changes.
    - **Default**: $\epsilon = 0.8$.

#### P-value and Strangeness

For each observation $i$, compute the p-value $p_i$ as:

$$
p_i = \frac{|\{j : \alpha_j > \alpha_i\}| + \theta|\{j : \alpha_j = \alpha_i\}|}{i}
$$

Where $\alpha_i$ is the strangeness of the $i$-th observation and $\theta$ is a tiebreaker parameter (typically between 0 and 1). If the strangeness is high, it indicates that the observation is unusual compared to past data, thus the p-value is low. On the other hand, if the strangeness is low, it indicates that the observation is typical, thus the p-value is high.

- **Low $p_i$**: Few past observations are more strange, indicating a potential change point.
- **High $p_i$**: Many past observations are more strange, suggesting normal variability.


### Centrality Measures

#### 1. Degree Centrality: "The Popularity Contest"

$$
C_D(v) = \frac{\text{degree}(v)}{N-1}
$$

Measures how many direct connections a node has relative to the maximum possible. For example, counting the number of followers of a celebrity. A celebrity with many followers has a high degree centrality score. A regular user with few connections has a low degree centrality score. A use case is identifying social influencers.

#### 2. Betweenness Centrality: "The Bridge Builder"

$$
C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
$$

Quantifies the number of times a node acts as a bridge along the shortest path between two other nodes. For example, a manager who connects different departments. A manager who connects many departments has a high betweenness centrality score. A manager who connects few departments has a low betweenness centrality score. A use case is finding information bottlenecks.

#### 3. Eigenvector Centrality: "Connected to VIPs"

$$
C_E(v) = \frac{1}{\lambda} \sum_{u \in G} A_{vu} C_E(u)
$$

Measures a node's influence based on the influence of its neighbors. For example, a paper cited by important papers has a high eigenvector centrality score. A paper with few citations has a low eigenvector centrality score. A use case is identifying key influencers.

#### 4. Closeness Centrality: "The Quick Communicator"

$$
C_C(v) = \frac{N-1}{\sum_{u \neq v} d(u,v)}
$$

Measures how quickly a node can access all other nodes in the network. For example, a central hospital has a high closeness centrality score. A remote clinic has a low closeness centrality score. A use case is optimal resource placement.

## References

1. Ho, S. S., & Wechsler, H. (2005). "A martingale framework for detecting changes in data streams by testing exchangeability." IEEE TPAMI.
2. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
3. Doob, J. L. (1953). Stochastic Processes. John Wiley & Sons. (Martingale Theory)
