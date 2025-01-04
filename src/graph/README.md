# Graph Generator for Dynamic Graph Sequences

This modules provides a **framework** for generating **dynamic graph sequences** using various **NetworkX** models. Each model has adjustable parameters that can evolve over time (through random or systematic changes), creating segments of graphs with distinct structural characteristics.  

## Overview of the Generator

**Key Features**  
1. **Generic Interface**: A single class, `GraphGenerator`, can produce dynamic sequences using multiple built-in graph models (e.g., Barabási-Albert, Erdős-Rényi, Watts-Strogatz, etc.).  
2. **Parameter Evolution**: Many parameters can change slightly at each time step (controlled by standard deviation fields ending in `_std`), simulating **gradual** structural changes.  
3. **Sudden Changes (Anomalies)**: You can specify **minimum** and **maximum** bounds for certain parameters (`min_*`, `max_*`), enabling **jump** changes at random "change points."  
4. **Consistent Node Labeling**: Generated graphs keep a consistent labeling strategy $0, 1, \ldots, n-1$.  
5. **Metadata**: Each sequence return includes **metadata** describing the parameter changes, detected segments, etc.

**Typical Workflow**  
1. **Define** the **model** (e.g., `"barabasi_albert"`, `"erdos_renyi"`, etc.).  
2. **Create** parameter objects (e.g., `BAParams`, `ERParams`) specifying:
   - The total sequence length.
   - The number of nodes $n$.
   - Ranges for anomaly injection (`min_*`, `max_*`).
   - Standard deviations for gradual evolution (`*_std`).
3. **Generate** a graph sequence by calling `generate_sequence()`.  
4. Get a dictionary of results including:
   - `graphs`: A list of adjacency matrices for each time step.
   - `change_points`: The time indices where parameters jump to new values.
   - `parameters`: The parameter sets actually used in each segment.
   - `metadata`: Additional info (e.g., evolving parameters).

## Graph Models

Below are the **core** graph models currently supported.

### 1. Barabási-Albert (BA) Model
- Start with a small seed network of $m_0$ nodes.  
- Each new node is added with $m$ edges, **preferentially** attaching to existing nodes with probability proportional to their degree.  
- Yields a **scale-free** network (power-law degree distribution).

$$
\text{Prob}(\text{new node connects to node } i) = \frac{\deg(i)}{\sum_j \deg(j)}
$$

**Examples**:  
- **Social networks** where "popular" individuals (high degree) attract more new connections over time.
- **Citation networks** where papers that already have many citations are more likely to receive new citations.

---

### 2. Watts-Strogatz (WS) Model
- Begin with a ring of $n$ nodes, each connected to its $k_\mathrm{nearest}$ immediate neighbors.  
- Rewire each edge with probability $p$ to a random node, introducing shortcuts.  
- Produces **small-world** networks: high clustering + short path lengths.

$$
\text{Rewire each edge with probability }p
$$

keeping the ring-lattice structure otherwise.

**Examples**:  
- **Friend-of-a-friend** social networks with occasional "long-distance" ties (small-world effect).
- **Neural networks** in the brain, where most connections are local but a few are global "shortcuts."

---

### 3. Erdős-Rényi (ER) Model
- Given $n$ nodes, each possible edge appears **independently** with probability $p$.  
- Generates binomial $G(n,p)$ graphs with a **Poisson-like degree distribution** for large $n$.

$$
\text{Edge}(i,j)\text{ exists with probability }p\text{, i.i.d.}
$$

**Examples**:  
- **Random communication links** among devices (each link formed with some probability).
- **Chemical reaction networks** where each potential reaction has a uniform activation probability.

---

### 4. Stochastic Block Model (SBM)
- Partition $n$ nodes into $k$ blocks (communities).  
- Edges within each block appear with probability $p_\mathrm{intra}$.  
- Edges across different blocks appear with probability $p_\mathrm{inter}$.  

$$
P(A_{ij} = 1) = 
\begin{cases}
p_\mathrm{intra}, & \text{if } i,j \text{ in same block}, \\
p_\mathrm{inter}, & \text{otherwise}.
\end{cases}
$$

**Examples**:  
- **Social communities** (people in the same group are more likely to connect).
- **Protein-protein interaction** networks with functional modules.

---

### 5. Random Core-Periphery (RCP)
- Divide nodes into a **core** (densely interconnected) and a **periphery** (sparse or no internal edges).  
- The probability of an edge depends on whether the nodes are both in the core, both in the periphery, or cross core/periphery.

$$
\begin{aligned}
&p(\text{edge within core}) = p_\mathrm{core},\\
&p(\text{edge within periphery}) = p_\mathrm{periph},\\
&p(\text{edge across core-periphery}) = p_\mathrm{core\_periph}.
\end{aligned}
$$

**Examples**:  
- **Corporate structure** with a core leadership team that interacts frequently, and peripheral departments that are less interconnected.

---

### 6. LFR Benchmark (LFR)
- Generates **benchmark networks** with **power-law** degree distribution and **power-law** community size distribution.  
- Controlled by the **mixing parameter** $\mu$ which sets fraction of edges crossing communities.

**Examples**:  
- Synthetic data for testing **community detection** algorithms where both node degrees and community sizes follow heavy-tailed distributions.

## Sequence Generation & Parameter Evolution

1. **Sequence Length**: Each sequence has `seq_len` graphs.  
2. **Change Points**: We choose up to `max_changes` times to "jump" the parameter values (randomly within `[min_*, max_*]`).  
3. **Gradual Evolution**: At each time step, parameters with `_std` are **perturbed** via a normal distribution. For instance, if `m_std=1.0`, then the Barabási-Albert `m` parameter might fluctuate by $\pm1$ each step (bounded by user-defined constraints).

**Hence** each segment is generated from distinct parameter values, allowing the graphs to drift or jump over time.

## Example Usage

```python
from graph.generator import GraphGenerator
from graph.params import BAParams

# 1. Create the generator
gen = GraphGenerator()

# 2. Configure BA parameters
params = BAParams(
    n=100,         # 100 nodes
    seq_len=50,    # 50 time steps
    min_segment=5, # each segment must be at least 5 steps long
    min_changes=1, # at least 1 jump change
    max_changes=3, # up to 3 jumps
    m=2,           # attach 2 edges per new node
    min_m=1,
    max_m=5,
    m_std=0.2,     # small fluctuations at each step
)

# 3. Generate the sequence
result = gen.generate_sequence(model="barabasi_albert", params=params, seed=42)

# 4. Access results
graphs = result["graphs"]          # list of adjacency matrices
change_points = result["change_points"]
param_history = result["parameters"]
metadata = result["metadata"]

print("Number of graphs:", len(graphs))
print("Change points at:", change_points)
print("First segment params:", param_history[0])
print("First metadata block:", metadata[0])
```

## References

- Barabási–Albert Model: Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439).
- Watts–Strogatz Model: Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of small-world networks. Nature, 393(6684).
- Erdős–Rényi Model: Erdős, P., & Rényi, A. (1959). On random graphs I. Publ. Math. Debrecen, 6.
- Stochastic Block Model: Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels. J. of the American Statistical Association, 76(373).
- Core-Periphery Structure: Borgatti, S. P., & Everett, M. G. (2000). Models of core/periphery structures. Social Networks, 21.
- LFR Benchmark: Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). Benchmark graphs for testing community detection algorithms. Physical Review E, 78(4).