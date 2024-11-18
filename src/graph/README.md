# Graph Generators

Understanding different graph models is crucial for simulating and analyzing complex networks. Below are detailed, intuitive explanations of various graph generators, including their mathematical foundations and how they are implemented in the `graph` module of our project.

---

## Barabási-Albert (BA) Model

The **Barabási-Albert (BA) model** is a mechanism for generating **scale-free networks** using a process known as **preferential attachment**. Scale-free networks are characterized by a power-law degree distribution, meaning that a few nodes (hubs) have a high degree, while most nodes have a low degree. This model captures the essence of many real-world networks, such as social networks, the World Wide Web, and biological systems.

### Preferential Attachment

The core idea behind the BA model is **"the rich get richer"**:

- Nodes with higher degrees have a higher probability of attracting new connections.
- This creates hubs that become increasingly connected over time.

### How the BA Model Works

1. **Initialization**:

   - Start with a small connected network of $m_0$ nodes.
   - These nodes can be connected in any arbitrary way.

2. **Growth Process**:

   - At each time step, add a new node with $m \leq m_0$ edges.
   - The new node connects to $m$ existing nodes.

3. **Preferential Attachment Rule**:

   - The probability $\Pi(k_i)$ that the new node connects to an existing node $i$ depends on the degree $k_i$ of node $i$:

     $$\Pi(k_i) = \frac{k_i}{\sum_j k_j}$$

   - Nodes with higher degrees are more likely to receive new connections.

### Mathematical Foundation

- **Degree Distribution**:

  - The network develops a **power-law degree distribution**:

    $$P(k) \sim k^{-\gamma}$$

- **Power-Law Behavior**:

  - Indicates that there is no characteristic degree in the network.
  - A small number of nodes (hubs) dominate the connectivity.

### Visual Illustration

Imagine building a network step by step:

1. **Starting Network**:

   ```lua
   Node 1 --- Node 2
   ```

2. **Adding a New Node (Node 3)**:

   - Node 3 connects to Node 1 or Node 2.
   - Since both have the same degree, the choice is random.

3. **Further Growth**:

   - As more nodes are added, nodes with higher degrees become more attractive.

4. **Example Graph after Several Iterations**:

   ```lua
            Node 4
            /
      Node 1 --- Node 2 --- Node 3
            \
            Node 5
   ```

   - Node 1 has a higher degree and thus attracts more connections.

### Implementation in the `graph` Module

```python
graphs = GraphGenerator().barabasi_albert(n=100, m1=2, m2=5, set1=50, set2=50)
```

- `n`: Total number of nodes.
- `m1` and `m2`: Number of edges to attach from a new node to existing nodes for two different graph sets.
- `set1` and `set2`: Number of graphs to generate with `m1` and `m2`, respectively.

---

## Barabási-Albert Internet (BA-I) Model

The **Barabási-Albert Internet (BA-I) model** extends the standard BA model by incorporating features specific to the Internet's topology. It aims to create networks that more accurately represent the structure and growth dynamics of the Internet.

### Key Differences from the Standard BA Model

- **Initial Graph**:

  - Starts with a base graph that mimics the Internet's Autonomous System (AS) level topology.

- **Growth Mechanism**:
  - New nodes connect using preferential attachment but consider the hierarchical and geographical constraints of the Internet.

### How the BA-I Model Works

1. **Initial Internet-Like Graph**:

   - Use a function like `nx.random_internet_as_graph(n)` to generate a base graph that resembles the Internet's AS topology.
   - Nodes represent autonomous systems, and edges represent connections between them.

2. **Adding New Nodes**:

   - New nodes are added one at a time.
   - They connect to existing nodes based on preferential attachment, considering factors like node degree and possibly other attributes like latency or bandwidth.

### Mathematical Considerations

- **Degree Distribution**:

  - Similar to the BA model but adjusted to fit the empirical data of Internet topology.

- **Hierarchical Structure**:
  - Incorporates a multi-tiered hierarchy, reflecting ISPs, regional networks, and local networks.

### Visual Illustration

**Simplified Internet-Like Graph**:

```lua
             [Tier 1 ISP]
                  |
        ---------------------
        |         |         |
   [ISP A]    [ISP B]    [ISP C]
      |           |          |
   [User 1]    [User 2]   [User 3]
```

### Implementation in the `graph` Module

```python
graphs = GraphGenerator().barabasi_albert_internet(n=100, m1=2, m2=4, set1=50, set2=50)
```

- `n`: Number of nodes in the base Internet-like graph.
- `m1` and `m2`: Attachment parameters for two different graph sets.
- `set1` and `set2`: Number of graphs to generate with different parameters.

---

## Erdős-Rényi (ER) Model

The **Erdős-Rényi (ER) model** is one of the foundational models for generating random graphs. It constructs a network by connecting nodes randomly, leading to a **Poisson degree distribution** in large graphs.

### How the ER Model Works

There are two classic formulations:

- **$G(n, p)$ Model**:

  - **Nodes**: Start with $n$ isolated nodes.
  - **Edge Probability $p$**:
    - For each possible pair of nodes, an edge is included with probability $p$.
  - **Total Possible Edges**: $\binom{n}{2} = \frac{n(n-1)}{2}$.

- **$G(n, M)$ Model**:

  - **Nodes**: Start with $n$ isolated nodes.
  - **Edges**:
    - Exactly $M$ edges are randomly selected from all possible edges.

### Mathematical Foundation

- **Degree Distribution**:

  - Follows a binomial distribution:

    $$P(k) = \binom{n-1}{k} p^k (1-p)^{n-1-k}$$

  - For large $n$ and small $p$, approximates a Poisson distribution:

    $$P(k) \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

    where $\lambda = p(n-1)$.

- **Expected Number of Edges**:

  $$E[M] = p \binom{n}{2}$$

### Properties

- **Randomness**:

  - Edges are placed independently of each other.

- **Phase Transition**:
  - A giant connected component emerges when $p$ crosses a critical threshold $p_c = \frac{\ln (n)}{n}$.

### Visual Illustration

- **Sparse ER Graph ($p$ small)**:

  ```lua
  Node 1     Node 2     Node 3

  Node 4 --- Node 5
  ```

  - Few edges; nodes are mostly disconnected.

- **Dense ER Graph ($p$ large)**:

  ```lua
   Node 1 --- Node 2 --- Node 3
       |        |        |
   Node 4 --- Node 5 --- Node 6
  ```

  - Many edges; the graph is likely connected.

### Implementation in the `graph` Module

```python
graphs = GraphGenerator().erdos_renyi(n=100, p1=0.05, p2=0.1, set1=50, set2=50)
```

- `n`: Number of nodes.
- `p1` and `p2`: Edge probabilities for two different graph sets.
- `set1` and `set2`: Number of graphs to generate with `p1` and `p2`.

---

## Newman-Watts-Strogatz (NWS) Model

The **Newman-Watts-Strogatz (NWS) model** is an enhancement of the original Watts-Strogatz model for generating **small-world networks**. It combines high clustering (like regular lattices) with short average path lengths (like random graphs), capturing the small-world phenomenon observed in many real networks.

### How the NWS Model Works

- **Start with a Ring Lattice**:

  - **Nodes**: Arrange $n$ nodes in a ring.
  - **Local Connections**:
    - Each node is connected to its $k$ nearest neighbors ($\frac{k}{2}$ on each side).

- **Adding Random Edges (Shortcuts)**:

  - **Rewiring Probability $p$**:
    - Instead of rewiring existing edges (as in the Watts-Strogatz model), the NWS model adds new edges between randomly selected pairs of nodes.
    - This avoids disconnected components and self-loops.

### Mathematical Foundation

- **Clustering Coefficient**:

  - Measures the likelihood that two neighbors of a node are also neighbors.
  - High in the initial lattice structure.

- **Average Path Length**:
  - The average number of steps along the shortest paths for all possible pairs of network nodes.
  - Reduced significantly by the addition of random shortcuts.

### Properties

- **High Clustering**:

  - Nodes tend to form tightly knit groups.

- **Short Average Path Length**:
  - Random shortcuts create "shortcuts" across the network, reducing the path length.

### Visual Illustration

- **Step 1: Ring Lattice**:

  ```lua
  Node 1 --- Node 2 --- Node 3 --- Node 4 --- Node 5 --- Node 6
    |                                               |
   (back to Node 1) ------------------------------|
  ```

  - Each node is connected to its immediate neighbors.

- **Step 2: Adding Shortcuts**:

  ```lua
   Node 1 --- Node 2 --- Node 3 --- Node 4 --- Node 5 --- Node 6
      |        \                              /          |
    (back to Node 1) ------------------------------|
  ```

  - Random edges (dashed lines) are added between non-neighboring nodes.

### Implementation in the `graph` Module

```python
graphs = GraphGenerator().newman_watts(n=100, k1=4, p1=0.1, k2=6, p2=0.2, set1=50, set2=50)
```

- `n`: Number of nodes.
- `k1` and `k2`: Each node is connected to $k$ nearest neighbors for two different graph sets.
- `p1` and `p2`: Probability of adding a new random edge for each node.
- `set1` and `set2`: Number of graphs to generate with different parameters.

---

## Conclusion

Understanding these graph models and their implementations is essential for simulating dynamic networks and studying phenomena like change point detection. Each model provides a different lens through which we can examine network structures:

- **Barabási-Albert (BA) Model**: Emphasizes preferential attachment and the emergence of hubs.
- **Barabási-Albert Internet (BA-I) Model**: Tailors the BA model to reflect the Internet's topology.
- **Erdős-Rényi (ER) Model**: Provides a foundation for random graphs with a binomial degree distribution.
- **Newman-Watts-Strogatz (NWS) Model**: Captures the small-world characteristics of high clustering and short path lengths.
