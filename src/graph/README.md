# Graph Models

Built on top of [NetworkX](https://networkx.org/), this module supports multiple graph models:

### Barabási-Albert (BA)
- **Mathematics**: Grows by adding nodes with $m$ edges, connecting to existing nodes with probability $\Pi(k_i) \sim k_i^{\alpha}$ where $k_i$ is node degree
- **Properties**: Power-law degree distribution $P(k) \sim k^{-\gamma}$ where $\gamma \approx 3$
- **Real-world Example**: World Wide Web, citation networks, protein interaction networks

### Erdős-Rényi (ER)
- **Mathematics**: Each edge exists with independent probability $p$
- **Properties**: Poisson degree distribution $P(k) = \frac{e^{-\lambda}\lambda^k}{k!}$ where $\lambda = p(n-1)$
- **Real-world Example**: Random connections in social networks, random chemical reactions

### Watts-Strogatz (WS)
- **Mathematics**: Start with ring lattice, rewire edges with probability $p$
- **Properties**: High clustering coefficient and low average path length
- **Real-world Example**: Neural networks, power grids, actor collaboration networks

### Random Regular (RR)
- **Mathematics**: Each node has exactly $d$ random connections
- **Properties**: Uniform degree distribution $P(k) = \delta_{k,d}$
- **Real-world Example**: Peer-to-peer networks, certain types of computer architectures

### Random Geometric (RG)
- **Mathematics**: Nodes connect if Euclidean distance $< r$ in $d$-dimensional space
- **Properties**: High clustering, spatial dependency
- **Real-world Example**: Wireless sensor networks, ecological interaction networks

### Stochastic Block Model (SBM)
- **Mathematics**: Nodes in same block connect with probability $p_{in}$, different blocks with $p_{out}$
- **Properties**: Community structure with $p_{in} > p_{out}$
- **Real-world Example**: Social communities, protein-protein interaction networks

### Random Core-Periphery (RCP)
- **Mathematics**: Dense core with probability $p_{core}$, sparse periphery with $p_{periph}$
- **Properties**: Hierarchical structure with $p_{core} > p_{core-periph} > p_{periph}$
- **Real-world Example**: Financial networks, transportation hubs

### Complete Graph (CG)
- **Mathematics**: Every node connects to every other node, with edge removal probability $p$
- **Properties**: Maximum density, degree $k = n-1$ for all nodes
- **Real-world Example**: Fully connected computer networks, complete social groups

### Dense Random Geometric (DRG)
- **Mathematics**: Similar to RG but with larger radius $r$ ensuring high density
- **Properties**: High clustering, high density, spatial correlation
- **Real-world Example**: Dense wireless networks, close-range communication systems

### Newman-Watts (NW)
- **Mathematics**: Ring lattice with additional random shortcuts probability $p$
- **Properties**: Small-world properties with preserved local structure
- **Real-world Example**: Social networks with both local and long-range connections

### Holme-Kim (HK)
- **Mathematics**: BA model with added triangle formation probability $p_{triad}$
- **Properties**: Scale-free with high clustering
- **Real-world Example**: Social media networks, scientific collaboration networks

### LFR Benchmark
- **Mathematics**: Communities with power-law degree ($\tau_1$) and size ($\tau_2$) distributions, mixing parameter $\mu$
- **Properties**: Realistic community structure with heterogeneous degree and community sizes
- **Real-world Example**: Large social networks, online communities with varying sizes
