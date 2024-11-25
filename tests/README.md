# Understanding Graphs

## Metrics and Structure

**Network Size and Basic Structure:**

```text
num_nodes: 50 - There are 50 nodes in the network
num_edges: 141 - There are 141 connections between nodes
avg_degree: 5.640 - On average, each node connects to about 6 other nodes
max_degree: 19 - The most connected node has 19 connections
min_degree: 1 - The least connected node has just 1 connection
density: 0.115 - About 11.5% of all possible connections actually exist
clustering_coeff: 0.236 - About 24% of the time, if A connects to B and C, B and C also connect to each other
```

**Importance Measures:**

```text
degree_centrality metrics - How many direct connections each node has (as a fraction of possible connections)

Mean: 0.115 - On average, nodes connect to 11.5% of other nodes
Max: 0.388 - The most connected node connects to 38.8% of other nodes
```

```text
betweenness_centrality metrics - How often a node acts as a bridge between others

Mean: 0.026 - On average, nodes control about 2.6% of information flow
Max: 0.195 - The biggest bridge controls 19.5% of information flow
```

```text
eigenvector_centrality metrics - How important a node's neighbors are

Mean: 0.120 - Average importance score is 0.12
Max: 0.348 - Most important node has score of 0.35
```

```text
closeness_centrality metrics - How quickly nodes can reach others

Mean: 0.449 - Average nodes can reach about 45% of the network efficiently
Max: 0.613 - Best-positioned node can reach 61% efficiently
```

**Network Structure:**

```text
svd_norm: 9.798 - A measure of network complexity
spectral_gap: 0.951 - How well-connected different parts of the network are [0, 1]
avg_shortest_path: 2.261 - On average, it takes about 2.3 steps to get from one node to another
diameter: 4 - The maximum number of steps needed between any two nodes
radius: 3 - The minimum number of steps needed to reach the furthest node from any starting node
```

The graphs are generated using the `networkx` library.

```text
- Colors and sizes show how connected nodes are.
- Dark colors mean a node has lots of connections.
- Light colors mean it has few connections.
```

## Barabási-Albert (BA) Graphs

The nodes in these graphs are numbered from 0 to 49, showing when they joined the network. Nodes that joined early (like 0, 1, 2) usually end up with the most connections. This happens because new nodes tend to connect to nodes that already have many connections **(rich-get-richer) or preferential attachment**. Think of it like popular social media accounts getting more followers simply because they already have lots of followers.

### Key features:

- Early nodes (low numbers) get more connections over time
- New nodes prefer to connect to well-connected nodes
- The oldest nodes often become the biggest hubs
- The degree distribution follows a power law $P(k) \propto k^{-\gamma}$

## Erdős-Rényi (ER) Graphs

These graphs have 50 nodes numbered from 0 to 49. Unlike BA graphs, the numbers don't mean anything special. Every node has the same chance of connecting to any other node. It's completely random, like flipping a coin for each possible connection.

### Key features:

- Numbers are just labels
- Connections form randomly
- No node has any advantage over others

## Newman-Watts (NW) Graphs

These graphs start with 30 nodes arranged in a circle, numbered from 0 to 29. At first, each node connects to its neighbors (like 1 connects to 0 and 2). Then we add random connections across the circle. We show the original circle connections in gray and the random ones as red dashed lines.

### Key features:

- Nodes start in a circle
- Each node connects to its neighbors first
- Random connections then bridge across the circle

### How Different Graphs Change

Each type of graph grows differently:

- BA graphs: Adding more connections per new node makes the hubs bigger
- ER graphs: Increasing the connection chance makes the whole graph more connected
- NW graphs: More random connections create more shortcuts across the circle

---

# Centrality Measures Evolution

When we look at networks, we measure how important different nodes are in several ways. Let's look at each type of measurement and how it works in different kinds of networks.

## Degree Centrality

This is the simplest measure - it just counts what fraction of other nodes a node connects to directly. It ranges from 0 to 1.

- In BA networks, nodes that joined early typically connect to more others. You can really see the rich-get-richer effect here - when we add more edges per new node, the well-connected nodes get even more connected.
- ER networks show a more even spread of connections. When we increase the chance of connections forming, we see more variation in how connected nodes are, but no node has a built-in advantage.
- NW networks start with each node connected to its neighbors. Random bridges then add extra connections. This creates a mix between regular patterns and random connections.

## Betweenness Centrality

This measures how many shortest paths between other nodes pass through each node. It helps find nodes that act like bridges or bottlenecks.

- BA networks: Hub nodes control most of the information flow
- ER networks: Bridge nodes appear randomly
- NW networks: The shortcut connections become important bridges

## Eigenvector Centrality

This looks at how important a node's neighbors are. If you connect to important nodes, you become more important too.

- BA networks: Older nodes and hubs rank highest
- ER networks: Importance spreads more evenly
- NW networks: A mix between regular structure and random variation

## Closeness Centrality

This shows how quickly information could spread from one node to all others.

- BA networks: Hub nodes can reach others quickly
- ER networks: Speed depends on how dense the connections are
- NW networks: Shortcuts help information spread fast

---

# Understanding SHAP Values in Change Point Detection

SHAP (SHapley Additive exPlanations) values help us understand how each network measure (like degree centrality or betweenness) contributes to detecting changes in the graph structure. They explain which aspects of the network changed and by how much.

## SHAP Values Over Time

- Shows how each centrality measure's contribution changes throughout the graph evolution
- Spikes in SHAP values align with network structure changes

## Feature Importance Heatmap

- Red = Positive contribution (supports detecting change)
- Blue = Negative contribution (against detecting change)
- Intensity shows magnitude of contribution

When a graph undergoes structural changes:

1. **Degree Centrality SHAP Values**
   - Positive spikes: Significant changes in how nodes connect
   - Negative values: Connection patterns remain stable
   - Large magnitude: Major redistribution of connections

2. **Betweenness Centrality SHAP Values**
   - Positive spikes: Changes in shortest paths and bridge nodes
   - Negative values: Path structures remain similar
   - Large magnitude: Major changes in information flow patterns

3. **Eigenvector Centrality SHAP Values**
   - Positive spikes: Changes in influence patterns
   - Negative values: Influence structures remain stable
   - Large magnitude: Major shifts in node importance

4. **Closeness Centrality SHAP Values**
   - Positive spikes: Changes in overall network distance patterns
   - Negative values: Distance relationships remain similar
   - Large magnitude: Major changes in network reachability

## SHAP Value Patterns at Change Points

At change points, SHAP values often show a characteristic "dip-then-spike" pattern:

### Initial Dip (Negative Values)
- Occurs immediately at the change point
- Indicates the old network patterns becoming temporarily unreliable
- Represents a transition phase where previous patterns are broken but new ones haven't formed

### Following Spike (Positive Values)
- Follows shortly after the initial dip
- Shows the centrality measures strongly detecting the new network structure
- Represents the moment when the structural change is fully captured

This pattern is particularly visible in BA graphs where:
1. Adding new hub nodes first disrupts existing centrality patterns (negative SHAP)
2. Then the new hub structure becomes clear (positive SHAP)

## Practical Example

If at a change point you observe:
- High positive SHAP value for degree centrality
- Negative SHAP value for betweenness centrality

This could indicate that while nodes gained/lost connections (degree changed), the overall path structure (betweenness) remained relatively stable. This might happen when new connections form but maintain similar routing patterns.

## SHAP Value Additivity

The sum of SHAP values equals the difference between the model's prediction and the average prediction.

$$ \sum_{i=1}^{n} \phi_i = \mathbb{E}[\hat{y}] - \hat{y} $$

This property ensures that SHAP values provide a complete explanation of how each feature contributes to detecting changes in the graph structure.
