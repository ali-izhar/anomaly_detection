# Spatio-Temporal Graph Neural Network For Anomaly Detection

## 1. Graph Fundamentals

### 1.1 Basic Definitions

A **graph** $ G = (V, E) $ consists of:
- **Vertex set** $ V $ with $ |V| = N $ vertices
- **Edge set** $ E \subseteq V \times V $
- **Adjacency matrix** $ A \in \{0,1\}^{N \times N} $ where $ A_{ij} = 1 $ if $ (i,j) \in E $

### 1.2 Degree Matrix and Graph Laplacian

**Degree Matrix** $ D \in \mathbb{R}^{N \times N} $:
- Diagonal matrix where $ D_{ii} = \sum_{j=1}^N A_{ij} $
- Represents the number of connections (degree) for each node
- Critical for normalization to prevent scale issues in highly connected nodes

$$
D = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_N \end{bmatrix}
$$

**Graph Laplacian** $ L = D - A $:

Laplacian measures how different a node's value is from its neighbors. The $D$ accounts for a node's overall connectivity (its "self-effect"). The $A$ removes direct contributions from neighbors, leaving only the "difference" between the node and its neighbors. Thus, the difference captures the "smoothness" of a function on the graph. High $L$-values indicate nodes whose value significantly differs from their neighbors. For example:

$$
L = D - A = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2 \end{bmatrix} - \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{bmatrix}
$$

Notice that the diagonal of $L$ is the degree matrix $D$, and the off-diagonal elements are the negation of the adjacency matrix $A$. The first row of $L$ is $L_{11} = 2$, $L_{12} = -1$, and $L_{13} = -1$ because $d_1 = 2$ and node 1 is connected to nodes 2 and 3.

- **Properties**:
  - The sum of the elements in each row of $L$ is 0, indicating that the total influence of a vertex's connections balances out its "self-effect" or degree.
  - Symmetric: $ L = L^T $
  - **Positive semi-definite**: $ x^T L x \geq 0 $ for all $ x $
  - Smallest eigenvalue is 0 with eigenvector $ \mathbf{1} $
  - Number of zero eigenvalues equals number of connected components

> [!Note]
> A matrix $L$ is positive semi-definite if for any vector $x$, the quadratic form $x^T L x \geq 0$. The quantity $x^T L x$ can be interpreted as a "measure of energy" in a graph. If this energy is non-negative for any input $x$, then the matrix is positive semi-definite. For the Laplacian, the energy is the sum of the squared differences between connected vertices:

$$
x^T L x = \sum_{i,j} A_{ij} (x_i - x_j)^2 \geq 0
$$

As a simple example, consider a graph with two nodes connected by an edge. We can write the Laplacian as:

$$
L = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

For any vector $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$, we have:

$$
x^T L x = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = x_1^2 - 2x_1x_2 + x_2^2 = (x_1 - x_2)^2 \geq 0
$$

### 1.3 Normalized Laplacian

The normalized Laplacian adjusts for varying vertex degrees, ensuring that each vertex contributes proportionally to its connectivity. This normalization is particularly useful when dealing with graphs where vertex degrees vary widely, as it prevents high-degree vertices from disproportionately influencing the analysis.

$$
\mathcal{L} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
$$

**Why Normalize?**
- Bounds eigenvalues: $ 0 \leq \lambda_i \leq 2 $
- Makes spectral properties independent of graph scale
- Essential for stable gradient flow in deep networks

**Proof (Eigenvalue Bounds)**:
For normalized Laplacian $ \mathcal{L} $:
- Let $ x $ be an eigenvector with eigenvalue $ \lambda $
- Then:
  $
  \lambda = x^T \mathcal{L} x = \sum_{i,j} A_{ij} \left( \frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}} \right)^2
  $
- Using Gershgorin circle theorem: $ 0 \leq \lambda \leq 2 $

## 2. Graph Convolution Layer

To understand graph convolutions, let's draw a parallel to the convolutional neural networks (CNNs) we use on images. A CNN operates on a regular, grid-like data such as images, where each pixel has a fixed position and neigbors are defined by their spatial relationship. For the convolutional operation, we apply filters (kernels) that slide over the grid, performing localized operations to capture spatial hierarchies and patterns.

Convolutions on graphs are more complex because graphs are not regular or grid-like. Nodes in a graph do not have a fixed spatial relationship, and the concept of "neighborhood" is defined by the graph structure. The convolution aggregates information from a node's neighbors to capture the graph's topology and node features.

### 2.1 Spectral Graph Convolution

The spectral convolution on graphs is defined in the Fourier domain as:

$$
g_\theta \star x = U g_\theta(\Lambda) U^T x
$$s

where:
- $ U $ contains eigenvectors of normalized Laplacian $ \mathcal{L} = U\Lambda U^T $
- $ \Lambda $ is diagonal matrix of eigenvalues
- $ g_\theta $ is learnable spectral filter

**Computational Issue**: $ O(N^3) $ complexity for eigendecomposition.

### 2.2 Polynomial Approximation

To address computational complexity, we use Chebyshev polynomials:

$$
g_\theta(\Lambda) \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{\Lambda})
$$

where:
- $ T_k $ is the Chebyshev polynomial of order k
- $ \tilde{\Lambda} = 2\Lambda/\lambda_{max} - I_N $ is the scaled eigenvalue matrix

### 2.3 First-Order Approximation

**Key Insight**: Limit to first-order neighborhood to reduce complexity.

1. First-order approximation ($ K=1 $):
   $
   g_\theta(\mathcal{L}) \approx \theta_0 + \theta_1(\mathcal{L} - I)
   $

2. Set $ \theta = \theta_0 = -\theta_1 $ and add self-loops:
   $
   \tilde{A} = A + I_N
   $

3. Renormalize:
   $
   \hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}
   $

**Final Layer Form**:
$
H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
$

**Complexity**: Reduced to $ O(|E|) $ from $ O(N^3) $.

### 2.4 Why This Works

1. **Localization**:
   - First-order approximation localizes convolution to 1-hop neighborhood (nodes within one edge away)
   - Captures local structural information efficiently

2. **Gradient Flow**:
   - Renormalization prevents vanishing/exploding gradients
   - Eigenvalues of $ \hat{A} $ are bounded in $ [-1,1] $

3. **Feature Smoothing**:
   - $ \hat{A}H $ averages node features with neighbors
   - Learnable weights $ W $ transform feature space

**Proof (Gradient Stability)**:
For normalized adjacency $ \hat{A} $:
$
\|\hat{A}\|_2 \leq 1
$
This bounds backpropagation and prevents exploding gradients.

## 3. Temporal Processing

### 3.1 Sequence Modeling Challenge

Graph sequences present two key challenges:
1. Temporal dependencies between graph snapshots
2. Variable-length sequences

### 3.2 Attention Mechanism

**Scaled Dot-Product Attention**:


The attention mechanism in neural networks enables models to dynamically focus on different parts of the input data, enhancing their ability to capture dependencies and contextual relationships. 

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $ Q $ (Query): Represents the current element seeking context.
- $ K $ (Key): Contains potential matches for the query within the input.
- $ V $ (Value): Holds the information corresponding to each key.

Consider this sentence: **The quick brown fox jumps over the lazy dog.** Let's say that we are currently processing the word "fox" and we want to determine which words are most relevant to "fox" in the context of the sentence.

- Query (Q): The vector representation of the target word, "fox," which seeks relevant information from the sentence.
- Keys (K): Vector representations of all words in the sentence, serving as references for matching against the query.
- Values (V): Typically identical to the keys in self-attention mechanisms, representing the information content of each word.

Compute the dot product between the query and each key. A high dot product indicates a strong similarity between "fox" and another word. A low dot product suggests lesser relevance. Then, divide each attention score by the square root of the dimensionality of the key vectors $ \sqrt{d_k} $. This scaling prevents the softmax function from producing extremely small gradients, ensuring stable training.

**Why Scale?**
For $ Q,K $ with mean 0, variance 1:
- $ QK^T $ has variance $ d_k $ (i.e. the dimensionality of the key vectors)
- Large $ d_k $ pushes softmax into regions of small gradients
- Scaling by $ \sqrt{d_k} $ maintains variance $ \approx 1 $

**Proof (Variance Analysis)**:
Let $ q_i, k_j \sim \mathcal{N}(0,1) $ be i.i.d. The, by definition of dot product,

$$
q^T k = \sum_{i=1}^{d_k} q_i k_i
$$

Since $ q_i $ and $ k_i $ are i.i.d. with mean 0 and variance 1, $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$ and $\text{Var}(q_i) = \text{Var}(k_i) = 1$. Therefore, $\mathbb{E}[q^T k] = 0$. The variance of the dot product is:

$$
\text{Var}(q^T k) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) 
$$

For independent random variables, the variance of their sum is the sum of their variances:

$$
\text{Var}(q^T k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)
$$

For independent $ q_i, k_i \sim \mathcal{N}(0,1) $:

$$
\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - \mathbb{E}[q_i k_i]^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = 1 \times 1 = 1
$$

Therefore,

$$
\text{Var}(q^T k) = \sum_{i=1}^{d_k} 1 = d_k
$$

As $ d_k $ increases, the variance of the dot product increases linearly. To prevent the softmax function from producing extremely small gradients, we scale the dot product by $ \frac{1}{\sqrt{d_k}} $


### 3.3 Multi-Head Attention

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**Why Multiple Heads?**
- Allows parallel attention to different subspaces
- Each head can specialize in different temporal patterns
- Increases model capacity without increasing sequence length

## 4. LSTM Decoder

### 4.1 Sequential Prediction

The LSTM decoder generates multi-step predictions using:

$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$

where:
- $ f_t $: forget gate
- $ i_t $: input gate
- $ o_t $: output gate
- $ c_t $: cell state
- $ h_t $: hidden state
- $ \odot $: element-wise multiplication

### 4.2 Autoregressive Prediction

For multi-step prediction:
$
y_t = W_{out}h_t + b_{out}
$

**Why Autoregressive?**
1. Captures temporal dependencies in predictions
2. Allows uncertainty propagation
3. Models non-linear dynamics

**Proof (Error Propagation)**:
For prediction horizon $ H $, error grows as:

$$
\|\epsilon_H\|_2 \leq (1 + \delta)^H\|\epsilon_1\|_2
$$
where $ \delta $ depends on Lipschitz constants of network components.

## 5. Martingale Framework

### 5.1 Theoretical Foundation

A sequence $ (M_n)_{n\geq 0} $ is a martingale if:
1. $ \mathbb{E}[|M_n|] < \infty $ for all $ n $
2. $ \mathbb{E}[M_n|\mathcal{F}_{n-1}] = M_{n-1} $

where $ \mathcal{F}_n $ is the filtration (information up to time $ n $).

### 5.2 Power Martingale for Change Detection

**Definition**:
$
M_n = \prod_{i=1}^n \epsilon p_i^{\epsilon-1}
$

where:
- $ p_i $: p-value at time i
- $ \epsilon $: sensitivity parameter (0,1)

**Properties**:
1. $ \mathbb{E}[M_n] = 1 $ under null hypothesis
2. $ M_n $ grows under alternative hypothesis
3. Parameter $ \epsilon $ controls false positive rate

**Proof (Martingale Property)**:
Under null hypothesis, for uniform p-values:
$
\mathbb{E}[\epsilon p^{\epsilon-1}] = \epsilon \int_0^1 p^{\epsilon-1}dp = 1
$

### 5.3 Strangeness-based P-values

For feature vector $ x_i $:
$
\alpha_i = \min_{j=1}^k \|x_i - \mu_j\|_2
$

P-value computation:
$
p(\theta) = \frac{\#\{\alpha_i > \alpha_n\} + \theta\#\{\alpha_i = \alpha_n\}}{n}
$

where $ \theta \sim U[0,1] $ for tie-breaking.

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.

2. Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.

4. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

5. Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). Wavelets on Graphs via Spectral Graph Theory. Applied and Computational Harmonic Analysis.

6. Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. NeurIPS.
