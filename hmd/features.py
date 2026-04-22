"""Graph → feature-vector extraction (Paper §III-A).

Each snapshot G_t becomes a permutation-invariant vector X_t ∈ R^K (K=8 by
default); the detector consumes the sequence {X_t} as its sole input.

Critical design choices
-----------------------
- **Per-graph scalars, not per-node vectors.** Nodes have no canonical label
  across time (especially on MIT Reality); a relabelling under isomorphism
  would otherwise register as "change". Mean-aggregation is the simplest
  permutation-invariant reduction and is 1-Lipschitz in any single node's
  feature (smoother than max under perturbation).
- **Exactly 8 features.** M^A = Σ_k M^(k) (Corollary 1) is budgeted by Ville's
  1/λ across the *sum*, so each feature inflates the Type-I floor. The 8 span
  local (degree, clustering), flow (betweenness, closeness, eigenvector),
  spectral-global (λ_2, λ_3−λ_2), and global-scalar (density) without
  redundancy. Density is our one departure from the paper's §III-A list
  (which enumerates 7); see `docs/design/choices.md`.
- **Unnormalized Laplacian L = D − A** matches paper §III-A. Eigenvalues
  partially correlate with mean degree (density-redundant); the non-conformity
  score re-ranks, so this is tolerable.
- **Empty / isolated / disconnected graphs return 0, never NaN**, because NaN
  poisons the K-means centroid and halts detection silently.
- **Eigenvector centrality** uses `nx.eigenvector_centrality_numpy` (dense
  eigh); power iteration can fail on disconnected graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import networkx as nx
import numpy as np
from tqdm import tqdm

from hmd._backend import xp  # re-exported for symmetry w/ rest of package

# xp is used below only for the small dense Laplacian eigensolve. Feature values
# are converted back to plain numpy floats because scipy / networkx / K-means
# consumers all expect host arrays.
_np = np  # alias to make intent explicit when we deliberately want host numpy


# ---------------------------------------------------------------------------
# Feature spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """A named, graph -> scalar feature.

    Frozen so specs are hashable and can be used as dict keys / cached.
    """

    name: str
    fn: Callable[[nx.Graph], float]


# ---------------------------------------------------------------------------
# Individual feature implementations
# Each returns a Python float (not np.float64) to keep downstream code simple.
# ---------------------------------------------------------------------------


def _mean_degree(g: nx.Graph) -> float:
    """Mean node degree: (1/n) Σ_v deg(v).

    Sensitive to: any edge-count change (density shift, edge insertion bursts).
    Empty or single-node graph -> 0.
    """
    n = g.number_of_nodes()
    if n == 0:
        return 0.0
    return float(2.0 * g.number_of_edges() / n)  # handshake lemma


def _mean_clustering(g: nx.Graph) -> float:
    """Mean local clustering coefficient C(v) = 2|{(u,w)∈E : u,w∈N(v)}| / (deg(v)(deg(v)-1)).

    Sensitive to: triadic closure changes — community formation/merging,
    small-world rewiring in WS networks.
    Nodes with deg < 2 contribute 0 (networkx default, matches paper §III-A).
    """
    if g.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(g))


def _mean_betweenness(g: nx.Graph) -> float:
    """Mean betweenness centrality: (1/n) Σ_v Σ_{s≠v≠t} σ_st(v)/σ_st.

    Sensitive to: bridge-node emergence — community merges create high-BC
    gateway nodes, which lifts the mean sharply.
    Returns normalized BC (divided by (n-1)(n-2)/2), so comparable across sizes.
    """
    if g.number_of_nodes() < 3:
        return 0.0
    bc = nx.betweenness_centrality(g, normalized=True)
    return float(_np.mean(list(bc.values())))


def _mean_closeness(g: nx.Graph) -> float:
    """Mean closeness centrality CC(v) = (|V|-1) / Σ_u d(v,u).

    Sensitive to: global diameter / reachability changes — ER density shifts,
    component fragmentation.
    Isolated v -> CC(v) = 0 (paper convention; networkx returns 0 for
    unreachable pairs when using the default wf_improved=True).
    """
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    cc = nx.closeness_centrality(g)
    return float(_np.mean(list(cc.values())))


def _mean_eigenvector(g: nx.Graph) -> float:
    """Mean eigenvector centrality EC(v) = v-th entry of dominant eigenvector of A.

    Sensitive to: hub formation (BA attach events), preferential-attachment
    regime shifts. The dominant eigenvector concentrates on high-centrality
    cores; its mean tracks how evenly "importance" is spread.

    Implementation note: we skip nx.eigenvector_centrality_numpy and do the
    dense eigendecomposition ourselves. The NetworkX wrapper routes through
    scipy.sparse.linalg.eigs for n ≥ 3, which raises TypeError when n is
    small (it refuses k ≥ N-1), and power iteration (the non-numpy variant)
    can stall on disconnected graphs. A direct eigh on the tiny dense
    adjacency is simpler and always works. Convention: take the dominant
    eigenvector, sign-normalized to be non-negative (Perron-Frobenius for
    a non-negative connected block; for disconnected graphs we flip the
    sign to the majority-positive orientation), then L2-normalize. Matches
    networkx's output up to sign.
    """
    n = g.number_of_nodes()
    if n == 0 or g.number_of_edges() == 0:
        return 0.0
    A = nx.to_numpy_array(g, dtype=_np.float64)
    # Real symmetric -> eigh. eigvec columns are orthonormal, eigvals ascending.
    evals, evecs = _np.linalg.eigh(A)
    v = evecs[:, -1]  # dominant eigenvector (largest eigenvalue)
    # Sign-normalize: Perron eigenvector is non-negative; eigh may flip sign.
    if v.sum() < 0:
        v = -v
    # eigh returns unit-norm; mean of unit-norm vector is a valid scalar feature.
    return float(v.mean())


def _laplacian_eigenvalues(g: nx.Graph) -> np.ndarray:
    """Sorted eigenvalues of L = D - A (unnormalized), ascending.

    Paper §III-A: L = D - A, 0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_|V|.
    Returns a numpy array of length n; callers pick off λ_2, λ_3 as needed.
    """
    n = g.number_of_nodes()
    if n == 0:
        return _np.zeros(0, dtype=_np.float64)
    # Dense is fine at N ≤ 50. xp lets a GPU backend take over if flipped.
    L = nx.laplacian_matrix(g).astype(_np.float64).toarray()
    L_xp = xp.asarray(L)
    # eigvalsh: symmetric real -> real eigenvalues, sorted ascending.
    evals = xp.linalg.eigvalsh(L_xp)
    if hasattr(evals, "get"):  # cupy -> host
        evals = evals.get()
    return _np.asarray(evals, dtype=_np.float64)


def _algebraic_connectivity(g: nx.Graph) -> float:
    """λ_2 of L = D - A (second-smallest eigenvalue; Fiedler value).

    Sensitive to: global connectivity — large on well-knit graphs, → 0 as the
    graph approaches disconnection. Responds to density changes and to the
    appearance of weak cuts.
    """
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    evals = _laplacian_eigenvalues(g)
    # evals[0] == 0 (up to fp noise) for any graph; λ_2 = evals[1].
    return float(evals[1])


def _spectral_gap(g: nx.Graph) -> float:
    """λ_3 − λ_2 of L = D - A.

    Sensitive to: community structure. Two well-separated clusters produce a
    near-degenerate {λ_1, λ_2} ≈ 0 with λ_3 well above — a large gap. A merge
    collapses the gap. Complementary to λ_2 alone.
    """
    n = g.number_of_nodes()
    if n < 3:
        return 0.0
    evals = _laplacian_eigenvalues(g)
    return float(evals[2] - evals[1])


def _density(g: nx.Graph) -> float:
    """Edge density |E| / (n(n-1)/2).

    Sensitive to: bulk edge-count changes. Scalar by construction (no node
    aggregation needed) so cheap and noise-free.
    """
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    return float(2.0 * g.number_of_edges() / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Default set (ordered — this ordering is part of the public contract, as
# downstream Shapley attribution ψ_k(t) indexes by position).
# ---------------------------------------------------------------------------


def default_set() -> list[FeatureSpec]:
    """The 8 features from Paper §III-A, in fixed order.

    Order is the public contract: Shapley attribution ψ_k indexes by position.
    """
    return [
        FeatureSpec("mean_degree", _mean_degree),
        FeatureSpec("mean_clustering", _mean_clustering),
        FeatureSpec("mean_betweenness", _mean_betweenness),
        FeatureSpec("mean_closeness", _mean_closeness),
        FeatureSpec("mean_eigenvector", _mean_eigenvector),
        FeatureSpec("algebraic_connectivity", _algebraic_connectivity),
        FeatureSpec("spectral_gap", _spectral_gap),
        FeatureSpec("density", _density),
    ]


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


def extract(
    graph: nx.Graph,
    spec: Sequence[FeatureSpec] | None = None,
) -> np.ndarray:
    """Extract the K-length feature vector from a single graph.

    Parameters
    ----------
    graph : nx.Graph
    spec  : Sequence[FeatureSpec] or None
        Feature set. Defaults to default_set() (Paper §III-A).

    Returns
    -------
    np.ndarray of shape (K,), dtype float64.
    """
    specs = default_set() if spec is None else list(spec)
    out = _np.empty(len(specs), dtype=_np.float64)
    for i, s in enumerate(specs):
        out[i] = s.fn(graph)
    return out


def extract_sequence(
    graphs: Sequence[nx.Graph],
    spec: Sequence[FeatureSpec] | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Extract a (T, K) feature matrix from a sequence of graphs.

    Parameters
    ----------
    graphs : Sequence[nx.Graph], length T
    spec   : Sequence[FeatureSpec] or None
    show_progress : bool
        Wrap the outer loop in tqdm. Off by default — noisy in unit tests.

    Returns
    -------
    np.ndarray of shape (T, K), dtype float64.
    """
    specs = default_set() if spec is None else list(spec)
    T = len(graphs)
    K = len(specs)
    out = _np.empty((T, K), dtype=_np.float64)
    iterator = tqdm(graphs, desc="extract_sequence") if show_progress else graphs
    for t, g in enumerate(iterator):
        for k, s in enumerate(specs):
            out[t, k] = s.fn(g)
    return out


def feature_names(spec: Sequence[FeatureSpec] | None = None) -> list[str]:
    """K feature names in the same order extract() returns their values."""
    specs = default_set() if spec is None else list(spec)
    return [s.name for s in specs]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    g = nx.erdos_renyi_graph(50, 0.1, seed=0)
    names = feature_names()
    vals = extract(g)
    width = max(len(n) for n in names)
    print(f"ER(50, 0.1, seed=0): n={g.number_of_nodes()}, m={g.number_of_edges()}")
    for n, v in zip(names, vals):
        print(f"  {n:<{width}s}  {v: .6f}")
