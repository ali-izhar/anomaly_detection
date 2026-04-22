"""Analytic correctness tests for graph feature extraction (Paper Sec III-A).

WHAT is tested
--------------
Each of the 8 per-graph features should satisfy:
  * Known analytic identities on canonical graphs (K_n, empty graph).
  * Permutation invariance -- the raison d'etre of per-graph SCALAR aggregation.

WHY this matters
----------------
The paper's Sec III-A explicitly motivates scalar aggregation (mean, global
statistic) to achieve permutation invariance: a relabelling of node IDs must
not register as a "structural change". If any one of the 8 default features
were not invariant, spurious martingale growth would happen whenever nodes
re-index -- the MIT Reality phone-ID reuse regime, for example, would
look like a change point at every snapshot.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from hmd import features as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_by_name(g: nx.Graph) -> dict[str, float]:
    """Return {name: value} for the default 8-feature set on g."""
    names = F.feature_names()
    vals = F.extract(g)
    return dict(zip(names, vals.tolist()))


# ---------------------------------------------------------------------------
# (1) Analytic identities
# ---------------------------------------------------------------------------


def test_mean_degree_handshake_lemma():
    """mean_degree(G) = 2 |E| / |V| by the handshake lemma."""
    g = nx.path_graph(10)  # 10 nodes, 9 edges
    vals = _extract_by_name(g)
    expected = 2 * 9 / 10
    assert abs(vals["mean_degree"] - expected) < 1e-12


def test_density_is_edges_over_n_choose_2():
    """density = |E| / C(|V|, 2)."""
    g = nx.cycle_graph(7)  # 7 nodes, 7 edges
    vals = _extract_by_name(g)
    expected = 7 / (7 * 6 / 2)
    assert abs(vals["density"] - expected) < 1e-12


def test_empty_graph_features_all_zero():
    """Empty graph (n nodes, 0 edges) -> all features 0 (by module convention)."""
    g = nx.empty_graph(10)  # 10 isolated nodes
    vals = _extract_by_name(g)
    for name, v in vals.items():
        assert v == 0.0, f"feature {name} on empty_graph(10) = {v}, expected 0"


def test_zero_node_graph_features_all_zero():
    g = nx.Graph()
    vals = _extract_by_name(g)
    for name, v in vals.items():
        assert v == 0.0, f"feature {name} on 0-node graph = {v}"


@pytest.mark.parametrize("n", [5, 8, 12])
def test_complete_graph_algebraic_connectivity_equals_n(n):
    """For K_n the Laplacian spectrum is {0, n, n, ..., n}.
    Hence lambda_2(K_n) = n (classical result; Chung, Spectral Graph Theory).
    Also spectral_gap = lambda_3 - lambda_2 = n - n = 0.
    """
    g = nx.complete_graph(n)
    vals = _extract_by_name(g)
    assert abs(vals["algebraic_connectivity"] - n) < 1e-9, (
        f"K_{n}: lambda_2 = {vals['algebraic_connectivity']}, expected {n}"
    )
    assert abs(vals["spectral_gap"]) < 1e-9, (
        f"K_{n}: spectral_gap = {vals['spectral_gap']}, expected 0"
    )


def test_complete_graph_density_is_one():
    g = nx.complete_graph(8)
    vals = _extract_by_name(g)
    assert abs(vals["density"] - 1.0) < 1e-12


def test_complete_graph_mean_degree_is_n_minus_1():
    g = nx.complete_graph(8)
    vals = _extract_by_name(g)
    assert abs(vals["mean_degree"] - 7.0) < 1e-12


# ---------------------------------------------------------------------------
# (2) Permutation invariance -- the core design claim.
# ---------------------------------------------------------------------------


def _relabel(g: nx.Graph, perm: list[int]) -> nx.Graph:
    nodes = list(g.nodes())
    mapping = {old: perm[i] for i, old in enumerate(nodes)}
    return nx.relabel_nodes(g, mapping, copy=True)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_features_permutation_invariant_on_er(seed):
    """For any graph G and relabelling sigma, features(G) = features(sigma(G)).

    This is the key property that makes the detector node-identity-agnostic.
    Tested on random Erdos-Renyi graphs across 5 seeds with different
    permutations.
    """
    rng = np.random.default_rng(seed)
    g = nx.erdos_renyi_graph(30, 0.2, seed=int(rng.integers(0, 2**31 - 1)))
    nodes = list(g.nodes())
    perm = list(rng.permutation(len(nodes)))
    g_rel = _relabel(g, perm)

    v0 = F.extract(g)
    v1 = F.extract(g_rel)
    # Some features rely on eigendecomposition with potential sign-flip; the
    # mean-aggregated features are sign-sensitive only via the Perron-Frobenius
    # sign-normalization (codebase enforces sum >= 0 convention). Allow a
    # tight numerical tolerance.
    np.testing.assert_allclose(v1, v0, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_features_permutation_invariant_on_sbm(seed):
    rng = np.random.default_rng(seed)
    g = nx.stochastic_block_model(
        [15, 15], [[0.9, 0.05], [0.05, 0.9]],
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    g = nx.Graph(g)  # drop block attribute
    perm = list(rng.permutation(g.number_of_nodes()))
    g_rel = _relabel(g, perm)

    v0 = F.extract(g)
    v1 = F.extract(g_rel)
    np.testing.assert_allclose(v1, v0, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# (3) Feature ordering contract.
# ---------------------------------------------------------------------------


def test_feature_names_ordering_is_public_contract():
    """Shapley attribution indexes by position; the order must be exactly this."""
    expected = [
        "mean_degree",
        "mean_clustering",
        "mean_betweenness",
        "mean_closeness",
        "mean_eigenvector",
        "algebraic_connectivity",
        "spectral_gap",
        "density",
    ]
    assert F.feature_names() == expected


def test_extract_sequence_shape():
    graphs = [nx.erdos_renyi_graph(10, 0.3, seed=i) for i in range(7)]
    X = F.extract_sequence(graphs)
    assert X.shape == (7, 8)
    assert X.dtype == np.float64
