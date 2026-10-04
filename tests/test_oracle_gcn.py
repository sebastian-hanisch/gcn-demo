"""Unabhängige Orakel für das GCN: dichte Schleifen-Referenz (Normierung, Vorwärtsrechnung, Glättung), Permutations-Äquivarianz, k-nächste-Nachbarn gegen scipy, Homophilie/Kantenzahl/Entfernungen gegen networkx.
Ergänzt test_algorithm.py (Handrechnung, Nachbarschleife, zentrale Differenzen) um Zufallsinstanzen und Kennzahlen des Vehikels."""

import numpy as np
import pytest

import gcn_algorithm as A
import gcn_scenario as S
import gcn_visualization as V


def _rand_graph(rng, n, p):
    M = np.triu((rng.random((n, n)) < p).astype(float), 1)
    return M + M.T


def _loop_layer(H, Ad, deg):
    out = np.zeros_like(H)
    for i in range(len(Ad)):
        agg = H[i] / deg[i]
        for j in np.flatnonzero(Ad[i]):
            agg = agg + H[j] / np.sqrt(deg[i] * deg[j])
        out[i] = agg
    return out


def test_normalization_forward_and_smoothing_match_a_dense_loop_on_random_graphs_including_isolated_nodes():
    rng = np.random.default_rng(1)
    for _ in range(60):
        n = int(rng.integers(1, 11))
        Ad = _rand_graph(rng, n, rng.random())
        deg = 1 + Ad.sum(axis=1)
        X = rng.normal(size=(n, 3))
        W = [rng.normal(size=(3, 4)), rng.normal(size=(4, 2))]
        Ah = A.normalized_adjacency(Ad)
        assert np.allclose(Ah, _loop_layer(np.eye(n), Ad, deg))
        ref = np.maximum(_loop_layer(X, Ad, deg) @ W[0], 0)
        ref = _loop_layer(ref, Ad, deg) @ W[1]
        assert np.allclose(A.forward(Ah, X, W)[2], ref)
        steps = int(rng.integers(0, 4))
        F = X.copy()
        for _ in range(steps):
            F = _loop_layer(F, Ad, deg)
        assert np.allclose(A.propagate(Ad, X, steps), F)


def test_forward_is_equivariant_under_renumbering_the_nodes():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(2, 12))
        Ad = _rand_graph(rng, n, 0.4)
        X = rng.normal(size=(n, 3))
        W = A.init_weights([3, 5, 2], int(rng.integers(100)))
        perm = rng.permutation(n)
        lg = A.forward(A.normalized_adjacency(Ad), X, W)[2]
        lp = A.forward(A.normalized_adjacency(Ad[np.ix_(perm, perm)]), X[perm], W)[2]
        assert np.allclose(lp, lg[perm])


def test_knn_graph_and_graph_statistics_agree_with_scipy_and_networkx():
    spatial = pytest.importorskip("scipy.spatial")
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(3)
    for t in range(40):
        n = int(rng.integers(8, 50))
        k = int(rng.integers(2, min(10, n - 1) + 1))
        xy = rng.random((n, 2)) * 100
        Ak = S.knn_adjacency(xy, k)
        _, idx = spatial.cKDTree(xy).query(xy, k=k + 1)
        R = np.zeros((n, n))
        for i in range(n):
            for j in idx[i]:
                if j != i:
                    R[i, j] = 1.0
        assert np.array_equal(Ak, np.maximum(R, R.T))
        B = S.rewire(Ak, float(rng.choice([0.0, 0.3, 1.0])), np.random.default_rng(t))
        assert np.array_equal(B, B.T) and np.diag(B).sum() == 0 and int(np.triu(B, 1).sum()) == int(np.triu(Ak, 1).sum())
        y = rng.integers(0, 3, size=n)
        g = S.Graph(xy, rng.normal(size=(n, 4)), y, B, 3, 0)
        G = nx.from_numpy_array(B)
        assert g.n_edges() == G.number_of_edges()
        same = [y[u] == y[v] for u, v in G.edges()]
        assert g.edge_homophily() == pytest.approx(float(np.mean(same)) if same else 1.0)
        src = int(rng.integers(n))
        hmax = int(rng.integers(1, 5))
        sp = nx.single_source_shortest_path_length(G, src)
        ref = np.array([sp[i] if (i in sp and sp[i] <= hmax) else -1 for i in range(n)])
        assert np.array_equal(V.hop_distances(B, src, hmax), ref)
