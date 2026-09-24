"""Vehikel (Reproduzierbarkeit, Homophilie, falsche Kanten, Aufteilung) und Auswertung (Analyse, drei Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import gcn_constants as C
import gcn_evaluation as E
import gcn_scenario as S


def test_generate_is_reproducible_and_shaped():
    a, b = S.generate(100, 3, 5, 1.5, 0.0, 4), S.generate(100, 3, 5, 1.5, 0.0, 4)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.A, b.A) and np.array_equal(a.y, b.y)
    assert a.X.shape == (100, C.N_FEATURES) and a.A.shape == (100, 100) and set(np.unique(a.y)) == {0, 1, 2}
    assert not np.array_equal(a.X, S.generate(100, 3, 5, 1.5, 0.0, 5).X)


def test_adjacency_is_symmetric_without_loops_and_each_node_has_at_least_k_neighbors():
    g = S.generate(150, 3, 6, 1.0, 0.0, 2)
    assert np.array_equal(g.A, g.A.T) and np.all(np.diag(g.A) == 0) and set(np.unique(g.A)) <= {0.0, 1.0}
    assert g.A.sum(axis=1).min() >= 6


def test_neighbors_are_spatially_closest():
    g = S.generate(60, 3, 4, 1.0, 0.0, 6)
    d = np.linalg.norm(g.xy[:, None] - g.xy[None], axis=2) + np.eye(60) * 1e9
    for i in range(60):
        nearest = set(np.argsort(d[i])[:4])
        assert nearest <= set(np.flatnonzero(g.A[i]))


def test_zones_are_spatially_coherent_so_neighbors_share_the_type():
    g = S.generate(200, 3, 5, 1.0, 0.0, 3)
    assert g.edge_homophily() > 0.85


def test_rewire_keeps_the_edge_count_and_lowers_the_homophily_monotonically():
    g = S.generate(200, 3, 5, 1.0, 0.0, 3)
    homs, counts = [], []
    for w in (0.0, 0.3, 0.6, 1.0):
        g2 = S.generate(200, 3, 5, 1.0, w, 3)
        homs.append(g2.edge_homophily())
        counts.append(g2.n_edges())
        assert np.array_equal(g2.A, g2.A.T) and np.all(np.diag(g2.A) == 0)
    assert counts == [g.n_edges()] * 4 and homs == sorted(homs, reverse=True) and homs[-1] < 0.45          # zufällige Kanten: etwa Summe der Klassenanteile zum Quadrat (~ 1/3 bis 0,4)


def test_rewire_replaces_exactly_the_requested_share_of_edges():
    rng = np.random.default_rng(1)
    g = S.generate(120, 3, 5, 1.0, 0.0, 2)
    B = S.rewire(g.A, 0.5, rng)
    kept = int((np.triu(g.A, 1) * np.triu(B, 1)).sum())
    assert kept >= round(0.5 * g.n_edges()) - 1 and int(np.triu(B, 1).sum()) == g.n_edges()


def test_split_gives_exactly_per_class_known_labels():
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    tr = S.split(y, 5, 1)
    assert tr.sum() == 15 and all(tr[y == c].sum() == 5 for c in range(3))
    assert np.array_equal(tr, S.split(y, 5, 1)) and not np.array_equal(tr, S.split(y, 5, 2))
    assert S.split(np.array([0, 0, 1]), 5, 1).sum() == 2                                            # weniger Kunden als gewünscht: es bleibt Prüfmenge übrig (hier je Typ mindestens ein bekannter)
    y8 = np.array([0] * 8 + [1] * 8 + [2] * 8)
    tr8 = S.split(y8, 20, 1)
    assert all(tr8[y8 == c].sum() == 6 for c in range(3)) and (~tr8).sum() == 6              # mindestens zwei unbekannte Kunden je Typ


def test_analyse_is_consistent():
    a = E.analyse(E.Settings(n=100, seed=3))
    assert a.pred_history.shape == (C.EPOCHS, 100) and np.array_equal(a.pred_history[-1], a.pred_gcn)
    assert a.acc_gcn == pytest.approx(a.gcn.history["test_acc"][-1]) and a.acc_mlp == pytest.approx(a.mlp.history["test_acc"][-1])
    assert 0 < a.majority_rate() < 1 and a.train_mask.sum() == 3 * 5


def test_settings_extremes_run():
    for st in (E.Settings(60, 2, 2, 2, 0.5, 0.0, 1, 0), E.Settings(400, 4, 20, 10, 4.0, 1.0, 8, 1), E.Settings(60, 4, 20, 2, 1.0, 0.0, 2, 2)):
        a = E.analyse(st)
        assert 0 <= a.acc_gcn <= 1 and 0 <= a.acc_mlp <= 1


def test_labels_experiment_shape_and_trend():
    rows = E.labels_experiment(levels=(2, 10), seeds=range(4))
    assert [r["labels"] for r in rows] == [2, 10] and all(r["n_seeds"] == 4 and 0 <= r["wins"] <= 4 for r in rows)
    assert all(r["gcn"] > r["mlp"] for r in rows)


def test_wrong_experiment_shape_and_trend():
    rows = E.wrong_experiment(levels=(0.0, 1.0), seeds=range(4))
    assert rows[0]["homophily"] > rows[1]["homophily"] and rows[0]["gcn"] > rows[1]["gcn"] and rows[0]["mlp"] == pytest.approx(rows[1]["mlp"])          # das MLP sieht die Kanten nie


def test_smoothing_experiment_shape():
    r = E.smoothing_experiment(steps=(0, 1, 8), depths=(1, 2), seeds=range(3))
    assert [x["steps"] for x in r["propagation"]] == [0, 1, 8] and r["propagation"][0]["spread"] == pytest.approx(1.0)
    assert r["propagation"][1]["spread"] < 1 and [x["layers"] for x in r["depth"]] == [1, 2]
