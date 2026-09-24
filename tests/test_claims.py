"""Jede Zahl aus PRESET_HELP und README: Presets auf die gezeigte Rundung (deterministisch, numpy-only), Mehr-Seed-Befunde mit Bändern (CI installiert die neueste numpy)."""

import numpy as np
import pytest

import gcn_constants as C
import gcn_evaluation as E
import gcn_presets as P


def analyse_preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["wrong"], p["layers"], p["seed"]))


# --- Presets: Bänder um die in PRESET_HELP gezeigten Werte (Rundung + Rechenfehler der neuesten numpy) -------------------------------------


def test_preset_standard():
    a = analyse_preset("Standardfall")
    assert a.acc_gcn == pytest.approx(0.881, abs=0.015) and a.acc_mlp == pytest.approx(0.686, abs=0.015)
    assert int((~a.train_mask).sum()) == 185 and a.graph.edge_homophily() == pytest.approx(0.924, abs=0.001) and a.graph.n_edges() == 608


def test_preset_two_labels():
    a = analyse_preset("Nur 2 Etiketten je Gebietstyp")
    assert int(a.train_mask.sum()) == 6 and a.acc_gcn == pytest.approx(0.814, abs=0.015) and a.acc_mlp == pytest.approx(0.500, abs=0.015) and a.majority_rate() == pytest.approx(0.407, abs=0.002)


def test_preset_wrong_neighbors():
    a = analyse_preset("Halb falsche Nachbarn")
    assert a.graph.edge_homophily() == pytest.approx(0.464, abs=0.001) and a.acc_gcn == pytest.approx(0.497, abs=0.015) and a.acc_mlp == pytest.approx(0.686, abs=0.015)
    assert a.acc_gcn < a.acc_mlp and a.gcn.history["train_acc"][-1] == pytest.approx(0.80, abs=0.07)                                   # lernt nicht einmal alle 15 bekannten Etiketten


def test_preset_noisy_features():
    a = analyse_preset("Stark verrauschte Merkmale")
    assert a.acc_mlp == pytest.approx(0.427, abs=0.015) and a.majority_rate() == pytest.approx(0.411, abs=0.002) and a.acc_gcn == pytest.approx(0.773, abs=0.015)


def test_preset_deep_network():
    a = analyse_preset("Tiefes Netz (8 Schichten)")
    b = analyse_preset("Standardfall")
    assert a.acc_gcn == pytest.approx(0.919, abs=0.015) and b.acc_gcn == pytest.approx(0.881, abs=0.015) and a.acc_gcn > b.acc_gcn - 0.03


def test_preset_four_types():
    a = analyse_preset("Vier Gebietstypen")
    assert a.acc_gcn == pytest.approx(0.775, abs=0.015) and a.acc_mlp == pytest.approx(0.321, abs=0.015) and a.majority_rate() == pytest.approx(0.339, abs=0.002)
    assert a.acc_mlp < a.majority_rate()                                                                     # das MLP ist schlechter als Raten


def test_spread_over_areas_is_large():
    """Einzelne Gebiete streuen stark: GCN und MLP über 8 Seeds (Standard-Einstellungen); der Graph gewinnt in jedem."""
    rows = [(E.analyse(E.Settings(seed=s)).acc_gcn, E.analyse(E.Settings(seed=s)).acc_mlp) for s in range(8)]
    g, m = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    assert g.min() == pytest.approx(0.627, abs=0.03) and g.max() == pytest.approx(0.941, abs=0.03) and m.min() == pytest.approx(0.395, abs=0.03) and m.max() == pytest.approx(0.686, abs=0.03)
    assert (g > m).all()


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labels_rows():
    return {r["labels"]: r for r in E.labels_experiment()}


def test_labels_experiment(labels_rows):
    r2, r20 = labels_rows[2], labels_rows[20]
    assert r2["gcn"] == pytest.approx(0.827, abs=0.03) and r2["mlp"] == pytest.approx(0.520, abs=0.03) and r20["gcn"] == pytest.approx(0.914, abs=0.03) and r20["mlp"] == pytest.approx(0.607, abs=0.03)
    assert all(r["wins"] >= 11 and r["n_seeds"] == 12 for r in labels_rows.values())                       # gemessen 12 von 12
    assert (r2["gcn"] - r2["mlp"]) > (r20["gcn"] - r20["mlp"]) - 0.02                                      # Vorsprung bei wenigen Etiketten mindestens so groß
    assert labels_rows[2]["gcn"] < labels_rows[10]["gcn"] < labels_rows[20]["gcn"] + 0.02                  # mehr Etiketten helfen


@pytest.fixture(scope="module")
def wrong_rows():
    return {r["wrong"]: r for r in E.wrong_experiment()}


def test_wrong_edges_experiment(wrong_rows):
    w = wrong_rows
    assert w[0.0]["homophily"] == pytest.approx(0.913, abs=0.01) and w[1.0]["homophily"] == pytest.approx(0.384, abs=0.02)
    assert w[0.0]["gcn"] == pytest.approx(0.879, abs=0.03) and w[1.0]["gcn"] == pytest.approx(0.462, abs=0.04) and all(w[x]["mlp"] == pytest.approx(w[0.0]["mlp"], abs=1e-12) for x in w) and w[0.0]["mlp"] == pytest.approx(0.578, abs=0.005)      # MLP sieht die Kanten nie
    assert w[0.6]["gcn"] > w[0.6]["mlp"] and w[0.8]["gcn"] < w[0.8]["mlp"] and w[1.0]["gcn"] < w[1.0]["mlp"]                     # Kipppunkt zwischen Homophilie 59 % und 49 %
    assert w[0.6]["gcn"] == pytest.approx(0.655, abs=0.04) and w[0.8]["gcn"] == pytest.approx(0.529, abs=0.04) and w[0.6]["homophily"] == pytest.approx(0.592, abs=0.02) and w[0.8]["homophily"] == pytest.approx(0.486, abs=0.02)
    assert w[0.0]["wins"] == 12 and w[0.6]["wins"] >= 7 and w[0.8]["wins"] <= 6 and w[1.0]["wins"] <= 3
    accs = [w[x]["gcn"] for x in sorted(w)]
    assert accs == sorted(accs, reverse=True)                                                                  # je mehr falsche Kanten, desto schlechter


@pytest.fixture(scope="module")
def smoothing():
    return E.smoothing_experiment()


def test_smoothing_experiment(smoothing):
    prop = {r["steps"]: r for r in smoothing["propagation"]}
    assert prop[0]["acc"] == pytest.approx(0.624, abs=0.03) and prop[0]["spread"] == pytest.approx(1.0)
    best = max(prop.values(), key=lambda r: r["acc"])
    assert best["steps"] in (4, 8, 16) and best["acc"] == pytest.approx(0.92, abs=0.03)                        # gemessen: Höchstwert nach 8 Schritten
    assert prop[128]["acc"] < best["acc"] - 0.08 and prop[128]["acc"] == pytest.approx(0.78, abs=0.04)          # danach fällt sie wieder
    assert prop[1]["acc"] == pytest.approx(0.856, abs=0.03) and prop[8]["acc"] == pytest.approx(0.920, abs=0.03)
    assert prop[128]["spread"] == pytest.approx(0.135, abs=0.03) and prop[1]["spread"] == pytest.approx(0.527, abs=0.03)
    spreads = [prop[k]["spread"] for k in sorted(prop)]
    assert spreads == sorted(spreads, reverse=True)                                                            # die Streuung schrumpft monoton
    depth = smoothing["depth"]
    accs = [r["gcn"] for r in depth]
    assert [r["layers"] for r in depth] == list(range(1, 9)) and min(accs) > 0.82 and max(accs) < 0.93
    assert accs[0] == pytest.approx(0.845, abs=0.03) and accs[1] == pytest.approx(0.879, abs=0.03)
    assert accs[1] > accs[0] and max(r["gcn_se"] for r in depth) < 0.035                                          # 2 Schichten besser als 1; kein Einbruch bis 8


def test_constants_used_in_the_texts():
    assert C.EPOCHS == 200 and C.HIDDEN == 16 and C.WEIGHT_DECAY == pytest.approx(5e-4) and C.LEARNING_RATE == pytest.approx(0.01)
