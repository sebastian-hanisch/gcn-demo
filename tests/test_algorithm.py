"""GCN-Kern: Normierung per Handrechnung, Vorwärtsrechnung gegen eine explizite Nachbarschleife, Gradienten gegen zentrale Differenzen, MLP-Gleichwertigkeit, Adam-Schritt, Glättung."""

import numpy as np
import pytest

import gcn_algorithm as A
import gcn_scenario as S


def path3():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)


def test_normalized_adjacency_by_hand_on_a_path_of_three_nodes():
    """Grade mit Schleife: 2, 3, 2. A_hat_00 = 1/2, A_hat_01 = 1/sqrt(6), A_hat_11 = 1/3, A_hat_02 = 0."""
    Ah = A.normalized_adjacency(path3())
    assert Ah[0, 0] == pytest.approx(1 / 2) and Ah[0, 1] == pytest.approx(1 / np.sqrt(6)) and Ah[1, 1] == pytest.approx(1 / 3) and Ah[0, 2] == 0.0
    assert np.allclose(Ah, Ah.T)


def test_normalized_adjacency_spectrum_and_stationary_vector():
    g = S.generate(80, 3, 6, 1.0, 0.0, 3)
    Ah = A.normalized_adjacency(g.A)
    ev = np.linalg.eigvalsh(Ah)
    assert ev.max() == pytest.approx(1.0, abs=1e-9) and ev.min() > -1.0
    d = (g.A + np.eye(80)).sum(axis=1)
    assert Ah @ np.sqrt(d) == pytest.approx(np.sqrt(d))                       # Eigenvektor zum Eigenwert 1


def test_forward_by_hand_for_one_layer():
    """W = Einheitsmatrix, X = (1, 2, 4) als eine Spalte: Ahat X = (1/2 + 2/sqrt6, 1/sqrt6 + 2/3 + 4/sqrt6, 2/sqrt6 + 4/2)."""
    X = np.array([[1.0], [2.0], [4.0]])
    Zs, H, logits = A.forward(A.normalized_adjacency(path3()), X, [np.array([[1.0]])])
    r6 = np.sqrt(6)
    assert logits[:, 0] == pytest.approx([1 / 2 + 2 / r6, 1 / r6 + 2 / 3 + 4 / r6, 2 / r6 + 4 / 2])


def test_forward_agrees_with_an_explicit_neighbor_loop_for_two_layers():
    g = S.generate(40, 3, 4, 1.0, 0.0, 5)
    W = A.init_weights([4, 8, 3], 1)
    deg = (g.A + np.eye(40)).sum(axis=1)

    def layer(H, Wl, relu):
        out = np.zeros((40, Wl.shape[1]))
        for i in range(40):
            agg = np.zeros(H.shape[1])
            for j in range(40):
                if g.A[i, j] or i == j:
                    agg += H[j] / np.sqrt(deg[i] * deg[j])
            out[i] = agg @ Wl
        return np.maximum(out, 0) if relu else out
    ref = layer(layer(g.X, W[0], True), W[1], False)
    assert A.forward(A.normalized_adjacency(g.A), g.X, W)[2] == pytest.approx(ref)


def test_loss_with_zero_weights_is_log_of_the_number_of_classes():
    g = S.generate(30, 3, 4, 1.0, 0.0, 1)
    tr = S.split(g.y, 3, 1)
    loss, grads = A.loss_and_grads(A.normalized_adjacency(g.A), g.X, g.y, tr, [np.zeros((4, 3))], 0.0)
    assert loss == pytest.approx(np.log(3))


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_gradients_agree_with_central_differences(layers):
    g = S.generate(30, 3, 4, 1.0, 0.0, 2)
    tr = S.split(g.y, 4, 2)
    Ah = A.normalized_adjacency(g.A)
    W = A.init_weights([4] + [6] * (layers - 1) + [3], 3)
    _, grads = A.loss_and_grads(Ah, g.X, g.y, tr, W, 5e-4)
    rng = np.random.default_rng(0)
    for l in range(layers):
        for _ in range(6):
            i, j = rng.integers(0, W[l].shape[0]), rng.integers(0, W[l].shape[1])
            h = 1e-6
            Wp = [w.copy() for w in W]
            Wm = [w.copy() for w in W]
            Wp[l][i, j] += h
            Wm[l][i, j] -= h
            num = (A.loss_and_grads(Ah, g.X, g.y, tr, Wp, 5e-4)[0] - A.loss_and_grads(Ah, g.X, g.y, tr, Wm, 5e-4)[0]) / (2 * h)
            assert grads[l][i, j] == pytest.approx(num, rel=1e-4, abs=1e-8)


def test_mlp_is_the_gcn_with_identity_propagation_and_matches_a_plain_mlp():
    g = S.generate(50, 3, 5, 1.0, 0.0, 4)
    W = A.init_weights([4, 8, 3], 2)
    plain = np.maximum(g.X @ W[0], 0) @ W[1]
    assert A.forward(np.eye(50), g.X, W)[2] == pytest.approx(plain)
    tr = S.split(g.y, 4, 4)
    m1 = A.train(g.A, g.X, g.y, tr, seed=2, use_graph=False, epochs=20)
    m2 = A.train(np.zeros((50, 50)), g.X, g.y, tr, seed=2, use_graph=True, epochs=20)          # keine Kanten: A_hat = I
    assert all(np.allclose(a, b) for a, b in zip(m1.weights, m2.weights))


def test_first_adam_step_moves_every_weight_by_about_the_learning_rate():
    g = S.generate(40, 3, 4, 1.0, 0.0, 6)
    tr = S.split(g.y, 4, 6)
    W0 = A.init_weights([4, 3], 1)
    Ah = A.normalized_adjacency(g.A)
    _, grads = A.loss_and_grads(Ah, g.X, g.y, tr, W0, 5e-4)
    m = A.train(g.A, g.X, g.y, tr, layers=1, epochs=1, seed=1, lr=0.01)
    step = m.weights[0] - W0[0]
    big = np.abs(grads[0]) > 1e-6
    assert step[big] == pytest.approx(-0.01 * np.sign(grads[0][big]), abs=1e-6)                # m_hat = g, v_hat = g^2 -> lr * g / |g|


def test_training_is_reproducible_reduces_the_loss_and_fits_the_known_nodes():
    g = S.generate(120, 3, 5, 1.0, 0.0, 8)
    tr = S.split(g.y, 6, 8)
    a = A.train(g.A, g.X, g.y, tr, seed=3)
    b = A.train(g.A, g.X, g.y, tr, seed=3)
    c = A.train(g.A, g.X, g.y, tr, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a.weights, b.weights)) and not np.allclose(a.weights[0], c.weights[0])
    assert a.history["loss"][-1] < 0.5 * a.history["loss"][0] and a.history["train_acc"][-1] == 1.0
    pred = A.predict(a, g.A, g.X)
    assert float((pred[~tr] == g.y[~tr]).mean()) == pytest.approx(a.history["test_acc"][-1])


def test_recorded_predictions_match_the_final_prediction():
    g = S.generate(60, 3, 5, 1.0, 0.0, 9)
    tr = S.split(g.y, 4, 9)
    m = A.train(g.A, g.X, g.y, tr, seed=1, epochs=30, record_pred=True)
    assert m.history["pred"].shape == (30, 60) and np.array_equal(m.history["pred"][-1], A.predict(m, g.A, g.X))


def test_propagation_converges_to_the_degree_weighted_constant_on_a_connected_graph():
    A0 = np.zeros((6, 6))
    for i in range(5):
        A0[i, i + 1] = A0[i + 1, i] = 1.0                                                     # Pfad 0-1-2-3-4-5, zusammenhängend
    X = np.random.default_rng(1).normal(size=(6, 2))
    F = A.propagate(A0, X, 2000)
    d = (A0 + np.eye(6)).sum(axis=1)
    v = np.sqrt(d) / np.linalg.norm(np.sqrt(d))
    assert F == pytest.approx(np.outer(v, v @ X), abs=1e-6)                                    # Projektion auf den Eigenvektor zum Eigenwert 1
    assert A.relative_spread(A.propagate(A0, X, 1), X) < 1 and A.relative_spread(F, X) < A.relative_spread(A.propagate(A0, X, 5), X)


def test_centroid_accuracy_and_relative_spread_by_hand():
    F = np.array([[0.0], [0.2], [5.0], [5.2], [0.1], [4.9]])
    y = np.array([0, 0, 1, 1, 0, 1])
    tr = np.array([True, False, True, False, False, False])
    assert A.centroid_accuracy(F, y, tr) == 1.0                                                # Mittelpunkte 0 und 5
    assert A.centroid_accuracy(np.array([[0.0], [0.2], [5.0], [0.3], [0.1], [4.9]]), y, tr) == pytest.approx(3 / 4)     # Kunde 3 liegt bei 0,3 statt 5,2
    assert A.relative_spread(F, F) == pytest.approx(1.0) and A.relative_spread(np.ones((6, 1)) * 3, F) == 0.0
