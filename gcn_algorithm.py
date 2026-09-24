"""GCN nach Kipf/Welling (2017), von Grund auf in numpy: Schichten H_neu = ReLU(A_hat H W) mit A_hat = D^-1/2 (A + I) D^-1/2, letzte Schicht ohne ReLU, Softmax-Kreuzentropie auf den bekannten Etiketten,
Gewichtszerfall, Adam. Mit A_hat = I ist dasselbe Netz ein gewöhnliches MLP (keine Nachbarn)."""

from dataclasses import dataclass, field

import numpy as np

import gcn_constants as C


def normalized_adjacency(A):
    """A_hat = D^-1/2 (A + I) D^-1/2 mit D = Gradmatrix von A + I."""
    B = A + np.eye(len(A))
    d = B.sum(axis=1)
    inv = 1.0 / np.sqrt(d)
    return B * inv[:, None] * inv[None, :]


def init_weights(sizes, seed):
    """Glorot-Gleichverteilung je Schicht (sizes = [F, H, ..., Klassen])."""
    rng = np.random.default_rng([seed, 31])
    out = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        lim = np.sqrt(6.0 / (a + b))
        out.append(rng.uniform(-lim, lim, size=(a, b)))
    return out


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(Ahat, X, weights):
    """Rückgabe (Z-Liste vor der Aktivierung, H-Liste inklusive Eingabe, Logits)."""
    H = [X]
    Zs = []
    for l, W in enumerate(weights):
        Z = Ahat @ H[-1] @ W
        Zs.append(Z)
        H.append(np.maximum(Z, 0.0) if l < len(weights) - 1 else Z)
    return Zs, H, H[-1]


def loss_and_grads(Ahat, X, y, train, weights, weight_decay):
    """Kreuzentropie über die bekannten Knoten plus Zerfall (weight_decay/2 * Summe W^2); Gradienten aller Gewichte."""
    Zs, H, logits = forward(Ahat, X, weights)
    P = softmax(logits)
    m = int(train.sum())
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    reg = 0.5 * weight_decay * sum(float((W ** 2).sum()) for W in weights)
    dZ = np.zeros_like(logits)
    dZ[idx] = P[idx]
    dZ[idx, y[idx]] -= 1.0
    dZ /= m
    grads = [None] * len(weights)
    for l in range(len(weights) - 1, -1, -1):
        AH = Ahat @ H[l]
        grads[l] = AH.T @ dZ + weight_decay * weights[l]
        if l > 0:
            dH = Ahat @ (dZ @ weights[l].T)
            dZ = dH * (Zs[l - 1] > 0)
    return ce + reg, grads


@dataclass
class Model:
    weights: list
    history: dict = field(default_factory=dict)


def train(A, X, y, train_mask, layers=C.DEFAULT_LAYERS, hidden=C.HIDDEN, epochs=C.EPOCHS, lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY, seed=0, use_graph=True, test_mask=None, record_pred=False):
    """Adam-Training. `use_graph=False` setzt A_hat = I (MLP). Verlauf je Epoche: Verlust, Trainings- und Testgenauigkeit (Test = alle nicht bekannten Knoten, nur zur Anzeige, nie zur Auswahl)."""
    n_classes = int(y.max()) + 1
    Ahat = normalized_adjacency(A) if use_graph else np.eye(len(A))
    sizes = [X.shape[1]] + [hidden] * (layers - 1) + [n_classes]
    W = init_weights(sizes, seed)
    mo = [np.zeros_like(w) for w in W]
    ve = [np.zeros_like(w) for w in W]
    b1, b2, eps = 0.9, 0.999, 1e-8
    test_mask = ~train_mask if test_mask is None else test_mask
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    preds = []
    for t in range(1, epochs + 1):
        loss, grads = loss_and_grads(Ahat, X, y, train_mask, W, weight_decay)
        for l in range(len(W)):
            mo[l] = b1 * mo[l] + (1 - b1) * grads[l]
            ve[l] = b2 * ve[l] + (1 - b2) * grads[l] ** 2
            W[l] = W[l] - lr * (mo[l] / (1 - b1 ** t)) / (np.sqrt(ve[l] / (1 - b2 ** t)) + eps)
        pred = forward(Ahat, X, W)[2].argmax(axis=1)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)                       # (Epochen, Knoten)
    return Model(W, hist)


def predict(model, A, X, use_graph=True):
    Ahat = normalized_adjacency(A) if use_graph else np.eye(len(A))
    return forward(Ahat, X, model.weights)[2].argmax(axis=1)


def hidden_representation(model, A, X, use_graph=True):
    """Ausgabe der vorletzten Schicht (bei einer Schicht: die Logits)."""
    Ahat = normalized_adjacency(A) if use_graph else np.eye(len(A))
    H = forward(Ahat, X, model.weights)[1]
    return H[-2] if len(model.weights) > 1 else H[-1]


def propagate(A, X, steps):
    """A_hat^steps X ohne jedes Lernen (reine Glättung der Merkmale über den Graphen)."""
    Ahat = normalized_adjacency(A)
    F = X.copy()
    for _ in range(steps):
        F = Ahat @ F
    return F


def centroid_accuracy(F, y, train_mask):
    """Nächster-Mittelpunkt-Klassifikator: Klassenmittelpunkte aus den bekannten Knoten, Genauigkeit auf den übrigen."""
    cents = np.array([F[train_mask & (y == c)].mean(axis=0) for c in range(int(y.max()) + 1)])
    pred = np.linalg.norm(F[:, None] - cents[None], axis=2).argmin(axis=1)
    return float((pred[~train_mask] == y[~train_mask]).mean())


def relative_spread(F, F0):
    """Streuung der (zentrierten) Zeilen relativ zur Streuung der Ausgangsmerkmale: 1 = unverändert, 0 = alle Knoten gleich."""
    return float(np.linalg.norm(F - F.mean(axis=0)) / np.linalg.norm(F0 - F0.mean(axis=0)))
