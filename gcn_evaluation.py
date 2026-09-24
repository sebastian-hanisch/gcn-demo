"""Auswertung: GCN gegen MLP (dasselbe Netz ohne Nachbarn) auf einem Liefergebiets-Graphen und drei Experimente (Zahl der Etiketten, falsche Kanten, Tiefe und Glättung)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import gcn_algorithm as A
import gcn_constants as C
import gcn_scenario as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    classes: int = C.DEFAULT_CLASSES
    labels: int = C.DEFAULT_LABELS
    neighbors: int = C.DEFAULT_NEIGHBORS
    noise: float = C.DEFAULT_NOISE
    wrong: float = C.DEFAULT_WRONG
    layers: int = C.DEFAULT_LAYERS
    seed: int = 7


@dataclass
class Analysis:
    settings: Settings
    graph: S.Graph
    train_mask: np.ndarray
    gcn: A.Model
    mlp: A.Model
    pred_history: np.ndarray        # GCN-Vorhersage je Epoche (Epochen, Knoten)
    pred_gcn: np.ndarray
    pred_mlp: np.ndarray

    def acc(self, pred):
        m = ~self.train_mask
        return float((pred[m] == self.graph.y[m]).mean())

    @property
    def acc_gcn(self):
        return self.acc(self.pred_gcn)

    @property
    def acc_mlp(self):
        return self.acc(self.pred_mlp)

    def majority_rate(self):
        m = ~self.train_mask
        return float(np.bincount(self.graph.y[m]).max() / m.sum())


@lru_cache(maxsize=64)
def analyse(settings):
    g = S.generate(settings.n, settings.classes, settings.neighbors, settings.noise, settings.wrong, settings.seed)
    tr = S.split(g.y, settings.labels, settings.seed)
    gcn = A.train(g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed, record_pred=True)
    mlp = A.train(g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed, use_graph=False)
    return Analysis(settings, g, tr, gcn, mlp, gcn.history["pred"], A.predict(gcn, g.A, g.X), A.predict(mlp, g.A, g.X, use_graph=False))


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _one(settings):
    g = S.generate(settings.n, settings.classes, settings.neighbors, settings.noise, settings.wrong, settings.seed)
    tr = S.split(g.y, settings.labels, settings.seed)
    gcn = A.train(g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed)
    mlp = A.train(g.A, g.X, g.y, tr, layers=settings.layers, seed=settings.seed, use_graph=False)
    return g, tr, gcn.history["test_acc"][-1], mlp.history["test_acc"][-1]


# --- Experiment 1: Zahl der bekannten Etiketten ------------------------------------------------------------------------------------------------


def labels_experiment(levels=None, seeds=None, base=None):
    levels = C.LABEL_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for m in levels:
        g_acc, m_acc = [], []
        for s in seeds:
            _, _, a, b = _one(Settings(base.n, base.classes, m, base.neighbors, base.noise, base.wrong, base.layers, s))
            g_acc.append(a)
            m_acc.append(b)
        gm, gse = _mean_se(g_acc)
        mm, mse = _mean_se(m_acc)
        rows.append({"labels": m, "gcn": gm, "gcn_se": gse, "mlp": mm, "mlp_se": mse, "wins": int(np.sum(np.array(g_acc) > np.array(m_acc))), "n_seeds": len(seeds)})
    return rows


# --- Experiment 2: falsche Kanten -------------------------------------------------------------------------------------------------------------


def wrong_experiment(levels=None, seeds=None, base=None):
    levels = C.WRONG_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for w in levels:
        g_acc, m_acc, hom = [], [], []
        for s in seeds:
            g, _, a, b = _one(Settings(base.n, base.classes, base.labels, base.neighbors, base.noise, w, base.layers, s))
            g_acc.append(a)
            m_acc.append(b)
            hom.append(g.edge_homophily())
        gm, gse = _mean_se(g_acc)
        mm, mse = _mean_se(m_acc)
        rows.append({"wrong": w, "homophily": float(np.mean(hom)), "gcn": gm, "gcn_se": gse, "mlp": mm, "mlp_se": mse, "wins": int(np.sum(np.array(g_acc) > np.array(m_acc))), "n_seeds": len(seeds)})
    return rows


# --- Experiment 3: Tiefe und Glättung -----------------------------------------------------------------------------------------------------------


def smoothing_experiment(steps=None, depths=None, seeds=None, base=None):
    """(a) Reine Glättung ohne Lernen: Genauigkeit des Nächster-Mittelpunkt-Klassifikators auf A_hat^k X und relative Streuung; (b) trainiertes GCN mit 1 bis 8 Schichten."""
    steps = C.PROP_STEPS if steps is None else steps
    depths = range(C.LAYERS_MIN, C.LAYERS_MAX + 1) if depths is None else depths
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    prop = {k: {"acc": [], "spread": []} for k in steps}
    for s in seeds:
        g = S.generate(base.n, base.classes, base.neighbors, base.noise, base.wrong, s)
        tr = S.split(g.y, base.labels, s)
        for k in steps:
            F = A.propagate(g.A, g.X, k)
            prop[k]["acc"].append(A.centroid_accuracy(F, g.y, tr))
            prop[k]["spread"].append(A.relative_spread(F, g.X))
    prop_rows = [{"steps": k, "acc": float(np.mean(v["acc"])), "spread": float(np.mean(v["spread"]))} for k, v in prop.items()]
    depth_rows = []
    for L in depths:
        accs = []
        for s in seeds:
            _, _, a, _ = _one(Settings(base.n, base.classes, base.labels, base.neighbors, base.noise, base.wrong, L, s))
            accs.append(a)
        m, se = _mean_se(accs)
        depth_rows.append({"layers": L, "gcn": m, "gcn_se": se})
    return {"propagation": prop_rows, "depth": depth_rows}
