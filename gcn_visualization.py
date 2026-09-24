"""Plotly-Abbildungen der GCN-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gcn_constants as C

PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2"]
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"
PURPLE = "#b279a2"
LINE_COLOR = "#4c78a8"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def hop_distances(A, source, max_hops):
    """Zahl der Kanten vom Knoten `source` bis zu jedem Knoten (breite Suche); nicht erreichbar oder weiter als max_hops: -1."""
    n = len(A)
    dist = np.full(n, -1)
    dist[source] = 0
    frontier = [source]
    for h in range(1, max_hops + 1):
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(A[u]):
                if dist[v] < 0:
                    dist[v] = h
                    nxt.append(v)
        frontier = nxt
    return dist


def build_map(a, epoch, mode="pred", node=None, hops=0):
    """Kunden im Gebiet: Farbe = vorhergesagter (oder wahrer) Gebietstyp, x = falsch vorhergesagt, schwarzer Ring = bekanntes Etikett; optional das Empfangsfeld eines Knotens."""
    g = a.graph
    pred = a.pred_history[epoch - 1] if mode == "pred" else g.y
    fig = go.Figure()
    iu = np.array(np.nonzero(np.triu(g.A, 1)))
    xs, ys = [], []
    for i, j in zip(*iu):
        xs += [g.xy[i, 0], g.xy[j, 0], None]
        ys += [g.xy[i, 1], g.xy[j, 1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=0.8), hoverinfo="skip", showlegend=False))
    if node is not None and hops > 0:
        d = hop_distances(g.A, node, hops)
        reach = np.flatnonzero(d >= 0)
        fig.add_trace(go.Scatter(x=g.xy[reach, 0], y=g.xy[reach, 1], mode="markers", marker=dict(size=19, color="rgba(245,133,24,0.28)"), name=f"Empfangsfeld ({hops} Schicht{'en' if hops > 1 else ''})", hoverinfo="skip"))
    for c in range(g.n_classes):
        ok = (pred == c) & (pred == g.y)
        bad = (pred == c) & (pred != g.y)
        fig.add_trace(go.Scatter(x=g.xy[ok, 0], y=g.xy[ok, 1], mode="markers", marker=dict(size=8, color=PALETTE[c], line=dict(color="white", width=0.6)), name=C.CLASS_NAMES[c],
                                 hovertemplate=f"{C.CLASS_NAMES[c]}<extra></extra>"))
        if mode == "pred" and bad.any():
            fig.add_trace(go.Scatter(x=g.xy[bad, 0], y=g.xy[bad, 1], mode="markers", marker=dict(size=10, color=PALETTE[c], symbol="x", line=dict(color=PALETTE[c], width=2)), showlegend=False,
                                     hovertemplate=f"vorhergesagt: {C.CLASS_NAMES[c]} (falsch)<extra></extra>"))
    tr = a.train_mask
    fig.add_trace(go.Scatter(x=g.xy[tr, 0], y=g.xy[tr, 1], mode="markers", marker=dict(size=14, symbol="circle-open", color="black", line=dict(width=2)), name="bekanntes Etikett"))
    if node is not None:
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", marker=dict(size=17, symbol="star", color=WARN, line=dict(color="black", width=1)), name="gewählter Kunde"))
    fig.update_xaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y")
    fig.update_yaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, 470).update_layout(legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=10, b=10))


def build_curves(a, epoch):
    """Links: Genauigkeit der unbekannten Knoten (GCN, MLP) und der bekannten (GCN) über die Epochen; rechts: Trainingsverlust des GCN."""
    E = len(a.gcn.history["loss"])
    xs = list(range(1, E + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Genauigkeit", "Trainingsverlust (GCN)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.gcn.history["test_acc"]], mode="lines", name="GCN, unbekannte Knoten", line=dict(color=PURPLE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.mlp.history["test_acc"]], mode="lines", name="MLP (ohne Nachbarn), unbekannte Knoten", line=dict(color=LINE_COLOR, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.gcn.history["train_acc"]], mode="lines", name="GCN, bekannte Knoten", line=dict(color=REF_COLOR, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=a.gcn.history["loss"], mode="lines", name="Verlust", line=dict(color=PURPLE, width=2.5), showlegend=False), row=1, col=2)
    for col in (1, 2):
        fig.add_vline(x=epoch, line=dict(color=WARN, dash="dash"), row=1, col=col)
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="Prozent", range=[0, 102], row=1, col=1)
    fig.update_yaxes(title_text="Kreuzentropie + Zerfall", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_labels(rows):
    xs = [r["labels"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["gcn"] for r in rows], error_y=dict(type="data", array=[100 * r["gcn_se"] for r in rows]), mode="lines+markers", name="GCN", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["mlp"] for r in rows], error_y=dict(type="data", array=[100 * r["mlp_se"] for r in rows]), mode="lines+markers", name="MLP (ohne Nachbarn)", line=dict(color=LINE_COLOR, width=2.5)))
    fig.update_xaxes(title_text="Bekannte Etiketten je Gebietstyp", type="log", tickvals=xs, ticktext=[str(x) for x in xs])
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, 100])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_wrong(rows):
    xs = [100 * r["homophily"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["gcn"] for r in rows], error_y=dict(type="data", array=[100 * r["gcn_se"] for r in rows]), mode="lines+markers", name="GCN", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * r["mlp"] for r in rows], error_y=dict(type="data", array=[100 * r["mlp_se"] for r in rows]), mode="lines+markers", name="MLP (ohne Nachbarn)", line=dict(color=LINE_COLOR, width=2.5)))
    fig.update_xaxes(title_text="Anteil der Kanten zwischen Kunden desselben Gebietstyps (%)", autorange="reversed")
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, 100])
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_smoothing(res):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reine Glättung ohne Lernen", "Trainiertes GCN nach Tiefe"), horizontal_spacing=0.13)
    prop = res["propagation"]
    labels = [str(r["steps"]) for r in prop]
    fig.add_trace(go.Scatter(x=labels, y=[100 * r["acc"] for r in prop], mode="lines+markers", name="Genauigkeit (nächster Mittelpunkt)", line=dict(color=PURPLE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=[100 * r["spread"] for r in prop], mode="lines+markers", name="Streuung der Merkmale (% des Ausgangs)", line=dict(color=WARN, width=2.5, dash="dot")), row=1, col=1)
    dep = res["depth"]
    fig.add_trace(go.Scatter(x=[r["layers"] for r in dep], y=[100 * r["gcn"] for r in dep], error_y=dict(type="data", array=[100 * r["gcn_se"] for r in dep]), mode="lines+markers", name="GCN", line=dict(color=PURPLE, width=2.5),
                             showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Glättungsschritte", row=1, col=1)
    fig.update_xaxes(title_text="Schichten", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="Prozent", range=[0, 102], row=1, col=1)
    fig.update_yaxes(title_text="Genauigkeit (%)", range=[50, 100], row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)
