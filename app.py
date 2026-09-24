"""GCN - Graph Convolutional Network - wie Nachbarn ein Etikett ersetzen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück der Graph-Neural-Network-Linie der "Konzepte"-Reihe (Wurzel): Kunden in einem Liefergebiet, ein Graph der räumlichen Nachbarn, vier Merkmale je Kunde - wenige bekannte Gebietstypen, der Rest wird
vorhergesagt.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import gcn_algorithm as A
import gcn_constants as C
import gcn_evaluation as E
from gcn_evaluation import Settings, analyse, labels_experiment, smoothing_experiment, wrong_experiment
from gcn_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from gcn_visualization import hop_distances, build_curves, build_labels, build_map, build_smoothing, build_wrong

st.set_page_config(page_title="GCN – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _labels(levels, seeds, base):
    return labels_experiment(levels=levels, seeds=seeds, base=base)


@st.cache_data(show_spinner=False)
def _wrong(levels, seeds, base):
    return wrong_experiment(levels=levels, seeds=seeds, base=base)


@st.cache_data(show_spinner=False)
def _smoothing(steps, seeds, base):
    return smoothing_experiment(steps=steps, seeds=seeds, base=base)


st.title("🕸️ GCN – Nachbarn statt Etiketten")
st.markdown(
    """
Von einem Liefergebiet kennen wir für nur wenige Kunden den **Gebietstyp** (Innenstadt, Vorstadt, ...); für alle anderen gibt es vier verrauschte Merkmale (Stopps je Stunde, Parksuche, Ladegewicht, Zeitfenster-Enge)
und die **räumlichen Nachbarn**. Ein **Graph Convolutional Network** (Kipf/Welling 2017) mischt in jeder Schicht die Merkmale eines Kunden mit denen seiner Nachbarn - gewichtet nach der Zahl der Nachbarn - und lernt
darauf ein kleines neuronales Netz. Dieselbe Rechnung ohne Nachbarn ist ein gewöhnliches **MLP**. Die Demo zeigt, wann der Graph hilft, wann er schadet, und was Tiefe mit den Merkmalen macht.
"""
)
st.caption(
    "Erstes Stück der **Graph-Neural-Network-Linie** der \"Konzepte\"-Reihe (Wurzel) und die erste Demo des Portfolios mit neuronalen Netzen auf Graphen. Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben. "
    "**Bezug zu OR:** dieselben Nachbarschaftsgraphen tragen Tourenplanung und Gebietszuschnitt; hier lernt ein Netz auf ihnen, statt dass ein Optimierer über sie läuft."
)

with st.expander("So funktioniert ein GCN", expanded=True):
    st.markdown(
        """
1. **Graph.** Kunden sind Knoten, die $k$ nächsten Nachbarn Kanten (Matrix $A$). Jeder Knoten bekommt eine Schleife zu sich selbst: $\\tilde A = A + I$.
2. **Normierung.** $\\hat A = D^{-1/2} \\tilde A D^{-1/2}$ mit $D$ = Zahl der Nachbarn plus 1. Ein Nachbar zählt mit $1/\\sqrt{(d_i+1)(d_j+1)}$ - alle Nachbarn ähnlich, unabhängig von ihrem Inhalt.
3. **Schicht.** $H' = \\mathrm{ReLU}(\\hat A\\,H\\,W)$: erst über den Graphen mitteln, dann mit einer lernbaren Matrix $W$ umrechnen. Nach $L$ Schichten sieht ein Knoten alle Knoten im Abstand $\\le L$.
4. **Lernen.** Nur die bekannten Etiketten zählen (Kreuzentropie), alle Knoten rechnen mit; Adam, Gewichtszerfall, feste 200 Epochen - nichts wird am Test abgestimmt.
5. **MLP zum Vergleich.** Setzt man $\\hat A = I$, bleibt dasselbe Netz ohne Nachbarn übrig; jede Differenz ist also allein dem Graphen zu verdanken.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_nodes = st.slider("Kunden", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Zahl der Kunden im Gebiet.")
    classes = st.slider("Gebietstypen", *bounds("classes_slider"), key="classes_slider", help="Zahl der Gebietstypen (räumlich zusammenhängend, nächstes von 2 bis 4 verdeckten Zentren).")
    labels = st.slider("Bekannte Etiketten je Gebietstyp", *bounds("labels_slider"), key="labels_slider", help="Nur diese Kunden verraten dem Netz ihren Gebietstyp; alle anderen werden vorhergesagt und zur Prüfung benutzt.")
    neighbors = st.slider("Nachbarn je Kunde", *bounds("neighbors_slider"), key="neighbors_slider", help="Jeder Kunde wird mit seinen k räumlich nächsten Kunden verbunden (Kanten symmetrisch).")
    noise = st.slider("Rauschen der Merkmale", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung der Merkmale um den Mittelwert ihres Gebietstyps. Je höher, desto weniger sagt ein einzelner Kunde über seinen Typ.")
    wrong = st.slider("Anteil falscher Kanten", *bounds("wrong_slider"), key="wrong_slider", step=C.WRONG_STEP, help="Anteil der Kanten, die durch zufällige ersetzt werden (z. B. veraltete Nachbarschaftslisten); die Zahl der Kanten bleibt.")
    layers = st.slider("Schichten", *bounds("layers_slider"), key="layers_slider", help="Zahl der GCN-Schichten (auch für das MLP, das dann ebenso tief ist).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Gebiet, Merkmale, bekannte Etiketten und Anfangsgewichte fest.")
    st.button("🎲 Neues Gebiet generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n_nodes), "classes_slider": int(classes), "labels_slider": int(labels), "neighbors_slider": int(neighbors), "noise_slider": round(float(noise), 2), "wrong_slider": round(float(wrong), 2),
                   "layers_slider": int(layers), "seed_input": int(seed)})

settings = Settings(int(n_nodes), int(classes), int(labels), int(neighbors), round(float(noise), 2), round(float(wrong), 2), int(layers), int(seed))
with st.spinner("Trainiere GCN und MLP..."):
    a = analyse(settings)
g = a.graph
n = g.n

# --- Das Liefergebiet ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Liefergebiet und was das Netz lernt")
c1, c2 = st.columns([1, 1])
with c1:
    epoch = st.slider("Trainingsepoche", 1, C.EPOCHS, C.EPOCHS, key="epoch_slider", help="Wie weit das GCN trainiert ist; die Karte zeigt seine Vorhersage nach dieser Epoche.")
with c2:
    mode = st.radio("Karte zeigt", ["Vorhersage des GCN", "Wahrer Gebietstyp"], horizontal=True, key="map_mode")
st.plotly_chart(build_map(a, epoch, "pred" if mode == "Vorhersage des GCN" else "truth"), width="stretch", key="map_chart")
pred_now = a.pred_history[epoch - 1]
unknown = ~a.train_mask
st.caption(
    f"{n} Kunden, {g.n_edges()} Kanten; {int(a.train_mask.sum())} bekannte Etiketten (schwarze Ringe). Ein Kreuz markiert eine falsche Vorhersage. Nach Epoche {epoch} liegt das GCN bei "
    f"{pct(float((pred_now[unknown] == g.y[unknown]).mean()))} richtig auf den {int(unknown.sum())} unbekannten Kunden. Von den Kanten verbinden {pct(g.edge_homophily())} Kunden desselben Gebietstyps (Homophilie)."
)

st.markdown("---")

# --- GCN gegen MLP ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 GCN gegen MLP: was bringt der Graph?")
m1, m2, m3 = st.columns(3)
m1.metric("GCN (mit Nachbarn)", pct(a.acc_gcn, 1), help="Genauigkeit auf den unbekannten Kunden nach 200 Epochen.")
m2.metric("MLP (ohne Nachbarn)", pct(a.acc_mlp, 1), delta=f"{de(100 * (a.acc_gcn - a.acc_mlp), 1)} Punkte für den Graphen", delta_color="off", help="Dasselbe Netz mit A_hat = I.")
m3.metric("Häufigster Typ (Raten)", pct(a.majority_rate(), 1), help="Anteil des häufigsten Gebietstyps unter den unbekannten Kunden.")
st.plotly_chart(build_curves(a, epoch), width="stretch", key="curves_chart")
diff = a.acc_gcn - a.acc_mlp
if diff > 0.03:
    st.success(f"✅ Der Graph hilft: {pct(a.acc_gcn, 1)} gegen {pct(a.acc_mlp, 1)} ohne Nachbarn. Beim Mitteln über die Nachbarn hebt sich das Rauschen der einzelnen Kunden auf, und Nachbarn haben meist denselben Gebietstyp (Homophilie {pct(g.edge_homophily())}).")
elif diff < -0.03:
    st.warning(f"⚠️ Der Graph schadet: {pct(a.acc_gcn, 1)} gegen {pct(a.acc_mlp, 1)} ohne Nachbarn. Nur {pct(g.edge_homophily())} der Kanten verbinden Kunden desselben Typs - das Mitteln mischt die Typen.")
else:
    st.info(f"Kein klarer Unterschied: {pct(a.acc_gcn, 1)} gegen {pct(a.acc_mlp, 1)} (Homophilie {pct(g.edge_homophily())}).")

st.markdown("---")

# --- Was ein Knoten sieht ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was ein Kunde von seinen Nachbarn sieht")
default_node = int(np.argmin(np.linalg.norm(g.xy - C.AREA / 2, axis=1)))
if "node_select" not in st.session_state or st.session_state["node_select"] >= n:
    st.session_state["node_select"] = default_node
node = st.number_input("Kunde Nr.", 0, n - 1, key="node_select", step=1, help="Nummer eines Kunden (0 bis n-1); die Karte zeigt sein Empfangsfeld nach der gewählten Zahl von Schichten.")
node = int(node)
st.plotly_chart(build_map(a, C.EPOCHS, "pred", node=node, hops=int(layers)), width="stretch", key="node_map")
Ahat = A.normalized_adjacency(g.A)
nbrs = np.flatnonzero(g.A[node])
order = np.argsort(-Ahat[node, nbrs])
rows = [{"Kunde": f"{node} (er selbst)", "Wahrer Typ": C.CLASS_NAMES[g.y[node]], "Gewicht in Â": de(Ahat[node, node], 3), "Gleicher Typ": "ja"}]
rows += [{"Kunde": str(int(nbrs[k])), "Wahrer Typ": C.CLASS_NAMES[g.y[nbrs[k]]], "Gewicht in Â": de(Ahat[node, nbrs[k]], 3), "Gleicher Typ": "ja" if g.y[nbrs[k]] == g.y[node] else "nein"} for k in order]
st.dataframe(rows, hide_index=True)
reach = int((hop_distances(g.A, node, int(layers)) >= 0).sum())
st.caption(
    f"Kunde {node} hat {len(nbrs)} Nachbarn; jedem gibt Â ein Gewicht von etwa 1/√((d+1)(d'+1)) - unabhängig davon, was der Nachbar mitteilt. Nach {int(layers)} Schicht{'en' if layers > 1 else ''} fließen Informationen von {reach} Kunden "
    f"(orange hinterlegt) in seine Vorhersage ein; er selbst wird {'richtig' if a.pred_gcn[node] == g.y[node] else 'falsch'} als {C.CLASS_NAMES[a.pred_gcn[node]]} vorhergesagt (wahr: {C.CLASS_NAMES[g.y[node]]}). "
    f"Von seinen Nachbarn haben {int((g.y[nbrs] == g.y[node]).sum())} von {len(nbrs)} denselben Gebietstyp."
)

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viele Etiketten braucht das Netz?")
st.caption(f"Standardgebiet (200 Kunden, 3 Typen, Rauschen {de(settings.noise)}), Etiketten je Typ von {C.LABEL_LEVELS[0]} bis {C.LABEL_LEVELS[-1]}; Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler).")
if st.button("Etiketten durchrechnen (dauert etwa 10 Sekunden)", key="labels_start"):
    st.session_state["labels_on"] = True
if st.session_state.get("labels_on"):
    base_l = Settings(200, 3, 5, 5, settings.noise, 0.0, 2, 0)
    rows_l = _labels(C.LABEL_LEVELS, C.EXP_SEEDS, base_l)
    st.plotly_chart(build_labels(rows_l), width="stretch", key="labels_chart")
    lo_, hi_ = rows_l[0], rows_l[-1]
    st.warning(
        f"**Befund:** Mit {lo_['labels']} Etiketten je Typ erreicht das GCN {pct(lo_['gcn'])} gegen {pct(lo_['mlp'])} beim MLP, mit {hi_['labels']} sind es {pct(hi_['gcn'])} gegen {pct(hi_['mlp'])}. Der Graph gewinnt in "
        f"{lo_['wins']} von {lo_['n_seeds']} Gebieten bei {lo_['labels']} Etiketten und in {hi_['wins']} von {hi_['n_seeds']} bei {hi_['labels']}. Der Abstand beträgt {de(100 * (lo_['gcn'] - lo_['mlp']), 0)} Punkte bei "
        f"{lo_['labels']} und {de(100 * (hi_['gcn'] - hi_['mlp']), 0)} bei {hi_['labels']} Etiketten: mehr Etiketten helfen beiden Netzen, ersetzen aber die Nachbarn nicht."
    )

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Was, wenn die Nachbarn nicht stimmen?")
st.caption(f"Derselbe Aufbau (5 Etiketten je Typ), aber ein Anteil der Kanten wird durch zufällige ersetzt: {', '.join(pct(x) for x in C.WRONG_LEVELS)}; Mittel über {len(C.EXP_SEEDS)} feste Seeds.")
if st.button("Falsche Kanten durchrechnen (dauert etwa 10 Sekunden)", key="wrong_start"):
    st.session_state["wrong_on"] = True
if st.session_state.get("wrong_on"):
    base_w = Settings(200, 3, 5, 5, settings.noise, 0.0, 2, 0)
    rows_w = _wrong(C.WRONG_LEVELS, C.EXP_SEEDS, base_w)
    st.plotly_chart(build_wrong(rows_w), width="stretch", key="wrong_chart")
    first_loss = next((r for r in rows_w if r["gcn"] < r["mlp"]), None)
    r0, rl = rows_w[0], rows_w[-1]
    st.warning(
        f"**Befund:** Ohne falsche Kanten (Homophilie {pct(r0['homophily'])}) schlägt das GCN das MLP mit {pct(r0['gcn'])} gegen {pct(r0['mlp'])}; bei vollständig zufälligen Kanten (Homophilie {pct(rl['homophily'])}) liegt es mit "
        f"{pct(rl['gcn'])} darunter (MLP {pct(rl['mlp'])}, gewinnt in {rl['wins']} von {rl['n_seeds']} Gebieten). "
        + (f"Im Mittel kippt es zwischen Homophilie {pct(rows_w[rows_w.index(first_loss) - 1]['homophily'])} und {pct(first_loss['homophily'])}. " if first_loss and rows_w.index(first_loss) > 0 else "")
        + "Ein GCN gewichtet alle Nachbarn gleich und kann falsche nicht erkennen - genau dort setzt die Aufmerksamkeit (GAT) im nächsten Stück an."
    )

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wird zu viel Mitteln zum Problem? Tiefe und Glättung")
st.caption(f"Links: die Merkmale werden k-mal mit Â multipliziert, ohne jedes Lernen; ein Nächster-Mittelpunkt-Klassifikator (Mittelpunkte aus den bekannten Kunden) bewertet sie. Rechts: ein trainiertes GCN mit 1 bis {C.LAYERS_MAX} Schichten. "
           f"{len(C.EXP_SEEDS)} feste Seeds.")
if st.button("Glättung durchrechnen (dauert etwa 20 Sekunden)", key="smooth_start"):
    st.session_state["smooth_on"] = True
if st.session_state.get("smooth_on"):
    base_s = Settings(200, 3, 5, 5, settings.noise, 0.0, 2, 0)
    res_s = _smoothing(C.PROP_STEPS, C.EXP_SEEDS, base_s)
    st.plotly_chart(build_smoothing(res_s), width="stretch", key="smooth_chart")
    prop = res_s["propagation"]
    best = max(prop, key=lambda r: r["acc"])
    last = prop[-1]
    dep = res_s["depth"]
    st.warning(
        f"**Befund:** Reines Mitteln hebt die Genauigkeit von {pct(prop[0]['acc'])} (Rohmerkmale) auf {pct(best['acc'])} nach {best['steps']} Schritten und lässt sie danach wieder fallen: {pct(last['acc'])} nach {last['steps']} Schritten, "
        f"während die Streuung der Merkmale auf {pct(last['spread'])} des Ausgangs schrumpft - alle Kunden nähern sich einander an (Über-Glättung). Ein trainiertes GCN spürt das bis {dep[-1]['layers']} Schichten nicht: "
        f"{pct(min(r['gcn'] for r in dep))} bis {pct(max(r['gcn'] for r in dep))} in allen Tiefen (Standardfehler bis {de(100 * max(r['gcn_se'] for r in dep), 1)} Punkte). Über-Glättung ist hier also real, bremst aber erst bei weit mehr Schritten; warum das trainierte Netz sie bis 8 Schichten "
        "übersteht (die Gewichte können ausgleichen, der Graph ist räumlich glatt), wurde nicht getrennt gemessen."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nachbarn haben meist denselben Typ (Homophilie)** | Bei zufälligen oder gegensätzlichen Nachbarn schadet das Mitteln (Experiment oben); ein GCN kann Kanten nicht nach ihrem Nutzen gewichten. | GAT, dann GATv2 (nächste Stücke) |
| **Alle Nachbarn zählen ähnlich (Gewicht nur aus den Graden)** | Wichtige und unwichtige Nachbarn werden gleich behandelt; das Gewicht hängt nie vom Inhalt ab. | GAT |
| **Der ganze Graph passt in den Speicher und ist beim Training bekannt** | Die Normierung nutzt den ganzen Graphen (transduktiv); für neue Knoten oder sehr große Graphen braucht man Stichproben der Nachbarschaft. | GraphSAGE |
| **Summe über Nachbarn genügt, um Strukturen zu unterscheiden** | Mitteln kann verschiedene Nachbarschaften mit gleichem Durchschnitt nicht unterscheiden. | GIN |
| **Lokale Nachbarschaft genügt** | Information aus weit entfernten Knoten muss durch viele Schichten wandern. | Graph Transformer |
| **Vier erzeugte Merkmale, Gebietstypen als Voronoi-Zonen** | Reale Daten sind unordentlicher; die Zahlen gelten für dieses Vehikel und die genannten Größen. | – |
"""
)
st.caption("Die Linie: GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer (Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Graph.** Knoten $i = 1..n$ mit Merkmalen $x_i \in \mathbb R^4$, Etikett $y_i \in \{1..K\}$ (nur für $m$ Knoten bekannt), Nachbarschaftsmatrix $A \in \{0,1\}^{n \times n}$ (symmetrisch, $k$ nächste Nachbarn).

**Normierung.** $\tilde A = A + I$, $\tilde D_{ii} = \sum_j \tilde A_{ij}$, $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$. Die Eigenwerte von $\hat A$ liegen in $(-1, 1]$; $\hat A^k X$ nähert sich für $k \to \infty$ je Zusammenhangskomponente
einem Vielfachen von $\tilde D^{1/2} \mathbf 1$: alle Knoten werden gleich (Über-Glättung).

**Schichten.** $H^{(0)} = X$, $H^{(l)} = \mathrm{ReLU}(\hat A H^{(l-1)} W^{(l)})$ für $l < L$, Logits $Z = \hat A H^{(L-1)} W^{(L)}$.

**Verlust.** $\mathcal L = -\frac1m \sum_{i \in \text{bekannt}} \log \mathrm{softmax}(Z_i)_{y_i} + \frac{\lambda}{2}\sum_l \lVert W^{(l)} \rVert_F^2$ mit $\lambda = 5 \cdot 10^{-4}$.

**Gradienten.** Mit $\delta^{(L)} = (\mathrm{softmax}(Z) - Y)/m$ auf den bekannten Zeilen: $\partial\mathcal L/\partial W^{(l)} = (\hat A H^{(l-1)})^\top \delta^{(l)} + \lambda W^{(l)}$ und
$\delta^{(l-1)} = \hat A\,(\delta^{(l)} W^{(l)\top}) \odot \mathbb 1[Z^{(l-1)} > 0]$ (Symmetrie von $\hat A$).

**MLP.** $\hat A = I$.

Implementiert in `gcn_algorithm.py` (Normierung, Vorwärts- und Rückwärtsrechnung, Adam), `gcn_scenario.py` (Vehikel), `gcn_evaluation.py` (Analyse, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
