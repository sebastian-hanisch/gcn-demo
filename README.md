# 🕸️ GCN – Nachbarn statt Etiketten

Erstes Stück der **Graph-Neural-Network-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und die **erste Demo des Portfolios
mit neuronalen Netzen auf Graphen**. Es ist die **Wurzel** der Linie (GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer; die übrigen Stücke sind noch nicht gebaut).

Vehikel **D "Liefergebiete"**: 60 bis 400 Kunden in einem Gebiet, das in räumlich zusammenhängende **Gebietstypen** zerfällt (Innenstadt, Vorstadt, Ländlich, Gewerbegebiet); vier verrauschte Merkmale je Kunde (Stopps je Stunde,
Parksuche, Ladegewicht, Zeitfenster-Enge); der Graph verbindet die $k$ räumlich nächsten Kunden. Nur wenige Gebietstypen sind bekannt (semi-überwachtes Lernen), alle übrigen Kunden werden vorhergesagt und zur Prüfung
benutzt. Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben.

**Bezug zu OR:** dieselben Nachbarschaftsgraphen tragen Tourenplanung und Gebietszuschnitt; hier lernt ein Netz auf ihnen, statt dass ein Optimierer über sie läuft.

## Warum dieses Problem

Ein **Graph Convolutional Network** (Kipf/Welling 2017) mischt in jeder Schicht die Merkmale eines Knotens mit denen seiner Nachbarn und lernt darauf ein kleines Netz: $H' = \mathrm{ReLU}(\hat A\,H\,W)$ mit
$\hat A = D^{-1/2}(A+I)D^{-1/2}$. Setzt man $\hat A = I$, bleibt dasselbe Netz **ohne Nachbarn** übrig (ein MLP) – jede Differenz ist also allein dem Graphen zu verdanken. Die Demo misst, **wann der Graph hilft, wann er
schadet und was Tiefe mit den Merkmalen macht** – und benennt damit die Schwächen, die die anderen Stücke der Linie aufgreifen.

## Modell

- **Vehikel** (`gcn_scenario.py`): Gebietstypen = nächstes von 2 bis 4 verdeckten Zentren (Mindestabstand), Merkmalsmittel je Typ = zufällige Einheitsvektoren, Merkmale = Mittel + Rauschen; Graph = symmetrische $k$-nächste-Nachbarn-Kanten;
  **falsche Kanten**: ein Anteil der Kanten wird durch zufällige ersetzt (Kantenzahl bleibt, Homophilie sinkt); je Gebietstyp $m$ bekannte Etiketten.
- **GCN** (`gcn_algorithm.py`): Normierung, Schichten, Softmax-Kreuzentropie auf den bekannten Knoten, Gewichtszerfall $5\cdot10^{-4}$, Adam (0,01), 200 Epochen, 16 verdeckte Einheiten, Glorot-Start; **kein Abstimmen am Test** (feste Epochenzahl,
  Test = alle unbekannten Kunden, nur zur Anzeige). Gradienten von Hand hergeleitet (siehe 📐-Abschnitt der App).

## Methodik

- **Handrechnungen:** Normierung auf einem Pfad mit drei Knoten ($\hat A_{00} = 1/2$, $\hat A_{01} = 1/\sqrt 6$, $\hat A_{11} = 1/3$), Vorwärtsrechnung einer Schicht mit den Werten (1, 2, 4).
- **Gegenproben:** Vorwärtsrechnung gegen eine explizite Nachbarschleife (zwei Schichten), **Gradienten gegen zentrale Differenzen** (1, 2, 3 Schichten, mit Gewichtszerfall), MLP-Gleichwertigkeit ($\hat A = I$ gegen ein unabhängig
  geschriebenes MLP), erster Adam-Schritt (jedes Gewicht bewegt sich um die Lernrate), Verlust $\log K$ bei Nullgewichten, Wiederholbarkeit, Spektrum von $\hat A$ (größter Eigenwert 1 mit Eigenvektor $\sqrt{d+1}$),
  Konvergenz von $\hat A^k X$ gegen die gradgewichtete Konstante.
- **Vorab-Messung:** vor dem Bau des Textes an 8 bis 12 Seeds geprüft, ob Vorsprung, Kipppunkt und Über-Glättung überhaupt zu sehen sind (Über-Glättung im trainierten Netz ist es bis 8 Schichten **nicht** – siehe Befunde).
- **Literatur** (nicht nachgebaut): Kipf/Welling 2017 ("Semi-supervised classification with graph convolutional networks", ICLR).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Hilft der Graph? (Standardfall: 200 Kunden, 3 Typen, 5 Etiketten je Typ, 5 Nachbarn, Rauschen 1,5, Seed 7) | GCN 88,1 % gegen MLP 68,6 % auf den 185 unbekannten Kunden; 92,4 % der 608 Kanten verbinden Kunden desselben Typs. | `test_preset_standard` |
| Wie stark streuen einzelne Gebiete? | Über 8 Seeds: GCN 63 bis 94 %, MLP 40 bis 69 %; der Graph gewinnt in jedem. Der Standardfall ist ein typisches, kein bestes Gebiet. | `test_spread_over_areas_is_large` |
| Wenige Etiketten (Mittel über 12 Seeds) | 2 Etiketten je Typ: GCN 82,7 % gegen MLP 52,0 %; 20 Etiketten: 91,4 % gegen 60,7 %. Der Graph gewinnt in allen 12 Gebieten bei jeder Etikettenzahl; der Abstand bleibt bei etwa 30 Punkten – mehr Etiketten helfen beiden, ersetzen die Nachbarn aber nicht. Nur 6 bekannte Kunden (Preset): 81,4 % gegen 50,0 %. | `test_labels_experiment`, `test_preset_two_labels` |
| Stark verrauschte Merkmale | Rauschen 3,0: MLP 42,7 % (Raten 41,1 %), GCN 77,3 %. Vier Typen: MLP 32,1 % – **schlechter als Raten** (33,9 %) –, GCN 77,5 %. | `test_preset_noisy_features`, `test_preset_four_types` |
| Was, wenn die Nachbarn nicht stimmen? (12 Seeds) | Mit dem Anteil zufällig ersetzter Kanten sinkt die Homophilie von 91 % auf 38 % und das GCN von 87,9 % auf 46,2 %; das MLP (sieht die Kanten nie) bleibt bei 57,8 %. **Kipppunkt** zwischen Homophilie 59 % (GCN 65,5 %, gewinnt in 9 von 12 Gebieten) und 49 % (GCN 52,9 %, gewinnt in 4 von 12). Preset "Halb falsche Nachbarn" (Homophilie 46,4 %): GCN 49,7 % gegen MLP 68,6 %, und das GCN lernt nicht einmal alle 15 bekannten Etiketten (Trainingsgenauigkeit 80 %). | `test_wrong_edges_experiment`, `test_preset_wrong_neighbors` |
| Wird zu viel Mitteln zum Problem? (reine Glättung, kein Lernen, 12 Seeds) | Nächster-Mittelpunkt-Genauigkeit auf $\hat A^k X$: 62,4 % (k = 0) → 85,6 % (1) → **92,0 % (8)** → 78,0 % (128); Streuung der Merkmale schrumpft von 100 % auf 52,7 % (1) und 13,5 % (128): alle Kunden werden einander ähnlich (Über-Glättung), aber langsam. | `test_smoothing_experiment` |
| Und im trainierten Netz? | **Kein Einbruch bis 8 Schichten**: 85 bis 89 % in allen Tiefen (1 Schicht 84,5 %, 2 Schichten 87,9 %), Standardfehler bis 2,7 Punkte. Warum das trainierte Netz die Glättung übersteht (Gewichte gleichen aus, der Graph ist räumlich glatt), wurde nicht getrennt gemessen. Preset "Tiefes Netz": 91,9 % (8 Schichten) gegen 88,1 % (2). | `test_smoothing_experiment`, `test_preset_deep_network` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nachbarn haben meist denselben Typ (Homophilie)** | Bei zufälligen oder gegensätzlichen Nachbarn schadet das Mitteln (gemessen); ein GCN kann Kanten nicht nach ihrem Nutzen gewichten. | GAT, dann GATv2 |
| **Alle Nachbarn zählen ähnlich** | Das Gewicht kommt nur aus den Graden, nie aus dem Inhalt des Nachbarn. | GAT |
| **Der ganze Graph passt in den Speicher und ist beim Training bekannt** | Die Normierung nutzt den ganzen Graphen (transduktiv); für neue Knoten oder große Graphen braucht man Stichproben der Nachbarschaft. Hier nicht gemessen. | GraphSAGE |
| **Mitteln genügt, um Strukturen zu unterscheiden** | Mitteln kann Nachbarschaften mit gleichem Durchschnitt nicht unterscheiden. Hier nicht gemessen. | GIN |
| **Die lokale Nachbarschaft genügt** | Information aus weit entfernten Knoten muss durch viele Schichten wandern. Hier nicht gemessen. | Graph Transformer |
| **Erzeugte Daten** | Vier Merkmale, Voronoi-Zonen, Rauschen: die Zahlen gelten für dieses Vehikel und die genannten Größen, nicht für reale Netze. | – |

Über-Glättung, Homophilie-Kipppunkt und der Vorsprung sind Befunde über dieses Vehikel; die drei zuletzt genannten Schwächen werden erst von den Nachfolgern gemessen.

## Tests

Pytest-Suite (`pytest tests/ -v`): Kern per Handrechnung und Gegenprobe (Normierung, Vorwärtsrechnung gegen Nachbarschleife, Gradienten gegen zentrale Differenzen, MLP-Gleichwertigkeit, Adam-Schritt, Glättung), Vehikel (Symmetrie,
räumliche Nachbarn, Homophilie, falsche Kanten, Aufteilung), Auswertung und Experimente (Form, Trends), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Epochen-Regler, Kundenwahl, Extremwerte, Experimente auf Abruf)
und `test_claims.py` (jede Zahl aus diesem README; Einzelgebiete mit Bändern von etwa 1,5 Punkten, Mehr-Seed-Zahlen mit größeren).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `gcn_constants.py` | Regler-Grenzen, Netz-Konstanten, Experiment-Seeds |
| `gcn_presets.py` | Permalink/Presets-Mechanik |
| `gcn_scenario.py` | Vehikel D (Gebiet, Merkmale, Graph, falsche Kanten, Aufteilung) |
| `gcn_algorithm.py` | GCN von Grund auf: Normierung, Vorwärts-/Rückwärtsrechnung, Adam, Glättung |
| `gcn_evaluation.py` | Analyse, drei Experimente |
| `gcn_visualization.py` | Plotly-Abbildungen (Karte, Kurven, Experimente) |

## Bewusst nicht umgesetzt

- Die übrigen fünf Stücke der Linie (GraphSAGE, GAT, GATv2, GIN, Graph Transformer).
- Dropout, Early Stopping und Validierungsknoten (feste Epochenzahl, nichts am Test abgestimmt).
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
