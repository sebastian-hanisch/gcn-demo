"""Konstanten der GCN-Demo: Vehikel D "Liefergebiete" (Kunden im Gebiet, Graph der räumlichen Nachbarn, Merkmale je Kunde, Etikett = Gebietstyp), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

AREA = 100.0
CLASS_NAMES = ("Innenstadt", "Vorstadt", "Ländlich", "Gewerbegebiet")
FEATURE_NAMES = ("Stopps je Stunde", "Parksuche (min)", "Ladegewicht (kg)", "Zeitfenster-Enge")
N_FEATURES = len(FEATURE_NAMES)

N_MIN, N_MAX, N_STEP, DEFAULT_N = 60, 400, 20, 200
CLASSES_MIN, CLASSES_MAX, DEFAULT_CLASSES = 2, 4, 3
LABELS_MIN, LABELS_MAX, DEFAULT_LABELS = 2, 20, 5
NEIGHBORS_MIN, NEIGHBORS_MAX, DEFAULT_NEIGHBORS = 2, 10, 5
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.5, 4.0, 0.25, 1.5
WRONG_MIN, WRONG_MAX, WRONG_STEP, DEFAULT_WRONG = 0.0, 1.0, 0.05, 0.0
LAYERS_MIN, LAYERS_MAX, DEFAULT_LAYERS = 1, 8, 2
HIDDEN = 16
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(12))
LABEL_LEVELS = (2, 3, 5, 10, 20)
WRONG_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PROP_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128)
