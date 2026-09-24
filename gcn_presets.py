"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. mo_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import gcn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "classes_slider": SettingSpec("classes", int, C.DEFAULT_CLASSES, C.CLASSES_MIN, C.CLASSES_MAX),
    "labels_slider": SettingSpec("labels", int, C.DEFAULT_LABELS, C.LABELS_MIN, C.LABELS_MAX),
    "neighbors_slider": SettingSpec("neighbors", int, C.DEFAULT_NEIGHBORS, C.NEIGHBORS_MIN, C.NEIGHBORS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "wrong_slider": SettingSpec("wrong", float, C.DEFAULT_WRONG, C.WRONG_MIN, C.WRONG_MAX),
    "layers_slider": SettingSpec("layers", int, C.DEFAULT_LAYERS, C.LAYERS_MIN, C.LAYERS_MAX),
    "seed_input": SettingSpec("seed", int, 7, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "classes": "classes_slider", "labels": "labels_slider", "neighbors": "neighbors_slider", "noise": "noise_slider", "wrong": "wrong_slider", "layers": "layers_slider", "seed": "seed_input"}
STEPS = {"n_slider": C.N_STEP, "noise_slider": C.NOISE_STEP, "wrong_slider": C.WRONG_STEP}

PRESETS = {
    "Standardfall": {"n": 200, "classes": 3, "labels": 5, "neighbors": 5, "noise": 1.5, "wrong": 0.0, "layers": 2, "seed": 7},
    "Nur 2 Etiketten je Gebietstyp": {"n": 200, "classes": 3, "labels": 2, "neighbors": 5, "noise": 1.5, "wrong": 0.0, "layers": 2, "seed": 7},
    "Halb falsche Nachbarn": {"n": 200, "classes": 3, "labels": 5, "neighbors": 5, "noise": 1.5, "wrong": 0.8, "layers": 2, "seed": 7},
    "Stark verrauschte Merkmale": {"n": 200, "classes": 3, "labels": 5, "neighbors": 5, "noise": 3.0, "wrong": 0.0, "layers": 2, "seed": 7},
    "Tiefes Netz (8 Schichten)": {"n": 200, "classes": 3, "labels": 5, "neighbors": 5, "noise": 1.5, "wrong": 0.0, "layers": 8, "seed": 7},
    "Vier Gebietstypen": {"n": 300, "classes": 4, "labels": 5, "neighbors": 5, "noise": 1.5, "wrong": 0.0, "layers": 2, "seed": 7},
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "200 Kunden, 3 Gebietstypen, 5 bekannte Etiketten je Typ, 5 Nachbarn, Rauschen 1,5: das GCN erreicht 88,1 % auf den 185 unbekannten Kunden, dasselbe Netz ohne Nachbarn (MLP) nur 68,6 %; 92,4 % der Kanten verbinden Kunden desselben Typs.",
    "Nur 2 Etiketten je Gebietstyp": "Nur 6 bekannte Kunden insgesamt: das GCN erreicht noch 81,4 %, das MLP nur 50,0 % (Raten: 40,7 %). Der Graph trägt die wenigen Etiketten weiter.",
    "Halb falsche Nachbarn": "80 % der Kanten sind durch zufällige ersetzt (Homophilie 46,4 %): das GCN fällt auf 49,7 % und liegt damit unter dem MLP (68,6 %) - es kann falsche Nachbarn nicht erkennen und kann nicht einmal alle 15 bekannten Etiketten lernen (Trainingsgenauigkeit 80 %).",
    "Stark verrauschte Merkmale": "Rauschen 3,0: einzelne Kunden verraten kaum noch ihren Typ. Das MLP kommt nur auf 42,7 % (Raten: 41,1 %), das GCN mit Nachbarn auf 77,3 %.",
    "Tiefes Netz (8 Schichten)": "Acht Schichten: das GCN erreicht 91,9 % (2 Schichten: 88,1 %) - kein Einbruch durch Über-Glättung; im Experiment unten zeigt reine Glättung ohne Lernen erst nach dutzenden Schritten Wirkung.",
    "Vier Gebietstypen": "300 Kunden, 4 Typen, 5 Etiketten je Typ: das GCN erreicht 77,5 %, das MLP nur 32,1 % - weniger als das Raten des häufigsten Typs (33,9 %).",
}
