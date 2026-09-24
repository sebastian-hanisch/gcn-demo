"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Epochen-Regler, Kundenwahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gcn_constants as C
import gcn_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_that_the_graph_helps():
    at = _run()
    _ok(at)
    assert at.metric and any("Der Graph hilft" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]
    assert at.get("plotly_chart")


def test_wrong_neighbors_preset_shows_the_graph_hurting():
    at = _run()
    next(b for b in at.button if b.key == "preset_Halb falsche Nachbarn").click().run()
    _ok(at)
    assert any("Der Graph schadet" in w.value for w in at.warning)


def test_epoch_slider_and_map_mode_run():
    at = _run(epoch_slider=1)
    _ok(at)
    at.radio(key="map_mode").set_value("Wahrer Gebietstyp").run()
    _ok(at)
    at.slider(key="epoch_slider").set_value(100).run()
    _ok(at)


def test_node_selection_survives_shrinking_n_and_shows_neighbors():
    at = _run(n_slider=400, node_select=399)
    _ok(at)
    at.slider(key="n_slider").set_value(100).run()
    _ok(at)
    assert at.session_state["node_select"] < 100
    assert any("Nachbarn; jedem gibt" in c.value for c in at.caption)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Gebiet generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["noise"] = "1.6"
    at.query_params["wrong"] = "0.33"
    at.query_params["layers"] = "0"
    at.query_params["labels"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["noise_slider"] == 1.5 and at.session_state["wrong_slider"] == 0.35
    assert at.session_state["layers_slider"] == C.LAYERS_MIN and at.session_state["labels_slider"] == C.DEFAULT_LABELS


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN, classes_slider=2, labels_slider=20), dict(n_slider=C.N_MAX, classes_slider=4, neighbors_slider=10), dict(layers_slider=C.LAYERS_MAX),
                                dict(wrong_slider=1.0), dict(noise_slider=C.NOISE_MAX), dict(neighbors_slider=C.NEIGHBORS_MIN, labels_slider=C.LABELS_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_labels_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "LABEL_LEVELS", (2, 10))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1, 2))
    at = _run()
    next(b for b in at.button if b.key == "labels_start").click().run()
    _ok(at)
    assert at.session_state["labels_on"] and any("Mit 2 Etiketten je Typ erreicht das GCN" in w.value for w in at.warning)


def test_wrong_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "WRONG_LEVELS", (0.0, 1.0))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1, 2))
    at = _run()
    next(b for b in at.button if b.key == "wrong_start").click().run()
    _ok(at)
    assert at.session_state["wrong_on"] and any("Ein GCN gewichtet alle Nachbarn gleich" in w.value for w in at.warning)


def test_smoothing_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "PROP_STEPS", (0, 1, 8, 32))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    monkeypatch.setattr(C, "LAYERS_MAX", 3)
    at = _run()
    next(b for b in at.button if b.key == "smooth_start").click().run()
    _ok(at)
    assert at.session_state["smooth_on"] and any("Reines Mitteln hebt die Genauigkeit" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value
