"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import gcn_constants as C
import gcn_evaluation as E
import gcn_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= p[key] <= spec.hi
            spec.caster(p[key])
        assert (p["n"] - C.N_MIN) % C.N_STEP == 0
        for key, state_key in (("noise", "noise_slider"), ("wrong", "wrong_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9


def test_default_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall"]
    assert E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["wrong"], p["layers"], p["seed"]) == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("layers_slider") == (C.LAYERS_MIN, C.LAYERS_MAX) and set(P.STEPS) == {"n_slider", "noise_slider", "wrong_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
