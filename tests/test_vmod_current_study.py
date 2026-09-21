from __future__ import annotations

import json

import pandas as pd

import vmod_current_study as study
from vmod_evaluation import action_reference_scales


def test_current_study_uses_nonoverlapping_external_protocol() -> None:
    assert study.read_selected_margin() == 0.003
    assert study.protocol_day_count(study.EXTERNAL_PROTOCOL_33, "selection") == 17
    assert study.protocol_day_count(study.EXTERNAL_PROTOCOL_33, "confirmation") == 349
    assert "EXTERNAL2018" in study.main_run_dir().name
    assert study.profile_windows_are_disjoint(
        study.DEVELOPMENT_PROFILES_33, study.EXTERNAL_PROFILES_33
    )


def test_current_study_has_matching_69_bus_external_inputs() -> None:
    assert study.EXTERNAL_PROTOCOL_69.is_dir()
    assert study.EXTERNAL_PROFILES_69.is_dir()
    assert study.protocol_day_count(study.EXTERNAL_PROTOCOL_69, "selection") == 17
    assert study.protocol_day_count(study.EXTERNAL_PROTOCOL_69, "confirmation") == 349
    assert study.profile_windows_are_disjoint(
        study.DEVELOPMENT_PROFILES_69, study.EXTERNAL_PROFILES_69
    )
    path = pd.read_csv(study.DEVELOPMENT_PROTOCOL_69 / "capacity_path.csv")
    assert len(path) == 15
    assert tuple(path.iloc[-1][["pv_s_scale", "svc_q_scale"]]) == (1.5, 1.5)
    audit = json.loads(
        (study.DEVELOPMENT_PROTOCOL_69 / "extension_decision_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["external_selection_output_observed"] is False
    assert audit["trigger"]["persisted_external_selection_output_count"] == 0
    assert action_reference_scales(69) == (1.5, 1.5)
    action_audit = json.loads(
        (
            study.DEVELOPMENT_PROTOCOL_69
            / "action_reference_correction_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert action_audit["external_selection_output_observed"] is False
    assert action_audit["correction_rule"]["external_data_used_to_choose_rule"] is False
    assert action_audit["correction_rule"]["reference_pv_s_scale"] == 1.5
