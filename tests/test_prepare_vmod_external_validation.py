from pathlib import Path

import pandas as pd

from prepare_vmod_external_validation import prepare_protocol, split_external_days


def test_external_split_is_disjoint_and_uses_all_days():
    splits = split_external_days(366, 20260921)
    assert len(splits["selection"]) == 17
    assert len(splits["confirmation"]) == 349
    assert not set(splits["selection"]) & set(splits["confirmation"])
    assert len(set(splits["selection"]) | set(splits["confirmation"])) == 366


def test_external_protocol_copies_only_validation_inputs(tmp_path: Path):
    profiles = tmp_path / "profiles"
    source = tmp_path / "source"
    destination = tmp_path / "protocol"
    profiles.mkdir()
    source.mkdir()
    pd.DataFrame(
        {"scenario_index": [0], "day_index": [2], "t0": [192]}
    ).to_csv(profiles / "selection_days.csv", index=False)
    pd.DataFrame(
        {"scenario_index": [0, 1], "day_index": [0, 1], "t0": [0, 96]}
    ).to_csv(profiles / "confirmation_days.csv", index=False)
    pd.DataFrame(
        {
            "capacity_label": ["P00"],
            "path_index": [0],
            "pv_s_scale": [0.3],
            "svc_q_scale": [0.0],
            "cap_total_mvar": [0.0],
        }
    ).to_csv(source / "capacity_path.csv", index=False)

    manifest = prepare_protocol(profiles, source, destination)

    assert manifest["selection_days"] == 1
    assert manifest["confirmation_days"] == 2
    assert not (destination / "fitting_days.csv").exists()
