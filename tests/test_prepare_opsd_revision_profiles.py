import numpy as np
import pandas as pd
import pytest

from prepare_opsd_revision_profiles import (
    LOAD_COLUMN,
    SOLAR_COLUMN,
    TIMESTAMP_COLUMN,
    build_profiles,
    day_hashes,
    extract_window,
    split_days,
)


def test_profiles_are_normalized_and_broadcast() -> None:
    frame = pd.DataFrame(
        {
            "DE_load_actual_entsoe_transparency": [2.0, 4.0, 1.0],
            "DE_solar_profile": [0.0, 0.5, 1.0],
        }
    )
    load, solar, stats = build_profiles(frame, load_columns=2, pv_columns=3)
    np.testing.assert_allclose(load[:, 0], [0.5, 1.0, 0.25])
    np.testing.assert_allclose(solar[:, 0], [0.0, 0.5, 1.0])
    assert load.shape == (3, 2)
    assert solar.shape == (3, 3)
    assert stats["load_reference_mw"] == 4.0


def test_profiles_can_reuse_frozen_normalisation_references() -> None:
    frame = pd.DataFrame(
        {
            "DE_load_actual_entsoe_transparency": [2.0, 4.0, 1.0],
            "DE_solar_profile": [0.0, 0.5, 1.0],
        }
    )
    load, solar, stats = build_profiles(
        frame,
        load_columns=1,
        pv_columns=1,
        load_reference=8.0,
        solar_reference=2.0,
    )

    np.testing.assert_allclose(load[:, 0], [0.25, 0.5, 0.125])
    np.testing.assert_allclose(solar[:, 0], [0.0, 0.25, 0.5])
    assert stats["load_values_above_reference"] == 0
    assert stats["solar_values_above_reference"] == 0


def test_frozen_split_is_disjoint_and_complete() -> None:
    splits = split_days(366, 20260918)
    assert {name: len(values) for name, values in splits.items()} == {
        "training": 50,
        "selection": 17,
        "confirmation": 299,
    }
    joined = np.concatenate(list(splits.values()))
    assert sorted(joined.tolist()) == list(range(366))


def test_joint_day_hash_detects_distinct_days() -> None:
    load = np.zeros((192, 2))
    solar = np.zeros((192, 1))
    solar[96:, 0] = 1.0
    hashes = day_hashes(load, solar)
    assert len(hashes) == 2
    assert len(set(hashes)) == 2


def test_extract_window_can_interpolate_and_audit_missing_values(tmp_path) -> None:
    timestamps = pd.date_range("2019-01-01", periods=96, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            LOAD_COLUMN: np.linspace(1.0, 2.0, 96),
            SOLAR_COLUMN: np.linspace(0.0, 1.0, 96),
        }
    )
    frame.loc[20:22, SOLAR_COLUMN] = np.nan
    source = tmp_path / "profiles.csv"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="missing load or solar"):
        extract_window(source, start="2019-01-01T00:00:00Z", days=1)

    repaired = extract_window(
        source,
        start="2019-01-01T00:00:00Z",
        days=1,
        missing_policy="interpolate_time",
    )
    assert not repaired[[LOAD_COLUMN, SOLAR_COLUMN]].isna().any().any()
    assert repaired.attrs["missing_values_before_repair"][SOLAR_COLUMN] == 3
