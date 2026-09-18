from pathlib import Path

import numpy as np
import pytest

import evaluate_crdc_policy
from evaluate_crdc_policy import load_day_indices, load_profiles_from_cfg


def test_checkpoint_profiles_override_legacy_arrays(tmp_path: Path) -> None:
    load = np.arange(192 * 2, dtype=float).reshape(192, 2)
    generation = np.arange(192, dtype=float).reshape(192, 1)
    load_path = tmp_path / "load.npy"
    generation_path = tmp_path / "generation.npy"
    np.save(load_path, load)
    np.save(generation_path, generation)

    actual_load, actual_generation = load_profiles_from_cfg(
        {
            "load_profile_path": str(load_path),
            "generation_profile_path": str(generation_path),
        }
    )

    np.testing.assert_array_equal(actual_load, load)
    np.testing.assert_array_equal(actual_generation, generation)


def test_checkpoint_profiles_require_both_paths(tmp_path: Path) -> None:
    path = tmp_path / "load.npy"
    np.save(path, np.ones((96, 2)))
    with pytest.raises(ValueError, match="both"):
        load_profiles_from_cfg({"load_profile_path": str(path)})


def test_checkpoint_profiles_require_matching_time_lengths(tmp_path: Path) -> None:
    load_path = tmp_path / "load.npy"
    generation_path = tmp_path / "generation.npy"
    np.save(load_path, np.ones((192, 2)))
    np.save(generation_path, np.ones((96, 1)))
    with pytest.raises(ValueError, match="equal time lengths"):
        load_profiles_from_cfg(
            {
                "load_profile_path": str(load_path),
                "generation_profile_path": str(generation_path),
            }
        )


def test_missing_provenance_paths_fall_back_to_release_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    input_dir = tmp_path / "data" / "inputs"
    src_dir.mkdir()
    input_dir.mkdir(parents=True)
    load = np.ones((192, 2), dtype=float)
    generation = np.full((192, 1), 0.5, dtype=float)
    np.save(input_dir / "load_release.npy", load)
    np.save(input_dir / "pv_release.npy", generation)
    monkeypatch.setattr(
        evaluate_crdc_policy,
        "__file__",
        str(src_dir / "evaluate_crdc_policy.py"),
    )

    actual_load, actual_generation = load_profiles_from_cfg(
        {
            "load_profile_path": str(tmp_path / "missing" / "load_release.npy"),
            "generation_profile_path": str(tmp_path / "missing" / "pv_release.npy"),
        }
    )

    np.testing.assert_array_equal(actual_load, load)
    np.testing.assert_array_equal(actual_generation, generation)


def test_day_indices_load_from_released_csv(tmp_path: Path) -> None:
    path = tmp_path / "confirmation_days.csv"
    path.write_text("scenario_index,day_index,t0\n0,2,192\n1,7,672\n", encoding="utf-8")
    assert load_day_indices(path) == [2, 7]


def test_day_indices_load_from_compatible_json(tmp_path: Path) -> None:
    path = tmp_path / "days.json"
    path.write_text('{"final_day_indices": [3, 9]}', encoding="utf-8")
    assert load_day_indices(path) == [3, 9]
