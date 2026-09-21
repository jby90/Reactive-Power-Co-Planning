import numpy as np
import pandas as pd

from ac_opf_nested_envelope import (
    adjacent_pair_loss_summary,
    loss_is_nonincreasing_with_tolerance,
)


def test_loss_monotonicity_accepts_solver_scale_noise():
    passed, maximum = loss_is_nonincreasing_with_tolerance(
        np.array([1.0, 0.8, 0.80002]), atol_mwh=1e-4
    )

    assert passed
    assert np.isclose(maximum, 2e-5)


def test_loss_monotonicity_rejects_material_increase():
    passed, maximum = loss_is_nonincreasing_with_tolerance(
        np.array([1.0, 0.8, 0.801]), atol_mwh=1e-4
    )

    assert not passed
    assert np.isclose(maximum, 1e-3)


def test_relative_tolerance_accepts_sub_tenth_percent_plateau_noise():
    passed, maximum = loss_is_nonincreasing_with_tolerance(
        np.array([1.5, 1.5005])
    )

    assert passed
    assert np.isclose(maximum, 5e-4)


def test_adjacent_pair_summary_uses_only_jointly_converged_days():
    episodes = pd.DataFrame(
        [
            {"scenario_index": 1, "candidate_id": 0, "daily_loss_mwh": 1.0, "pf_fail": False},
            {"scenario_index": 2, "candidate_id": 0, "daily_loss_mwh": 1.1, "pf_fail": True},
            {"scenario_index": 1, "candidate_id": 1, "daily_loss_mwh": 0.9, "pf_fail": False},
            {"scenario_index": 2, "candidate_id": 1, "daily_loss_mwh": 0.8, "pf_fail": False},
            {"scenario_index": 1, "candidate_id": 2, "daily_loss_mwh": 0.90002, "pf_fail": False},
            {"scenario_index": 2, "candidate_id": 2, "daily_loss_mwh": 0.7, "pf_fail": False},
        ]
    )

    summary = adjacent_pair_loss_summary(episodes)

    assert summary.paired_converged_days.tolist() == [1, 2]
    assert summary.nonincreasing_within_tolerance.tolist() == [True, True]
    assert np.isclose(summary.iloc[1].maximum_positive_increase_mwh, 2e-5)
