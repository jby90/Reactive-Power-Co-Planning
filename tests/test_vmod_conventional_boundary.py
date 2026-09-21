import pandas as pd

from run_vmod_conventional_capacity_boundaries import (
    all_higher_points_pass,
    confirmation_point_passed,
    merge_droop_tuning_tables,
)


def test_confirmation_point_requires_zero_events_failures_and_sub_one_percent_bound():
    assert confirmation_point_passed(0, 349, 0)
    assert not confirmation_point_passed(1, 349, 0)
    assert not confirmation_point_passed(0, 349, 1)
    assert not confirmation_point_passed(0, 200, 0)


def test_conventional_boundary_rejects_higher_capacity_reversal() -> None:
    frame = pd.DataFrame(
        {
            "method": ["droop"] * 4,
            "path_index": [0, 1, 2, 3],
            "passed_selection_gate": [False, True, False, True],
        }
    )
    assert not all_higher_points_pass(frame, "droop", 1)
    assert all_higher_points_pass(frame, "droop", 3)
    assert not all_higher_points_pass(frame, "no_control", None)


def test_droop_refinement_merges_unique_configs_and_uses_lexicographic_best():
    common = {
        "full_injection": 0.95,
        "deadband_low": 0.98,
        "deadband_high": 1.02,
        "full_absorption": 1.05,
        "evaluated_days": 40,
        "pf_fail_days": 0,
    }
    coarse = pd.DataFrame(
        [
            {
                **common,
                "action_scale": 0.10,
                "event_days": 2,
                "mean_daily_line_loss_mwh": 1.5,
            },
            {
                **common,
                "action_scale": 0.25,
                "event_days": 4,
                "mean_daily_line_loss_mwh": 1.4,
            },
        ]
    )
    refinement = pd.DataFrame(
        [
            {
                **common,
                "action_scale": 0.10,
                "event_days": 2,
                "mean_daily_line_loss_mwh": 1.5,
            },
            {
                **common,
                "action_scale": 0.14,
                "event_days": 2,
                "mean_daily_line_loss_mwh": 1.3,
            },
        ]
    )

    merged = merge_droop_tuning_tables(coarse, refinement)

    assert len(merged) == 3
    assert merged.iloc[0].action_scale == 0.14
