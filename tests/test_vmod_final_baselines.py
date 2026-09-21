import pandas as pd
import pytest

from run_vmod_final_baselines import (
    build_path_selection_jobs,
    paired_matched_seed_interval,
    paired_seed_interval,
)


def test_build_path_selection_jobs_creates_matched_cartesian_product():
    days = pd.DataFrame({"day": [7, 11]})
    path = pd.DataFrame(
        {
            "path_index": [0, 1, 2],
            "capacity_label": ["P00", "P01", "P02"],
        }
    )

    jobs = build_path_selection_jobs(days, path)

    assert len(jobs) == 6
    assert jobs.candidate_id.tolist() == [0, 1, 2, 0, 1, 2]
    assert jobs.groupby("day").path_index.apply(list).to_dict() == {
        7: [0, 1, 2],
        11: [0, 1, 2],
    }
    assert jobs.groupby("path_index").day.apply(list).to_dict() == {
        0: [7, 11],
        1: [7, 11],
        2: [7, 11],
    }


def test_build_path_selection_jobs_rejects_empty_or_unindexed_inputs():
    with pytest.raises(ValueError):
        build_path_selection_jobs(pd.DataFrame(), pd.DataFrame({"path_index": [0]}))
    with pytest.raises(ValueError):
        build_path_selection_jobs(pd.DataFrame({"day": [1]}), pd.DataFrame())
    with pytest.raises(ValueError):
        build_path_selection_jobs(
            pd.DataFrame({"day": [1]}), pd.DataFrame({"capacity_label": ["P00"]})
        )


def test_unmatched_conventional_days_are_rejected():
    proposed = pd.DataFrame(
        {
            "seed": [42, 42],
            "day": [1, 2],
            "line_loss_mwh": [1.0, 1.1],
        }
    )
    baseline = pd.DataFrame(
        {"day": [1], "baseline_loss_mwh": [1.2]}
    )

    with pytest.raises(ValueError, match="exactly matched"):
        paired_seed_interval(proposed, baseline, "baseline_loss_mwh")


def test_unmatched_seedwise_baseline_days_are_rejected():
    proposed = pd.DataFrame(
        {
            "seed": [42, 42],
            "day": [1, 2],
            "event": [0, 0],
            "line_loss_mwh": [1.0, 1.1],
        }
    )
    baseline = pd.DataFrame(
        {
            "seed": [42],
            "day": [1],
            "event": [0],
            "line_loss_mwh": [1.2],
        }
    )

    with pytest.raises(ValueError, match="exactly matched"):
        paired_matched_seed_interval(proposed, baseline)
