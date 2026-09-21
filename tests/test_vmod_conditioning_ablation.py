import numpy as np
import pytest

from run_vmod_conditioning_ablation import (
    contrast_summary,
    matched_fixed_dataset_indices,
    paired_seed_contrasts,
    resample_to_count,
)


def test_fixed_capacity_resampling_matches_each_split_budget_without_leakage():
    days = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    candidate_mask = np.array([True, True, True, True, True, True, False, False])
    validation_days = np.array([3])

    selected = matched_fixed_dataset_indices(
        days,
        candidate_mask,
        validation_days,
        target_training_examples=10,
        target_validation_examples=4,
        seed=17,
    )

    selected_days = days[selected]
    assert len(selected) == 14
    assert np.count_nonzero(selected_days == 3) == 4
    assert np.count_nonzero(selected_days != 3) == 10
    assert set(selected_days) <= {1, 2, 3}
    np.testing.assert_array_equal(
        selected,
        matched_fixed_dataset_indices(
            days,
            candidate_mask,
            validation_days,
            10,
            4,
            seed=17,
        ),
    )


def test_resampling_rejects_empty_or_nonpositive_budgets():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="empty"):
        resample_to_count(np.array([], dtype=int), 3, rng)
    with pytest.raises(ValueError, match="positive"):
        resample_to_count(np.array([1, 2]), 0, rng)


def test_conditioning_contrasts_require_exact_day_pairing():
    import pandas as pd

    shared = pd.DataFrame(
        [
            {
                "seed": seed,
                "day": day,
                "event": 0,
                "line_loss_mwh": 1.0 + 0.01 * (seed - 42),
            }
            for seed in range(42, 47)
            for day in (1, 2)
        ]
    )
    fixed = pd.DataFrame(
        [
            {
                "seed": seed,
                "day": day,
                "event": 1 if day == 2 else 0,
                "line_loss_mwh": 1.1,
            }
            for seed in range(42, 47)
            for day in (1, 2)
        ]
    )

    rows = paired_seed_contrasts(shared, fixed, "confirmation")
    summary = contrast_summary(rows)

    assert len(rows) == 5
    assert all(row["split"] == "confirmation" for row in rows)
    assert all(row["paired_days"] == 2 for row in rows)
    assert all(row["shared_event_days"] == 0 for row in rows)
    assert all(row["fixed_event_days"] == 1 for row in rows)
    assert summary["shared_minus_fixed_loss_mean_mwh"] < 0

    with pytest.raises(RuntimeError, match="exactly matched confirmation"):
        paired_seed_contrasts(shared, fixed.loc[fixed.day == 1], "confirmation")
