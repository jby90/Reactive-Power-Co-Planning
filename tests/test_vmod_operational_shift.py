import numpy as np
import pandas as pd

from evaluate_vmod_operational_shift import (
    finite_or_none,
    observed_state,
    replicate_event_rate_summary,
    shift_random_seed,
)


def test_shift_realisation_is_paired_across_actor_seeds():
    base = 2026092001

    seed_42_realisation = shift_random_seed(base, day=118, shift_id=7)
    seed_46_realisation = shift_random_seed(base, day=118, shift_id=7)

    assert seed_42_realisation == seed_46_realisation
    assert seed_42_realisation != shift_random_seed(base, day=119, shift_id=7)
    assert seed_42_realisation != shift_random_seed(base, day=118, shift_id=8)


def test_observation_noise_is_reproducible_for_a_paired_shift():
    state = np.linspace(0.8, 1.2, 13)
    first = observed_state(
        state,
        buses=4,
        rng=np.random.default_rng(1234),
        voltage_std=0.002,
        power_relative_std=0.02,
    )
    second = observed_state(
        state,
        buses=4,
        rng=np.random.default_rng(1234),
        voltage_std=0.002,
        power_relative_std=0.02,
    )

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, state)


def test_nonfinite_shift_outputs_are_json_safe():
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("-inf")) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(0.951) == 0.951


def test_shift_replicate_interval_uses_matched_realisations():
    episodes = pd.DataFrame(
        {
            "shift_id": [0, 0, 1, 1, 2, 2],
            "event": [0, 0, 1, 0, 1, 1],
        }
    )

    result = replicate_event_rate_summary(episodes)

    assert result["shift_replicates"] == 3
    assert result["replicate_event_rates"] == [0.0, 0.5, 1.0]
    assert result["replicate_mean_event_rate"] == 0.5
    assert result["replicate_interval_available"] is True
    assert 0.0 <= result["replicate_level_event_rate_ci95_low"] <= 1.0
    assert 0.0 <= result["replicate_level_event_rate_ci95_high"] <= 1.0
    assert "percentile bootstrap" in result[
        "replicate_level_event_rate_interval_method"
    ]
