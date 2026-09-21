import pandas as pd

from run_vmod_env69_after_bidirectional import (
    development_range_has_passing_point,
    unconfirmed_boundary_payload,
)


def test_failed_portability_selection_is_retained_as_negative_evidence():
    selection = {
        "boundary_identified": False,
        "selected_path_index": None,
        "all_higher_points_pass": False,
    }

    payload = unconfirmed_boundary_payload(selection)

    assert payload["boundary_confirmed"] is False
    assert payload["confirmation_opened"] is False
    assert payload["points"] == []
    assert payload["selection_summary"] == selection
    assert "did not identify" in payload["reason"]


def test_external_selection_requires_a_development_passing_point():
    failed = pd.DataFrame(
        {"all_seeds_zero_events": [False, False], "evaluated_seed_days": [50, 50]}
    )
    passing = pd.DataFrame(
        {"all_seeds_zero_events": [False, True], "evaluated_seed_days": [50, 50]}
    )

    assert not development_range_has_passing_point(failed)
    assert development_range_has_passing_point(passing)
