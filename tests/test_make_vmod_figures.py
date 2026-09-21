import pandas as pd
import pytest

from make_vmod_figures import all_capacity_feasible_days


def test_all_capacity_feasible_days_requires_complete_safe_support():
    episodes = pd.DataFrame(
        [
            {"day": 1, "candidate_id": 0, "pf_fail": 0, "risk": 0},
            {"day": 1, "candidate_id": 1, "pf_fail": 0, "risk": 0},
            {"day": 2, "candidate_id": 0, "pf_fail": 0, "risk": 0},
            {"day": 2, "candidate_id": 1, "pf_fail": 1, "risk": 1},
            {"day": 3, "candidate_id": 0, "pf_fail": 0, "risk": 0},
            {"day": 4, "candidate_id": 0, "pf_fail": 0, "risk": 0},
            {"day": 4, "candidate_id": 1, "pf_fail": 0, "risk": 0},
            {"day": 4, "candidate_id": 1, "pf_fail": 0, "risk": 0},
        ]
    )

    assert all_capacity_feasible_days(episodes, 2).tolist() == [1]


def test_all_capacity_feasible_days_validates_schema():
    with pytest.raises(ValueError, match="missing columns"):
        all_capacity_feasible_days(pd.DataFrame({"day": [1]}), 1)
