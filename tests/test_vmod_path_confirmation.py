import pytest

from run_vmod_path_confirmation import common_confirmation_days


def test_confirmation_day_count_is_derived_from_outputs():
    points = [
        {"seed_summaries": [{"evaluated_days": 349}, {"evaluated_days": 349}]},
        {"seed_summaries": [{"evaluated_days": 349}, {"evaluated_days": 349}]},
    ]
    assert common_confirmation_days(points) == 349


def test_confirmation_day_count_rejects_inconsistent_outputs():
    points = [
        {"seed_summaries": [{"evaluated_days": 349}, {"evaluated_days": 299}]}
    ]
    with pytest.raises(ValueError, match="Inconsistent"):
        common_confirmation_days(points)
