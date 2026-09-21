import pandas as pd

from run_vmod_path_selection import (
    teacher_coverage_eligible,
    upper_path_is_monotone,
)


def test_teacher_coverage_is_a_selection_gate():
    assert teacher_coverage_eligible(40, 32)
    assert not teacher_coverage_eligible(40, 31)
    assert not teacher_coverage_eligible(0, 0)


def test_upper_path_monotonicity_rejects_a_higher_capacity_reversal():
    monotone = pd.DataFrame(
        {"path_index": [0, 1, 2, 3], "passed_selection_gate": [False, True, True, True]}
    )
    reversal = monotone.copy()
    reversal.loc[reversal.path_index == 2, "passed_selection_gate"] = False

    assert upper_path_is_monotone(monotone, 1)
    assert not upper_path_is_monotone(reversal, 1)
    assert not upper_path_is_monotone(monotone, None)
