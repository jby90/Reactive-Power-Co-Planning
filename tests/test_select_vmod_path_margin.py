import pandas as pd

from select_vmod_path_margin import candidate_path_gate


def _coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [0, 1, 2, 3],
            "attempted_days": [40, 40, 40, 40],
            "converged_days": [20, 40, 40, 40],
        }
    )


def test_path_margin_gate_requires_buffer_and_no_higher_reversal():
    path = pd.DataFrame(
        {
            "path_index": [0, 1, 2, 3],
            "all_seeds_zero_events": [False, True, True, True],
            "all_seeds_zero_pf_failures": [True, True, True, True],
            "worst_vmin": [0.94, 0.9512, 0.9511, 0.952],
        }
    )
    result = candidate_path_gate(path, _coverage())
    assert result["first_passing_path_index"] == 1
    assert result["all_higher_points_pass"]
    assert result["passed_path_gate"]

    path.loc[path.path_index == 2, "worst_vmin"] = 0.95099
    result = candidate_path_gate(path, _coverage())
    assert not result["all_higher_points_pass"]
    assert not result["passed_path_gate"]


def test_path_margin_gate_rejects_sparse_teacher_support():
    path = pd.DataFrame(
        {
            "path_index": [0, 1, 2, 3],
            "all_seeds_zero_events": [True, True, True, True],
            "all_seeds_zero_pf_failures": [True, True, True, True],
            "worst_vmin": [0.952, 0.952, 0.952, 0.952],
        }
    )
    result = candidate_path_gate(path, _coverage())
    assert result["first_passing_path_index"] == 1
