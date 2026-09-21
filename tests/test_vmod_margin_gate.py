from run_vmod_bidirectional_margin_study import candidate_passed


def test_margin_gate_requires_zero_events_and_zero_pf_failures() -> None:
    assert candidate_passed(
        {
            "all_seeds_zero_events": True,
            "all_seeds_zero_pf_failures": True,
            "adequate_calibration_teacher_support": True,
        }
    )
    assert not candidate_passed(
        {
            "all_seeds_zero_events": True,
            "all_seeds_zero_pf_failures": False,
            "adequate_calibration_teacher_support": True,
        }
    )
    assert not candidate_passed(
        {
            "all_seeds_zero_events": False,
            "all_seeds_zero_pf_failures": True,
            "adequate_calibration_teacher_support": True,
        }
    )
    assert not candidate_passed(
        {
            "all_seeds_zero_events": True,
            "all_seeds_zero_pf_failures": True,
            "adequate_calibration_teacher_support": False,
        }
    )
