import pandas as pd

from evaluate_opf_initialisation_baseline import voltage_tail_summary


def test_voltage_tail_summary_counts_only_inside_near_limit_band():
    frame = pd.DataFrame(
        {
            "minimum_voltage_pu": [0.9499, 0.95, 0.9505, 0.951, 0.9511],
            "maximum_voltage_pu": [1.0489, 1.049, 1.0495, 1.05, 1.0501],
        }
    )

    result = voltage_tail_summary(frame)

    assert result["days_with_minimum_voltage_within_0p001_pu_of_lower_limit"] == 3
    assert result["days_with_maximum_voltage_within_0p001_pu_of_upper_limit"] == 3
    assert result["daily_minimum_voltage_p01_pu"] < 0.95
    assert result["daily_maximum_voltage_p99_pu"] > 1.05
