# VMOD Final Claim-Gate Report

Core story supported: **True**

## Gates

- PASS: `selection_feasibility_is_monotone_above_boundary`
- PASS: `main_adjacent_boundary_confirmed`
- PASS: `per_seed_finite_sample_upper_below_one_percent`
- FAIL: `vmod_capacity_strictly_lower_than_confirmed_droop`
- PASS: `vmod_capacity_no_greater_than_confirmed_pilot_droop`
- FAIL: `vmod_capacity_strictly_lower_than_confirmed_no_control`
- FAIL: `vmod_lower_capacity_also_has_lower_paired_loss_than_droop`
- PASS: `vmod_has_lower_paired_loss_than_pilot_droop`
- PASS: `margin_calibration_completed_without_data_leakage`
- PASS: `positive_voltage_margin_reduces_confirmation_events`
- FAIL: `capacity_conditioning_reduces_confirmation_events`
- PASS: `capacity_conditioning_not_worse_for_confirmation_events`
- PASS: `selected_point_has_adequate_teacher_trajectory_support`
- PASS: `ac_nodal_balance_verified`
- PASS: `online_path_below_one_percent_of_control_interval`
- PASS: `timing_diagnostic_below_one_percent_on_33_69_118_bus_cases`
- FAIL: `sixty_nine_bus_boundary_confirmed`

## Claim discipline

- A failed gate removes the corresponding superiority claim; it is not repaired by wording.
- The selected point is the lowest confirmed tested path point, not a global or economic optimum.
- Shift and second-feeder results are empirical scope checks, not formal or field guarantees.
