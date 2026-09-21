import numpy as np
import pytest

from evaluate_raw_non_rl_baselines import action_for, volt_var_droop
import tune_raw_droop_baseline


@pytest.mark.parametrize(
    ("voltage", "expected"),
    [
        (0.94, 1.0),
        (0.95, 1.0),
        (0.965, 0.5),
        (1.0, 0.0),
        (1.035, -0.5),
        (1.05, -1.0),
        (1.06, -1.0),
    ],
)
def test_volt_var_droop_breakpoints(voltage, expected):
    assert volt_var_droop(voltage) == pytest.approx(expected)


def test_droop_tuner_uses_current_vmod_environment_builder():
    assert tune_raw_droop_baseline.build_env.__module__ == "vmod_evaluation"


def test_default_droop_grid_remains_frozen():
    assert tune_raw_droop_baseline.SCALES == (0.10, 0.25, 0.50, 0.75, 1.00)
    assert len(tune_raw_droop_baseline.CURVES) == 3


def test_pilot_grid_adds_a_stronger_feeder_wide_curve():
    assert tune_raw_droop_baseline.PILOT_CURVES[-1] == (0.95, 0.99, 1.01, 1.05)


def test_pilot_droop_uses_limiting_bus_voltage_for_all_devices():
    observation = np.array([1.00, 0.95, 1.01, 0.0, 0.0, 0.0])
    action = action_for(
        "pilot_droop",
        observation,
        device_buses=[0, 2],
        n_bus=3,
        action_scale=0.5,
        droop_breakpoints=(0.95, 0.98, 1.02, 1.05),
    )
    np.testing.assert_allclose(action, [0.5, 0.5])
