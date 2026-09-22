import numpy as np

from vmod_sensitivity_qp import (
    SensitivityQPConfig,
    representative_profile_step,
    voltage_objective_and_gradient,
)


def test_voltage_objective_gradient_matches_finite_difference():
    voltage = np.array([0.94, 1.01, 1.06])
    sensitivity = np.array([[0.02, 0.01], [0.01, 0.03], [0.015, 0.02]])
    current = np.array([0.1, -0.2])
    q = np.array([0.2, -0.1])
    config = SensitivityQPConfig(1000.0, 1.0, 0.1)
    _, gradient = voltage_objective_and_gradient(
        q, voltage, current, sensitivity, config
    )
    numerical = np.zeros_like(q)
    epsilon = 1e-6
    for index in range(len(q)):
        plus = q.copy()
        minus = q.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        f_plus, _ = voltage_objective_and_gradient(
            plus, voltage, current, sensitivity, config
        )
        f_minus, _ = voltage_objective_and_gradient(
            minus, voltage, current, sensitivity, config
        )
        numerical[index] = (f_plus - f_minus) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-5)


def test_representative_profile_step_is_drawn_from_requested_days():
    load = np.arange(5 * 96, dtype=float)[:, None]
    generation = np.ones_like(load)
    step = representative_profile_step(load, generation, [1, 3])
    assert 96 <= step < 2 * 96 or 3 * 96 <= step < 4 * 96
