import numpy as np

import Env
from vmod_evaluation import action_reference_scales, build_env


def build(scale: float, parameterization: str) -> Env.grid_case:
    env = Env.grid_case(
        33,
        np.ones((96, 32)),
        np.zeros((96, 3)),
        [17, 21, 24],
        [32],
        enable_pq_curve=True,
        pv_s_scale=scale,
        svc_q_scale=scale,
        action_parameterization=parameterization,
    )
    env.reset_at_step(10)
    return env


def test_reference_mvar_does_not_mechanically_scale_same_action() -> None:
    low = build(0.7, "reference_mvar")
    high = build(1.5, "reference_mvar")
    action = np.array([0.1, -0.1, 0.05, -0.5])
    np.testing.assert_allclose(low.action_clip(action), high.action_clip(action), atol=1e-12)


def test_relative_parameterization_scales_with_capacity() -> None:
    low = build(0.7, "relative")
    high = build(1.5, "relative")
    action = np.array([0.1, -0.1, 0.05, -0.5])
    assert not np.allclose(low.action_clip(action), high.action_clip(action))


def test_physical_q_inverse_round_trip() -> None:
    env = build(0.7, "reference_mvar")
    q_target = np.array([0.25, -0.10, 0.30, 0.60])
    action = env.physical_q_to_action(q_target)
    np.testing.assert_allclose(env.action_clip(action), q_target, atol=1e-12)


def test_svc_absorption_ratio_changes_lower_bound_only() -> None:
    env = Env.grid_case(
        33,
        np.ones((96, 32)),
        np.zeros((96, 3)),
        [17, 21, 24],
        [32],
        svc_q_scale=0.5,
        svc_absorption_ratio=1.0,
    )
    svc_index = env._idx_svc_sgen[0]
    assert env.model.sgen.loc[svc_index, "min_q_mvar"] == -1.0
    assert env.model.sgen.loc[svc_index, "max_q_mvar"] == 1.0


def test_69_bus_reference_map_spans_frozen_extended_path() -> None:
    load = np.load("data/vmod/profiles69/load_15min_366d.npy")[:96]
    generation = np.load("data/vmod/profiles69/pv_15min_366d.npy")[:96]
    env = build_env(
        69,
        np.asarray([1.5, 1.5, 0.0]),
        [],
        1.0,
        "reference_mvar",
        load,
        generation,
        1.0,
    )
    env.reset_at_step(0)
    low, high = env._action_mapping_bounds()

    assert action_reference_scales(69) == (1.5, 1.5)
    np.testing.assert_allclose(low, env.model.sgen.min_q_mvar.to_numpy())
    np.testing.assert_allclose(high, env.model.sgen.max_q_mvar.to_numpy())
