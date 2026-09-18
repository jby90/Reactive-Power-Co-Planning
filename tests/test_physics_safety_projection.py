from pathlib import Path

import numpy as np

import Env
from physics_safety_projection import ACPowerFlowSafetyProjector


ROOT = Path(__file__).resolve().parents[1]


def profile_path(released_name: str, legacy_name: str) -> Path:
    released = ROOT / "data" / "inputs" / released_name
    return released if released.is_file() else ROOT / legacy_name


class LinearPreviewProjector(ACPowerFlowSafetyProjector):
    def _preview(self, env, action):
        del env
        action = np.asarray(action)
        return np.array([0.94 + 0.02 * action[0], 1.0 + 0.01 * action[1]])


def test_safe_action_is_unchanged():
    projector = LinearPreviewProjector(lower_voltage=0.939, upper_voltage=1.02)
    raw = np.array([0.1, -0.2])
    result = projector.project(None, raw)
    np.testing.assert_allclose(result.action, raw)
    assert not result.intervened
    assert result.success


def test_projection_finds_minimum_linear_correction():
    projector = LinearPreviewProjector(
        lower_voltage=0.9505,
        upper_voltage=1.0495,
        finite_difference_step=0.02,
    )
    result = projector.project(None, np.array([0.0, 0.0]))
    assert result.intervened
    assert result.success
    assert result.action[0] >= 0.525 - 1e-5
    assert abs(result.action[1]) < 1e-5
    assert result.projected_vmin >= 0.9505 - 1e-8


def test_preview_matches_next_environment_step():
    env = Env.grid_case(
        33,
        np.load(profile_path("load_15min_366d.npy", "load96.npy")),
        np.load(profile_path("pv_15min_366d.npy", "gen96.npy")),
        [17, 21, 24],
        [32],
        enable_pq_curve=True,
        pv_s_scale=0.7,
        svc_q_scale=0.7,
        cap_buses=[20, 8],
        cap_q_mvar=[0.125, 0.125],
    )
    env.reset_at_step(15 * 96)
    projector = ACPowerFlowSafetyProjector()
    action = np.array([0.1, 0.2, 0.3, 0.4])
    preview = projector._preview(env, action)
    next_state, *_ = env.step_model(action)
    np.testing.assert_allclose(preview, next_state[:33], atol=1e-9, rtol=0.0)
