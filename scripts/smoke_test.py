"""Run one deterministic environment step with a released WG-CVaR checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_crdc_policy import build_env, load_actor  # noqa: E402


def main() -> None:
    theta = np.asarray([0.7, 0.7, 0.0], dtype=np.float32)
    env = build_env(33, theta, [20, 8], 1.0)
    actor, checkpoint = load_actor(
        ROOT / "models" / "wg_cvar" / "seed42",
        len(env.observation_space),
        len(env.action_space),
        torch.device("cpu"),
    )
    state = env.reset_at_step(0)
    with torch.no_grad():
        action, _ = actor._pi(torch.as_tensor(state, dtype=torch.float32), torch.from_numpy(theta))
    next_state, *_ = env.step_model(action.numpy())
    assert np.isfinite(np.asarray(next_state, dtype=float)).all()
    print(f"Smoke test passed with {checkpoint['algorithm']} at theta={theta.tolist()}.")


if __name__ == "__main__":
    main()
