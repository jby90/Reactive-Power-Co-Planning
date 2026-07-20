"""Fast integrity and reproducibility checks for the public release."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import Env  # noqa: E402
from PPO_theta_robust import ActorCritic  # noqa: E402
from PPO_theta_robust_film_curriculum import FiLMActorCritic  # noqa: E402


EXPECTED_HASHES = {
    "case33_bw.mat": "b90e392c9a86c3f9",
    "load96.npy": "d536ec5f92252c9e",
    "gen96.npy": "bc0c97a1185ed784",
    "two33load.npy": "2e2fb7e64b8efa53",
    "two33gen.npy": "5dff4d92d5093342",
}


def _short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def verify_inputs() -> None:
    for name, expected in EXPECTED_HASHES.items():
        actual = _short_hash(Env.DATA_DIR / name)
        assert actual == expected, f"Hash mismatch for {name}: {actual} != {expected}"

    load = np.load(Env.DATA_DIR / "load96.npy")
    generation = np.load(Env.DATA_DIR / "gen96.npy")
    assert load.shape == generation.shape == (38496,)
    assert np.load(Env.DATA_DIR / "two33load.npy").shape == (38496, 32)
    assert np.load(Env.DATA_DIR / "two33gen.npy").shape == (38496, 3)


def verify_environment() -> None:
    load = np.load(Env.DATA_DIR / "load96.npy")
    generation = np.load(Env.DATA_DIR / "gen96.npy")
    env = Env.grid_case(33, load, generation, [17, 21, 24], [32])
    result = env.step_model(np.zeros(4, dtype=float))
    assert len(env.observation_space) == 103
    assert len(env.action_space) == 4
    assert np.isfinite(np.asarray(result[1], dtype=float)).all()
    assert np.isfinite(float(result[8]))


def verify_models() -> None:
    film = FiLMActorCritic(103, 4, theta_dim=3, hidden=(256, 256))
    film.load_state_dict(torch.load(ROOT / "models" / "film" / "actor.pth", map_location="cpu", weights_only=True))

    blind = ActorCritic(103, 4, hidden=(256, 256))
    blind.load_state_dict(torch.load(ROOT / "models" / "blind" / "actor.pth", map_location="cpu", weights_only=True))

    concat = ActorCritic(106, 4, hidden=(256, 256))
    concat.load_state_dict(torch.load(ROOT / "models" / "concat" / "actor.pth", map_location="cpu", weights_only=True))


def verify_reported_values() -> None:
    base = ROOT / "data" / "results" / "cleaner_energy"
    comparison = pd.read_csv(base / "controller_comparison" / "best_plans.csv").set_index("controller")
    np.testing.assert_allclose(comparison.loc["film", "annual_loss_mwh"], 382.99994065819266)
    np.testing.assert_allclose(comparison.loc["blind", "annual_loss_mwh"], 424.385778746996)
    np.testing.assert_allclose(comparison.loc["concat", "annual_loss_mwh"], 378.997370742389)
    assert (comparison["pr_violation_any"] == 0.0).all()

    low = pd.read_csv(base / "resource_efficient" / "best_plans.csv")
    high = pd.read_csv(base / "high_redundancy" / "best_plans.csv")
    low = low[(low["controller"] == "film") & np.isclose(low["pv_generation_scale"], 1.0)].iloc[0]
    high = high[(high["controller"] == "film") & np.isclose(high["pv_generation_scale"], 1.0)].iloc[0]
    np.testing.assert_allclose(100.0 * (1.0 - low["asset_cost_index"] / high["asset_cost_index"]), 40.3973505987, rtol=1e-8)
    np.testing.assert_allclose(100.0 * (1.0 - low["annual_loss_mwh"] / high["annual_loss_mwh"]), 35.9165457351, rtol=1e-8)

    planning = pd.read_csv(ROOT / "data" / "results" / "capacity_planning" / "summary_candidates.csv")
    assert len(planning) == 2000
    assert planning[["pv_s_scale", "svc_q_scale", "cap_total_mvar"]].drop_duplicates().shape[0] == 2000


def main() -> None:
    verify_inputs()
    verify_environment()
    verify_models()
    verify_reported_values()
    print("Release verification passed: inputs, environment, models, and reported values are consistent.")


if __name__ == "__main__":
    main()
