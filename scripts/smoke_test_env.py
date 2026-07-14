"""
Smoke test for Env.grid_case.

Run:
  python smoke_test_env.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import Env


def _test_one(env_name: int, id_iber: list[int], id_svc: list[int]) -> None:
    load = np.load(Env.DATA_DIR / "load96.npy")
    gen = np.load(Env.DATA_DIR / "gen96.npy")
    env = Env.grid_case(env_name, load, gen, id_iber, id_svc)
    a = np.zeros(env.action_space.shape[0], dtype=float)
    out = env.step_model(a)
    reward = out[1]
    grid_loss = out[8]
    violation = out[3]
    print(f"env={env_name}\treward={reward}\tgrid_loss={grid_loss}\tviolation={violation}")
    print(
        "generated:",
        [p.name for p in [Env.DATA_DIR / f"two{env.n_bus}load.npy", Env.DATA_DIR / f"two{env.n_bus}gen.npy"] if p.exists()],
    )


def main() -> None:
    _test_one(33, [17, 21, 24], [32])


if __name__ == "__main__":
    main()
