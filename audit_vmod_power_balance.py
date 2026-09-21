"""Audit nonlinear AC nodal-balance residuals for a raw VMOD actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vmod_actor import CapacityConditionedActor, load_policy_initialisation
from vmod_evaluation import (
    build_env,
    load_day_indices,
    parse_floats,
    parse_ints,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", required=True)
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initialisation_dir", type=Path)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--svc_absorption_ratio", type=float, default=1.0)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.initialisation_dir is None):
        parser.error("Provide exactly one of --checkpoint or --initialisation_dir")

    torch.set_num_threads(1)
    theta = np.asarray(parse_floats(args.theta), dtype=np.float32)
    if theta.shape != (3,):
        parser.error("--theta must contain three values")
    load = np.load(args.profile_dir / "load_15min_366d.npy")
    generation = np.load(args.profile_dir / "pv_15min_366d.npy")
    env = build_env(
        args.env,
        theta,
        parse_ints(args.cap_buses),
        1.0,
        "reference_mvar",
        load,
        generation,
        args.svc_absorption_ratio,
    )
    env.audit_power_balance = True
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    rows = []
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.initialisation_dir is not None
        else [-1]
    )
    failure_rows = []
    for seed in seeds:
        checkpoint = (
            args.checkpoint
            if seed == -1
            else args.initialisation_dir / f"policy_init_seed{seed}.pth"
        )
        actor = CapacityConditionedActor(
            len(env.observation_space),
            len(env.action_space),
            [0.225, 0.0, 0.0],
            [1.5, 1.5, 1.0],
            log_std_init=-3.0,
        )
        load_policy_initialisation(actor, str(checkpoint))
        actor.eval()
        seed_failures = 0
        for day in load_day_indices(args.days_metadata):
            state = env.reset_at_step(int(day) * 96)
            for step in range(96):
                with torch.no_grad():
                    action, _ = actor._pi(
                        torch.as_tensor(state, dtype=torch.float32), theta_tensor
                    )
                try:
                    state = env.step_model(action.numpy())[-1]
                except Exception:
                    seed_failures += 1
                    break
                rows.append(
                    {
                        "seed": seed,
                        "day": int(day),
                        "step": step,
                        "maximum_active_balance_residual_mw": float(
                            env.last_p_balance_residual_mw
                        ),
                        "maximum_reactive_balance_residual_mvar": float(
                            env.last_q_balance_residual_mvar
                        ),
                    }
                )
        failure_rows.append({"seed": seed, "power_flow_failures": seed_failures})
    frame = pd.DataFrame(rows)
    per_seed = []
    for failure in failure_rows:
        seed_frame = frame.loc[frame.seed == failure["seed"]]
        per_seed.append(
            {
                **failure,
                "audited_steps": int(len(seed_frame)),
                "maximum_active_balance_residual_mw": float(
                    seed_frame.maximum_active_balance_residual_mw.max()
                ),
                "maximum_reactive_balance_residual_mvar": float(
                    seed_frame.maximum_reactive_balance_residual_mvar.max()
                ),
            }
        )
    summary = {
        "environment": args.env,
        "theta": theta.tolist(),
        "seeds": seeds,
        "audited_steps": int(len(frame)),
        "power_flow_failures": int(
            sum(row["power_flow_failures"] for row in failure_rows)
        ),
        "maximum_active_balance_residual_mw": float(
            frame.maximum_active_balance_residual_mw.max()
        ),
        "maximum_reactive_balance_residual_mvar": float(
            frame.maximum_reactive_balance_residual_mvar.max()
        ),
        "solver": "pandapower backward/forward sweep nonlinear AC power flow",
        "slack_treatment": "benchmark feeder external grid is the AC slack source",
        "per_seed": per_seed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "step_residuals.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
