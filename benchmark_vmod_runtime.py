"""Benchmark raw VMOD actor inference and nonlinear AC execution latency."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import pandapower
import torch

from vmod_actor import CapacityConditionedActor, load_policy_initialisation
from vmod_evaluation import (
    build_env,
    load_day_indices,
    parse_floats,
    parse_ints,
)


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "maximum": float(data.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", default="0.45,0.375,0")
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    theta = np.asarray(parse_floats(args.theta), dtype=np.float32)
    days = load_day_indices(args.days_metadata)
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
    actor = CapacityConditionedActor(
        len(env.observation_space),
        len(env.action_space),
        [0.225, 0.0, 0.0],
        [1.5, 1.5, 1.0],
        log_std_init=-3.0,
    )
    load_policy_initialisation(actor, str(args.checkpoint))
    actor.eval()
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)

    # Warm up tensor allocation and the actor before collecting timings.
    warm_observation = torch.as_tensor(env.reset_at_step(0), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(100):
            actor._pi(warm_observation, theta_tensor)

    rows = []
    actor_ms: list[float] = []
    ac_ms: list[float] = []
    total_ms: list[float] = []
    for day in days:
        observation = env.reset_at_step(int(day) * 96)
        for step in range(96):
            total_start = perf_counter_ns()
            actor_start = perf_counter_ns()
            with torch.no_grad():
                action, _ = actor._pi(
                    torch.as_tensor(observation, dtype=torch.float32), theta_tensor
                )
            actor_elapsed = (perf_counter_ns() - actor_start) / 1e6
            ac_start = perf_counter_ns()
            transition = env.step_model(action.numpy())
            ac_elapsed = (perf_counter_ns() - ac_start) / 1e6
            total_elapsed = (perf_counter_ns() - total_start) / 1e6
            observation = transition[-1]
            actor_ms.append(actor_elapsed)
            ac_ms.append(ac_elapsed)
            total_ms.append(total_elapsed)
            rows.append(
                {
                    "day_index": int(day),
                    "step": step,
                    "actor_inference_ms": actor_elapsed,
                    "ac_execution_ms": ac_elapsed,
                    "total_ms": total_elapsed,
                }
            )

    checkpoint_bytes = args.checkpoint.stat().st_size
    summary = {
        "algorithm": "VMOD raw feed-forward policy",
        "theta": theta.tolist(),
        "environment": args.env,
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "days": len(days),
        "steps": len(rows),
        "actor_inference_ms": distribution(actor_ms),
        "nonlinear_ac_execution_ms": distribution(ac_ms),
        "total_online_path_ms": distribution(total_ms),
        "control_interval_ms": 900_000.0,
        "maximum_budget_fraction": max(total_ms) / 900_000.0,
        "checkpoint_bytes": checkpoint_bytes,
        "trainable_policy_parameters": int(
            sum(parameter.numel() for parameter in actor.policy_parameters())
        ),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pandapower": pandapower.__version__,
            "numpy": np.__version__,
        },
        "measurement_note": (
            "Single-process CPU benchmark. The online VMOD path contains no "
            "optimisation, safety projection, or online OPF."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "steps.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
