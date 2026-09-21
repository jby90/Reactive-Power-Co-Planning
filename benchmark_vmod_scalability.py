"""Benchmark VMOD architecture and AC-solver latency across feeder sizes.

This is a computational scaling diagnostic only. It uses a zero-command actor
and deterministic synthetic operating traces, so it must not be used as
evidence of control quality, voltage feasibility, or a capacity boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import pandapower
import torch
try:
    from pandapower.converter.matpower import from_mpc
except ImportError:  # pragma: no cover - compatibility with older pandapower
    from pandapower.converter import from_mpc

from vmod_actor import CapacityConditionedActor
from vmod_evaluation import build_env, device_ids, parse_ints


ROOT = Path(__file__).resolve().parent
CASE_FILES = {33: "case33_bw.mat", 69: "case69.mat", 118: "case1180zh.mat"}
CONTROL_INTERVAL_MS = 900_000.0


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "maximum": float(data.max()),
    }


def synthetic_profiles(env_id: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic dimensionally valid traces for timing only."""
    network = from_mpc(
        str(ROOT / CASE_FILES[env_id]),
        f_hz=50,
        casename_mpc_file="mpc",
        validate_conversion=False,
    )
    phase = np.linspace(0.0, 2.0 * np.pi, steps + 1, endpoint=False)
    load_factor = 0.75 + 0.05 * np.sin(phase)
    pv_generation_mw = 0.50 + 0.05 * np.cos(phase)
    load = np.repeat(load_factor[:, None], len(network.load), axis=1)
    pv = np.repeat(pv_generation_mw[:, None], len(device_ids(env_id)[0]), axis=1)
    return load, pv


def benchmark_environment(env_id: int, steps: int) -> tuple[dict, pd.DataFrame]:
    load, pv = synthetic_profiles(env_id, steps)
    theta = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    env = build_env(
        env_id,
        theta,
        [],
        1.0,
        "reference_mvar",
        load,
        pv,
        1.0,
    )
    actor = CapacityConditionedActor(
        len(env.observation_space),
        len(env.action_space),
        [0.225, 0.0, 0.0],
        [1.5, 1.5, 1.0],
        log_std_init=-3.0,
    )
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    actor.eval()
    theta_tensor = torch.as_tensor(theta, dtype=torch.float32)
    observation = env.reset_at_step(0)
    with torch.no_grad():
        for _ in range(50):
            actor._pi(torch.as_tensor(observation, dtype=torch.float32), theta_tensor)

    rows = []
    actor_ms: list[float] = []
    ac_ms: list[float] = []
    total_ms: list[float] = []
    for step in range(steps):
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
                "environment": env_id,
                "step": step,
                "actor_inference_ms": actor_elapsed,
                "ac_execution_ms": ac_elapsed,
                "total_ms": total_elapsed,
            }
        )

    summary = {
        "environment": env_id,
        "buses": int(len(env.model.bus)),
        "in_service_branches": int(env.model.line.in_service.astype(bool).sum()),
        "loads": int(len(env.model.load)),
        "controlled_devices": int(len(env.action_space)),
        "observation_dimension": int(len(env.observation_space)),
        "steps": steps,
        "actor_inference_ms": distribution(actor_ms),
        "nonlinear_ac_execution_ms": distribution(ac_ms),
        "total_online_path_ms": distribution(total_ms),
        "maximum_budget_fraction": max(total_ms) / CONTROL_INTERVAL_MS,
        "trainable_policy_parameters": int(
            sum(parameter.numel() for parameter in actor.policy_parameters())
        ),
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default="33,69,118")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 20:
        raise ValueError("At least 20 timed steps are required")

    torch.set_num_threads(1)
    environments = parse_ints(args.envs)
    invalid = sorted(set(environments) - set(CASE_FILES))
    if invalid:
        raise ValueError(f"Unsupported environments: {invalid}")

    summaries = []
    tables = []
    for env_id in environments:
        summary, table = benchmark_environment(env_id, args.steps)
        summaries.append(summary)
        tables.append(table)

    payload = {
        "diagnostic_scope": (
            "Architecture-and-AC-solver timing only. Zero actor commands and "
            "synthetic deterministic traces are used; no control-performance, "
            "safety, portability, or capacity-boundary claim is supported."
        ),
        "control_interval_ms": CONTROL_INTERVAL_MS,
        "environments": summaries,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pandapower": pandapower.__version__,
            "numpy": np.__version__,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out_dir / "steps.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
