"""Evaluate the frozen OPF-imitation actors before any RL fine-tuning."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vmod_actor import CapacityConditionedActor, load_policy_initialisation
from vmod_evaluation import (
    build_env,
    evaluate_day,
    load_day_indices,
    parse_floats,
    parse_ints,
)


def voltage_tail_summary(frame: pd.DataFrame) -> dict:
    """Summarise daily voltage tails relative to the reported 0.95--1.05 band."""
    return {
        "daily_minimum_voltage_p01_pu": float(
            frame.minimum_voltage_pu.quantile(0.01)
        ),
        "daily_maximum_voltage_p99_pu": float(
            frame.maximum_voltage_pu.quantile(0.99)
        ),
        "days_with_minimum_voltage_within_0p001_pu_of_lower_limit": int(
            frame.minimum_voltage_pu.between(0.95, 0.951, inclusive="both").sum()
        ),
        "days_with_maximum_voltage_within_0p001_pu_of_upper_limit": int(
            frame.maximum_voltage_pu.between(1.049, 1.05, inclusive="both").sum()
        ),
    }


def evaluate_seed(payload: tuple) -> tuple[dict, list[dict]]:
    # Each seed already runs in its own process; avoid nested BLAS thread storms.
    torch.set_num_threads(1)
    (
        seed,
        theta_values,
        days,
        profile_dir,
        initialisation_dir,
        svc_absorption_ratio,
        env_id,
        cap_buses,
    ) = payload
    theta = np.asarray(theta_values, dtype=np.float32)
    profile_root = Path(profile_dir)
    load = np.load(profile_root / "load_15min_366d.npy")
    generation = np.load(profile_root / "pv_15min_366d.npy")
    env = build_env(
        int(env_id),
        theta,
        list(cap_buses),
        1.0,
        "reference_mvar",
        load,
        generation,
        float(svc_absorption_ratio),
    )
    actor = CapacityConditionedActor(
        len(env.observation_space),
        len(env.action_space),
        [0.225, 0.0, 0.0],
        [1.5, 1.5, 1.0],
        log_std_init=-3.0,
    )
    load_policy_initialisation(
        actor, str(Path(initialisation_dir) / f"policy_init_seed{seed}.pth")
    )
    actor.eval()
    rows = [
        {
            "seed": int(seed),
            **evaluate_day(env, actor, theta, int(day), torch.device("cpu")),
        }
        for day in days
    ]
    frame = pd.DataFrame(rows)
    summary = {
        "seed": int(seed),
        "evaluated_days": int(len(frame)),
        "event_days": int(frame.event.sum()),
        "pf_fail_days": int(frame.pf_fail.sum()),
        "mean_daily_line_loss_mwh": float(frame.line_loss_mwh.mean()),
        "worst_minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "worst_maximum_voltage_pu": float(frame.maximum_voltage_pu.max()),
        **voltage_tail_summary(frame),
        "mean_absolute_normalized_action": float(
            frame.mean_absolute_normalized_action.mean()
        ),
        "mean_rms_normalized_action": float(frame.rms_normalized_action.mean()),
        "normalized_action_saturation_fraction": float(
            frame.normalized_action_saturation_fraction.mean()
        ),
        "mean_daily_absolute_reactive_throughput_mvarh": float(
            frame.absolute_reactive_throughput_mvarh.mean()
        ),
        "maximum_absolute_reactive_command_mvar": float(
            frame.maximum_absolute_reactive_command_mvar.max()
        ),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", default="0.45,0.375,0")
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--initialisation_dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    theta = parse_floats(args.theta)
    if len(theta) != 3:
        parser.error("--theta must contain three values")
    days = load_day_indices(args.days_metadata)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cap_buses = parse_ints(args.cap_buses)
    jobs = [
        (
            seed,
            theta,
            days,
            str(args.profile_dir.resolve()),
            str(args.initialisation_dir.resolve()),
            args.svc_absorption_ratio,
            args.env,
            cap_buses,
        )
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        outputs = list(executor.map(evaluate_seed, jobs, chunksize=1))
    summaries = pd.DataFrame(output[0] for output in outputs)
    daily = pd.DataFrame(row for output in outputs for row in output[1])
    aggregate = {
        "theta": theta,
        "seeds": seeds,
        "evaluated_days_per_seed": len(days),
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "environment": args.env,
        "capacitor_buses": cap_buses,
        "all_seeds_zero_events": bool((summaries.event_days == 0).all()),
        "all_seeds_zero_pf_failures": bool((summaries.pf_fail_days == 0).all()),
        "total_event_days": int(summaries.event_days.sum()),
        "total_pf_fail_days": int(summaries.pf_fail_days.sum()),
        "maximum_seed_event_days": int(summaries.event_days.max()),
        "mean_seed_loss_mwh": float(summaries.mean_daily_line_loss_mwh.mean()),
        "mean_seed_absolute_reactive_throughput_mvarh": float(
            summaries.mean_daily_absolute_reactive_throughput_mvarh.mean()
        ),
        "mean_seed_normalized_action_saturation_fraction": float(
            summaries.normalized_action_saturation_fraction.mean()
        ),
        "seed_summaries": summaries.to_dict("records"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_dir / "daily.csv", index=False)
    summaries.to_csv(args.out_dir / "seed_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
