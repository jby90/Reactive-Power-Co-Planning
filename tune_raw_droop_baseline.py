"""Tune a local Volt-VAR droop baseline on training days, then freeze it."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_raw_non_rl_baselines import evaluate_day
from vmod_evaluation import build_env, load_day_indices, parse_floats, parse_ints


CURVES = (
    (0.94, 0.97, 1.03, 1.06),
    (0.94, 0.98, 1.02, 1.06),
    (0.95, 0.98, 1.02, 1.05),
)
PILOT_CURVES = (
    *CURVES,
    (0.95, 0.99, 1.01, 1.05),
)
SCALES = (0.10, 0.25, 0.50, 0.75, 1.00)


def evaluate_configuration(payload: tuple) -> dict:
    (
        scale,
        curve,
        days,
        load_path,
        generation_path,
        theta_values,
        env_id,
        cap_buses,
        svc_absorption_ratio,
        method,
    ) = payload
    theta = np.asarray(theta_values, dtype=np.float32)
    load = np.load(load_path)
    generation = np.load(generation_path)
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
    rows = [
        evaluate_day(
            env,
            method,
            theta,
            int(day),
            float(scale),
            tuple(float(value) for value in curve),
        )
        for day in days
    ]
    frame = pd.DataFrame(rows)
    return {
        "method": str(method),
        "action_scale": float(scale),
        "full_injection": float(curve[0]),
        "deadband_low": float(curve[1]),
        "deadband_high": float(curve[2]),
        "full_absorption": float(curve[3]),
        "evaluated_days": int(len(frame)),
        "event_days": int(frame.event.sum()),
        "pf_fail_days": int(frame.pf_fail.sum()),
        "mean_daily_line_loss_mwh": float(frame.line_loss_mwh.mean()),
        "worst_minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "worst_maximum_voltage_pu": float(frame.maximum_voltage_pu.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_days", type=Path, required=True)
    parser.add_argument("--load_profile", type=Path, required=True)
    parser.add_argument("--generation_profile", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--theta", default="0.45,0.375,0")
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument(
        "--method",
        choices=("droop", "pilot_droop"),
        default="droop",
        help="Conventional feedback signal to tune on development days.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(value) for value in SCALES),
        help=(
            "Comma-separated normalized droop scales. The default reproduces "
            "the frozen coarse grid; alternate values support development-only "
            "grid-refinement audits."
        ),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    theta = parse_floats(args.theta)
    if len(theta) != 3:
        parser.error("--theta must contain three values")
    cap_buses = parse_ints(args.cap_buses)
    scales = parse_floats(args.scales)
    if not scales or any(value <= 0.0 or value > 1.0 for value in scales):
        parser.error("--scales must contain values in (0, 1]")
    if len(set(scales)) != len(scales):
        parser.error("--scales must not contain duplicates")

    days = load_day_indices(args.training_days)
    curves = PILOT_CURVES if args.method == "pilot_droop" else CURVES
    jobs = [
        (
            scale,
            curve,
            days,
            str(args.load_profile.resolve()),
            str(args.generation_profile.resolve()),
            theta,
            args.env,
            cap_buses,
            args.svc_absorption_ratio,
            args.method,
        )
        for curve in curves
        for scale in scales
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(evaluate_configuration, jobs, chunksize=1))
    table = pd.DataFrame(rows).sort_values(
        ["pf_fail_days", "event_days", "mean_daily_line_loss_mwh"],
        kind="stable",
    )
    best = table.iloc[0].to_dict()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "training_grid.csv", index=False)
    payload = {
        "selection_rule": "lexicographic PF failures, event days, then mean line loss",
        "method": args.method,
        "theta": theta,
        "environment": args.env,
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "training_day_count": len(days),
        "configurations": len(table),
        "breakpoint_curves": [list(curve) for curve in curves],
        "action_scales": scales,
        "best": best,
    }
    (args.out_dir / "frozen_droop.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
