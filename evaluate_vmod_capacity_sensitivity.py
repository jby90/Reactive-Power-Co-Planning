"""Evaluate frozen VMOD actors on a declared capacity-point sensitivity table."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vmod_actor import CapacityConditionedActor, load_policy_initialisation
from vmod_evaluation import build_env, evaluate_day, load_day_indices, parse_ints


def evaluate_job(payload: tuple) -> tuple[dict, list[dict]]:
    torch.set_num_threads(1)
    (
        seed,
        point,
        days,
        profile_dir,
        policy_dir,
        svc_absorption_ratio,
        cap_buses,
    ) = payload
    theta = np.asarray(
        [point["pv_s_scale"], point["svc_q_scale"], point.get("cap_total_mvar", 0.0)],
        dtype=np.float32,
    )
    profile_root = Path(profile_dir)
    load = np.load(profile_root / "load_15min_366d.npy")
    generation = np.load(profile_root / "pv_15min_366d.npy")
    env = build_env(
        33,
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
    load_policy_initialisation(actor, str(Path(policy_dir) / f"policy_init_seed{seed}.pth"))
    actor.eval()
    rows = []
    for day in days:
        result = evaluate_day(env, actor, theta, int(day), torch.device("cpu"))
        rows.append(
            {
                "path_id": point["path_id"],
                "point_order": int(point["point_order"]),
                "capacity_label": point["capacity_label"],
                "pv_s_scale": float(point["pv_s_scale"]),
                "svc_q_scale": float(point["svc_q_scale"]),
                "seed": int(seed),
                **result,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "path_id": point["path_id"],
        "point_order": int(point["point_order"]),
        "capacity_label": point["capacity_label"],
        "pv_s_scale": float(point["pv_s_scale"]),
        "svc_q_scale": float(point["svc_q_scale"]),
        "seed": int(seed),
        "evaluated_days": int(len(frame)),
        "event_days": int(frame.event.sum()),
        "pf_fail_days": int(frame.pf_fail.sum()),
        "mean_daily_line_loss_mwh": float(frame.line_loss_mwh.mean()),
        "worst_minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "worst_maximum_voltage_pu": float(frame.maximum_voltage_pu.max()),
        "mean_daily_absolute_reactive_throughput_mvarh": float(
            frame.absolute_reactive_throughput_mvarh.mean()
        ),
    }
    return summary, rows


def validate_table(frame: pd.DataFrame) -> None:
    required = {
        "path_id",
        "point_order",
        "capacity_label",
        "pv_s_scale",
        "svc_q_scale",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Capacity table missing columns: {missing}")
    for path_id, group in frame.groupby("path_id", sort=False):
        ordered = group.sort_values("point_order")
        if ordered.point_order.duplicated().any():
            raise ValueError(f"Duplicate point_order in path {path_id}")
        if (ordered.pv_s_scale.diff().dropna() < 0).any():
            raise ValueError(f"PV scale decreases in path {path_id}")
        if (ordered.svc_q_scale.diff().dropna() < 0).any():
            raise ValueError(f"SVC scale decreases in path {path_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity_table", type=Path, required=True)
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--policy_dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--svc_absorption_ratio", type=float, default=1.0)
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    points = pd.read_csv(args.capacity_table)
    validate_table(points)
    days = load_day_indices(args.days_metadata)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cap_buses = parse_ints(args.cap_buses)
    jobs = [
        (
            seed,
            point._asdict(),
            days,
            str(args.profile_dir.resolve()),
            str(args.policy_dir.resolve()),
            args.svc_absorption_ratio,
            cap_buses,
        )
        for point in points.itertuples(index=False)
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        outputs = list(executor.map(evaluate_job, jobs, chunksize=1))

    seed_summary = pd.DataFrame(output[0] for output in outputs)
    daily = pd.DataFrame(row for output in outputs for row in output[1])
    point_summary = (
        seed_summary.groupby(
            [
                "path_id",
                "point_order",
                "capacity_label",
                "pv_s_scale",
                "svc_q_scale",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            total_event_days=("event_days", "sum"),
            maximum_seed_event_days=("event_days", "max"),
            total_pf_fail_days=("pf_fail_days", "sum"),
            mean_seed_loss_mwh=("mean_daily_line_loss_mwh", "mean"),
            worst_minimum_voltage_pu=("worst_minimum_voltage_pu", "min"),
            worst_maximum_voltage_pu=("worst_maximum_voltage_pu", "max"),
            mean_seed_reactive_throughput_mvarh=(
                "mean_daily_absolute_reactive_throughput_mvarh",
                "mean",
            ),
        )
    )
    point_summary["all_seeds_zero_events"] = point_summary.total_event_days.eq(0)
    point_summary["all_seeds_zero_pf_failures"] = point_summary.total_pf_fail_days.eq(0)
    point_summary["operational_pass"] = (
        point_summary.all_seeds_zero_events
        & point_summary.all_seeds_zero_pf_failures
    )
    selected = []
    for path_id, group in point_summary.groupby("path_id", sort=False):
        ordered = group.sort_values("point_order")
        passing = ordered.loc[ordered.operational_pass]
        first = None if passing.empty else passing.iloc[0]
        selected.append(
            {
                "path_id": path_id,
                "first_operational_pass_label": (
                    None if first is None else first.capacity_label
                ),
                "first_operational_pass_order": (
                    None if first is None else int(first.point_order)
                ),
                "all_higher_tested_points_pass": bool(
                    first is not None
                    and ordered.loc[
                        ordered.point_order >= first.point_order, "operational_pass"
                    ].all()
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_dir / "daily.csv", index=False)
    seed_summary.to_csv(args.out_dir / "seed_summary.csv", index=False)
    point_summary.to_csv(args.out_dir / "point_summary.csv", index=False)
    payload = {
        "scope": (
            "Frozen-actor operational sensitivity only; no teacher-support gate, "
            "investment objective, or global-optimality claim."
        ),
        "capacity_table": args.capacity_table.as_posix(),
        "days_metadata": args.days_metadata.as_posix(),
        "profile_dir": args.profile_dir.as_posix(),
        "seeds": seeds,
        "evaluated_days_per_seed": len(days),
        "paths": selected,
        "points": point_summary.to_dict("records"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
