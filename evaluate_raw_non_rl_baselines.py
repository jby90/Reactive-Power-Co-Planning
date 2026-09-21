"""Evaluate deterministic no-control and local Volt-VAR droop baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vmod_evaluation import (
    DT_HOURS,
    STEPS_PER_DAY,
    build_env,
    load_day_indices,
    parse_floats,
    parse_ints,
)


def volt_var_droop(
    voltage: float,
    full_injection: float = 0.95,
    deadband_low: float = 0.98,
    deadband_high: float = 1.02,
    full_absorption: float = 1.05,
) -> float:
    """IEEE-1547-style piecewise-linear local Volt-VAR action."""
    if voltage <= full_injection:
        return 1.0
    if voltage < deadband_low:
        return (deadband_low - voltage) / (deadband_low - full_injection)
    if voltage <= deadband_high:
        return 0.0
    if voltage < full_absorption:
        return -(voltage - deadband_high) / (full_absorption - deadband_high)
    return -1.0


def action_for(
    method: str,
    observation: np.ndarray,
    device_buses: list[int],
    n_bus: int,
    action_scale: float,
    droop_breakpoints: tuple[float, float, float, float],
) -> np.ndarray:
    if method == "no_control":
        return np.zeros(len(device_buses), dtype=np.float32)
    voltage = np.asarray(observation[:n_bus], dtype=float)
    if method == "pilot_droop":
        # A feeder-wide pilot signal is a stronger conventional comparator than
        # purely local droop when the limiting bus has no controllable device.
        pilot_action = action_scale * volt_var_droop(
            float(voltage.min()), *droop_breakpoints
        )
        return np.full(len(device_buses), pilot_action, dtype=np.float32)
    return np.asarray(
        [
            action_scale
            * volt_var_droop(float(voltage[bus]), *droop_breakpoints)
            for bus in device_buses
        ],
        dtype=np.float32,
    )


def evaluate_day(
    env,
    method: str,
    theta: np.ndarray,
    day: int,
    action_scale: float,
    droop_breakpoints: tuple[float, float, float, float],
) -> dict:
    del theta
    device_buses = list(env.id_iber) + list(env.id_svc)
    observation = env.reset_at_step(int(day) * STEPS_PER_DAY)
    under_steps = 0
    over_steps = 0
    pf_fail = 0
    daily_loss = 0.0
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    absolute_normalized_action = 0.0
    squared_normalized_action = 0.0
    saturated_actions = 0
    action_count = 0
    absolute_reactive_throughput_mvarh = 0.0
    maximum_absolute_reactive_command_mvar = 0.0
    for _ in range(STEPS_PER_DAY):
        action = action_for(
            method,
            observation,
            device_buses,
            len(env.model.bus),
            action_scale,
            droop_breakpoints,
        )
        clipped_command = np.clip(action, -1.0, 1.0)
        try:
            next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = (
                env.step_model(action)
            )
        except Exception:
            pf_fail = 1
            break
        executed_q = np.asarray(next_obs[-len(action) :], dtype=float)
        absolute_normalized_action += float(np.abs(clipped_command).sum())
        squared_normalized_action += float(np.square(clipped_command).sum())
        saturated_actions += int(np.count_nonzero(np.abs(clipped_command) >= 0.999))
        action_count += int(clipped_command.size)
        absolute_reactive_throughput_mvarh += float(np.abs(executed_q).sum()) * DT_HOURS
        maximum_absolute_reactive_command_mvar = max(
            maximum_absolute_reactive_command_mvar,
            float(np.abs(executed_q).max(initial=0.0)),
        )
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        minimum_voltage = min(minimum_voltage, float(vmin))
        maximum_voltage = max(maximum_voltage, float(vmax))
        daily_loss += max(0.0, float(-grid_loss)) * DT_HOURS
        observation = new_state
    return {
        "method": method,
        "day": int(day),
        "event": int(bool(under_steps or over_steps or pf_fail)),
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "minimum_voltage_pu": minimum_voltage,
        "maximum_voltage_pu": maximum_voltage,
        "line_loss_mwh": daily_loss,
        "mean_absolute_normalized_action": (
            absolute_normalized_action / action_count if action_count else np.nan
        ),
        "rms_normalized_action": (
            np.sqrt(squared_normalized_action / action_count)
            if action_count
            else np.nan
        ),
        "normalized_action_saturation_fraction": (
            saturated_actions / action_count if action_count else np.nan
        ),
        "absolute_reactive_throughput_mvarh": absolute_reactive_throughput_mvarh,
        "maximum_absolute_reactive_command_mvar": (
            maximum_absolute_reactive_command_mvar
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", required=True)
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--load_profile", type=Path, required=True)
    parser.add_argument("--generation_profile", type=Path, required=True)
    parser.add_argument("--methods", default="droop,no_control")
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--full_injection", type=float, default=0.95)
    parser.add_argument("--deadband_low", type=float, default=0.98)
    parser.add_argument("--deadband_high", type=float, default=1.02)
    parser.add_argument("--full_absorption", type=float, default=1.05)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    theta = np.asarray(parse_floats(args.theta), dtype=np.float32)
    if theta.shape != (3,):
        parser.error("--theta must contain exactly three values")
    load = np.load(args.load_profile)
    generation = np.load(args.generation_profile)
    days = load_day_indices(args.days_metadata)
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unsupported = set(methods) - {"droop", "pilot_droop", "no_control"}
    if unsupported:
        parser.error(f"Unsupported methods: {sorted(unsupported)}")
    rows = []
    droop_breakpoints = (
        args.full_injection,
        args.deadband_low,
        args.deadband_high,
        args.full_absorption,
    )
    for method in methods:
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
        rows.extend(
            evaluate_day(
                env,
                method,
                theta,
                day,
                args.action_scale,
                droop_breakpoints,
            )
            for day in days
        )
    daily = pd.DataFrame(rows)
    summary = (
        daily.groupby("method", as_index=False)
        .agg(
            evaluated_days=("day", "size"),
            event_days=("event", "sum"),
            pf_fail_days=("pf_fail", "sum"),
            mean_daily_line_loss_mwh=("line_loss_mwh", "mean"),
            worst_minimum_voltage_pu=("minimum_voltage_pu", "min"),
            worst_maximum_voltage_pu=("maximum_voltage_pu", "max"),
            mean_absolute_normalized_action=(
                "mean_absolute_normalized_action",
                "mean",
            ),
            mean_rms_normalized_action=("rms_normalized_action", "mean"),
            normalized_action_saturation_fraction=(
                "normalized_action_saturation_fraction",
                "mean",
            ),
            mean_daily_absolute_reactive_throughput_mvarh=(
                "absolute_reactive_throughput_mvarh",
                "mean",
            ),
            maximum_absolute_reactive_command_mvar=(
                "maximum_absolute_reactive_command_mvar",
                "max",
            ),
        )
        .assign(
            theta=str(theta.tolist()),
            action_scale=args.action_scale,
            droop_breakpoints=str(droop_breakpoints),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_dir / "daily.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    payload = summary.to_dict("records")
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
