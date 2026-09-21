"""Shared feeder construction and deterministic VMOD evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import Env


STEPS_PER_DAY = 96
DT_HOURS = 0.25


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def device_ids(env_id: int) -> tuple[list[int], list[int]]:
    if env_id == 33:
        return [17, 21, 24], [32]
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    if env_id == 118:
        return [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]
    raise ValueError(f"Unsupported environment: {env_id}")


def action_reference_scales(env_id: int) -> tuple[float, float]:
    """Return fixed MVAr-map scales spanning each frozen VMOD test path."""
    if env_id == 33:
        return 1.0, 1.0
    if env_id == 69:
        return 1.5, 1.5
    if env_id == 118:
        return 1.0, 1.0
    raise ValueError(f"Unsupported environment: {env_id}")


def load_day_indices(path: str | Path) -> list[int]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
        if "day_index" not in table.columns:
            raise ValueError(f"Day table must contain a day_index column: {source}")
        return [int(day) for day in table["day_index"].tolist()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("final_day_indices", "day_indices"):
        if key in payload:
            return [int(day) for day in payload[key]]
    raise ValueError(f"Day metadata must define final_day_indices or day_indices: {source}")


def build_env(
    env_id: int,
    theta: np.ndarray,
    cap_buses: list[int],
    pv_generation_scale: float,
    action_parameterization: str = "relative",
    load_pu: np.ndarray | None = None,
    gene_pu: np.ndarray | None = None,
    svc_absorption_ratio: float = 0.0,
) -> Env.grid_case:
    if load_pu is None or gene_pu is None:
        raise ValueError("VMOD evaluation requires explicit load and generation profiles")
    id_iber, id_svc = device_ids(env_id)
    reference_pv_s_scale, reference_svc_q_scale = action_reference_scales(env_id)
    cap_total = float(theta[2])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    return Env.grid_case(
        env_id,
        load_pu,
        gene_pu,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=float(theta[0]),
        pv_generation_scale=float(pv_generation_scale),
        svc_q_scale=float(theta[1]),
        svc_absorption_ratio=float(svc_absorption_ratio),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization=action_parameterization,
        reference_pv_s_scale=reference_pv_s_scale,
        reference_svc_q_scale=reference_svc_q_scale,
    )


def evaluate_day(
    env: Env.grid_case,
    actor: torch.nn.Module,
    theta: np.ndarray,
    day: int,
    device: torch.device,
) -> dict:
    observation = env.reset_at_step(int(day) * STEPS_PER_DAY)
    under_steps = 0
    over_steps = 0
    pf_fail = 0
    line_loss_mwh = 0.0
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    absolute_normalized_action = 0.0
    squared_normalized_action = 0.0
    saturated_actions = 0
    action_count = 0
    absolute_reactive_throughput_mvarh = 0.0
    maximum_absolute_reactive_command_mvar = 0.0
    for _ in range(STEPS_PER_DAY):
        with torch.no_grad():
            action, _ = actor._pi(
                torch.as_tensor(observation, dtype=torch.float32, device=device),
                torch.as_tensor(theta, dtype=torch.float32, device=device),
            )
        command = action.detach().cpu().numpy()
        clipped_command = np.clip(command, -1.0, 1.0)
        try:
            next_observation, _, _, _, _, _, vmax, vmin, grid_loss, new_state = (
                env.step_model(command)
            )
        except Exception:
            pf_fail = 1
            break
        executed_q = np.asarray(next_observation[-len(command) :], dtype=float)
        absolute_normalized_action += float(np.abs(clipped_command).sum())
        squared_normalized_action += float(np.square(clipped_command).sum())
        saturated_actions += int(np.count_nonzero(np.abs(clipped_command) >= 0.999))
        action_count += int(clipped_command.size)
        absolute_reactive_throughput_mvarh += float(np.abs(executed_q).sum()) * DT_HOURS
        maximum_absolute_reactive_command_mvar = max(
            maximum_absolute_reactive_command_mvar,
            float(np.abs(executed_q).max(initial=0.0)),
        )
        voltage = np.asarray(next_observation[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        minimum_voltage = min(minimum_voltage, float(vmin))
        maximum_voltage = max(maximum_voltage, float(vmax))
        line_loss_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
        observation = new_state
    return {
        "day": int(day),
        "event": int(bool(under_steps or over_steps or pf_fail)),
        "under_event": int(bool(under_steps or pf_fail)),
        "over_event": int(bool(over_steps or pf_fail)),
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "minimum_voltage_pu": minimum_voltage,
        "maximum_voltage_pu": maximum_voltage,
        "line_loss_mwh": line_loss_mwh,
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
        "maximum_absolute_reactive_command_mvar": maximum_absolute_reactive_command_mvar,
    }
