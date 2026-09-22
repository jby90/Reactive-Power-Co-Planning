"""Fixed-sensitivity online Volt--VAR controller used as a VMOD baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandapower as pp
from scipy.optimize import minimize

from vmod_evaluation import DT_HOURS, STEPS_PER_DAY


@dataclass(frozen=True)
class SensitivityQPConfig:
    band_weight: float
    centering_weight: float
    reactive_weight: float
    movement_weight: float = 0.01
    voltage_margin_pu: float = 0.003

    def to_dict(self) -> dict:
        return asdict(self)


def voltage_objective_and_gradient(
    q: np.ndarray,
    voltage: np.ndarray,
    q_current: np.ndarray,
    sensitivity: np.ndarray,
    config: SensitivityQPConfig,
) -> tuple[float, np.ndarray]:
    """Return a smooth convex penalty and its analytical gradient."""
    predicted = voltage + sensitivity @ (q - q_current)
    lower = 0.95 + config.voltage_margin_pu
    upper = 1.05 - config.voltage_margin_pu
    below = np.maximum(lower - predicted, 0.0)
    above = np.maximum(predicted - upper, 0.0)
    centred = predicted - 1.0
    movement = q - q_current
    objective = (
        config.band_weight * (below @ below + above @ above)
        + config.centering_weight * (centred @ centred)
        + config.reactive_weight * (q @ q)
        + config.movement_weight * (movement @ movement)
    )
    voltage_gradient = (
        -2.0 * config.band_weight * below
        + 2.0 * config.band_weight * above
        + 2.0 * config.centering_weight * centred
    )
    gradient = (
        sensitivity.T @ voltage_gradient
        + 2.0 * config.reactive_weight * q
        + 2.0 * config.movement_weight * movement
    )
    return float(objective), np.asarray(gradient, dtype=float)


class SensitivityQPController:
    """Solve a four-variable fixed-sensitivity problem at each control step."""

    def __init__(self, sensitivity: np.ndarray, config: SensitivityQPConfig):
        self.sensitivity = np.asarray(sensitivity, dtype=float)
        self.config = config
        self.solver_failures = 0
        self.solve_times_ms: list[float] = []

    def action(self, env, observation: np.ndarray) -> np.ndarray:
        n_bus = len(env.model.bus)
        voltage = np.asarray(observation[:n_bus], dtype=float)
        q_current = np.asarray(env.model.sgen.q_mvar, dtype=float)
        low = np.asarray(env.model.sgen.min_q_mvar, dtype=float)
        high = np.asarray(env.model.sgen.max_q_mvar, dtype=float)
        initial = np.clip(q_current, low, high)
        started = perf_counter()
        result = minimize(
            lambda q: voltage_objective_and_gradient(
                q, voltage, q_current, self.sensitivity, self.config
            ),
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(low, high)),
            options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
        )
        self.solve_times_ms.append((perf_counter() - started) * 1000.0)
        if not result.success or not np.all(np.isfinite(result.x)):
            self.solver_failures += 1
            target_q = initial
        else:
            target_q = np.clip(result.x, low, high)
        return env.physical_q_to_action(target_q)


def representative_profile_step(
    load: np.ndarray, generation: np.ndarray, days: list[int]
) -> int:
    """Choose the development step nearest the multivariate profile centroid."""
    indices = np.concatenate(
        [np.arange(day * STEPS_PER_DAY, (day + 1) * STEPS_PER_DAY) for day in days]
    )
    features = np.column_stack((load[indices], generation[indices])).astype(float)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    distance = np.square((features - centre) / scale).mean(axis=1)
    return int(indices[int(np.argmin(distance))])


def estimate_voltage_sensitivity(
    env, step: int, perturbation_mvar: float = 0.01
) -> tuple[np.ndarray, dict]:
    """Estimate one fixed AC voltage/reactive sensitivity on a development point."""
    env.reset_at_step(step)
    controlled = list(env.model.sgen.index)
    base_q = np.asarray(env.model.sgen.q_mvar, dtype=float).copy()
    base_voltage = np.asarray(env.model.res_bus.vm_pu, dtype=float).copy()
    low = np.asarray(env.model.sgen.min_q_mvar, dtype=float)
    high = np.asarray(env.model.sgen.max_q_mvar, dtype=float)
    columns = []
    realised_perturbations = []
    for position, index in enumerate(controlled):
        positive_room = high[position] - base_q[position]
        negative_room = base_q[position] - low[position]
        if positive_room >= perturbation_mvar:
            delta = perturbation_mvar
        elif negative_room >= perturbation_mvar:
            delta = -perturbation_mvar
        else:
            delta = max(positive_room, -negative_room, key=abs)
        if abs(delta) <= 1e-9:
            raise RuntimeError(f"Device {position} has no reactive perturbation room")
        env.model.sgen.loc[controlled, "q_mvar"] = base_q
        env.model.sgen.loc[index, "q_mvar"] = base_q[position] + delta
        pp.runpp(env.model, algorithm="bfsw")
        voltage = np.asarray(env.model.res_bus.vm_pu, dtype=float)
        columns.append((voltage - base_voltage) / delta)
        realised_perturbations.append(float(delta))
    env.model.sgen.loc[controlled, "q_mvar"] = base_q
    pp.runpp(env.model, algorithm="bfsw")
    sensitivity = np.column_stack(columns)
    return sensitivity, {
        "representative_step": int(step),
        "perturbation_mvar": float(perturbation_mvar),
        "realised_perturbations_mvar": realised_perturbations,
        "shape": list(sensitivity.shape),
        "maximum_absolute_sensitivity_pu_per_mvar": float(
            np.abs(sensitivity).max()
        ),
    }


def evaluate_controller_day(env, controller: SensitivityQPController, day: int) -> dict:
    observation = env.reset_at_step(int(day) * STEPS_PER_DAY)
    under_steps = 0
    over_steps = 0
    pf_fail = 0
    line_loss_mwh = 0.0
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    reactive_throughput = 0.0
    maximum_q = 0.0
    solver_failures_before = controller.solver_failures
    solve_count_before = len(controller.solve_times_ms)
    for _ in range(STEPS_PER_DAY):
        command = controller.action(env, observation)
        try:
            next_observation, _, _, _, _, _, vmax, vmin, grid_loss, new_state = (
                env.step_model(command)
            )
        except Exception:
            pf_fail = 1
            break
        executed_q = np.asarray(next_observation[-len(command) :], dtype=float)
        voltage = np.asarray(next_observation[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        minimum_voltage = min(minimum_voltage, float(vmin))
        maximum_voltage = max(maximum_voltage, float(vmax))
        line_loss_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
        reactive_throughput += float(np.abs(executed_q).sum()) * DT_HOURS
        maximum_q = max(maximum_q, float(np.abs(executed_q).max(initial=0.0)))
        observation = new_state
    timings = controller.solve_times_ms[solve_count_before:]
    return {
        "day": int(day),
        "event": int(bool(under_steps or over_steps or pf_fail)),
        "under_steps": int(under_steps),
        "over_steps": int(over_steps),
        "pf_fail": int(pf_fail),
        "solver_failures": int(controller.solver_failures - solver_failures_before),
        "minimum_voltage_pu": float(minimum_voltage),
        "maximum_voltage_pu": float(maximum_voltage),
        "line_loss_mwh": float(line_loss_mwh),
        "absolute_reactive_throughput_mvarh": float(reactive_throughput),
        "maximum_absolute_reactive_command_mvar": float(maximum_q),
        "mean_online_optimisation_ms": float(np.mean(timings)) if timings else np.nan,
        "p95_online_optimisation_ms": float(np.quantile(timings, 0.95)) if timings else np.nan,
    }
