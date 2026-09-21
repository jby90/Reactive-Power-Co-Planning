"""Evaluate raw VMOD actors under physical and telemetry distribution shifts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vmod_actor import CapacityConditionedActor, load_policy_initialisation
from vmod_evaluation import build_env, load_day_indices, parse_floats
from vmod_statistics import percentile_bootstrap_mean_interval


STEPS = 96
DT_HOURS = 0.25
TOPOLOGY_RECONFIGURATIONS = {
    "none": None,
    "tie32_open18": (32, 18),
    "tie33_open11": (33, 11),
}

_ACTORS: dict[int, CapacityConditionedActor] = {}
_LOAD: np.ndarray | None = None
_GENERATION: np.ndarray | None = None
_THETA: np.ndarray | None = None
_SVC_ABSORPTION_RATIO = 0.0


def finite_or_none(value: float) -> float | None:
    """Return a JSON-safe finite scalar, or ``None`` when unavailable."""
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def shift_random_seed(base_seed: int, day: int, shift_id: int) -> int:
    """Return a seed shared by every actor for one day/shift replicate."""
    return int(base_seed + int(day) * 1009 + int(shift_id))


def replicate_event_rate_summary(episodes: pd.DataFrame) -> dict:
    """Summarise uncertainty across matched physical-shift realisations."""
    rates = (
        episodes.groupby("shift_id", sort=True).event.mean().to_numpy(dtype=float)
    )
    if len(rates) == 1:
        mean = float(rates[0])
        low = high = mean
        interval_available = False
    else:
        mean, low, high = percentile_bootstrap_mean_interval(rates)
        interval_available = True
    return {
        "shift_replicates": int(len(rates)),
        "replicate_mean_event_rate": mean,
        "replicate_level_event_rate_ci95_low": low,
        "replicate_level_event_rate_ci95_high": high,
        "replicate_interval_available": interval_available,
        "replicate_level_event_rate_interval_method": (
            "percentile bootstrap over matched shift realisations; 10000 "
            "resamples; seed 20260921"
        ),
        "replicate_event_rates": rates.tolist(),
    }


def initialise_worker(
    initialisation_dir: str,
    profile_dir: str,
    theta_values: list[float],
    seeds: list[int],
    svc_absorption_ratio: float,
) -> None:
    global _ACTORS, _LOAD, _GENERATION, _THETA, _SVC_ABSORPTION_RATIO
    torch.set_num_threads(1)
    profile_root = Path(profile_dir)
    _LOAD = np.load(profile_root / "load_15min_366d.npy")
    _GENERATION = np.load(profile_root / "pv_15min_366d.npy")
    _THETA = np.asarray(theta_values, dtype=np.float32)
    _SVC_ABSORPTION_RATIO = float(svc_absorption_ratio)
    dummy = build_env(
        33,
        _THETA,
        [20, 8],
        1.0,
        "reference_mvar",
        _LOAD,
        _GENERATION,
        _SVC_ABSORPTION_RATIO,
    )
    _ACTORS = {}
    for seed in seeds:
        actor = CapacityConditionedActor(
            len(dummy.observation_space),
            len(dummy.action_space),
            [0.225, 0.0, 0.0],
            [1.5, 1.5, 1.0],
            log_std_init=-3.0,
        )
        load_policy_initialisation(
            actor, str(Path(initialisation_dir) / f"policy_init_seed{seed}.pth")
        )
        actor.eval()
        _ACTORS[int(seed)] = actor


def observed_state(
    state: np.ndarray,
    buses: int,
    rng: np.random.Generator,
    voltage_std: float,
    power_relative_std: float,
) -> np.ndarray:
    measured = np.asarray(state, dtype=float).copy()
    if voltage_std > 0:
        measured[:buses] += rng.normal(0.0, voltage_std, buses)
    if power_relative_std > 0:
        measured[buses : 3 * buses] *= rng.normal(
            1.0, power_relative_std, 2 * buses
        )
    return measured


def evaluate_job(job: tuple) -> dict:
    (
        seed,
        day,
        shift_id,
        random_seed,
        r_std,
        x_std,
        load_std,
        pv_std,
        voltage_measurement_std,
        power_measurement_relative_std,
        observation_delay_steps,
        topology_reconfiguration,
    ) = job
    assert _LOAD is not None and _GENERATION is not None and _THETA is not None
    actor = _ACTORS[int(seed)]
    rng = np.random.default_rng(int(random_seed))
    load_multiplier = float(np.clip(rng.normal(1.0, load_std), 0.7, 1.3))
    pv_multiplier = float(np.clip(rng.normal(1.0, pv_std), 0.7, 1.3))
    env = build_env(
        33,
        _THETA,
        [20, 8],
        pv_multiplier,
        "reference_mvar",
        _LOAD,
        _GENERATION,
        _SVC_ABSORPTION_RATIO,
    )
    r_multiplier = np.clip(rng.normal(1.0, r_std, len(env.model.line)), 0.5, 1.5)
    x_multiplier = np.clip(rng.normal(1.0, x_std, len(env.model.line)), 0.5, 1.5)
    env.model.line.loc[:, "r_ohm_per_km"] *= r_multiplier
    env.model.line.loc[:, "x_ohm_per_km"] *= x_multiplier
    env.init_load_p_mw *= load_multiplier
    env.init_load_q_mvar *= load_multiplier
    topology = TOPOLOGY_RECONFIGURATIONS[str(topology_reconfiguration)]
    if topology is not None:
        close_line, open_line = topology
        env.model.line.loc[close_line, "in_service"] = True
        env.model.line.loc[open_line, "in_service"] = False

    under_steps = 0
    over_steps = 0
    pf_fail = 0
    line_loss_mwh = 0.0
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    try:
        state = env.reset_at_step(int(day) * STEPS)
        state_history = [np.asarray(state, dtype=float).copy()]
        for _ in range(STEPS):
            delayed_index = max(0, len(state_history) - 1 - observation_delay_steps)
            measurement = observed_state(
                state_history[delayed_index],
                len(env.model.bus),
                rng,
                voltage_measurement_std,
                power_measurement_relative_std,
            )
            with torch.no_grad():
                action, _ = actor._pi(
                    torch.as_tensor(measurement, dtype=torch.float32),
                    torch.as_tensor(_THETA, dtype=torch.float32),
                )
            next_state, _, _, _, _, _, vmax, vmin, grid_loss, state = env.step_model(
                action.numpy()
            )
            state_history.append(np.asarray(state, dtype=float).copy())
            voltage = np.asarray(next_state[: len(env.model.bus)], dtype=float)
            under_steps += int(np.any(voltage < 0.95))
            over_steps += int(np.any(voltage > 1.05))
            minimum_voltage = min(minimum_voltage, float(vmin))
            maximum_voltage = max(maximum_voltage, float(vmax))
            line_loss_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
    except Exception:
        pf_fail = 1
        # A partial trajectory is not a valid daily energy observation.
        line_loss_mwh = np.nan

    return {
        "seed": int(seed),
        "day": int(day),
        "shift_id": int(shift_id),
        "random_seed": int(random_seed),
        "event": int(bool(under_steps or over_steps or pf_fail)),
        "under_steps": int(under_steps),
        "over_steps": int(over_steps),
        "pf_fail": int(pf_fail),
        "minimum_voltage_pu": finite_or_none(minimum_voltage),
        "maximum_voltage_pu": finite_or_none(maximum_voltage),
        "line_loss_mwh": finite_or_none(line_loss_mwh),
        "load_multiplier": load_multiplier,
        "pv_multiplier": pv_multiplier,
        "mean_r_multiplier": float(np.mean(r_multiplier)),
        "mean_x_multiplier": float(np.mean(x_multiplier)),
        "topology_reconfiguration": str(topology_reconfiguration),
        "observation_delay_steps": int(observation_delay_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta", default="0.45,0.375,0")
    parser.add_argument("--days_metadata", type=Path, required=True)
    parser.add_argument("--profile_dir", type=Path, required=True)
    parser.add_argument("--initialisation_dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--shift_replicates", type=int, default=20)
    parser.add_argument("--shift_seed", type=int, default=2026092001)
    parser.add_argument("--r_std", type=float, default=0.05)
    parser.add_argument("--x_std", type=float, default=0.05)
    parser.add_argument("--load_std", type=float, default=0.02)
    parser.add_argument("--pv_std", type=float, default=0.02)
    parser.add_argument("--voltage_measurement_std", type=float, default=0.0)
    parser.add_argument("--power_measurement_relative_std", type=float, default=0.0)
    parser.add_argument("--observation_delay_steps", type=int, default=0)
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument(
        "--topology_reconfiguration",
        choices=tuple(TOPOLOGY_RECONFIGURATIONS),
        default="none",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    theta = parse_floats(args.theta)
    if len(theta) != 3:
        parser.error("--theta must contain three values")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    days = load_day_indices(args.days_metadata)
    jobs = []
    for seed in seeds:
        for day in days:
            for shift_id in range(args.shift_replicates):
                jobs.append(
                    (
                        seed,
                        day,
                        shift_id,
                        shift_random_seed(args.shift_seed, day, shift_id),
                        args.r_std,
                        args.x_std,
                        args.load_std,
                        args.pv_std,
                        args.voltage_measurement_std,
                        args.power_measurement_relative_std,
                        args.observation_delay_steps,
                        args.topology_reconfiguration,
                    )
                )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialise_worker,
        initargs=(
            str(args.initialisation_dir.resolve()),
            str(args.profile_dir.resolve()),
            theta,
            seeds,
            args.svc_absorption_ratio,
        ),
    ) as executor:
        rows = list(executor.map(evaluate_job, jobs, chunksize=1))

    episodes = pd.DataFrame(rows)
    seed_summary = (
        episodes.groupby("seed", sort=True)
        .agg(
            evaluated_shift_days=("event", "size"),
            event_days=("event", "sum"),
            pf_fail_days=("pf_fail", "sum"),
            mean_daily_line_loss_mwh=("line_loss_mwh", "mean"),
            worst_minimum_voltage_pu=("minimum_voltage_pu", "min"),
            worst_maximum_voltage_pu=("maximum_voltage_pu", "max"),
        )
        .reset_index()
    )
    seed_summary["event_rate"] = (
        seed_summary.event_days / seed_summary.evaluated_shift_days
    )
    seed_event_rate_mean, seed_event_rate_low, seed_event_rate_high = (
        percentile_bootstrap_mean_interval(seed_summary.event_rate.to_numpy())
    )
    replicate_summary = replicate_event_rate_summary(episodes)
    seed_records = seed_summary.to_dict("records")
    for row in seed_records:
        for column in (
            "mean_daily_line_loss_mwh",
            "worst_minimum_voltage_pu",
            "worst_maximum_voltage_pu",
        ):
            row[column] = finite_or_none(row[column])
    summary = {
        "theta": theta,
        "seeds": seeds,
        "days": len(days),
        "shift_replicates": args.shift_replicates,
        "evaluated_shift_days": int(len(episodes)),
        "event_days": int(episodes.event.sum()),
        "event_rate": float(episodes.event.mean()),
        "seed_mean_event_rate": seed_event_rate_mean,
        "seed_level_event_rate_ci95_low": seed_event_rate_low,
        "seed_level_event_rate_ci95_high": seed_event_rate_high,
        "seed_level_event_rate_interval_method": (
            "percentile bootstrap over independently trained actors; 10000 "
            "resamples; seed 20260921"
        ),
        **replicate_summary,
        "pf_fail_days": int(episodes.pf_fail.sum()),
        "mean_daily_line_loss_mwh": finite_or_none(episodes.line_loss_mwh.mean()),
        "worst_minimum_voltage_pu": finite_or_none(
            episodes.minimum_voltage_pu.min()
        ),
        "worst_maximum_voltage_pu": finite_or_none(
            episodes.maximum_voltage_pu.max()
        ),
        "shift": {
            "r_std": args.r_std,
            "x_std": args.x_std,
            "load_std": args.load_std,
            "pv_std": args.pv_std,
            "voltage_measurement_std": args.voltage_measurement_std,
            "power_measurement_relative_std": args.power_measurement_relative_std,
            "observation_delay_steps": args.observation_delay_steps,
            "topology_reconfiguration": args.topology_reconfiguration,
        },
        "interpretation": (
            "Empirical out-of-distribution stress test of a raw actor without "
            "online model use; not a formal robustness or safety guarantee."
        ),
        "seed_summaries": seed_records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    seed_summary.to_csv(args.out_dir / "seed_summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
