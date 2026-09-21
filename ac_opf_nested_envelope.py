"""Capacity-monotone AC-optimized control envelope for nested device limits."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pandapower as pp

import Env
from ac_opf_vvc_baseline import DT_HOURS, STEPS, _direct_ac_optimisation, _prepare_opf
from vmod_evaluation import device_ids, parse_ints


LOSS_MONOTONICITY_ATOL_MWH = 1e-4
LOSS_MONOTONICITY_RTOL = 1e-3


def loss_is_nonincreasing_with_tolerance(
    values: np.ndarray | pd.Series,
    atol_mwh: float = LOSS_MONOTONICITY_ATOL_MWH,
    rtol: float = LOSS_MONOTONICITY_RTOL,
) -> tuple[bool, float]:
    array = np.asarray(values, dtype=float)
    differences = np.diff(array)
    tolerances = atol_mwh + rtol * np.abs(array[:-1])
    maximum_positive_increase = float(np.maximum(differences, 0.0).max(initial=0.0))
    return bool(np.all(differences <= tolerances)), maximum_positive_increase


def adjacent_pair_loss_summary(
    episodes: pd.DataFrame,
    atol_mwh: float = LOSS_MONOTONICITY_ATOL_MWH,
    rtol: float = LOSS_MONOTONICITY_RTOL,
) -> pd.DataFrame:
    """Compare adjacent capacities only on days where both OPFs converged."""
    rows = []
    candidate_ids = sorted(int(value) for value in episodes.candidate_id.unique())
    for lower_id, upper_id in zip(candidate_ids[:-1], candidate_ids[1:]):
        lower = episodes.loc[
            episodes.candidate_id == lower_id,
            ["scenario_index", "daily_loss_mwh", "pf_fail"],
        ].rename(
            columns={
                "daily_loss_mwh": "lower_loss_mwh",
                "pf_fail": "lower_pf_fail",
            }
        )
        upper = episodes.loc[
            episodes.candidate_id == upper_id,
            ["scenario_index", "daily_loss_mwh", "pf_fail"],
        ].rename(
            columns={
                "daily_loss_mwh": "upper_loss_mwh",
                "pf_fail": "upper_pf_fail",
            }
        )
        paired = lower.merge(upper, on="scenario_index", validate="one_to_one")
        paired = paired.loc[
            ~paired.lower_pf_fail.astype(bool)
            & ~paired.upper_pf_fail.astype(bool)
            & np.isfinite(paired.lower_loss_mwh)
            & np.isfinite(paired.upper_loss_mwh)
        ].copy()
        differences = paired.upper_loss_mwh - paired.lower_loss_mwh
        tolerances = atol_mwh + rtol * paired.lower_loss_mwh.abs()
        relative_increases = np.maximum(differences, 0.0) / paired.lower_loss_mwh.abs()
        rows.append(
            {
                "lower_candidate_id": lower_id,
                "upper_candidate_id": upper_id,
                "paired_converged_days": int(len(paired)),
                "mean_lower_loss_mwh": (
                    None if paired.empty else float(paired.lower_loss_mwh.mean())
                ),
                "mean_upper_loss_mwh": (
                    None if paired.empty else float(paired.upper_loss_mwh.mean())
                ),
                "mean_upper_minus_lower_loss_mwh": (
                    None if paired.empty else float(differences.mean())
                ),
                "maximum_positive_increase_mwh": (
                    None if paired.empty else float(max(0.0, differences.max()))
                ),
                "maximum_relative_increase": (
                    None if paired.empty else float(relative_increases.max())
                ),
                "days_exceeding_tolerance": int((differences > tolerances).sum()),
                "nonincreasing_within_tolerance": bool(
                    not paired.empty and (differences <= tolerances).all()
                ),
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=4)
def _load_profiles(load_path: str, generation_path: str) -> tuple[np.ndarray, np.ndarray]:
    load = np.load(load_path)
    generation = np.load(generation_path)
    if len(load) != len(generation):
        raise ValueError("Load and generation profiles must have equal time lengths")
    return load, generation


def _build_env(
    row,
    load_path: str,
    generation_path: str,
    svc_absorption_ratio: float,
    env_id: int,
    cap_buses: list[int],
) -> Env.grid_case:
    load, generation = _load_profiles(load_path, generation_path)
    id_iber, id_svc = device_ids(env_id)
    cap_total = float(row.cap_total_mvar)
    cap_q = (
        [cap_total / len(cap_buses)] * len(cap_buses)
        if cap_total > 0 and cap_buses
        else []
    )
    return Env.grid_case(
        env_id,
        load,
        generation,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=float(row.pv_s_scale),
        svc_q_scale=float(row.svc_q_scale),
        svc_absorption_ratio=float(svc_absorption_ratio),
        cap_buses=cap_buses if cap_q else None,
        cap_q_mvar=cap_q or None,
    )


def _feasible_voltage(voltage: np.ndarray, lower: float, upper: float) -> bool:
    return bool(voltage.min() >= lower - 1e-6 and voltage.max() <= upper + 1e-6)


def _solve_day(payload: tuple) -> tuple[list[dict], list[dict]]:
    (
        row_dicts,
        lower_voltage,
        upper_voltage,
        load_path,
        generation_path,
        svc_absorption_ratio,
        env_id,
        cap_buses,
        allow_direct_fallback,
    ) = payload
    candidates = pd.DataFrame(row_dicts)
    required = {
        "candidate_id",
        "scenario_index",
        "t0",
        "pv_s_scale",
        "svc_q_scale",
        "cap_total_mvar",
    }
    missing = sorted(required - set(candidates.columns))
    if candidates.empty or missing:
        raise ValueError(
            "Nested-envelope day payload is empty or missing columns: "
            f"{missing}"
        )
    candidates = candidates.sort_values(
        ["pv_s_scale", "svc_q_scale", "candidate_id"]
    )
    if candidates.cap_total_mvar.nunique() != 1:
        raise ValueError("A nested envelope requires fixed capacitor capacity")
    envs = {
        int(row.candidate_id): _build_env(
            row,
            load_path,
            generation_path,
            svc_absorption_ratio,
            env_id,
            cap_buses,
        )
        for row in candidates.itertuples(index=False)
    }
    episode_state = {
        int(row.candidate_id): {
            "loss": 0.0,
            "vmin": np.inf,
            "vmax": -np.inf,
            "under_steps": 0,
            "over_steps": 0,
            "pf_fail": 0,
            "solve_seconds": 0.0,
            "inherited_steps": 0,
            "direct_fallback_steps": 0,
        }
        for row in candidates.itertuples(index=False)
    }
    step_rows: list[dict] = []
    t0 = int(candidates.iloc[0].t0)

    for offset in range(STEPS):
        inherited_q: np.ndarray | None = None
        inherited_loss = np.inf
        for row in candidates.itertuples(index=False):
            candidate_id = int(row.candidate_id)
            env = envs[candidate_id]
            state = episode_state[candidate_id]
            start = perf_counter()
            try:
                env.reset_at_step(t0 + offset)
                low_q = env.model.sgen.min_q_mvar.to_numpy(dtype=float)
                high_q = env.model.sgen.max_q_mvar.to_numpy(dtype=float)
                best_q: np.ndarray | None = None
                best_voltage: np.ndarray | None = None
                best_loss = np.inf
                solver = ""

                if inherited_q is not None and np.all(inherited_q >= low_q - 1e-9) and np.all(
                    inherited_q <= high_q + 1e-9
                ):
                    env.model.sgen.loc[:, "q_mvar"] = inherited_q
                    pp.runpp(env.model, algorithm="bfsw", numba=False)
                    voltage = env.model.res_bus.vm_pu.to_numpy(dtype=float)
                    loss = float(env.model.res_line.pl_mw.sum())
                    if _feasible_voltage(voltage, lower_voltage, upper_voltage):
                        best_q = inherited_q.copy()
                        best_voltage = voltage.copy()
                        best_loss = loss
                        solver = "inherited_nested_dispatch"

                _prepare_opf(env.model, lower_voltage, upper_voltage)
                try:
                    pp.runopp(
                        env.model,
                        verbose=False,
                        calculate_voltage_angles=False,
                        init="pf",
                    )
                    voltage = env.model.res_bus.vm_pu.to_numpy(dtype=float)
                    loss = float(env.model.res_line.pl_mw.sum())
                    if _feasible_voltage(voltage, lower_voltage, upper_voltage) and loss < best_loss:
                        best_q = env.model.res_sgen.q_mvar.to_numpy(dtype=float).copy()
                        best_voltage = voltage.copy()
                        best_loss = loss
                        solver = "pandapower_ac_opf"
                except Exception:
                    pass

                if best_q is None:
                    if not allow_direct_fallback:
                        raise RuntimeError("AC-OPF did not return a feasible dispatch")
                    state["direct_fallback_steps"] += 1
                    best_q, best_voltage, best_loss, _, _ = _direct_ac_optimisation(
                        env, lower_voltage, upper_voltage
                    )
                    solver = "direct_slsqp_ac_pf"

                assert best_voltage is not None
                inherited_q = best_q.copy()
                inherited_loss = best_loss
                elapsed = perf_counter() - start
                state["solve_seconds"] += elapsed
                state["inherited_steps"] += int(solver == "inherited_nested_dispatch")
                state["loss"] += max(0.0, best_loss) * DT_HOURS
                state["vmin"] = min(state["vmin"], float(best_voltage.min()))
                state["vmax"] = max(state["vmax"], float(best_voltage.max()))
                state["under_steps"] += int(best_voltage.min() < 0.95 - 1e-8)
                state["over_steps"] += int(best_voltage.max() > 1.05 + 1e-8)
                step_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario_index": int(row.scenario_index),
                        "step": offset,
                        "time_index": t0 + offset,
                        "solver": solver,
                        "loss_mw": best_loss,
                        "vmin": float(best_voltage.min()),
                        "vmax": float(best_voltage.max()),
                        "solve_ms": elapsed * 1000.0,
                        **{f"q_mvar_{index}": float(value) for index, value in enumerate(best_q)},
                    }
                )
            except Exception as exc:
                state["pf_fail"] = 1
                step_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario_index": int(row.scenario_index),
                        "step": offset,
                        "time_index": t0 + offset,
                        "solver": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                inherited_q = None
                inherited_loss = np.inf

    episode_rows = []
    for row in candidates.itertuples(index=False):
        state = episode_state[int(row.candidate_id)]
        episode_rows.append(
            {
                "candidate_id": int(row.candidate_id),
                "scenario_index": int(row.scenario_index),
                "day_index": int(row.t0 // STEPS),
                "t0": int(row.t0),
                "pv_s_scale": float(row.pv_s_scale),
                "svc_q_scale": float(row.svc_q_scale),
                "cap_total_mvar": float(row.cap_total_mvar),
                "risk": int(bool(state["under_steps"] or state["over_steps"] or state["pf_fail"])),
                "under_steps": int(state["under_steps"]),
                "over_steps": int(state["over_steps"]),
                "pf_fail": int(state["pf_fail"]),
                "daily_loss_mwh": float(state["loss"]),
                "vmin": float(state["vmin"]),
                "vmax": float(state["vmax"]),
                "solve_total_seconds": float(state["solve_seconds"]),
                "inherited_steps": int(state["inherited_steps"]),
                "direct_fallback_steps": int(state["direct_fallback_steps"]),
            }
        )
    return episode_rows, step_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lower_voltage", type=float, default=0.95)
    parser.add_argument("--upper_voltage", type=float, default=1.05)
    parser.add_argument("--load_profile", type=Path, default=Path("load96.npy"))
    parser.add_argument("--generation_profile", type=Path, default=Path("gen96.npy"))
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument(
        "--cap_buses",
        default="20,8",
        help="Comma-separated fixed-shunt buses; ignored when capacity is zero.",
    )
    parser.add_argument(
        "--no_direct_fallback",
        action="store_true",
        help="Mark non-converged AC-OPF steps as failed instead of fitting a direct SLSQP fallback.",
    )
    args = parser.parse_args()
    cap_buses = parse_ints(args.cap_buses)
    jobs = pd.read_csv(args.jobs_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "day_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    outputs = []
    for scenario_index, group in jobs.groupby("scenario_index", sort=True):
        episode_path = shard_dir / f"scenario_{int(scenario_index):04d}_episodes.json"
        steps_path = shard_dir / f"scenario_{int(scenario_index):04d}_steps.csv"
        if episode_path.exists() and steps_path.exists():
            outputs.append(
                (
                    json.loads(episode_path.read_text(encoding="utf-8")),
                    pd.read_csv(steps_path).to_dict("records"),
                )
            )
            continue
        payload = (
            group.to_dict("records"),
            args.lower_voltage,
            args.upper_voltage,
            str(args.load_profile.resolve()),
            str(args.generation_profile.resolve()),
            args.svc_absorption_ratio,
            args.env,
            cap_buses,
            not args.no_direct_fallback,
        )
        groups.append((int(scenario_index), payload))
    # Pandapower and SciPy keep native solver state. Recycling each Windows
    # worker after one day prevents long-lived process-pool failures.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        max_tasks_per_child=1,
    ) as executor:
        futures = {
            executor.submit(_solve_day, payload): scenario_index
            for scenario_index, payload in groups
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            output = future.result()
            scenario_index = futures[future]
            episode_path = shard_dir / f"scenario_{scenario_index:04d}_episodes.json"
            steps_path = shard_dir / f"scenario_{scenario_index:04d}_steps.csv"
            temporary_episode = episode_path.with_suffix(".json.tmp")
            temporary_steps = steps_path.with_suffix(".csv.tmp")
            temporary_episode.write_text(
                json.dumps(output[0], indent=2), encoding="utf-8"
            )
            pd.DataFrame(output[1]).to_csv(temporary_steps, index=False)
            temporary_episode.replace(episode_path)
            temporary_steps.replace(steps_path)
            outputs.append(output)
            print(
                f"completed {completed}/{len(groups)} missing day trajectories",
                flush=True,
            )
    outputs.sort(key=lambda output: int(output[0][0]["scenario_index"]))
    episodes = pd.DataFrame(row for output in outputs for row in output[0])
    steps = pd.DataFrame(row for output in outputs for row in output[1])
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    steps.to_csv(args.out_dir / "steps.csv", index=False)

    by_capacity = episodes.groupby(
        ["candidate_id", "pv_s_scale", "svc_q_scale", "cap_total_mvar"], as_index=False
    ).agg(
        evaluated_days=("scenario_index", "nunique"),
        risk=("risk", "mean"),
        mean_daily_loss_mwh=("daily_loss_mwh", "mean"),
        max_daily_loss_mwh=("daily_loss_mwh", "max"),
        worst_vmin=("vmin", "min"),
        worst_vmax=("vmax", "max"),
        mean_solve_seconds=("solve_total_seconds", "mean"),
        inherited_steps=("inherited_steps", "sum"),
        direct_fallback_steps=("direct_fallback_steps", "sum"),
    )
    by_capacity["loss_increase_from_previous_mwh"] = by_capacity.mean_daily_loss_mwh.diff()
    by_capacity.to_csv(args.out_dir / "capacity_summary.csv", index=False)
    capacity_count = int(episodes.candidate_id.nunique())
    day_status = episodes.groupby("scenario_index").agg(
        capacity_levels=("candidate_id", "nunique"),
        pf_failures=("pf_fail", "sum"),
    )
    matched_days = day_status.index[
        (day_status.capacity_levels == capacity_count) & (day_status.pf_failures == 0)
    ]
    matched_episodes = episodes.loc[episodes.scenario_index.isin(matched_days)]
    matched_by_capacity = (
        matched_episodes.groupby(
            ["candidate_id", "pv_s_scale", "svc_q_scale", "cap_total_mvar"],
            as_index=False,
        )
        .agg(
            evaluated_days=("scenario_index", "nunique"),
            mean_daily_loss_mwh=("daily_loss_mwh", "mean"),
            worst_vmin=("vmin", "min"),
            worst_vmax=("vmax", "max"),
        )
        .sort_values("candidate_id")
    )
    matched_by_capacity["loss_increase_from_previous_mwh"] = (
        matched_by_capacity.mean_daily_loss_mwh.diff()
    )
    matched_by_capacity.to_csv(
        args.out_dir / "capacity_summary_matched_days.csv", index=False
    )
    adjacent_pairs = adjacent_pair_loss_summary(episodes)
    adjacent_pairs.to_csv(
        args.out_dir / "adjacent_pair_loss_summary.csv", index=False
    )
    monotone = None
    maximum_positive_loss_increase_mwh = None
    if len(matched_days) > 0 and len(matched_by_capacity) == capacity_count:
        (
            monotone,
            maximum_positive_loss_increase_mwh,
        ) = loss_is_nonincreasing_with_tolerance(
            matched_by_capacity.mean_daily_loss_mwh
        )
    summary = {
        "days": int(episodes.scenario_index.nunique()),
        "capacity_levels": capacity_count,
        "all_capacity_converged_matched_days": int(len(matched_days)),
        "monotone_nonincreasing_loss": monotone,
        "loss_monotonicity_tolerance_mwh": LOSS_MONOTONICITY_ATOL_MWH,
        "loss_monotonicity_relative_tolerance": LOSS_MONOTONICITY_RTOL,
        "maximum_positive_loss_increase_mwh": maximum_positive_loss_increase_mwh,
        "all_adjacent_pairs_nonincreasing_within_tolerance": bool(
            not adjacent_pairs.empty
            and adjacent_pairs.nonincreasing_within_tolerance.all()
        ),
        "minimum_adjacent_pair_matched_days": (
            0
            if adjacent_pairs.empty
            else int(adjacent_pairs.paired_converged_days.min())
        ),
        "risk": float(episodes.risk.mean()),
        "direct_fallback_steps": int(episodes.direct_fallback_steps.sum()),
        "inherited_steps": int(episodes.inherited_steps.sum()),
        "load_profile": str(args.load_profile),
        "generation_profile": str(args.generation_profile),
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "environment": args.env,
        "capacitor_buses": cap_buses,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
