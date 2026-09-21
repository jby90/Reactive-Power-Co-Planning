"""Centralized AC-OPF Volt-VAR baseline for matched planning scenarios.

The active-power injections, fixed capacitors, device locations, and reactive
limits match the learned-controller environment. The objective minimizes slack
active-power import with all controllable active injections fixed, which is
equivalent to minimizing network active-power loss for a fixed operating state.
"""

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
from scipy.optimize import Bounds, NonlinearConstraint, minimize

import Env
from vmod_evaluation import device_ids


STEPS = 96
DT_HOURS = 0.25


@lru_cache(maxsize=4)
def _load_profiles(load_path: str, generation_path: str) -> tuple[np.ndarray, np.ndarray]:
    load = np.load(load_path)
    generation = np.load(generation_path)
    if len(load) != len(generation):
        raise ValueError("Load and generation profiles must have equal time lengths")
    return load, generation


def _prepare_opf(net, lower_voltage: float, upper_voltage: float) -> None:
    net.bus.loc[:, "min_vm_pu"] = lower_voltage
    net.bus.loc[:, "max_vm_pu"] = upper_voltage
    net.sgen.loc[:, "controllable"] = True
    # Active PV injections are exogenous. Only reactive dispatch is optimized.
    net.sgen.loc[:, "min_p_mw"] = net.sgen.p_mw.to_numpy(dtype=float)
    net.sgen.loc[:, "max_p_mw"] = net.sgen.p_mw.to_numpy(dtype=float)
    if len(net.poly_cost):
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
    if len(net.pwl_cost):
        net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    pp.create_poly_cost(
        net,
        int(net.ext_grid.index[0]),
        "ext_grid",
        cp1_eur_per_mw=1.0,
    )


def _direct_ac_optimisation(
    env: Env.grid_case,
    lower_voltage: float,
    upper_voltage: float,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Nonlinear AC optimization directly in physical reactive-power units."""
    low_q = env.model.sgen.min_q_mvar.to_numpy(dtype=float)
    high_q = env.model.sgen.max_q_mvar.to_numpy(dtype=float)
    zero_q = np.clip(np.zeros_like(low_q), low_q, high_q)
    cache_q: np.ndarray | None = None
    cache_voltage: np.ndarray | None = None
    cache_loss = np.inf
    power_flow_calls = 0

    def evaluate(q_mvar: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal cache_q, cache_voltage, cache_loss, power_flow_calls
        candidate = np.clip(np.asarray(q_mvar, dtype=float), low_q, high_q)
        if cache_q is not None and np.array_equal(candidate, cache_q):
            assert cache_voltage is not None
            return cache_voltage, cache_loss
        env.model.sgen.loc[:, "q_mvar"] = candidate
        try:
            pp.runpp(env.model, algorithm="bfsw", numba=False)
            cache_voltage = env.model.res_bus.vm_pu.to_numpy(dtype=float).copy()
            cache_loss = float(env.model.res_line.pl_mw.sum())
        except Exception:
            # A deterministic infeasible surrogate lets SLSQP retreat from
            # extreme-Q points at which the load flow itself diverges.
            cache_voltage = np.zeros(len(env.model.bus), dtype=float)
            cache_loss = 1e3 + float(np.square(candidate).sum())
        power_flow_calls += 1
        cache_q = candidate.copy()
        return cache_voltage, cache_loss

    def objective(q_mvar: np.ndarray) -> float:
        return evaluate(q_mvar)[1]

    def voltage(q_mvar: np.ndarray) -> np.ndarray:
        return evaluate(q_mvar)[0]

    starts = [
        zero_q,
        np.clip(0.25 * high_q + 0.75 * zero_q, low_q, high_q),
        np.clip(0.25 * low_q + 0.75 * zero_q, low_q, high_q),
    ]
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    start_time = perf_counter()
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=Bounds(low_q, high_q),
            constraints=[NonlinearConstraint(voltage, lower_voltage, upper_voltage)],
            options={"ftol": 1e-9, "maxiter": 150, "disp": False},
        )
        candidate = np.clip(np.asarray(result.x, dtype=float), low_q, high_q)
        candidate_voltage, candidate_loss = evaluate(candidate)
        feasible = bool(
            candidate_voltage.min() >= lower_voltage - 1e-6
            and candidate_voltage.max() <= upper_voltage + 1e-6
        )
        if feasible and (best is None or candidate_loss < best[2]):
            best = candidate.copy(), candidate_voltage.copy(), candidate_loss
    elapsed = perf_counter() - start_time
    if best is None:
        raise RuntimeError("Both AC OPF and direct nonlinear AC optimization failed")
    q_mvar, final_voltage, final_loss = best
    env.model.sgen.loc[:, "q_mvar"] = q_mvar
    pp.runpp(env.model, algorithm="bfsw", numba=False)
    return q_mvar, final_voltage, final_loss, power_flow_calls + 1, elapsed


def _evaluate_job(
    job: tuple[int, int, float, float, float, int, float, float, str, str, float]
) -> tuple[dict, list[dict]]:
    (
        candidate_id,
        scenario_index,
        pv_scale,
        svc_scale,
        cap_total,
        t0,
        lower_voltage,
        upper_voltage,
        load_path,
        generation_path,
        svc_absorption_ratio,
    ) = job
    load, generation = _load_profiles(load_path, generation_path)
    id_iber, id_svc = device_ids(33)
    cap_buses = [20, 8]
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 else []
    env = Env.grid_case(
        33,
        load,
        generation,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=pv_scale,
        svc_q_scale=svc_scale,
        svc_absorption_ratio=svc_absorption_ratio,
        cap_buses=cap_buses if cap_q else None,
        cap_q_mvar=cap_q or None,
    )

    step_rows: list[dict] = []
    daily_loss_mwh = 0.0
    pf_fail = 0
    under_steps = over_steps = 0
    vmin = np.inf
    vmax = -np.inf
    solve_seconds = 0.0
    opf_fallback_steps = 0

    for offset in range(STEPS):
        time_index = int(t0 + offset)
        try:
            env.reset_at_step(time_index)
            _prepare_opf(env.model, lower_voltage, upper_voltage)
            solve_start = perf_counter()
            solver = "pandapower_ac_opf"
            power_flow_calls = 0
            try:
                pp.runopp(
                    env.model,
                    verbose=False,
                    calculate_voltage_angles=False,
                    init="pf",
                )
                voltage = env.model.res_bus.vm_pu.to_numpy(dtype=float)
                loss_mw = float(env.model.res_line.pl_mw.sum())
                feasible = bool(
                    env.model.OPF_converged
                    and voltage.min() >= lower_voltage - 1e-6
                    and voltage.max() <= upper_voltage + 1e-6
                )
                if not feasible:
                    raise RuntimeError("Built-in AC OPF returned an infeasible solution")
            except Exception:
                opf_fallback_steps += 1
                solver = "direct_slsqp_ac_pf"
                _, voltage, loss_mw, power_flow_calls, _ = _direct_ac_optimisation(
                    env, lower_voltage, upper_voltage
                )
            elapsed = perf_counter() - solve_start
            solve_seconds += elapsed
            step_vmin = float(voltage.min())
            step_vmax = float(voltage.max())
            q_dispatch = env.model.res_sgen.q_mvar.to_numpy(dtype=float)
        except Exception as exc:
            pf_fail = 1
            step_rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario_index": scenario_index,
                    "step": offset,
                    "time_index": time_index,
                    "converged": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

        under = int(step_vmin < 0.95 - 1e-8)
        over = int(step_vmax > 1.05 + 1e-8)
        under_steps += under
        over_steps += over
        vmin = min(vmin, step_vmin)
        vmax = max(vmax, step_vmax)
        daily_loss_mwh += max(0.0, loss_mw) * DT_HOURS
        step_rows.append(
            {
                "candidate_id": candidate_id,
                "scenario_index": scenario_index,
                "step": offset,
                "time_index": time_index,
                "converged": 1,
                "loss_mw": loss_mw,
                "vmin": step_vmin,
                "vmax": step_vmax,
                "solve_ms": elapsed * 1000.0,
                "solver": solver,
                "power_flow_calls": power_flow_calls,
                "pv_q_abs_mvar": float(np.abs(q_dispatch[: len(id_iber)]).sum()),
                "svc_q_mvar": float(q_dispatch[len(id_iber) :].sum()),
                "reporting_margin_pu": min(step_vmin - 0.95, 1.05 - step_vmax),
                "error": "",
            }
        )

    episode = {
        "candidate_id": candidate_id,
        "scenario_index": scenario_index,
        "day_index": int(t0 // STEPS),
        "t0": t0,
        "pv_s_scale": pv_scale,
        "svc_q_scale": svc_scale,
        "cap_total_mvar": cap_total,
        "risk": int(bool(under_steps or over_steps or pf_fail)),
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "daily_loss_mwh": daily_loss_mwh,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "solve_total_seconds": solve_seconds,
        "solve_mean_ms": 1000.0 * solve_seconds / max(1, len(step_rows)),
        "opf_fallback_steps": opf_fallback_steps,
    }
    return episode, step_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lower_voltage", type=float, default=0.95)
    parser.add_argument("--upper_voltage", type=float, default=1.05)
    parser.add_argument("--load_profile", type=Path, required=True)
    parser.add_argument("--generation_profile", type=Path, required=True)
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    args = parser.parse_args()

    source = pd.read_csv(args.jobs_csv)
    required = [
        "candidate_id",
        "scenario_index",
        "pv_s_scale",
        "svc_q_scale",
        "cap_total_mvar",
        "t0",
    ]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Missing job columns: {missing}")
    jobs = [
        (
            int(row.candidate_id),
            int(row.scenario_index),
            float(row.pv_s_scale),
            float(row.svc_q_scale),
            float(row.cap_total_mvar),
            int(row.t0),
            float(args.lower_voltage),
            float(args.upper_voltage),
            str(args.load_profile.resolve()),
            str(args.generation_profile.resolve()),
            float(args.svc_absorption_ratio),
        )
        for row in source[required].itertuples(index=False)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "job_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    def shard_paths(job: tuple) -> tuple[Path, Path]:
        candidate_id, scenario_index = int(job[0]), int(job[1])
        stem = f"candidate_{candidate_id:04d}_scenario_{scenario_index:04d}"
        return shard_dir / f"{stem}_episode.json", shard_dir / f"{stem}_steps.csv"

    outputs: list[tuple[dict, list[dict]]] = []
    missing_jobs = []
    for job in jobs:
        episode_path, steps_path = shard_paths(job)
        if episode_path.exists() and steps_path.exists():
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            step_rows = pd.read_csv(steps_path).to_dict("records")
            outputs.append((episode, step_rows))
        else:
            missing_jobs.append(job)

    if missing_jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            max_tasks_per_child=1,
        ) as executor:
            futures = {executor.submit(_evaluate_job, job): job for job in missing_jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                output = future.result()
                episode_path, steps_path = shard_paths(futures[future])
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
                    f"completed {completed}/{len(missing_jobs)} remaining AC-OPF jobs",
                    flush=True,
                )

    outputs.sort(key=lambda item: (item[0]["candidate_id"], item[0]["scenario_index"]))
    episodes = pd.DataFrame([item[0] for item in outputs])
    steps = pd.DataFrame(row for item in outputs for row in item[1])
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    steps.to_csv(args.out_dir / "steps.csv", index=False)
    summary = {
        "jobs": len(episodes),
        "risk": float(episodes.risk.mean()),
        "pf_fail_rate": float(episodes.pf_fail.mean()),
        "mean_daily_loss_mwh": float(episodes.daily_loss_mwh.mean()),
        "worst_vmin": float(episodes.vmin.min()),
        "worst_vmax": float(episodes.vmax.max()),
        "mean_solve_ms": float(steps.loc[steps.converged == 1, "solve_ms"].mean()),
        "p95_solve_ms": float(steps.loc[steps.converged == 1, "solve_ms"].quantile(0.95)),
        "max_solve_ms": float(steps.loc[steps.converged == 1, "solve_ms"].max()),
        "opf_fallback_steps": int(episodes.opf_fallback_steps.sum()),
        "lower_voltage": args.lower_voltage,
        "upper_voltage": args.upper_voltage,
        "load_profile": str(args.load_profile),
        "generation_profile": str(args.generation_profile),
        "svc_absorption_ratio": args.svc_absorption_ratio,
        "objective": "minimum slack active-power import with fixed active injections",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
