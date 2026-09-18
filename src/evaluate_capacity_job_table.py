"""Evaluate raw or safety-projected controllers on a capacity/day job table."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

if not hasattr(np, "Inf"):
    # Compatibility for the NumPy-2 checkpoint environment and pandapower 2.10.
    np.Inf = np.inf
    np.NaN = np.nan
    np.float_ = np.float64
    fallback = Path.home() / "anaconda3" / "Lib" / "site-packages"
    if fallback.exists():
        sys.path.append(str(fallback))
if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

import Env
from evaluate_crdc_policy import device_ids, load_actor, load_profiles_from_cfg
from physics_safety_projection import ACPowerFlowSafetyProjector


STEPS = 96
DT_HOURS = 0.25
_ACTOR = None
_CFG = None
_PROJECTOR = None
_LOAD = None
_GENERATION = None
_PROJECTED = False


def _initialise_worker(
    checkpoint_path: str, projected: bool, projection_config: dict
) -> None:
    global _ACTOR, _CFG, _PROJECTOR, _LOAD, _GENERATION, _PROJECTED
    torch.set_num_threads(1)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _CFG = payload["cfg"]
    _LOAD, _GENERATION = load_profiles_from_cfg(_CFG, Path(checkpoint_path))
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    dummy = Env.grid_case(env_id, _LOAD, _GENERATION, id_iber, id_svc)
    _ACTOR, _ = load_actor(
        Path(checkpoint_path).parent.parent,
        len(dummy.observation_space),
        len(dummy.action_space),
        torch.device("cpu"),
    )
    _PROJECTED = bool(projected)
    _PROJECTOR = ACPowerFlowSafetyProjector(**projection_config) if projected else None


def _evaluate_job(job: tuple[int, int, float, float, float, int]) -> dict:
    candidate_id, scenario_index, pv_scale, svc_scale, cap_total, t0 = job
    assert _ACTOR is not None and _CFG is not None
    assert _LOAD is not None and _GENERATION is not None
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    cap_buses = list(_CFG.get("cap_buses") or [])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    env = Env.grid_case(
        env_id,
        _LOAD,
        _GENERATION,
        id_iber,
        id_svc,
        enable_pq_curve=bool(_CFG["enable_pq_curve"]),
        pv_s_scale=pv_scale,
        svc_q_scale=svc_scale,
        svc_absorption_ratio=float(_CFG.get("svc_absorption_ratio", 0.0)),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization=str(_CFG.get("action_parameterization", "relative")),
        audit_power_balance=True,
    )
    obs = env.reset_at_step(t0)
    theta = torch.tensor([pv_scale, svc_scale, cap_total], dtype=torch.float32)
    under_steps = over_steps = pf_fail = 0
    intervention_steps = projection_failures = previews = 0
    projection_iterations = 0
    reporting_margin_steps = projection_margin_steps = 0
    corrections: list[float] = []
    actor_seconds: list[float] = []
    projection_seconds: list[float] = []
    preview_seconds: list[float] = []
    optimiser_seconds: list[float] = []
    execution_seconds: list[float] = []
    balance_audit_seconds: list[float] = []
    loss_energy_mwh = 0.0
    vmin = np.inf
    vmax = -np.inf
    raw_vmin = np.inf
    raw_vmax = -np.inf
    max_p_balance_residual_mw = 0.0
    max_q_balance_residual_mvar = 0.0

    for _ in range(STEPS):
        actor_start = perf_counter()
        with torch.no_grad():
            raw_action, _ = _ACTOR._pi(torch.tensor(obs, dtype=torch.float32), theta)
        actor_seconds.append(perf_counter() - actor_start)
        action = raw_action.numpy()
        try:
            if _PROJECTED:
                assert _PROJECTOR is not None
                projection = _PROJECTOR.project(env, action)
                action = projection.action
                intervention_steps += int(projection.intervened)
                projection_failures += int(not projection.success)
                previews += int(projection.power_flow_previews)
                projection_iterations += int(projection.iterations)
                corrections.append(float(projection.correction_norm))
                projection_seconds.append(float(projection.total_seconds))
                preview_seconds.append(float(projection.preview_seconds))
                optimiser_seconds.append(float(projection.optimiser_seconds))
                raw_vmin = min(raw_vmin, float(projection.raw_vmin))
                raw_vmax = max(raw_vmax, float(projection.raw_vmax))
                reporting_margin_steps += int(
                    min(projection.projected_vmin - 0.95, 1.05 - projection.projected_vmax)
                    <= 0.001
                )
                projection_margin_steps += int(
                    min(
                        projection.projected_vmin - _PROJECTOR.lower_voltage,
                        _PROJECTOR.upper_voltage - projection.projected_vmax,
                    )
                    <= 0.001
                )
            execution_start = perf_counter()
            next_obs, _, _, _, _, _, step_vmax, step_vmin, grid_loss, new_state = env.step_model(
                action
            )
            measured_execution = perf_counter() - execution_start
            balance_audit_seconds.append(float(env.last_balance_audit_seconds))
            execution_seconds.append(
                max(0.0, measured_execution - float(env.last_balance_audit_seconds))
            )
            max_p_balance_residual_mw = max(
                max_p_balance_residual_mw,
                float(env.last_p_balance_residual_mw),
            )
            max_q_balance_residual_mvar = max(
                max_q_balance_residual_mvar,
                float(env.last_q_balance_residual_mvar),
            )
        except Exception:
            pf_fail = 1
            break
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        vmin = min(vmin, float(step_vmin))
        vmax = max(vmax, float(step_vmax))
        loss_energy_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
        obs = new_state

    if not _PROJECTED:
        raw_vmin, raw_vmax = vmin, vmax
    completed_steps = min(len(actor_seconds), len(execution_seconds))
    online_seconds = [
        actor_seconds[index]
        + (projection_seconds[index] if index < len(projection_seconds) else 0.0)
        + execution_seconds[index]
        for index in range(completed_steps)
    ]
    return {
        "candidate_id": int(candidate_id),
        "scenario_index": int(scenario_index),
        "day_index": int(t0 // STEPS),
        "t0": int(t0),
        "pv_s_scale": float(pv_scale),
        "svc_q_scale": float(svc_scale),
        "cap_total_mvar": float(cap_total),
        "risk": int(bool(under_steps or over_steps or pf_fail)),
        "under_risk": int(bool(under_steps or pf_fail)),
        "over_risk": int(bool(over_steps or pf_fail)),
        "under_steps": int(under_steps),
        "over_steps": int(over_steps),
        "pf_fail": int(pf_fail),
        "daily_loss_mwh": float(loss_energy_mwh),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "raw_vmin": float(raw_vmin),
        "raw_vmax": float(raw_vmax),
        "intervention_steps": int(intervention_steps),
        "intervention_rate": float(intervention_steps / STEPS),
        "projection_failures": int(projection_failures),
        "mean_correction_norm": float(np.mean(corrections)) if corrections else 0.0,
        "max_correction_norm": float(np.max(corrections)) if corrections else 0.0,
        "power_flow_previews": int(previews),
        "projection_iterations": int(projection_iterations),
        "steps_within_0_001_of_reporting_limit": int(reporting_margin_steps),
        "steps_within_0_001_of_projection_guard": int(projection_margin_steps),
        "actor_total_seconds": float(np.sum(actor_seconds)),
        "actor_mean_ms": float(1000.0 * np.mean(actor_seconds)) if actor_seconds else 0.0,
        "projection_total_seconds": float(np.sum(projection_seconds)),
        "projection_mean_ms": float(1000.0 * np.mean(projection_seconds)) if projection_seconds else 0.0,
        "preview_total_seconds": float(np.sum(preview_seconds)),
        "optimiser_total_seconds": float(np.sum(optimiser_seconds)),
        "execution_total_seconds": float(np.sum(execution_seconds)),
        "execution_mean_ms": float(1000.0 * np.mean(execution_seconds)) if execution_seconds else 0.0,
        "balance_audit_total_seconds": float(np.sum(balance_audit_seconds)),
        "online_path_total_seconds": float(
            np.sum(actor_seconds) + np.sum(projection_seconds) + np.sum(execution_seconds)
        ),
        "completed_steps": int(completed_steps),
        "max_p_balance_residual_mw": float(max_p_balance_residual_mw),
        "max_q_balance_residual_mvar": float(max_q_balance_residual_mvar),
        "_actor_step_ms": [1000.0 * value for value in actor_seconds[:completed_steps]],
        "_projection_step_ms": (
            [1000.0 * value for value in projection_seconds[:completed_steps]]
            if projection_seconds
            else [0.0] * completed_steps
        ),
        "_execution_step_ms": [
            1000.0 * value for value in execution_seconds[:completed_steps]
        ],
        "_online_step_ms": [1000.0 * value for value in online_seconds],
    }


def timing_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capacity/day controller evaluator")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--jobs_csv", required=True)
    parser.add_argument("--projected", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lower_voltage", type=float, default=0.9505)
    parser.add_argument("--upper_voltage", type=float, default=1.0495)
    parser.add_argument("--finite_difference_step", type=float, default=0.02)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--optimiser_tolerance", type=float, default=1e-9)
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
        )
        for row in source[required].itertuples(index=False)
    ]
    projection_config = {
        "lower_voltage": args.lower_voltage,
        "upper_voltage": args.upper_voltage,
        "finite_difference_step": args.finite_difference_step,
        "max_iterations": args.max_iterations,
        "optimiser_tolerance": args.optimiser_tolerance,
    }
    checkpoint = str(Path(args.run_dir) / "models" / "checkpoint.pth")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise_worker,
        initargs=(checkpoint, bool(args.projected), projection_config),
    ) as executor:
        rows = list(executor.map(_evaluate_job, jobs, chunksize=2))
    timing_columns = (
        "_actor_step_ms",
        "_projection_step_ms",
        "_execution_step_ms",
        "_online_step_ms",
    )
    timings = {
        column: np.asarray(
            [value for row in rows for value in row.pop(column)], dtype=float
        )
        for column in timing_columns
    }
    results = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "episodes.csv", index=False)
    summary = {
        "jobs": len(results),
        "projected": bool(args.projected),
        "risk": float(results["risk"].mean()),
        "pf_fail_rate": float(results["pf_fail"].mean()),
        "mean_daily_loss_mwh": float(results["daily_loss_mwh"].mean()),
        "mean_intervention_rate": float(results["intervention_rate"].mean()),
        "projection_failure_steps": int(results["projection_failures"].sum()),
        "projection_activation_steps": int(results["intervention_steps"].sum()),
        "projection_iterations": int(results["projection_iterations"].sum()),
        "power_flow_previews": int(results["power_flow_previews"].sum()),
        "steps_within_0_001_of_reporting_limit": int(
            results["steps_within_0_001_of_reporting_limit"].sum()
        ),
        "steps_within_0_001_of_projection_guard": int(
            results["steps_within_0_001_of_projection_guard"].sum()
        ),
        "actor_mean_ms": float(
            1000.0 * results["actor_total_seconds"].sum()
            / max(int(results["completed_steps"].sum()), 1)
        ),
        "projection_mean_ms": float(
            1000.0 * results["projection_total_seconds"].sum()
            / max(int(results["completed_steps"].sum()), 1)
        ),
        "execution_mean_ms": float(
            1000.0 * results["execution_total_seconds"].sum()
            / max(int(results["completed_steps"].sum()), 1)
        ),
        "balance_audit_mean_ms": float(
            1000.0 * results["balance_audit_total_seconds"].sum()
            / max(int(results["completed_steps"].sum()), 1)
        ),
        "online_path_mean_ms": float(
            1000.0 * results["online_path_total_seconds"].sum()
            / max(int(results["completed_steps"].sum()), 1)
        ),
        "completed_steps": int(results["completed_steps"].sum()),
        "timing_ms": {
            "actor": timing_distribution(timings["_actor_step_ms"]),
            "projection": timing_distribution(timings["_projection_step_ms"]),
            "ac_execution": timing_distribution(timings["_execution_step_ms"]),
            "online_total": timing_distribution(timings["_online_step_ms"]),
        },
        "worst_vmin": float(results["vmin"].min()),
        "worst_vmax": float(results["vmax"].max()),
        "max_p_balance_residual_mw": float(results["max_p_balance_residual_mw"].max()),
        "max_q_balance_residual_mvar": float(
            results["max_q_balance_residual_mvar"].max()
        ),
        "jobs_csv": str(Path(args.jobs_csv)),
        "projection_config": projection_config if args.projected else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
