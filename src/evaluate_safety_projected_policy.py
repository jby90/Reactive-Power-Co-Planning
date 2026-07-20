"""Evaluate a deterministic policy with the frozen AC safety projection."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import Env
from checkpoint_compat import install_numpy_pickle_aliases
from evaluate_crdc_policy import device_ids, load_actor, parse_floats, parse_ints
from physics_safety_projection import ACPowerFlowSafetyProjector


STEPS = 96
DT_HOURS = 0.25
_ACTOR = None
_CFG = None
_PROJECTOR = None


def _initialise_worker(checkpoint_path: str, projection_config: dict) -> None:
    global _ACTOR, _CFG, _PROJECTOR
    torch.set_num_threads(1)
    install_numpy_pickle_aliases()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _CFG = payload["cfg"]
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    dummy = Env.grid_case(
        env_id,
        np.load(Env.DATA_DIR / "load96.npy"),
        np.load(Env.DATA_DIR / "gen96.npy"),
        id_iber,
        id_svc,
    )
    _ACTOR, _ = load_actor(
        Path(checkpoint_path).parent.parent,
        len(dummy.observation_space),
        len(dummy.action_space),
        torch.device("cpu"),
    )
    _PROJECTOR = ACPowerFlowSafetyProjector(**projection_config)


def _evaluate_job(job: tuple[int, float, float, float, int]) -> dict:
    index, pv_scale, svc_scale, cap_total, t0 = job
    assert _ACTOR is not None and _CFG is not None and _PROJECTOR is not None
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    cap_buses = list(_CFG.get("cap_buses") or [])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    env = Env.grid_case(
        env_id,
        np.load(Env.DATA_DIR / "load96.npy"),
        np.load(Env.DATA_DIR / "gen96.npy"),
        id_iber,
        id_svc,
        enable_pq_curve=bool(_CFG["enable_pq_curve"]),
        pv_s_scale=pv_scale,
        svc_q_scale=svc_scale,
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
    )
    obs = env.reset_at_step(t0)
    theta = torch.tensor([pv_scale, svc_scale, cap_total], dtype=torch.float32)
    under_steps = over_steps = pf_fail = 0
    interventions = projection_failures = previews = 0
    correction_norms: list[float] = []
    losses: list[float] = []
    minimum_voltage, maximum_voltage = np.inf, -np.inf
    raw_minimum_voltage, raw_maximum_voltage = np.inf, -np.inf

    for _ in range(STEPS):
        with torch.no_grad():
            raw_action, _ = _ACTOR._pi(torch.tensor(obs, dtype=torch.float32), theta)
        try:
            projection = _PROJECTOR.project(env, raw_action.numpy())
            interventions += int(projection.intervened)
            projection_failures += int(not projection.success)
            previews += projection.power_flow_previews
            correction_norms.append(projection.correction_norm)
            raw_minimum_voltage = min(raw_minimum_voltage, projection.raw_vmin)
            raw_maximum_voltage = max(raw_maximum_voltage, projection.raw_vmax)
            next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = env.step_model(
                projection.action
            )
        except Exception:
            pf_fail = 1
            break
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        losses.append(max(0.0, float(-grid_loss)))
        minimum_voltage = min(minimum_voltage, float(vmin))
        maximum_voltage = max(maximum_voltage, float(vmax))
        obs = new_state

    loss_mean = float(np.mean(losses)) if losses else np.nan
    return {
        "job_index": index,
        "pv_s_scale": pv_scale,
        "svc_q_scale": svc_scale,
        "cap_total_mvar": cap_total,
        "t0": t0,
        "risk": int(bool(under_steps or over_steps or pf_fail)),
        "under_risk": int(bool(under_steps or pf_fail)),
        "over_risk": int(bool(over_steps or pf_fail)),
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "loss_mean_mw": loss_mean,
        "energy_mwh": loss_mean * STEPS * DT_HOURS,
        "vmin": minimum_voltage,
        "vmax": maximum_voltage,
        "raw_vmin": raw_minimum_voltage,
        "raw_vmax": raw_maximum_voltage,
        "intervention_steps": interventions,
        "intervention_rate": interventions / STEPS,
        "projection_failures": projection_failures,
        "mean_correction_norm": float(np.mean(correction_norms)),
        "max_correction_norm": float(np.max(correction_norms)),
        "power_flow_previews": previews,
    }


def _jobs_from_args(args: argparse.Namespace) -> list[tuple[int, float, float, float, int]]:
    if args.jobs_csv:
        source = pd.read_csv(args.jobs_csv)
        required = ["pv_s_scale", "svc_q_scale", "cap_total_mvar", "t0"]
        return [
            (
                index,
                float(row.pv_s_scale),
                float(row.svc_q_scale),
                float(row.cap_total_mvar),
                int(row.t0),
            )
            for index, row in source[required].iterrows()
        ]
    theta = parse_floats(args.theta)
    if len(theta) != 3:
        raise ValueError("--theta must have three values")
    metadata = json.loads(Path(args.days_metadata).read_text(encoding="utf-8"))
    return [
        (index, theta[0], theta[1], theta[2], int(day) * STEPS)
        for index, day in enumerate(metadata["final_day_indices"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="AC safety-projected policy evaluation")
    parser.add_argument("--run_dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jobs_csv")
    source.add_argument("--days_metadata")
    parser.add_argument("--theta", default="0.7,0.7,0.25")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lower_voltage", type=float, default=0.9505)
    parser.add_argument("--upper_voltage", type=float, default=1.0495)
    parser.add_argument("--finite_difference_step", type=float, default=0.02)
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--optimiser_tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    jobs = _jobs_from_args(args)
    checkpoint = str(Path(args.run_dir) / "models" / "checkpoint.pth")
    projection_config = {
        "lower_voltage": args.lower_voltage,
        "upper_voltage": args.upper_voltage,
        "finite_difference_step": args.finite_difference_step,
        "max_iterations": args.max_iterations,
        "optimiser_tolerance": args.optimiser_tolerance,
    }
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise_worker,
        initargs=(checkpoint, projection_config),
    ) as executor:
        rows = list(executor.map(_evaluate_job, jobs, chunksize=1))
    results = pd.DataFrame(rows).sort_values("job_index")
    summary = {
        "jobs": len(results),
        "risk": float(results["risk"].mean()),
        "under_risk": float(results["under_risk"].mean()),
        "over_risk": float(results["over_risk"].mean()),
        "pf_fail_rate": float(results["pf_fail"].mean()),
        "mean_loss_mw": float(results["loss_mean_mw"].mean()),
        "mean_energy_mwh": float(results["energy_mwh"].mean()),
        "worst_vmin": float(results["vmin"].min()),
        "worst_vmax": float(results["vmax"].max()),
        "raw_worst_vmin": float(results["raw_vmin"].min()),
        "raw_worst_vmax": float(results["raw_vmax"].max()),
        "mean_intervention_rate": float(results["intervention_rate"].mean()),
        "jobs_with_intervention": int((results["intervention_steps"] > 0).sum()),
        "projection_failure_steps": int(results["projection_failures"].sum()),
        "mean_correction_norm": float(results["mean_correction_norm"].mean()),
        "max_correction_norm": float(results["max_correction_norm"].max()),
        "mean_power_flow_previews_per_job": float(results["power_flow_previews"].mean()),
        "projection_config": projection_config,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "episodes.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
