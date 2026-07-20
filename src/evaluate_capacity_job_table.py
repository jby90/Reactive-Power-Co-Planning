"""Evaluate raw or safety-projected controllers on a capacity/day job table."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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
from checkpoint_compat import install_numpy_pickle_aliases
from evaluate_crdc_policy import device_ids, load_actor
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
    _LOAD = np.load(Env.DATA_DIR / "load96.npy")
    _GENERATION = np.load(Env.DATA_DIR / "gen96.npy")
    install_numpy_pickle_aliases()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _CFG = payload["cfg"]
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
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
    )
    obs = env.reset_at_step(t0)
    theta = torch.tensor([pv_scale, svc_scale, cap_total], dtype=torch.float32)
    under_steps = over_steps = pf_fail = 0
    intervention_steps = projection_failures = previews = 0
    corrections: list[float] = []
    loss_energy_mwh = 0.0
    vmin = np.inf
    vmax = -np.inf
    raw_vmin = np.inf
    raw_vmax = -np.inf

    for _ in range(STEPS):
        with torch.no_grad():
            raw_action, _ = _ACTOR._pi(torch.tensor(obs, dtype=torch.float32), theta)
        action = raw_action.numpy()
        try:
            if _PROJECTED:
                assert _PROJECTOR is not None
                projection = _PROJECTOR.project(env, action)
                action = projection.action
                intervention_steps += int(projection.intervened)
                projection_failures += int(not projection.success)
                previews += int(projection.power_flow_previews)
                corrections.append(float(projection.correction_norm))
                raw_vmin = min(raw_vmin, float(projection.raw_vmin))
                raw_vmax = max(raw_vmax, float(projection.raw_vmax))
            next_obs, _, _, _, _, _, step_vmax, step_vmin, grid_loss, new_state = env.step_model(
                action
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
        "worst_vmin": float(results["vmin"].min()),
        "worst_vmax": float(results["vmax"].max()),
        "jobs_csv": str(Path(args.jobs_csv)),
        "projection_config": projection_config if args.projected else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
