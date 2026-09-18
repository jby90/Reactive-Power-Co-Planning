"""Evaluate nominal-model projection against a perturbed executed feeder."""

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
TOPOLOGY_RECONFIGURATIONS = {
    "none": None,
    "tie32_open18": (32, 18),
    "tie33_open11": (33, 11),
}
_ACTOR = None
_CFG = None
_PROJECTOR = None
_LOAD = None
_GENERATION = None


def _initialise_worker(checkpoint_path: str, projection_config: dict) -> None:
    global _ACTOR, _CFG, _PROJECTOR, _LOAD, _GENERATION
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
    _PROJECTOR = ACPowerFlowSafetyProjector(**projection_config)


def _build_env(
    pv_scale: float,
    svc_scale: float,
    cap_total: float,
    pv_generation_scale: float,
) -> Env.grid_case:
    assert _CFG is not None
    assert _LOAD is not None and _GENERATION is not None
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    cap_buses = list(_CFG.get("cap_buses") or [])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    return Env.grid_case(
        env_id,
        _LOAD,
        _GENERATION,
        id_iber,
        id_svc,
        enable_pq_curve=bool(_CFG["enable_pq_curve"]),
        pv_s_scale=pv_scale,
        pv_generation_scale=pv_generation_scale,
        svc_q_scale=svc_scale,
        svc_absorption_ratio=float(_CFG.get("svc_absorption_ratio", 0.0)),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization=str(_CFG.get("action_parameterization", "relative")),
    )


def _evaluate_job(job: tuple) -> tuple[dict, list[dict]]:
    (
        candidate_id,
        scenario_index,
        pv_scale,
        svc_scale,
        cap_total,
        t0,
        mismatch_id,
        mismatch_seed,
        r_std,
        x_std,
        load_std,
        pv_std,
        topology_reconfiguration,
    ) = job
    assert _ACTOR is not None and _CFG is not None and _PROJECTOR is not None
    rng = np.random.default_rng(int(mismatch_seed))
    nominal = _build_env(pv_scale, svc_scale, cap_total, 1.0)
    actual_pv_multiplier = float(np.clip(rng.normal(1.0, pv_std), 0.7, 1.3))
    actual = _build_env(pv_scale, svc_scale, cap_total, actual_pv_multiplier)
    r_multiplier = np.clip(
        rng.normal(1.0, r_std, len(actual.model.line)), 0.5, 1.5
    )
    x_multiplier = np.clip(
        rng.normal(1.0, x_std, len(actual.model.line)), 0.5, 1.5
    )
    load_multiplier = float(np.clip(rng.normal(1.0, load_std), 0.7, 1.3))
    actual.model.line.loc[:, "r_ohm_per_km"] *= r_multiplier
    actual.model.line.loc[:, "x_ohm_per_km"] *= x_multiplier
    topology = TOPOLOGY_RECONFIGURATIONS[str(topology_reconfiguration)]
    if topology is not None:
        close_line, open_line = topology
        actual.model.line.loc[close_line, "in_service"] = True
        actual.model.line.loc[open_line, "in_service"] = False
    actual.init_line_r_ohm_per_km = actual.model.line.r_ohm_per_km.copy()
    actual.init_line_x_ohm_per_km = actual.model.line.x_ohm_per_km.copy()
    actual.init_load_p_mw *= load_multiplier
    actual.init_load_q_mvar *= load_multiplier

    nominal_obs = nominal.reset_at_step(t0)
    actual_obs = actual.reset_at_step(t0)
    del nominal_obs
    theta = torch.tensor([pv_scale, svc_scale, cap_total], dtype=torch.float32)
    under_steps = over_steps = pf_fail = 0
    interventions = projection_failures = 0
    loss_energy = 0.0
    actual_vmin = np.inf
    actual_vmax = -np.inf
    steps: list[dict] = []

    for offset in range(STEPS):
        with torch.no_grad():
            raw_action, _ = _ACTOR._pi(torch.tensor(actual_obs, dtype=torch.float32), theta)
        try:
            projection = _PROJECTOR.project(nominal, raw_action.numpy())
            interventions += int(projection.intervened)
            projection_failures += int(not projection.success)
            actual_next, _, _, _, _, _, vmax, vmin, grid_loss, actual_new = actual.step_model(
                projection.action
            )
            _, _, _, _, _, _, _, _, _, nominal_new = nominal.step_model(projection.action)
        except Exception as exc:
            pf_fail = 1
            steps.append(
                {
                    "candidate_id": candidate_id,
                    "scenario_index": scenario_index,
                    "mismatch_id": mismatch_id,
                    "step": offset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        actual_voltage = np.asarray(actual_next[: len(actual.model.bus)], dtype=float)
        under = int(np.any(actual_voltage < 0.95))
        over = int(np.any(actual_voltage > 1.05))
        under_steps += under
        over_steps += over
        actual_vmin = min(actual_vmin, float(vmin))
        actual_vmax = max(actual_vmax, float(vmax))
        loss_energy += max(0.0, float(-grid_loss)) * DT_HOURS
        steps.append(
            {
                "candidate_id": candidate_id,
                "scenario_index": scenario_index,
                "mismatch_id": mismatch_id,
                "step": offset,
                "nominal_projected_vmin": projection.projected_vmin,
                "nominal_projected_vmax": projection.projected_vmax,
                "actual_vmin": float(vmin),
                "actual_vmax": float(vmax),
                "actual_margin_pu": min(float(vmin) - 0.95, 1.05 - float(vmax)),
                "intervened": int(projection.intervened),
                "projection_success": int(projection.success),
                "error": "",
            }
        )
        actual_obs = actual_new
        del nominal_new

    episode = {
        "candidate_id": int(candidate_id),
        "scenario_index": int(scenario_index),
        "mismatch_id": int(mismatch_id),
        "mismatch_seed": int(mismatch_seed),
        "pv_s_scale": float(pv_scale),
        "svc_q_scale": float(svc_scale),
        "cap_total_mvar": float(cap_total),
        "r_std": float(r_std),
        "x_std": float(x_std),
        "load_std": float(load_std),
        "pv_std": float(pv_std),
        "topology_reconfiguration": str(topology_reconfiguration),
        "mean_r_multiplier": float(np.mean(r_multiplier)),
        "mean_x_multiplier": float(np.mean(x_multiplier)),
        "load_multiplier": load_multiplier,
        "pv_multiplier": actual_pv_multiplier,
        "risk": int(bool(under_steps or over_steps or pf_fail)),
        "under_steps": int(under_steps),
        "over_steps": int(over_steps),
        "pf_fail": int(pf_fail),
        "projection_failures": int(projection_failures),
        "intervention_rate": float(interventions / STEPS),
        "daily_loss_mwh": float(loss_energy),
        "actual_vmin": float(actual_vmin),
        "actual_vmax": float(actual_vmax),
    }
    return episode, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mismatch_replicates", type=int, default=5)
    parser.add_argument("--mismatch_seed", type=int, default=2026091801)
    parser.add_argument("--r_std", type=float, default=0.05)
    parser.add_argument("--x_std", type=float, default=0.05)
    parser.add_argument("--load_std", type=float, default=0.02)
    parser.add_argument("--pv_std", type=float, default=0.02)
    parser.add_argument(
        "--topology_reconfiguration",
        choices=tuple(TOPOLOGY_RECONFIGURATIONS),
        default="none",
    )
    parser.add_argument("--lower_voltage", type=float, default=0.9505)
    parser.add_argument("--upper_voltage", type=float, default=1.0495)
    parser.add_argument("--finite_difference_step", type=float, default=0.02)
    args = parser.parse_args()
    source = pd.read_csv(args.jobs_csv)
    jobs = []
    for row in source.itertuples(index=False):
        for mismatch_id in range(args.mismatch_replicates):
            seed = args.mismatch_seed + int(row.scenario_index) * 1009 + mismatch_id
            jobs.append(
                (
                    int(row.candidate_id),
                    int(row.scenario_index),
                    float(row.pv_s_scale),
                    float(row.svc_q_scale),
                    float(row.cap_total_mvar),
                    int(row.t0),
                    mismatch_id,
                    seed,
                    args.r_std,
                    args.x_std,
                    args.load_std,
                    args.pv_std,
                    args.topology_reconfiguration,
                )
            )
    projection_config = {
        "lower_voltage": args.lower_voltage,
        "upper_voltage": args.upper_voltage,
        "finite_difference_step": args.finite_difference_step,
    }
    checkpoint = args.run_dir / "models" / "checkpoint.pth"
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise_worker,
        initargs=(str(checkpoint), projection_config),
    ) as executor:
        outputs = list(executor.map(_evaluate_job, jobs, chunksize=1))
    episodes = pd.DataFrame(row for output in outputs for row in [output[0]])
    steps = pd.DataFrame(row for output in outputs for row in output[1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    steps.to_csv(args.out_dir / "steps.csv", index=False)
    summary = {
        "jobs": len(episodes),
        "risk": float(episodes.risk.mean()),
        "pf_fail_rate": float(episodes.pf_fail.mean()),
        "projection_failure_steps": int(episodes.projection_failures.sum()),
        "mean_intervention_rate": float(episodes.intervention_rate.mean()),
        "mean_daily_loss_mwh": float(episodes.daily_loss_mwh.mean()),
        "worst_actual_vmin": float(episodes.actual_vmin.min()),
        "worst_actual_vmax": float(episodes.actual_vmax.max()),
        "mismatch": {
            "r_std": args.r_std,
            "x_std": args.x_std,
            "load_std": args.load_std,
            "pv_std": args.pv_std,
            "replicates": args.mismatch_replicates,
            "topology_reconfiguration": args.topology_reconfiguration,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
