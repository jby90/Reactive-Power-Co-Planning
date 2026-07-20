"""Evaluate CRDC-PPO on an existing matched theta/time-offset job table."""

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
from PPO_theta_crdc import ConcatDirectionalActorCritic
from PPO_theta_concat_corrected import BlindScalarActorCritic, ConcatScalarActorCritic


STEPS = 96
DT_HOURS = 0.25
_ACTOR = None
_CFG = None


def _device_ids(env_id: int):
    if env_id == 33:
        return [17, 21, 24], [32]
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    return [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]


def _initialise_worker(checkpoint_path: str) -> None:
    global _ACTOR, _CFG
    torch.set_num_threads(1)
    install_numpy_pickle_aliases()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _CFG = payload["cfg"]
    id_iber, id_svc = _device_ids(int(_CFG["env"]))
    dummy = Env.grid_case(
        int(_CFG["env"]),
        np.load(Env.DATA_DIR / "load96.npy"),
        np.load(Env.DATA_DIR / "gen96.npy"),
        id_iber,
        id_svc,
    )
    actor_class = {
        "CRDC-PPO": ConcatDirectionalActorCritic,
        "WG-CVaR-PPO": ConcatDirectionalActorCritic,
        "Concat-Scalar-PPO-Corrected": ConcatScalarActorCritic,
        "Blind-Scalar-PPO-Corrected": BlindScalarActorCritic,
    }.get(payload.get("algorithm"))
    if actor_class is None:
        raise ValueError(f"Unsupported checkpoint algorithm: {payload.get('algorithm')}")
    _ACTOR = actor_class(
        len(dummy.observation_space),
        len(dummy.action_space),
        [_CFG["pv_s_scale_min"], _CFG["svc_q_scale_min"], _CFG["cap_total_min"]],
        [_CFG["pv_s_scale_max"], _CFG["svc_q_scale_max"], _CFG["cap_total_max"]],
        log_std_init=_CFG["log_std_init"],
    )
    _ACTOR.load_state_dict(payload["ac"])
    _ACTOR.eval()


def _evaluate_job(job: tuple[int, float, float, float, int]) -> dict:
    index, pv_scale, svc_scale, cap_total, t0 = job
    assert _ACTOR is not None and _CFG is not None
    env_id = int(_CFG["env"])
    id_iber, id_svc = _device_ids(env_id)
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
    violation_steps = 0
    under_steps = 0
    over_steps = 0
    pf_fail = 0
    losses = []
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    for _ in range(STEPS):
        with torch.no_grad():
            action, _ = _ACTOR._pi(torch.tensor(obs, dtype=torch.float32), theta)
        try:
            next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = env.step_model(action.numpy())
        except Exception:
            pf_fail = 1
            break
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        under = int(np.any(voltage < 0.95))
        over = int(np.any(voltage > 1.05))
        under_steps += under
        over_steps += over
        violation_steps += int(bool(under or over))
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
        "risk": int(bool(violation_steps or pf_fail)),
        "under_risk": int(bool(under_steps or pf_fail)),
        "over_risk": int(bool(over_steps or pf_fail)),
        "viol_rate": violation_steps / STEPS,
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "loss_mean_mw": loss_mean,
        "energy_mwh": loss_mean * STEPS * DT_HOURS,
        "vmin": minimum_voltage,
        "vmax": maximum_voltage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched CRDC-PPO uniform/stress evaluation")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--jobs_csv", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.jobs_csv)
    required = ["pv_s_scale", "svc_q_scale", "cap_total_mvar", "t0"]
    jobs = [
        (index, float(row.pv_s_scale), float(row.svc_q_scale), float(row.cap_total_mvar), int(row.t0))
        for index, row in source[required].iterrows()
    ]
    checkpoint = str(Path(args.run_dir) / "models" / "checkpoint.pth")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_initialise_worker, initargs=(checkpoint,)
    ) as executor:
        rows = list(executor.map(_evaluate_job, jobs, chunksize=1))
    results = pd.DataFrame(rows).sort_values("job_index")
    summary = {
        "jobs": len(results),
        "risk": float(results["risk"].mean()),
        "under_risk": float(results["under_risk"].mean()),
        "over_risk": float(results["over_risk"].mean()),
        "pf_fail_rate": float(results["pf_fail"].mean()),
        "mean_violation_step_rate": float(results["viol_rate"].mean()),
        "mean_loss_mw": float(results["loss_mean_mw"].mean()),
        "mean_energy_mwh": float(results["energy_mwh"].mean()),
        "worst_vmin": float(results["vmin"].min()),
        "worst_vmax": float(results["vmax"].max()),
        "source_jobs_csv": str(Path(args.jobs_csv)),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "episodes.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
