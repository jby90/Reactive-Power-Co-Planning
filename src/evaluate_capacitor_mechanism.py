"""Trace electrical mechanisms for matched fixed-capacitor candidate pairs."""

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
_ACTOR = None
_CFG = None
_PROJECTOR = None
_LOAD = None
_PV = None


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 10000,
) -> tuple[float, float]:
    """Return a deterministic percentile CI for a paired-day mean."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    batch = 500
    for start in range(0, samples, batch):
        stop = min(start + batch, samples)
        draw = rng.integers(0, len(finite), size=(stop - start, len(finite)))
        means[start:stop] = finite[draw].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarise_paired_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarise capacitor-on minus capacitor-off changes by matched pair."""
    rows = []
    delta_columns = [column for column in deltas if column.startswith("delta_")]
    for pair_id, group in deltas.groupby("pair_id", sort=True):
        for metric_index, column in enumerate(delta_columns):
            values = group[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lower, upper = bootstrap_mean_ci(
                finite,
                seed=2026091900 + int(pair_id) * 100 + metric_index,
            )
            rows.append(
                {
                    "pair_id": int(pair_id),
                    "metric": column.removeprefix("delta_"),
                    "paired_days": int(len(finite)),
                    "mean_delta": float(np.mean(finite)),
                    "median_delta": float(np.median(finite)),
                    "bootstrap_95_ci_low": float(lower),
                    "bootstrap_95_ci_high": float(upper),
                    "fraction_negative": float(np.mean(finite < 0)),
                    "fraction_positive": float(np.mean(finite > 0)),
                    "fraction_zero": float(np.mean(finite == 0)),
                }
            )
    return pd.DataFrame(rows)


def _initialise_worker(checkpoint_path: str, projection_config: dict) -> None:
    global _ACTOR, _CFG, _PROJECTOR, _LOAD, _PV
    torch.set_num_threads(1)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _CFG = payload["cfg"]
    _LOAD, _PV = load_profiles_from_cfg(_CFG, Path(checkpoint_path))
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    dummy = Env.grid_case(env_id, _LOAD, _PV, id_iber, id_svc)
    _ACTOR, _ = load_actor(
        Path(checkpoint_path).parent.parent,
        len(dummy.observation_space),
        len(dummy.action_space),
        torch.device("cpu"),
    )
    _PROJECTOR = ACPowerFlowSafetyProjector(**projection_config)


def _evaluate_job(job: tuple) -> tuple[dict, list[dict]]:
    pair_id, candidate_id, scenario_index, pv_scale, svc_scale, cap_total, t0 = job
    assert _ACTOR is not None and _CFG is not None and _PROJECTOR is not None
    assert _LOAD is not None and _PV is not None
    env_id = int(_CFG["env"])
    id_iber, id_svc = device_ids(env_id)
    cap_buses = list(_CFG.get("cap_buses") or [])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total and cap_buses else []
    env = Env.grid_case(
        env_id,
        _LOAD,
        _PV,
        id_iber,
        id_svc,
        enable_pq_curve=bool(_CFG["enable_pq_curve"]),
        pv_s_scale=pv_scale,
        svc_q_scale=svc_scale,
        svc_absorption_ratio=float(_CFG.get("svc_absorption_ratio", 0.0)),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization=str(_CFG.get("action_parameterization", "relative")),
        audit_physical_trace=True,
    )
    obs = env.reset_at_step(t0)
    theta = torch.tensor([pv_scale, svc_scale, cap_total], dtype=torch.float32)
    rows = []
    pf_fail = 0
    for offset in range(STEPS):
        with torch.no_grad():
            raw_action, _ = _ACTOR._pi(torch.tensor(obs, dtype=torch.float32), theta)
        try:
            projection = _PROJECTOR.project(env, raw_action.numpy())
            next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = env.step_model(
                projection.action
            )
        except Exception as exc:
            pf_fail = 1
            rows.append(
                {
                    "pair_id": pair_id,
                    "candidate_id": candidate_id,
                    "scenario_index": scenario_index,
                    "step": offset,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
        trace = env.last_executed_trace
        sgen_q = np.asarray(trace["sgen_q_mvar"], dtype=float)
        pv_q = sgen_q[np.asarray(env._idx_iber_sgen, dtype=int)]
        svc_q = sgen_q[np.asarray(env._idx_svc_sgen, dtype=int)]
        ext_q = float(trace["ext_grid_q_mvar"])
        branch_q = 0.5 * float(
            np.abs(np.asarray(trace["line_q_from_mvar"], dtype=float)).sum()
            + np.abs(np.asarray(trace["line_q_to_mvar"], dtype=float)).sum()
        )
        rows.append(
            {
                "pair_id": int(pair_id),
                "candidate_id": int(candidate_id),
                "scenario_index": int(scenario_index),
                "step": offset,
                "cap_total_mvar": float(cap_total),
                "loss_mw": max(0.0, float(-grid_loss)),
                "vmin": float(vmin),
                "vmax": float(vmax),
                "pv_q_sum_mvar": float(pv_q.sum()),
                "pv_q_abs_sum_mvar": float(np.abs(pv_q).sum()),
                "pv_absorption_mvar": float(-np.minimum(pv_q, 0.0).sum()),
                "svc_q_sum_mvar": float(svc_q.sum()),
                "fixed_cap_injection_mvar": float(cap_total),
                "ext_grid_q_mvar": ext_q,
                "branch_abs_q_flow_mvar": branch_q,
                "sum_line_current_squared_ka2": float(
                    np.square(np.asarray(trace["line_i_ka"], dtype=float)).sum()
                ),
                "intervened": int(projection.intervened),
                "projection_success": int(projection.success),
                "correction_norm": float(projection.correction_norm),
                "error": "",
            }
        )
        obs = new_state
    valid = pd.DataFrame(rows)
    if "loss_mw" in valid:
        valid = valid[valid.loss_mw.notna()]
    return (
        {
            "pair_id": int(pair_id),
            "candidate_id": int(candidate_id),
            "scenario_index": int(scenario_index),
            "pv_s_scale": float(pv_scale),
            "svc_q_scale": float(svc_scale),
            "cap_total_mvar": float(cap_total),
            "pf_fail": int(pf_fail),
            "completed_steps": len(valid),
            "daily_loss_mwh": float(valid.loss_mw.sum() * DT_HOURS) if len(valid) else np.nan,
            "mean_ext_grid_q_mvar": float(valid.ext_grid_q_mvar.mean()) if len(valid) else np.nan,
            "mean_branch_abs_q_flow_mvar": float(valid.branch_abs_q_flow_mvar.mean()) if len(valid) else np.nan,
            "mean_line_current_squared_ka2": float(valid.sum_line_current_squared_ka2.mean()) if len(valid) else np.nan,
            "mean_pv_absorption_mvar": float(valid.pv_absorption_mvar.mean()) if len(valid) else np.nan,
            "mean_svc_q_mvar": float(valid.svc_q_sum_mvar.mean()) if len(valid) else np.nan,
            "worst_vmin": float(valid.vmin.min()) if len(valid) else np.nan,
            "worst_vmax": float(valid.vmax.max()) if len(valid) else np.nan,
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lower_voltage", type=float, default=0.9505)
    parser.add_argument("--upper_voltage", type=float, default=1.0495)
    parser.add_argument("--finite_difference_step", type=float, default=0.02)
    args = parser.parse_args()
    source = pd.read_csv(args.jobs_csv)
    required = [
        "pair_id", "candidate_id", "scenario_index", "pv_s_scale",
        "svc_q_scale", "cap_total_mvar", "t0",
    ]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Missing job columns: {missing}")
    jobs = [tuple(row) for row in source[required].itertuples(index=False, name=None)]
    checkpoint = args.run_dir / "models" / "checkpoint.pth"
    projection_config = {
        "lower_voltage": args.lower_voltage,
        "upper_voltage": args.upper_voltage,
        "finite_difference_step": args.finite_difference_step,
    }
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise_worker,
        initargs=(str(checkpoint), projection_config),
    ) as executor:
        outputs = list(executor.map(_evaluate_job, jobs, chunksize=1))
    episodes = pd.DataFrame(item[0] for item in outputs)
    steps = pd.DataFrame(row for item in outputs for row in item[1])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out_dir / "episodes.csv", index=False)
    steps.to_csv(args.out_dir / "steps.csv", index=False)
    metrics = [
        "daily_loss_mwh", "mean_ext_grid_q_mvar", "mean_branch_abs_q_flow_mvar",
        "mean_line_current_squared_ka2", "mean_pv_absorption_mvar",
        "mean_svc_q_mvar", "worst_vmin", "worst_vmax",
    ]
    paired = episodes.pivot_table(
        index=["pair_id", "scenario_index"], columns="cap_total_mvar", values=metrics
    )
    cap_levels = sorted(episodes.cap_total_mvar.unique())
    if len(cap_levels) != 2:
        raise ValueError("Mechanism analysis requires exactly two capacitor levels")
    deltas = pd.DataFrame(index=paired.index)
    for metric in metrics:
        deltas[f"delta_{metric}"] = paired[(metric, cap_levels[1])] - paired[(metric, cap_levels[0])]
    deltas = deltas.reset_index()
    deltas.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    pair_summary = summarise_paired_deltas(deltas)
    pair_summary.to_csv(args.out_dir / "pair_summary.csv", index=False)
    summary = {
        "jobs": len(episodes),
        "pairs": int(episodes.pair_id.nunique()),
        "days": int(episodes.scenario_index.nunique()),
        "capacitor_levels_mvar": cap_levels,
        "pf_failures": int(episodes.pf_fail.sum()),
        "mean_paired_deltas": {
            column: float(deltas[column].mean())
            for column in deltas
            if column.startswith("delta_")
        },
        "pair_summary_csv": "pair_summary.csv",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
