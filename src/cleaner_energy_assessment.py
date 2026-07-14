"""Evaluate resource-efficient PV integration with adaptive and local Volt-VAR control.

The script reports only quantities directly produced by the electrical simulation:
day-level empirical risk, physical line-loss energy (MWh), accepted PV injection,
and a user-defined normalised asset-cost index. It deliberately does not infer
carbon emissions or life-cycle impacts from these operational metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import Env
from PPO_theta_robust_film_curriculum import FiLMActorCritic
from PPO_theta_robust import ActorCritic as MLPActorCritic


DT_HOURS = 0.25
STEPS_PER_DAY = 96


def _parse_float_list(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _device_ids(env_id: int) -> tuple[List[int], List[int]]:
    if env_id == 33:
        return [17, 21, 24], [32]
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    if env_id == 118:
        return [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]
    raise ValueError(f"Unsupported environment: {env_id}")


def _infer_film_architecture(state_dict: dict) -> tuple[int, int, int]:
    return (
        int(state_dict["pi_fc1.weight"].shape[0]),
        int(state_dict["pi_fc2.weight"].shape[0]),
        int(state_dict["pi_film1.scale.weight"].shape[1]),
    )


def _resolve_actor_path(run_dir: Path) -> Path:
    """Accept either a released model directory or a legacy training-run directory."""
    direct = run_dir / "actor.pth"
    return direct if direct.exists() else run_dir / "models" / "actor.pth"


def _load_film_actor(run_dir: Path, obs_dim: int, act_dim: int, device: torch.device) -> FiLMActorCritic:
    actor_path = _resolve_actor_path(run_dir)
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
    payload = torch.load(actor_path, map_location=device)
    state_dict = payload.get("ac", payload) if isinstance(payload, dict) else payload
    h1, h2, theta_dim = _infer_film_architecture(state_dict)
    actor = FiLMActorCritic(obs_dim, act_dim, theta_dim=theta_dim, hidden=(h1, h2))
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor


def _load_mlp_actor(run_dir: Path, obs_dim: int, act_dim: int, device: torch.device) -> MLPActorCritic:
    actor_path = _resolve_actor_path(run_dir)
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
    payload = torch.load(actor_path, map_location=device)
    state_dict = payload.get("ac", payload) if isinstance(payload, dict) else payload
    hidden = (int(state_dict["pi_fc1.weight"].shape[0]), int(state_dict["pi_fc2.weight"].shape[0]))
    actor = MLPActorCritic(obs_dim, act_dim, hidden=hidden)
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor


def _make_env(
    env_id: int,
    load_pu: np.ndarray,
    gene_pu: np.ndarray,
    id_iber: List[int],
    id_svc: List[int],
    *,
    theta: np.ndarray,
    pv_generation_scale: float,
    cap_buses: List[int],
) -> Env.grid_case:
    cap_total = float(theta[2])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    return Env.grid_case(
        env_id,
        load_pu,
        gene_pu,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=float(theta[0]),
        pv_generation_scale=float(pv_generation_scale),
        svc_q_scale=float(theta[1]),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
    )


def _film_action(actor: FiLMActorCritic, obs: np.ndarray, theta: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
        action, _ = actor._pi(obs_t, theta_t)
    return action.detach().cpu().numpy()


def _mlp_action(actor: MLPActorCritic, obs: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, _ = actor._pi(obs_t)
    return action.detach().cpu().numpy()


def _droop_action(obs: np.ndarray, device_buses: List[int], n_bus: int) -> np.ndarray:
    voltage = np.asarray(obs[:n_bus], dtype=float)
    actions = []
    for bus in device_buses:
        v = float(voltage[bus])
        if v < 0.98:
            action = 1.0
        elif v < 1.00:
            action = (1.00 - v) / 0.02
        elif v <= 1.00:
            action = 0.0
        elif v <= 1.02:
            action = -(v - 1.00) / 0.02
        else:
            action = -1.0
        actions.append(float(np.clip(action, -1.0, 1.0)))
    return np.asarray(actions, dtype=float)


@dataclass(frozen=True)
class Metrics:
    pr_violation_any: float
    pr_pf_fail_any: float
    daily_loss_mwh: float
    daily_pv_energy_mwh: float


def _run_day(
    env: Env.grid_case,
    controller: str,
    actors: dict[str, torch.nn.Module],
    theta: np.ndarray,
    day_idx: int,
    device: torch.device,
    device_buses: List[int],
) -> tuple[int, int, float, float]:
    obs = env.reset_at_step(int(day_idx) * STEPS_PER_DAY)
    violation_any = 0
    pf_fail_any = 0
    loss_mwh = 0.0
    pv_energy_mwh = 0.0

    for _ in range(STEPS_PER_DAY):
        if controller == "film":
            action = _film_action(actors["film"], obs, theta, device)  # type: ignore[arg-type]
        elif controller == "blind":
            action = _mlp_action(actors["blind"], obs, device)  # type: ignore[arg-type]
        elif controller == "concat":
            action = _mlp_action(actors["concat"], np.concatenate([obs, theta]), device)  # type: ignore[arg-type]
        elif controller == "droop":
            action = _droop_action(obs, device_buses, len(env.model.bus))
        else:
            raise ValueError(f"Unknown controller: {controller}")
        try:
            next_obs, _, _, _, _, _, _, _, grid_loss, new_state = env.step_model(action)
        except Exception:
            pf_fail_any = 1
            violation_any = 1
            break
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        violation_any |= int(np.any(voltage < 0.95) or np.any(voltage > 1.05))
        loss_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
        pv_power = np.asarray(env.model.sgen.loc[env._idx_iber_sgen, "p_mw"], dtype=float)
        pv_energy_mwh += float(np.maximum(pv_power, 0.0).sum()) * DT_HOURS
        obs = new_state
    return violation_any, pf_fail_any, loss_mwh, pv_energy_mwh


def _evaluate(
    env: Env.grid_case,
    controller: str,
    actors: dict[str, torch.nn.Module],
    theta: np.ndarray,
    days: np.ndarray,
    device: torch.device,
    device_buses: List[int],
) -> Metrics:
    values = [_run_day(env, controller, actors, theta, int(day), device, device_buses) for day in days]
    arr = np.asarray(values, dtype=float)
    return Metrics(
        pr_violation_any=float(arr[:, 0].mean()),
        pr_pf_fail_any=float(arr[:, 1].mean()),
        daily_loss_mwh=float(arr[:, 2].mean()),
        daily_pv_energy_mwh=float(arr[:, 3].mean()),
    )


def _cost_index(theta: np.ndarray, costs: np.ndarray) -> float:
    return float(np.dot(theta, costs))


def _safe(metrics: Metrics, eps_v: float, eps_pf: float) -> bool:
    return metrics.pr_violation_any <= eps_v + 1e-12 and metrics.pr_pf_fail_any <= eps_pf + 1e-12


def _plot_results(best: pd.DataFrame, out_dir: Path) -> None:
    labels = {
        "film": "FiLM-PPO co-planning",
        "blind": "Blind PPO planning",
        "concat": "Concat PPO planning",
        "droop": "Volt-VAR droop planning",
    }
    colors = {"film": "#176B87", "blind": "#6A994E", "concat": "#8E5EA2", "droop": "#C8543C"}
    feasible = best[best["feasible"] == 1].copy()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    controller_order = [c for c in ("film", "blind", "concat", "droop") if c in set(best["controller"])]
    for y, controller in enumerate(controller_order):
        data = best[best["controller"] == controller].sort_values("pv_generation_scale")
        feasible_data = data[data["feasible"] == 1]
        infeasible_data = data[data["feasible"] == 0]
        if len(feasible_data):
            ax.scatter(feasible_data["pv_generation_scale"], np.full(len(feasible_data), y), s=82, color="#176B87", marker="o", label="Feasible" if y == 0 else None)
        if len(infeasible_data):
            ax.scatter(infeasible_data["pv_generation_scale"], np.full(len(infeasible_data), y), s=82, color="#C8543C", marker="x", linewidths=2.2, label="No feasible plan in evaluated grid" if y == 0 else None)
    ax.set_yticks(range(len(controller_order)), [labels[c] for c in controller_order])
    ax.set_xlabel("PV generation scale relative to the base profile")
    ax.set_title("Empirical feasibility over the evaluated capacity grid")
    ax.grid(True, axis="x", alpha=0.28)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "feasibility_envelope.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for column, ylabel, filename in [
        ("asset_cost_index", "Normalised asset-cost index", "planning_cost_vs_pv.png"),
        ("annual_loss_mwh", "Annual line-loss energy (MWh)", "annual_loss_vs_pv.png"),
        ("annual_pv_energy_mwh", "Annual accepted PV energy (MWh)", "accepted_pv_energy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for controller in controller_order:
            data = feasible[feasible["controller"] == controller].sort_values("pv_generation_scale")
            if len(data):
                ax.plot(data["pv_generation_scale"], data[column], "o-", lw=2.1, color=colors[controller], label=labels[controller])
        ax.set_xlabel("PV generation scale relative to the base profile")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleaner-energy assessment for adaptive reactive-power co-planning.")
    parser.add_argument("--actor_run_dir", required=True, help="FiLM model directory containing actor.pth, or a legacy run directory containing models/actor.pth")
    parser.add_argument("--blind_run_dir", default="", help="Optional Blind PPO model or legacy run directory")
    parser.add_argument("--concat_run_dir", default="", help="Optional Concat PPO model or legacy run directory")
    parser.add_argument("--include_droop", action="store_true", help="Include the fixed local Volt-VAR droop controller.")
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pv_generation_levels", default="0.8,1.0,1.2,1.4")
    parser.add_argument("--pv_s_levels", default="0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--svc_levels", default="0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--cap_levels", default="0.0,0.25,0.5")
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--n_days_coarse", type=int, default=5)
    parser.add_argument("--n_days_final", type=int, default=30)
    parser.add_argument("--refine_topk", type=int, default=12)
    parser.add_argument("--eps_v", type=float, default=0.05)
    parser.add_argument("--eps_pf", type=float, default=0.0)
    parser.add_argument("--cost_pv", type=float, default=50.0)
    parser.add_argument("--cost_svc", type=float, default=50.0)
    parser.add_argument("--cost_cap", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--out_dir", default="runs/CLEANER_ENERGY_ASSESSMENT")
    args = parser.parse_args()

    device = torch.device(args.device)
    load_pu = np.load(Env.DATA_DIR / "load96.npy")
    gene_pu = np.load(Env.DATA_DIR / "gen96.npy")
    id_iber, id_svc = _device_ids(int(args.env))
    dummy = Env.grid_case(args.env, load_pu, gene_pu, id_iber, id_svc)
    actors: dict[str, torch.nn.Module] = {
        "film": _load_film_actor(Path(args.actor_run_dir), len(dummy.observation_space), len(dummy.action_space), device)
    }
    controller_names = ["film"]
    if str(args.blind_run_dir).strip():
        actors["blind"] = _load_mlp_actor(Path(args.blind_run_dir), len(dummy.observation_space), len(dummy.action_space), device)
        controller_names.append("blind")
    if str(args.concat_run_dir).strip():
        actors["concat"] = _load_mlp_actor(Path(args.concat_run_dir), len(dummy.observation_space) + 3, len(dummy.action_space), device)
        controller_names.append("concat")
    if bool(args.include_droop):
        controller_names.append("droop")
    device_buses = id_iber + id_svc
    cap_buses = _parse_int_list(args.cap_buses)
    costs = np.asarray([args.cost_pv, args.cost_svc, args.cost_cap], dtype=float)
    pv_generation_levels = _parse_float_list(args.pv_generation_levels)
    candidates = [
        np.asarray([pv_s, svc, cap], dtype=np.float32)
        for cap in _parse_float_list(args.cap_levels)
        for pv_s in _parse_float_list(args.pv_s_levels)
        for svc in _parse_float_list(args.svc_levels)
    ]
    n_days_total = min(len(load_pu), len(gene_pu)) // STEPS_PER_DAY - 1
    rng = np.random.default_rng(args.seed)
    final_days = np.sort(rng.choice(np.arange(n_days_total), size=min(args.n_days_final, n_days_total), replace=False))
    coarse_days = final_days[: min(args.n_days_coarse, len(final_days))]
    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    best_rows = []
    total_jobs = len(pv_generation_levels) * len(controller_names) * len(candidates)
    completed = 0
    started = time.time()
    for pv_generation_scale in pv_generation_levels:
        for controller in controller_names:
            coarse_rows = []
            for theta in candidates:
                env = _make_env(args.env, load_pu, gene_pu, id_iber, id_svc, theta=theta, pv_generation_scale=pv_generation_scale, cap_buses=cap_buses)
                metrics = _evaluate(env, controller, actors, theta, coarse_days, device, device_buses)
                row = {
                    "controller": controller,
                    "pv_generation_scale": float(pv_generation_scale),
                    "pv_s_scale": float(theta[0]),
                    "svc_q_scale": float(theta[1]),
                    "cap_total_mvar": float(theta[2]),
                    "asset_cost_index": _cost_index(theta, costs),
                    "stage": "coarse",
                    **metrics.__dict__,
                    "feasible": int(_safe(metrics, args.eps_v, args.eps_pf)),
                }
                coarse_rows.append(row)
                completed += 1
                if completed % 20 == 0 or completed == total_jobs:
                    rate = completed / max(time.time() - started, 1e-9)
                    print(f"[coarse] {completed}/{total_jobs} candidates | {rate:.2f} candidates/s")
            coarse = pd.DataFrame(coarse_rows)
            selected = pd.concat([
                coarse[coarse["feasible"] == 1].sort_values(["asset_cost_index", "daily_loss_mwh"]).head(args.refine_topk),
                coarse.sort_values(["pr_violation_any", "pr_pf_fail_any", "asset_cost_index"]).head(args.refine_topk),
            ]).drop_duplicates(subset=["pv_s_scale", "svc_q_scale", "cap_total_mvar"])
            final_rows = []
            for _, selected_row in selected.iterrows():
                theta = selected_row[["pv_s_scale", "svc_q_scale", "cap_total_mvar"]].to_numpy(dtype=np.float32)
                env = _make_env(args.env, load_pu, gene_pu, id_iber, id_svc, theta=theta, pv_generation_scale=pv_generation_scale, cap_buses=cap_buses)
                metrics = _evaluate(env, controller, actors, theta, final_days, device, device_buses)
                row = {
                    "controller": controller,
                    "pv_generation_scale": float(pv_generation_scale),
                    "pv_s_scale": float(theta[0]),
                    "svc_q_scale": float(theta[1]),
                    "cap_total_mvar": float(theta[2]),
                    "asset_cost_index": _cost_index(theta, costs),
                    "stage": "refined",
                    **metrics.__dict__,
                    "feasible": int(_safe(metrics, args.eps_v, args.eps_pf)),
                }
                final_rows.append(row)
            trial = pd.concat([coarse, pd.DataFrame(final_rows)], ignore_index=True)
            all_rows.append(trial)
            refined = trial[trial["stage"] == "refined"]
            feasible = refined[refined["feasible"] == 1].sort_values(["asset_cost_index", "daily_loss_mwh"])
            if len(feasible):
                best = feasible.iloc[0].to_dict()
                best["annual_loss_mwh"] = float(best["daily_loss_mwh"]) * 365.0
                best["annual_pv_energy_mwh"] = float(best["daily_pv_energy_mwh"]) * 365.0
            else:
                best = {
                    "controller": controller,
                    "pv_generation_scale": float(pv_generation_scale),
                    "feasible": 0,
                    "annual_loss_mwh": np.nan,
                    "annual_pv_energy_mwh": np.nan,
                }
            best_rows.append(best)

    candidates_df = pd.concat(all_rows, ignore_index=True)
    best_df = pd.DataFrame(best_rows)
    candidates_df.to_csv(out_dir / "candidate_results.csv", index=False)
    best_df.to_csv(out_dir / "best_plans.csv", index=False)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({
            "dt_hours": DT_HOURS,
            "annualisation_days": 365,
            "risk_definition": "fraction of sampled days containing at least one voltage violation or power-flow failure",
            "asset_cost_index": "normalised index with user-specified coefficients; it is not a market price or lifecycle footprint",
            "pv_energy_definition": "accepted active PV injection; the environment contains no active-power curtailment action",
            "coarse_day_indices": coarse_days.tolist(),
            "final_day_indices": final_days.tolist(),
            "candidate_grid": {
                "pv_generation_levels": pv_generation_levels,
                "pv_s_levels": _parse_float_list(args.pv_s_levels),
                "svc_levels": _parse_float_list(args.svc_levels),
                "cap_levels": _parse_float_list(args.cap_levels),
            },
            "args": vars(args),
        }, handle, indent=2)
    _plot_results(best_df, out_dir)
    print(f"saved results to: {out_dir}")


if __name__ == "__main__":
    main()
