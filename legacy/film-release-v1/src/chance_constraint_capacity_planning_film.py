"""
Chance-Constraint Capacity Planning (Outer Loop) with a strong inner controller (FiLM-PPO).

Goal:
  min CAPEX(theta)
  s.t.  Pr(violation > 0) <= eps_v
        Pr(pf_fail  > 0) <= eps_pf

We estimate probabilities by evaluating many day-scenarios (shared day indices for fairness)
using deterministic (mean) actions of the trained policy.

Theta:
  theta = [pv_s_scale, svc_q_scale, cap_total_mvar]
  - pv_s_scale: scales PV inverter apparent power ratings (P-Q curve)
  - svc_q_scale: scales SVC/SVG reactive capacity
  - cap_total_mvar: distributed capacitors total Mvar (equally split over cap_buses)

Outputs (under out_dir):
  summary_candidates.csv
  best_by_epsilon.csv
  frontier_by_epsilon.csv
  figures/
    feasible_map_cap*.png
    capex_vs_reliability.png
    capex_vs_risk_scatter.png
    loss_vs_capex_feasible.png

Usage:
  python chance_constraint_capacity_planning_film.py --actor_run_dir runs/PPO_THETA_FILM/.../run_name --env 33
"""

from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure non-interactive backend (prevents Windows backend hangs)
import matplotlib.pyplot as plt
import torch

import Env
import PPO_theta_robust_film_curriculum as film


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def _parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if str(x).strip() != ""]


def _infer_env_id_from_run_dir(run_dir: Path, default: int = 33) -> int:
    for part in run_dir.parts:
        if part.startswith("env") and "_seed" in part:
            try:
                return int(part.split("_seed")[0].replace("env", ""))
            except Exception:
                pass
    return default


def _device_ids(env_id: int) -> tuple[list[int], list[int]]:
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    if env_id == 33:
        return [17, 21, 24], [32]
    return [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]


def _infer_film_arch_from_state_dict(sd: dict) -> tuple[int, int, int]:
    # (h1, h2, theta_dim)
    try:
        h1 = int(sd["pi_fc1.weight"].shape[0])
        h2 = int(sd["pi_fc2.weight"].shape[0])
        theta_dim = int(sd["pi_film1.scale.weight"].shape[1])
        return h1, h2, theta_dim
    except Exception as e:
        raise RuntimeError("Failed to infer FiLM architecture from checkpoint state_dict.") from e


def capex(theta: np.ndarray, *, cost_pv: float, cost_svc: float, cost_cap: float, mode: str) -> float:
    pv, svc, cap_total = float(theta[0]), float(theta[1]), float(theta[2])
    mode = str(mode).lower().strip()
    if mode == "incremental":
        return float(cost_pv) * max(0.0, pv - 1.0) + float(cost_svc) * max(0.0, svc - 1.0) + float(cost_cap) * cap_total
    # absolute
    return float(cost_pv) * pv + float(cost_svc) * svc + float(cost_cap) * cap_total


def _sample_days(rng: np.random.Generator, n_days_total: int, k: int, *, avoid: np.ndarray | None = None) -> np.ndarray:
    """
    Best-effort day sampling with minimal duplicates.
    If avoid is provided, we try not to sample those indices (when possible).
    """
    n_days_total = int(n_days_total)
    k = int(k)
    if k <= 0:
        return np.asarray([], dtype=int)
    if n_days_total <= 0:
        raise ValueError("n_days_total must be positive")

    avoid_set = set(np.asarray(avoid, dtype=int).tolist()) if avoid is not None and len(avoid) > 0 else set()
    pool = np.asarray([i for i in range(n_days_total) if i not in avoid_set], dtype=int)
    if len(pool) == 0:
        # fallback: nothing to avoid
        pool = np.arange(n_days_total, dtype=int)
    if k <= len(pool):
        return rng.choice(pool, size=k, replace=False)
    # not enough unique days; allow replacement
    return rng.choice(pool, size=k, replace=True)


def _make_env(
    env_id: int,
    load_pu: np.ndarray,
    gene_pu: np.ndarray,
    id_iber: List[int],
    id_svc: List[int],
    *,
    enable_pq_curve: bool,
    pv_s_scale: float,
    svc_q_scale: float,
    cap_buses: List[int],
    cap_total_mvar: float,
) -> Env.grid_case:
    if len(cap_buses) > 0 and cap_total_mvar > 0:
        per = float(cap_total_mvar) / float(len(cap_buses))
        cap_q = [per for _ in cap_buses]
    else:
        cap_buses, cap_q = [], []

    env = Env.grid_case(
        env_id,
        load_pu,
        gene_pu,
        id_iber,
        id_svc,
        enable_pq_curve=bool(enable_pq_curve),
        pv_s_scale=float(pv_s_scale),
        svc_q_scale=float(svc_q_scale),
        cap_buses=cap_buses if len(cap_buses) > 0 else None,
        cap_q_mvar=cap_q if len(cap_q) > 0 else None,
    )
    env.name = env_id
    return env


@torch.no_grad()
def _act_mean(ac: film.FiLMActorCritic, obs: np.ndarray, theta: np.ndarray, device: torch.device) -> np.ndarray:
    obs_t = torch.FloatTensor(obs.astype(np.float32)).to(device)
    th_t = torch.FloatTensor(theta.astype(np.float32)).to(device)
    mu, _ = ac._pi(obs_t, th_t)
    return mu.detach().cpu().numpy()


def run_one_day(
    *,
    env: Env.grid_case,
    ac: film.FiLMActorCritic,
    device: torch.device,
    theta: np.ndarray,
    day_idx: int,
    vmin: float,
    vmax: float,
    action_scale: float,
    fail_fast: bool,
    stride: int,
) -> tuple[int, int, float]:
    """
    Returns:
      viol_any (0/1), pf_any (0/1), loss_sum_pos (float)
    """
    # align to this day start
    env.step_n = int(day_idx) * 96
    env.done = False
    obs = env.reset()  # env.reset() does not reset step_n; it just returns a placeholder obs.
    n_bus = len(env.model.bus)

    viol_any = 0
    pf_any = 0
    loss_sum = 0.0

    stride = max(1, int(stride))
    n_steps = int(np.ceil(96 / float(stride)))
    for k in range(n_steps):
        # subsample time steps to speed up chance estimation
        t_in_day = min(95, k * stride)
        env.step_n = int(day_idx) * 96 + int(t_in_day)
        a = _act_mean(ac, obs, theta, device=device)
        a = np.asarray(a, dtype=float) * float(action_scale)
        a = np.clip(a, -1.0, 1.0)
        try:
            next_obs, reward_vec, done, viol, _, _, _, _, grid_loss, new_state = env.step_model(a)
        except Exception:
            pf_any = 1
            viol_any = 1
            if fail_fast:
                break
            else:
                continue

        vm = np.asarray(next_obs[:n_bus], dtype=float)
        if np.any(vm > float(vmax)) or np.any(vm < float(vmin)):
            viol_any = 1
        loss_sum += float(-grid_loss)  # positive loss proxy
        obs = new_state

    return int(viol_any), int(pf_any), float(loss_sum)


@dataclass
class Candidate:
    pv: float
    svc: float
    cap_total: float


def plot_feasible_maps(
    *,
    out_dir: Path,
    df: pd.DataFrame,
    pv_levels: List[float],
    svc_levels: List[float],
    cap_levels: List[float],
    eps_v: float,
    eps_pf: float,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for cap in cap_levels:
        d0 = df[np.isclose(df["cap_total_mvar"].astype(float), float(cap))].copy()
        if len(d0) == 0:
            continue
        piv = (
            d0.pivot_table(index="svc_q_scale", columns="pv_s_scale", values="feasible", aggfunc="mean")
            .reindex(index=svc_levels, columns=pv_levels)
        )
        Z = np.asarray(piv.values, dtype=float)
        X, Y = np.meshgrid(np.asarray(pv_levels, dtype=float), np.asarray(svc_levels, dtype=float))
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 6))
        ax.pcolormesh(X, Y, Z, shading="nearest", cmap=plt.get_cmap("Greens"), vmin=0.0, vmax=1.0)
        ax.contour(X, Y, Z, levels=[0.5], colors=["black"], linewidths=1.8)
        ax.set_xlabel("PV scale")
        ax.set_ylabel("SVC scale")
        ax.set_title(f"Feasible region (Pr(viol>0)≤{eps_v:.3g}, Pr(pf>0)≤{eps_pf:.3g}) @ cap_total={cap:g}")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"feasible_map_cap{cap:g}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _plot_extra_figures(
    *,
    out_dir: Path,
    df: pd.DataFrame,
    cap_levels: List[float],
    eps_v: float,
    eps_pf: float,
) -> None:
    """Extra high-signal figures to tell the 'planning' story."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Feasible ratio vs cap_total
    rows = []
    for cap in cap_levels:
        d0 = df[np.isclose(df["cap_total_mvar"].astype(float), float(cap))].copy()
        if len(d0) == 0:
            continue
        rows.append({"cap_total_mvar": float(cap), "feasible_ratio": float(d0["feasible"].mean())})
    if len(rows) > 0:
        dfr = pd.DataFrame(rows).sort_values("cap_total_mvar")
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(dfr["cap_total_mvar"].values, dfr["feasible_ratio"].values, "-o", lw=2)
        ax.set_xlabel("cap_total_mvar")
        ax.set_ylabel("feasible ratio (over PV×SVC grid)")
        ax.set_title(f"Feasible region size vs capacitor level (eps_v={eps_v:g}, eps_pf={eps_pf:g})")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "feasible_ratio_vs_cap_total.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 2) Joint risk scatter (Pr(viol) vs Pr(pf)), color = CAPEX
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.4))
    x = np.asarray(df["pr_violation_any"].values, dtype=float)
    y = np.asarray(df["pr_pf_fail_any"].values, dtype=float)
    c = np.asarray(df["capex"].values, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    sc = ax.scatter(x[m], y[m], c=c[m], s=14, cmap="viridis", alpha=0.75)
    ax.axvline(float(eps_v), color="red", lw=1.5, ls="--")
    ax.axhline(float(eps_pf), color="red", lw=1.5, ls="--")
    ax.set_xlabel("Pr(violation > 0)")
    ax.set_ylabel("Pr(pf_fail > 0)")
    ax.set_title("Risk map (color = CAPEX), dashed = constraints")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="CAPEX")
    fig.tight_layout()
    fig.savefig(fig_dir / "risk_joint_scatter_capex.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) CAPEX vs loss, colored by Pr(viol) for feasible points (if any)
    feas = df[df["feasible"] == 1].copy()
    if len(feas) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
        xx = np.asarray(feas["capex"].values, dtype=float)
        yy = np.asarray(feas["loss_mean_pos"].values, dtype=float)
        cc = np.asarray(feas["pr_violation_any"].values, dtype=float)
        mm = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(cc)
        sc = ax.scatter(xx[mm], yy[mm], c=cc[mm], s=18, cmap="magma", alpha=0.85)
        ax.set_xlabel("CAPEX")
        ax.set_ylabel("Expected grid loss (positive proxy)")
        ax.set_title("Feasible points: loss vs CAPEX (color = Pr(violation>0))")
        ax.grid(True, alpha=0.3)
        fig.colorbar(sc, ax=ax, label="Pr(violation>0)")
        fig.tight_layout()
        fig.savefig(fig_dir / "loss_vs_capex_feasible_colored_by_risk.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chance-constraint capacity planning with FiLM-PPO inner controller.")
    ap.add_argument("--actor_run_dir", type=str, required=True, help="PPO_THETA_FILM run directory containing models/actor.pth or checkpoint.pth.")
    ap.add_argument("--env", type=int, default=-1, choices=[-1, 33, 69, 118])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (default: runs/CC_PLAN_PPO_THETA_FILM/...)")

    # grid
    ap.add_argument("--pv_levels", type=str, default="", help="Comma-separated PV scales. If empty, use linspace(pv_min,pv_max,res).")
    ap.add_argument("--svc_levels", type=str, default="", help="Comma-separated SVC scales. If empty, use linspace(svc_min,svc_max,res).")
    ap.add_argument("--cap_levels", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--res", type=int, default=20)
    ap.add_argument("--pv_min", type=float, default=0.7)
    ap.add_argument("--pv_max", type=float, default=1.5)
    ap.add_argument("--svc_min", type=float, default=0.7)
    ap.add_argument("--svc_max", type=float, default=1.5)

    # evaluation scenarios
    ap.add_argument("--n_scenarios", type=int, default=30, help="How many random days to estimate probabilities (final).")
    ap.add_argument("--n_scenarios_coarse", type=int, default=5, help="Coarse screening scenarios for all candidates.")
    ap.add_argument("--topk_refine", type=int, default=150, help="Refine only top-K lowest CAPEX candidates after coarse screening.")
    ap.add_argument(
        "--refine_feasible_max",
        type=int,
        default=300,
        help="Additionally refine at most this many coarse-feasible candidates (lowest CAPEX among coarse-feasible). "
        "This prevents refine from exploding when coarse constraints are loose.",
    )
    ap.add_argument("--stride", type=int, default=2, help="Time subsampling stride within a day (1=full 96 steps, 2=~48 steps, 4=~24 steps).")
    ap.add_argument("--enable_pq_curve", action="store_true")
    ap.add_argument("--cap_buses", type=str, default="20,8")
    ap.add_argument("--vmin", type=float, default=0.95)
    ap.add_argument("--vmax", type=float, default=1.05)
    ap.add_argument("--action_scale", type=float, default=1.0)
    ap.add_argument("--fail_fast", action="store_true")

    # chance constraints
    ap.add_argument("--eps_v", type=float, default=0.01, help="Constraint: Pr(violation>0) <= eps_v.")
    ap.add_argument("--eps_pf", type=float, default=0.01, help="Constraint: Pr(pf_fail>0) <= eps_pf.")
    ap.add_argument("--eps_list", type=str, default="0.2,0.1,0.05,0.02,0.01", help="Epsilon list for reliability-cost curve.")

    # CAPEX model
    ap.add_argument("--capex_mode", type=str, default="absolute", choices=["absolute", "incremental"])
    ap.add_argument("--cost_pv", type=float, default=1.0)
    ap.add_argument("--cost_svc", type=float, default=1.0)
    ap.add_argument("--cost_cap", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    actor_run_dir = Path(args.actor_run_dir)
    env_id = int(args.env) if int(args.env) != -1 else _infer_env_id_from_run_dir(actor_run_dir, default=33)

    # load model
    actor_path = actor_run_dir / "actor.pth"
    if not actor_path.exists():
        actor_path = actor_run_dir / "models" / "actor.pth"
    ckpt_path = actor_run_dir / "models" / "checkpoint.pth"
    if actor_path.exists():
        ckpt = torch.load(str(actor_path), map_location="cpu")
        sd = ckpt
    elif ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt["ac"] if isinstance(ckpt, dict) and "ac" in ckpt else ckpt
    else:
        raise FileNotFoundError(f"Cannot find actor.pth in {actor_run_dir} or its models subdirectory")

    load_pu = np.load(Env.DATA_DIR / "load96.npy")
    gene_pu = np.load(Env.DATA_DIR / "gen96.npy")
    n_days_total = int(len(load_pu) // 96)
    if n_days_total <= 0:
        raise RuntimeError("load96.npy length is too short.")
    # Env.step_model internally peeks one step ahead (t2=self.step_n), so the last day index can overflow.
    # Keep a safe range of [0, n_days_total-2] when possible.
    n_days_safe = max(1, n_days_total - 1)

    id_iber, id_svc = _device_ids(env_id)
    env0 = Env.grid_case(env_id, load_pu, gene_pu, id_iber, id_svc)
    obs_dim = int(env0.observation_space.shape[0])
    act_dim = int(env0.action_space.shape[0])
    h1, h2, theta_dim = _infer_film_arch_from_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = film.FiLMActorCritic(obs_dim, act_dim, theta_dim=int(theta_dim), hidden=(int(h1), int(h2))).to(device)
    ac.load_state_dict(sd)
    ac.eval()

    # grid levels
    if str(args.pv_levels).strip():
        pv_levels = _parse_float_list(args.pv_levels)
    else:
        pv_levels = np.linspace(float(args.pv_min), float(args.pv_max), int(args.res)).tolist()
    if str(args.svc_levels).strip():
        svc_levels = _parse_float_list(args.svc_levels)
    else:
        svc_levels = np.linspace(float(args.svc_min), float(args.svc_max), int(args.res)).tolist()
    cap_levels = _parse_float_list(args.cap_levels)
    pv_levels = [float(x) for x in pv_levels]
    svc_levels = [float(x) for x in svc_levels]
    cap_levels = [float(x) for x in cap_levels]

    cap_buses = _parse_int_list(str(args.cap_buses))
    candidates = [Candidate(pv=pv, svc=svc, cap_total=cap) for cap in cap_levels for pv in pv_levels for svc in svc_levels]

    # shared scenario days (fairness)
    n_scen_final = max(1, int(args.n_scenarios))
    n_scen_coarse = max(1, min(int(args.n_scenarios_coarse), n_scen_final))
    day_indices_coarse = _sample_days(rng, n_days_safe, n_scen_coarse)
    # extra days for refinement (avoid duplicates as best-effort; allow replacement only when needed)
    n_extra = max(0, n_scen_final - n_scen_coarse)
    day_indices_extra = _sample_days(rng, n_days_safe, n_extra, avoid=day_indices_coarse) if n_extra > 0 else np.asarray([], dtype=int)

    # output dir
    if str(args.out_dir).strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("runs") / "CC_PLAN_PPO_THETA_FILM" / f"env{env_id}_seed{int(args.seed)}" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    try:
        # --- pass 1: coarse screening for all candidates ---
        rows = []
        t0 = time.time()
        for i, c in enumerate(candidates):
            theta = np.asarray([c.pv, c.svc, c.cap_total], dtype=np.float32)
            env = _make_env(
                env_id,
                load_pu,
                gene_pu,
                id_iber,
                id_svc,
                enable_pq_curve=bool(args.enable_pq_curve),
                pv_s_scale=float(c.pv),
                svc_q_scale=float(c.svc),
                cap_buses=list(cap_buses),
                cap_total_mvar=float(c.cap_total),
            )

            viol_any = 0
            pf_any = 0
            loss_sum = 0.0
            for d in day_indices_coarse.tolist():
                v1, p1, l1 = run_one_day(
                    env=env,
                    ac=ac,
                    device=device,
                    theta=theta,
                    day_idx=int(d),
                    vmin=float(args.vmin),
                    vmax=float(args.vmax),
                    action_scale=float(args.action_scale),
                    fail_fast=bool(args.fail_fast),
                    stride=int(args.stride),
                )
                viol_any += int(v1)
                pf_any += int(p1)
                loss_sum += float(l1)

            pr_v = viol_any / float(n_scen_coarse)
            pr_pf = pf_any / float(n_scen_coarse)
            loss_mean = loss_sum / float(n_scen_coarse)
            capex_val = capex(
                theta,
                cost_pv=float(args.cost_pv),
                cost_svc=float(args.cost_svc),
                cost_cap=float(args.cost_cap),
                mode=str(args.capex_mode),
            )

            row = {
                "pv_s_scale": float(c.pv),
                "svc_q_scale": float(c.svc),
                "cap_total_mvar": float(c.cap_total),
                "n_scenarios": int(n_scen_coarse),
                "pr_violation_any": float(pr_v),
                "pr_pf_fail_any": float(pr_pf),
                "loss_mean_pos": float(loss_mean),
                "capex": float(capex_val),
                "pass": "coarse",
            }
            rows.append(row)

            log_every = max(1, int(args.log_every))
            if (i + 1) % log_every == 0 or (i + 1) == len(candidates):
                dt = max(time.time() - t0, 1e-9)
                rate = (i + 1) / dt
                eta = (len(candidates) - (i + 1)) / max(rate, 1e-9)
                print(f"[eval] {i+1}/{len(candidates)} | {rate:.3g} cand/s | ETA {eta/60:.1f} min")

        df = pd.DataFrame(rows)
    except KeyboardInterrupt:
        # Save partial coarse results for debugging / resuming.
        out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else Path("runs") / "CC_PLAN_PPO_THETA_FILM" / f"env{env_id}_seed{int(args.seed)}" / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows if "rows" in locals() else []).to_csv(out_dir / "summary_candidates_partial.csv", index=False)
        print(f"\ninterrupted: saved partial coarse results to {out_dir / 'summary_candidates_partial.csv'}")
        return

    # choose refine set:
    # - always include coarse-feasible (under target eps) because they are most important for min-CAPEX feasible solution
    # - plus top-K by CAPEX as a safety net, so we don't miss near-feasible low-cost points
    topk = max(1, int(args.topk_refine))
    df_coarse = df[df["pass"] == "coarse"].copy()
    coarse_feas = df_coarse[
        (df_coarse["pr_violation_any"] <= float(args.eps_v) + 1e-12) & (df_coarse["pr_pf_fail_any"] <= float(args.eps_pf) + 1e-12)
    ].copy()
    # Cap the number of coarse-feasible points we refine; otherwise if eps is loose, almost everything becomes "feasible"
    # and refine becomes prohibitively expensive.
    refine_feas_max = max(0, int(args.refine_feasible_max))
    if refine_feas_max > 0 and len(coarse_feas) > refine_feas_max:
        coarse_feas = coarse_feas.sort_values(["capex", "loss_mean_pos"], ascending=[True, True]).head(refine_feas_max).copy()
    low_cap = df_coarse.sort_values(["capex", "pr_violation_any", "pr_pf_fail_any"], ascending=[True, True, True]).head(topk).copy()
    key = ["pv_s_scale", "svc_q_scale", "cap_total_mvar"]
    df_ref = pd.concat([coarse_feas, low_cap], ignore_index=True).drop_duplicates(subset=key, keep="first")
    print(f"[refine] selected {len(df_ref)}/{len(df_coarse)} candidates (coarse_feasible_kept={len(coarse_feas)}, low_cap={len(low_cap)})")

    # --- pass 2: refine top-K with extra scenarios ---
    if n_extra > 0 and len(df_ref) > 0:
        rows2 = []
        t1 = time.time()
        n_ref = int(len(df_ref))
        for j, r in df_ref.iterrows():
            pv = float(r["pv_s_scale"]); svc = float(r["svc_q_scale"]); cap = float(r["cap_total_mvar"])
            theta = np.asarray([pv, svc, cap], dtype=np.float32)
            env = _make_env(
                env_id,
                load_pu,
                gene_pu,
                id_iber,
                id_svc,
                enable_pq_curve=bool(args.enable_pq_curve),
                pv_s_scale=pv,
                svc_q_scale=svc,
                cap_buses=list(cap_buses),
                cap_total_mvar=cap,
            )
            viol_any = int(round(float(r["pr_violation_any"]) * float(n_scen_coarse)))
            pf_any = int(round(float(r["pr_pf_fail_any"]) * float(n_scen_coarse)))
            loss_sum = float(r["loss_mean_pos"]) * float(n_scen_coarse)

            # early stop threshold for infeasibility under target eps
            max_viol_allowed = int(np.floor(float(args.eps_v) * float(n_scen_final) + 1e-12))
            max_pf_allowed = int(np.floor(float(args.eps_pf) * float(n_scen_final) + 1e-12))

            for d in day_indices_extra.tolist():
                v1, p1, l1 = run_one_day(
                    env=env,
                    ac=ac,
                    device=device,
                    theta=theta,
                    day_idx=int(d),
                    vmin=float(args.vmin),
                    vmax=float(args.vmax),
                    action_scale=float(args.action_scale),
                    fail_fast=bool(args.fail_fast),
                    stride=int(args.stride),
                )
                viol_any += int(v1)
                pf_any += int(p1)
                loss_sum += float(l1)
                # early stop: once either chance constraint is already violated, further scenarios won't rescue feasibility
                if (viol_any > max_viol_allowed) or (pf_any > max_pf_allowed):
                    break

            pr_v = viol_any / float(n_scen_final)
            pr_pf = pf_any / float(n_scen_final)
            loss_mean = loss_sum / float(n_scen_final)
            capex_val = float(r["capex"])
            rows2.append(
                {
                    "pv_s_scale": pv,
                    "svc_q_scale": svc,
                    "cap_total_mvar": cap,
                    "n_scenarios": int(n_scen_final),
                    "pr_violation_any": float(pr_v),
                    "pr_pf_fail_any": float(pr_pf),
                    "loss_mean_pos": float(loss_mean),
                    "capex": float(capex_val),
                    "pass": "refined",
                }
            )

            # refine progress
            log_every_ref = max(1, int(args.log_every))
            done_ref = len(rows2)
            if (done_ref % log_every_ref == 0) or (done_ref == n_ref):
                dt = max(time.time() - t1, 1e-9)
                rate = done_ref / dt
                eta = (n_ref - done_ref) / max(rate, 1e-9)
                print(f"[refine] {done_ref}/{n_ref} | {rate:.3g} cand/s | ETA {eta/60:.1f} min")

        df2 = pd.DataFrame(rows2)
        # merge refined back (prefer refined rows)
        df = (
            pd.concat([df, df2], ignore_index=True)
            .sort_values(["pass"], ascending=True)
            .drop_duplicates(subset=key, keep="last")
            .reset_index(drop=True)
        )

    # feasibility under target eps
    df["feasible"] = ((df["pr_violation_any"] <= float(args.eps_v) + 1e-12) & (df["pr_pf_fail_any"] <= float(args.eps_pf) + 1e-12)).astype(int)
    df.to_csv(out_dir / "summary_candidates.csv", index=False)

    # best feasible for (eps_v, eps_pf)
    feas = df[df["feasible"] == 1].copy()
    if len(feas) > 0:
        best = feas.sort_values(["capex", "loss_mean_pos"], ascending=[True, True]).iloc[0].to_dict()
        pd.DataFrame([best]).to_csv(out_dir / "best_by_epsilon.csv", index=False)
    else:
        pd.DataFrame([]).to_csv(out_dir / "best_by_epsilon.csv", index=False)

    # reliability-cost curve over eps list (fix eps_pf same as args.eps_pf unless provided)
    eps_list = _parse_float_list(str(args.eps_list))
    frontier = []
    for epsv in eps_list:
        feas2 = df[(df["pr_violation_any"] <= float(epsv) + 1e-12) & (df["pr_pf_fail_any"] <= float(args.eps_pf) + 1e-12)].copy()
        if len(feas2) == 0:
            frontier.append({"eps_v": float(epsv), "eps_pf": float(args.eps_pf), "capex_min": float("nan"), "loss_mean_pos": float("nan")})
            continue
        b = feas2.sort_values(["capex", "loss_mean_pos"], ascending=[True, True]).iloc[0]
        frontier.append({"eps_v": float(epsv), "eps_pf": float(args.eps_pf), "capex_min": float(b["capex"]), "loss_mean_pos": float(b["loss_mean_pos"])})
    df_front = pd.DataFrame(frontier)
    df_front.to_csv(out_dir / "frontier_by_epsilon.csv", index=False)

    # plots
    try:
        plot_feasible_maps(
            out_dir=out_dir,
            df=df,
            pv_levels=pv_levels,
            svc_levels=svc_levels,
            cap_levels=cap_levels,
            eps_v=float(args.eps_v),
            eps_pf=float(args.eps_pf),
        )

        # capex vs reliability curve
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(df_front["eps_v"].values, df_front["capex_min"].values, "-o", lw=2)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("eps_v (smaller = higher reliability)")
        ax.set_ylabel("min CAPEX satisfying chance constraints")
        ax.set_title("Reliability-Cost Curve (chance constraint)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "capex_vs_reliability.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # scatter: risk vs capex
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(df["capex"].values, df["pr_violation_any"].values, s=14, alpha=0.6)
        axes[0].set_xlabel("CAPEX"); axes[0].set_ylabel("Pr(violation>0)")
        axes[0].grid(True, alpha=0.3)
        axes[1].scatter(df["capex"].values, df["pr_pf_fail_any"].values, s=14, alpha=0.6)
        axes[1].set_xlabel("CAPEX"); axes[1].set_ylabel("Pr(pf_fail>0)")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("Risk vs CAPEX (all candidates)")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "capex_vs_risk_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        # loss vs capex (feasible)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        if len(feas) > 0:
            ax.scatter(feas["capex"].values, feas["loss_mean_pos"].values, s=16, alpha=0.7)
        ax.set_xlabel("CAPEX")
        ax.set_ylabel("Expected grid loss (positive proxy)")
        ax.set_title("Loss vs CAPEX (feasible points)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "loss_vs_capex_feasible.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        _plot_extra_figures(
            out_dir=out_dir,
            df=df,
            cap_levels=cap_levels,
            eps_v=float(args.eps_v),
            eps_pf=float(args.eps_pf),
        )
    except Exception as e:
        print("warning: plotting failed:", repr(e))

    print(f"saved: {out_dir}")
    if len(feas) > 0:
        print("best feasible theta:", float(best["pv_s_scale"]), float(best["svc_q_scale"]), float(best["cap_total_mvar"]))
        print("best capex:", float(best["capex"]), "Pr(v):", float(best["pr_violation_any"]), "Pr(pf):", float(best["pr_pf_fail_any"]))
        print("loss_mean_pos:", float(best["loss_mean_pos"]))
    else:
        print("no feasible candidate under given eps constraints.")


if __name__ == "__main__":
    main()
