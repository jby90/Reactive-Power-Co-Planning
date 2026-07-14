"""
Evaluate a trained PPO theta-conditioned policy (deterministic mean action).

Loads actor from a PPO_THETA run directory and evaluates across many theta samples.

Outputs:
  <run_dir>/eval/
    eval_episodes.csv
    eval_episodes.png
"""

from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import Env


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


class ActorCritic(nn.Module):
    """Must match PPO_theta_robust.py network definition."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), log_std_init: float = -0.5):
        super().__init__()
        h1, h2 = hidden
        self.pi_fc1 = nn.Linear(obs_dim, h1)
        self.pi_fc2 = nn.Linear(h1, h2)
        self.pi_mu = nn.Linear(h2, act_dim)
        self.pi_log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        self.v_fc1 = nn.Linear(obs_dim, h1)
        self.v_fc2 = nn.Linear(h1, h2)
        self.v_out = nn.Linear(h2, 1)

    def _pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.pi_fc1(obs))
        x = F.relu(self.pi_fc2(x))
        mu = torch.tanh(self.pi_mu(x))
        log_std = torch.clamp(self.pi_log_std, -20, 2)
        return mu, log_std

    @torch.no_grad()
    def act_mean(self, obs: torch.Tensor) -> np.ndarray:
        mu, _ = self._pi(obs)
        return mu.detach().cpu().numpy()


def _sample_theta(rng: np.random.Generator, pv_rng: Tuple[float, float], svc_rng: Tuple[float, float], cap_rng: Tuple[float, float]) -> np.ndarray:
    pv = float(rng.uniform(pv_rng[0], pv_rng[1]))
    svc = float(rng.uniform(svc_rng[0], svc_rng[1]))
    cap = float(rng.uniform(cap_rng[0], cap_rng[1]))
    return np.asarray([pv, svc, cap], dtype=np.float32)


def _make_env(
    env_name: int,
    load_pu: np.ndarray,
    gene_pu: np.ndarray,
    id_iber: List[int],
    id_svc: List[int],
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
        env_name,
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
    env.name = env_name
    return env


@dataclass
class EvalCfg:
    env: int = 33
    seed: int = 42
    episodes: int = 50
    episode_len: int = 96
    days_per_episode: int = 1  # episode_len should typically be 96 * days

    enable_pq_curve: bool = True
    pv_s_min: float = 0.95
    pv_s_max: float = 1.05
    svc_q_min: float = 0.95
    svc_q_max: float = 1.05
    cap_total_min: float = 0.0
    cap_total_max: float = 0.5
    cap_buses: List[int] | None = None

    vmin: float = 0.95
    vmax: float = 1.05
    action_scale: float = 1.0  # optional damping of actor output


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PPO theta-conditioned policy.")
    ap.add_argument("--run_dir", type=str, required=True, help="PPO_THETA run directory (contains models/actor.pth).")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--days", type=int, default=1, help="Days per evaluation episode (episode_len = 96*days).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable_pq_curve", action="store_true")
    ap.add_argument("--pv_s_min", type=float, default=0.95)
    ap.add_argument("--pv_s_max", type=float, default=1.05)
    ap.add_argument("--svc_q_min", type=float, default=0.95)
    ap.add_argument("--svc_q_max", type=float, default=1.05)
    ap.add_argument("--cap_total_min", type=float, default=0.0)
    ap.add_argument("--cap_total_max", type=float, default=0.5)
    ap.add_argument("--cap_buses", type=str, default="")
    ap.add_argument("--vmin", type=float, default=0.95)
    ap.add_argument("--vmax", type=float, default=1.05)
    ap.add_argument("--action_scale", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    actor_path = run_dir / "actor.pth"
    if not actor_path.exists():
        actor_path = run_dir / "models" / "actor.pth"
    if not actor_path.exists():
        raise FileNotFoundError(f"actor not found: {actor_path}")

    # infer env id from path: .../env33_seed42_gamma0.9/...
    env_id = 33
    for part in run_dir.parts:
        if part.startswith("env") and "_seed" in part:
            try:
                env_id = int(part.split("_seed")[0].replace("env", ""))
            except Exception:
                pass

    cfg = EvalCfg(
        env=env_id,
        seed=int(args.seed),
        episodes=int(args.episodes),
        days_per_episode=int(args.days),
        episode_len=96 * int(args.days),
        enable_pq_curve=bool(args.enable_pq_curve),
        pv_s_min=float(args.pv_s_min),
        pv_s_max=float(args.pv_s_max),
        svc_q_min=float(args.svc_q_min),
        svc_q_max=float(args.svc_q_max),
        cap_total_min=float(args.cap_total_min),
        cap_total_max=float(args.cap_total_max),
        cap_buses=_parse_int_list(args.cap_buses) if args.cap_buses.strip() else None,
        vmin=float(args.vmin),
        vmax=float(args.vmax),
        action_scale=float(args.action_scale),
    )

    rng = np.random.default_rng(cfg.seed)

    load_pu = np.load(Env.DATA_DIR / "load96.npy")
    gene_pu = np.load(Env.DATA_DIR / "gen96.npy")
    if cfg.env == 69:
        id_iber = [5, 23, 44, 57]
        id_svc = [13]
    elif cfg.env == 33:
        id_iber = [17, 21, 24]
        id_svc = [32]
    else:
        id_iber = [33, 50, 53, 68, 74, 97, 107, 111]
        id_svc = [44, 104]

    # Build one env to get dims
    theta0 = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    env0 = _make_env(cfg.env, load_pu, gene_pu, id_iber, id_svc, cfg.enable_pq_curve, 1.0, 1.0, cfg.cap_buses or [], 0.0)
    obs_dim = int(env0.observation_space.shape[0]) + 3
    act_dim = int(env0.action_space.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = ActorCritic(obs_dim, act_dim).to(device)
    ac.load_state_dict(torch.load(str(actor_path), map_location=device))
    ac.eval()

    rows = []
    for ep in range(cfg.episodes):
        theta = _sample_theta(
            rng,
            (cfg.pv_s_min, cfg.pv_s_max),
            (cfg.svc_q_min, cfg.svc_q_max),
            (cfg.cap_total_min, cfg.cap_total_max),
        )
        env = _make_env(
            cfg.env,
            load_pu,
            gene_pu,
            id_iber,
            id_svc,
            cfg.enable_pq_curve,
            float(theta[0]),
            float(theta[1]),
            cfg.cap_buses or [],
            float(theta[2]),
        )
        s = env.reset()
        n_bus = len(env.model.bus)

        pf_fail = 0.0
        viol = 0.0
        gl = 0.0

        for t in range(cfg.episode_len):
            obs_aug = np.concatenate([s.astype(np.float32), theta.astype(np.float32)], axis=0)
            a_mean = ac.act_mean(torch.FloatTensor(obs_aug).to(device))
            a_mean = np.asarray(a_mean, dtype=float) * float(cfg.action_scale)
            a_mean = np.clip(a_mean, -1.0, 1.0)
            try:
                next_state, reward_vec, done, violation, violM, violN, vM, vN, grid_loss, new_state = env.step_model(a_mean)
            except Exception:
                pf_fail += 1.0
                viol += 1.0
                break

            vm = np.asarray(next_state[:n_bus], dtype=float)
            vflag = 1.0 if ((vm < cfg.vmin).any() or (vm > cfg.vmax).any()) else 0.0
            viol += vflag
            gl += float(grid_loss)
            s = new_state

        rows.append(
            {
                "episode": ep,
                "pv_s_scale": float(theta[0]),
                "svc_q_scale": float(theta[1]),
                "cap_total_mvar": float(theta[2]),
                "violation_count": float(viol),
                "grid_loss_sum": float(gl),
                "pf_fail": float(pf_fail),
            }
        )

    import pandas as pd

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(eval_dir / "eval_episodes.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(df["violation_count"].values)
    axes[0].set_title("violation_count per episode")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["grid_loss_sum"].values)
    axes[1].set_title("grid_loss_sum per episode")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["pf_fail"].values)
    axes[2].set_title("pf_fail per episode")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"eval: {run_dir.name} (episodes={cfg.episodes}, days/ep={cfg.days_per_episode})")
    fig.tight_layout()
    fig.savefig(eval_dir / "eval_episodes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {eval_dir}")
    print(
        f"means: viol={df.violation_count.mean():.4g}  grid_loss_sum={df.grid_loss_sum.mean():.4g}  pf_fail={df.pf_fail.mean():.4g}"
    )


if __name__ == "__main__":
    main()
