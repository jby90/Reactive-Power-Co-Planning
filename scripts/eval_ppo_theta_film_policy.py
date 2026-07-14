"""
Evaluate a trained PPO FiLM(theta) policy with deterministic mean actions.

Loads actor from a PPO_THETA_FILM run directory and evaluates across many theta samples.

Outputs:
  <run_dir>/eval/
    eval_episodes.csv
    eval_episodes.png
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import sys
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import Env
import PPO_theta_robust_film_curriculum as film


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


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


def _sample_theta(
    rng: np.random.Generator,
    pv_rng: Tuple[float, float],
    svc_rng: Tuple[float, float],
    cap_rng: Tuple[float, float],
) -> np.ndarray:
    pv = float(rng.uniform(pv_rng[0], pv_rng[1]))
    svc = float(rng.uniform(svc_rng[0], svc_rng[1]))
    cap = float(rng.uniform(cap_rng[0], cap_rng[1]))
    return np.asarray([pv, svc, cap], dtype=np.float32)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate PPO FiLM(theta) policy (mean action).")
    ap.add_argument("--run_dir", type=str, required=True, help="PPO_THETA_FILM run directory (contains models/actor.pth).")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--days", type=int, default=1, help="Days per evaluation episode (episode_len=96*days).")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--enable_pq_curve", action="store_true")
    ap.add_argument("--cap_buses", type=str, default="")
    ap.add_argument("--pv_s_min", type=float, default=0.7)
    ap.add_argument("--pv_s_max", type=float, default=1.5)
    ap.add_argument("--svc_q_min", type=float, default=0.7)
    ap.add_argument("--svc_q_max", type=float, default=1.5)
    ap.add_argument("--cap_total_min", type=float, default=0.0)
    ap.add_argument("--cap_total_max", type=float, default=1.0)

    ap.add_argument("--vmin", type=float, default=0.95)
    ap.add_argument("--vmax", type=float, default=1.05)
    ap.add_argument("--action_scale", type=float, default=1.0, help="Optional damping of actor output.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    actor_path = run_dir / "actor.pth"
    if not actor_path.exists():
        actor_path = run_dir / "models" / "actor.pth"
    if not actor_path.exists():
        raise FileNotFoundError(f"actor not found: {actor_path}")

    env_id = _infer_env_id_from_run_dir(run_dir, default=33)
    id_iber, id_svc = _device_ids(env_id)
    cap_buses = _parse_int_list(args.cap_buses) if str(args.cap_buses).strip() else []

    load_pu = np.load(Env.DATA_DIR / "load96.npy")
    gene_pu = np.load(Env.DATA_DIR / "gen96.npy")

    # infer dims
    env0 = Env.grid_case(env_id, load_pu, gene_pu, id_iber, id_svc)
    obs_dim = int(env0.observation_space.shape[0])
    act_dim = int(env0.action_space.shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = film.FiLMActorCritic(obs_dim, act_dim, theta_dim=3).to(device)
    ac.load_state_dict(torch.load(str(actor_path), map_location=device))
    ac.eval()

    rng = np.random.default_rng(int(args.seed))
    episode_len = 96 * int(args.days)
    rows = []

    for ep in range(int(args.episodes)):
        theta = _sample_theta(
            rng,
            (float(args.pv_s_min), float(args.pv_s_max)),
            (float(args.svc_q_min), float(args.svc_q_max)),
            (float(args.cap_total_min), float(args.cap_total_max)),
        )

        env = _make_env(
            env_id,
            load_pu,
            gene_pu,
            id_iber,
            id_svc,
            enable_pq_curve=bool(args.enable_pq_curve),
            pv_s_scale=float(theta[0]),
            svc_q_scale=float(theta[1]),
            cap_buses=list(cap_buses),
            cap_total_mvar=float(theta[2]),
        )
        s = env.reset()
        n_bus = len(env.model.bus)

        pf_fail = 0.0
        viol = 0.0
        gl = 0.0

        for _t in range(int(episode_len)):
            a_mean = _act_mean(ac, s, theta, device=device)
            a_mean = np.asarray(a_mean, dtype=float) * float(args.action_scale)
            a_mean = np.clip(a_mean, -1.0, 1.0)
            try:
                next_state, reward_vec, done, violation, violM, violN, vM, vN, grid_loss, new_state = env.step_model(a_mean)
            except Exception:
                pf_fail += 1.0
                viol += 1.0
                break

            vm = np.asarray(next_state[:n_bus], dtype=float)
            vflag = 1.0 if ((vm < float(args.vmin)).any() or (vm > float(args.vmax)).any()) else 0.0
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
    fig.suptitle(f"eval(FiLM): {run_dir.name} (episodes={len(df)}, days/ep={int(args.days)})")
    fig.tight_layout()
    fig.savefig(eval_dir / "eval_episodes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {eval_dir}")
    print(f"means: viol={df.violation_count.mean():.4g}  grid_loss_sum={df.grid_loss_sum.mean():.4g}  pf_fail={df.pf_fail.mean():.4g}")


if __name__ == "__main__":
    main()
