"""
PPO for varying capacity parameters theta: learn pi(a | s, theta).

Design goal:
  Train a policy that remains effective under changing capacity configurations theta,
  via domain randomization + conditioning.

Key ideas:
  - Use on-policy PPO (more robust than off-policy DDPG under non-stationarity).
  - Append theta to observation: obs_aug = [obs, theta]
    theta = [pv_s_scale, svc_q_scale, cap_total_mvar]
  - Resample theta at episode boundary (default 96 steps) and rebuild the env
    so the capacity changes actually affect dynamics/constraints.
  - Treat theta change / PF failure as terminal for GAE/returns to avoid bootstrapping
    across different environments.

Outputs (aligned style):
  runs/PPO_THETA/env{env}_seed{seed}_gamma{gamma}/<run_name>/
    tb/ csv/ figures/ models/
"""

from __future__ import annotations

import os
import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Env

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ma(a, w: int = 200) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if len(a) <= w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def relu_violation_sums(vm: np.ndarray, vmin: float, vmax: float) -> tuple[float, float, int]:
    m = float(np.maximum(0.0, vm - vmax).sum())
    n = float(np.maximum(0.0, vmin - vm).sum())
    flag = 1 if (m + n) > 0 else 0
    return m, n, flag


def map_action_to_q(env, a_norm: np.ndarray) -> np.ndarray:
    low = np.asarray(env.model.sgen.min_q_mvar, dtype=float)
    high = np.asarray(env.model.sgen.max_q_mvar, dtype=float)
    scale = (high - low) / 2.0
    reloc = high - scale
    q = a_norm * scale + reloc
    return np.clip(q, low, high)


class ActorCritic(nn.Module):
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

    def _v(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.v_fc1(obs))
        x = F.relu(self.v_fc2(x))
        return self.v_out(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float, float]:
        mu, log_std = self._pi(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-7)
        logp = logp.sum(-1).item()
        v = self._v(obs).item()
        return a.cpu().numpy(), mu.cpu().numpy(), logp, v

    def logp_and_v(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._pi(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        eps = 1e-6
        a = torch.clamp(act, -1 + eps, 1 - eps)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-7)
        logp = logp.sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self._v(obs)
        return logp, entropy, v


@dataclass
class PPOConfig:
    env: int = 33
    seed: int = 42
    gamma: float = 0.9
    run_name: str = ""

    # PPO core
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_iters: int = 10
    minibatch_size: int = 256
    steps_per_update: int = 2048
    num_frames: int = 96 * 300

    # scalarization
    reward_v_weight: float = 200.0  # weight for voltage term (Env reward_v is negative)
    vmin: float = 0.95
    vmax: float = 1.05
    pf_fail_penalty: float = 1000.0

    # exploration anneal
    log_std_init: float = -0.5
    log_std_final: float = -2.0
    entropy_coef: float = 0.0

    # theta randomization
    enable_pq_curve: bool = True
    blind_theta: bool = False  # if True: do NOT append theta to observation (theta-blind baseline)
    pv_s_scale_min: float = 0.95
    pv_s_scale_max: float = 1.05
    svc_q_scale_min: float = 0.95
    svc_q_scale_max: float = 1.05
    cap_total_min: float = 0.0
    cap_total_max: float = 0.5
    cap_buses: List[int] | None = None  # distribute cap_total equally over these buses

    episode_len: int = 96

    # misc
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    log_every: int = 200


class ThetaEnvManager:
    def __init__(self, cfg: PPOConfig, load_pu: np.ndarray, gene_pu: np.ndarray, id_iber: List[int], id_svc: List[int]):
        self.cfg = cfg
        self.load_pu = load_pu
        self.gene_pu = gene_pu
        self.id_iber = id_iber
        self.id_svc = id_svc
        self.theta = np.zeros(3, dtype=np.float32)
        self.env = None
        self.env_det = None
        self.state = None
        self.state_det = None
        self.episode_step = 0
        self.resample_and_reset()

    def _sample_theta(self) -> np.ndarray:
        cfg = self.cfg
        pv = np.random.uniform(cfg.pv_s_scale_min, cfg.pv_s_scale_max)
        svc = np.random.uniform(cfg.svc_q_scale_min, cfg.svc_q_scale_max)
        cap = np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
        return np.asarray([pv, svc, cap], dtype=np.float32)

    def _build_env(self, theta: np.ndarray):
        cfg = self.cfg
        cap_buses = list(cfg.cap_buses) if cfg.cap_buses is not None else []
        cap_total = float(theta[2])
        if len(cap_buses) > 0 and cap_total > 0:
            per = cap_total / float(len(cap_buses))
            cap_q = [per for _ in cap_buses]
        else:
            cap_buses, cap_q = [], []

        env = Env.grid_case(
            cfg.env,
            self.load_pu,
            self.gene_pu,
            self.id_iber,
            self.id_svc,
            enable_pq_curve=bool(cfg.enable_pq_curve),
            pv_s_scale=float(theta[0]),
            svc_q_scale=float(theta[1]),
            cap_buses=cap_buses if len(cap_buses) > 0 else None,
            cap_q_mvar=cap_q if len(cap_q) > 0 else None,
        )
        env.name = cfg.env
        return env

    def obs_aug(self) -> np.ndarray:
        assert self.state is not None
        if bool(self.cfg.blind_theta):
            return self.state.astype(np.float32)
        return np.concatenate([self.state.astype(np.float32), self.theta.astype(np.float32)], axis=0)

    def resample_and_reset(self) -> None:
        self.theta = self._sample_theta()
        self.env = self._build_env(self.theta)
        self.env_det = self._build_env(self.theta)
        self.state = self.env.reset()
        self.state_det = self.env_det.reset()
        self.episode_step = 0

    def step(self, action: np.ndarray, action_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """
        Returns:
          next_obs_aug (sampled env), reward_vec_sampled, reward_vec_det, done, info
        done is True at episode boundary or PF fail.
        """
        cfg = self.cfg
        assert self.env is not None
        assert self.env_det is not None
        assert self.state is not None
        assert self.state_det is not None

        info = {"pf_fail_s": 0.0, "pf_fail_d": 0.0}
        done = False

        # --- deterministic mirror step (mean action) ---
        try:
            next_state_d, reward_vec_d, _done_d, violation_d, violM_d, violN_d, vM_d, vN_d, grid_loss_d, new_state_d = self.env_det.step_model(
                action_mean
            )
        except Exception:
            info["pf_fail_d"] = 1.0
            reward_vec_d = np.asarray([-cfg.pf_fail_penalty, -cfg.pf_fail_penalty], dtype=np.float32)
            next_state_d = self.state_det
            new_state_d = self.state_det
            done = True

        # --- sampled step ---
        try:
            next_state, reward_vec, _done, violation, violM, violN, vM, vN, grid_loss, new_state = self.env.step_model(action)
        except Exception:
            # PF fail => terminal with big penalty
            info["pf_fail_s"] = 1.0
            reward_vec = np.asarray([-cfg.pf_fail_penalty, -cfg.pf_fail_penalty], dtype=np.float32)
            next_state = self.state
            new_state = self.state
            done = True

        # update internal states (before potential reset)
        self.state = new_state
        self.state_det = new_state_d
        self.episode_step += 1

        if self.episode_step >= cfg.episode_len:
            done = True

        if bool(cfg.blind_theta):
            obs_aug_next = next_state.astype(np.float32)
        else:
            obs_aug_next = np.concatenate([next_state.astype(np.float32), self.theta.astype(np.float32)], axis=0)

        # per-step violation flags for logging (counts "any violation" like your other scripts)
        try:
            n_bus = len(self.env.model.bus)
            vm_s = np.asarray(next_state[:n_bus], dtype=float)
            _, _, vflag_s = relu_violation_sums(vm_s, cfg.vmin, cfg.vmax)
        except Exception:
            vflag_s = 1
        try:
            n_bus = len(self.env_det.model.bus)
            vm_d = np.asarray(next_state_d[:n_bus], dtype=float)
            _, _, vflag_d = relu_violation_sums(vm_d, cfg.vmin, cfg.vmax)
        except Exception:
            vflag_d = 1
        info["vflag_s"] = float(vflag_s)
        info["vflag_d"] = float(vflag_d)

        # when done because of boundary, resample theta for next call
        if done:
            self.resample_and_reset()

        return obs_aug_next, reward_vec, reward_vec_d, done, info


class PPOTrainerTheta:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

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

        self.id_iber = id_iber
        self.id_svc = id_svc

        self.mgr = ThetaEnvManager(cfg, load_pu, gene_pu, id_iber, id_svc)
        self.obs_dim = int(self.mgr.obs_aug().shape[0])
        self.act_dim = int(self.mgr.env.action_space.shape[0])  # type: ignore

        self.ac = ActorCritic(self.obs_dim, self.act_dim, log_std_init=cfg.log_std_init).to(self.device)
        self.pi_opt = optim.Adam(
            list(self.ac.pi_fc1.parameters()) + list(self.ac.pi_fc2.parameters()) + list(self.ac.pi_mu.parameters()) + [self.ac.pi_log_std],
            lr=cfg.pi_lr,
        )
        self.vf_opt = optim.Adam(
            list(self.ac.v_fc1.parameters()) + list(self.ac.v_fc2.parameters()) + list(self.ac.v_out.parameters()),
            lr=cfg.vf_lr,
        )

        gamma_str = str(cfg.gamma)
        group_dir = Path("runs") / "PPO_THETA" / f"env{cfg.env}_seed{cfg.seed}_gamma{gamma_str}"
        run_name = cfg.run_name if cfg.run_name else time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = group_dir / run_name
        self.tb_dir = self.run_dir / "tb"
        self.csv_dir = self.run_dir / "csv"
        self.fig_dir = self.run_dir / "figures"
        self.model_dir = self.run_dir / "models"
        for d in (self.tb_dir, self.csv_dir, self.fig_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tb_dir)) if SummaryWriter is not None else None

        self.total_step = 0
        self.state_aug = self.mgr.obs_aug()

        self.prev_q_sample = np.zeros(self.act_dim, dtype=float)
        self.prev_q_mean = np.zeros(self.act_dim, dtype=float)

    def _anneal_log_std(self) -> None:
        cfg = self.cfg
        frac = min(max(self.total_step / max(cfg.num_frames, 1), 0.0), 1.0)
        target = cfg.log_std_init + frac * (cfg.log_std_final - cfg.log_std_init)
        with torch.no_grad():
            self.ac.pi_log_std.data.fill_(float(target))

    def _scalar_reward(self, reward_vec: np.ndarray) -> float:
        # Env returns [reward_p (= -loss), reward_v (= -violation_amplitude)]
        return float(reward_vec[0] + self.cfg.reward_v_weight * reward_vec[1])

    def save_models(self) -> None:
        torch.save(self.ac.state_dict(), str(self.model_dir / "actor.pth"))
        torch.save({"cfg": self.cfg.__dict__, "ac": self.ac.state_dict(), "total_step": self.total_step}, str(self.model_dir / "checkpoint.pth"))

    def _write_csv_and_figures(
        self,
        scores,
        viol_counts,
        grid_loss_sums,
        scores0,
        viol_counts0,
        grid_loss_sums0,
        pf_fails,
        pf_fails0,
        actor_losses,
        value_losses,
        entropies,
    ) -> None:
        import pandas as pd

        pd.DataFrame(
            {"scores": scores, "violation_sum_s": viol_counts, "grid_loss_sum_s": grid_loss_sums, "pf_fail_sum_s": pf_fails}
        ).to_csv(self.csv_dir / "train.csv", index=False)
        pd.DataFrame(
            {"scorest": scores0, "violation_sum_st": viol_counts0, "grid_loss_sum_st": grid_loss_sums0, "pf_fail_sum_st": pf_fails0}
        ).to_csv(
            self.csv_dir / "traintest.csv", index=False
        )
        pd.DataFrame({"actor_losses": actor_losses, "critic1_losses": value_losses, "critic2_losses": entropies}).to_csv(
            self.csv_dir / "trainloss.csv", index=False
        )

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
        axes = axes.reshape(2, 3)
        axes[0, 0].plot(scores); axes[0, 0].set_title("train/episode_score")
        axes[0, 1].plot(grid_loss_sums); axes[0, 1].set_title("train/episode_grid_loss_sum")
        axes[0, 2].plot(viol_counts); axes[0, 2].set_title("train/episode_violation_count")
        axes[1, 0].plot(scores0); axes[1, 0].set_title("train_action0/episode_score")
        axes[1, 1].plot(grid_loss_sums0); axes[1, 1].set_title("train_action0/episode_grid_loss_sum")
        axes[1, 2].plot(viol_counts0); axes[1, 2].set_title("train_action0/episode_violation_count")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("episode")
        fig.suptitle(self.run_dir.name, y=1.02)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "episode_metrics.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(_ma(actor_losses), label="policy_loss (MA200)")
        ax.plot(_ma(value_losses), label="value_loss (MA200)")
        ax.plot(_ma(entropies), label="entropy (MA200)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.fig_dir / "loss_metrics.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def train(self) -> None:
        cfg = self.cfg
        writer = self.writer

        # episode logs
        scores, viol_counts, grid_loss_sums, pf_fails = [], [], [], []
        scores0, viol_counts0, grid_loss_sums0, pf_fails0 = [], [], [], []
        actor_losses, value_losses, entropies = [], [], []

        score = 0.0
        viol_cnt = 0.0
        gl_sum = 0.0
        pf_fail_ep = 0.0
        score0 = 0.0
        viol_cnt0 = 0.0
        gl_sum0 = 0.0
        pf_fail_ep0 = 0.0

        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

        t_start = time.time()
        t_last = t_start

        if writer is not None:
            writer.add_text("meta/algo", "PPO_theta_robust")
            writer.add_text("meta/theta", "theta=[pv_s_scale, svc_q_scale, cap_total_mvar]")
            writer.add_scalar("meta/reward_v_weight", float(cfg.reward_v_weight), 0)

        while self.total_step < cfg.num_frames:
            self._anneal_log_std()

            obs_buf.clear()
            act_buf.clear()
            logp_buf.clear()
            val_buf.clear()
            rew_buf.clear()
            done_buf.clear()

            steps_this = min(cfg.steps_per_update, cfg.num_frames - self.total_step)

            for _ in range(int(steps_this)):
                self.total_step += 1
                obs_t = torch.FloatTensor(self.state_aug).to(self.device)
                a_sample, a_mean, logp, v = self.ac.act(obs_t)

                # sampled
                next_obs_aug, reward_vec, reward_vec0, done, info = self.mgr.step(a_sample, a_mean)
                r_scalar = self._scalar_reward(reward_vec)
                r_scalar0 = self._scalar_reward(reward_vec0)

                # episode metrics from step info (computed on next_state before any reset)
                score0 += float(r_scalar0)
                score += float(r_scalar)
                viol_cnt += float(info.get("vflag_s", 0.0))
                gl_sum += float(reward_vec[0])
                pf_fail_ep += float(info.get("pf_fail_s", 0.0))
                viol_cnt0 += float(info.get("vflag_d", 0.0))
                gl_sum0 += float(reward_vec0[0])
                pf_fail_ep0 += float(info.get("pf_fail_d", 0.0))

                # buffers
                obs_buf.append(self.state_aug.copy())
                act_buf.append(a_sample.copy())
                logp_buf.append(logp)
                val_buf.append(v)
                rew_buf.append(r_scalar)
                done_buf.append(1.0 if done else 0.0)

                # advance rollout state: if done, manager has already reset to a new theta/env
                self.state_aug = self.mgr.obs_aug() if done else next_obs_aug

                if writer is not None:
                    writer.add_scalar("step/reward_p", float(reward_vec[0]), self.total_step)
                    writer.add_scalar("step/reward_v", float(reward_vec[1]), self.total_step)
                    writer.add_scalar("step/reward_scalar", float(r_scalar), self.total_step)
                    writer.add_scalar("step/pf_fail", float(info.get("pf_fail_s", 0.0)), self.total_step)
                    writer.add_scalar("step/pf_fail_action0", float(info.get("pf_fail_d", 0.0)), self.total_step)
                    writer.add_scalar("theta/pv_s_scale", float(self.mgr.theta[0]), self.total_step)
                    writer.add_scalar("theta/svc_q_scale", float(self.mgr.theta[1]), self.total_step)
                    writer.add_scalar("theta/cap_total_mvar", float(self.mgr.theta[2]), self.total_step)

                if cfg.log_every > 0 and self.total_step % cfg.log_every == 0:
                    now = time.time()
                    dt = max(now - t_last, 1e-9)
                    total_dt = max(now - t_start, 1e-9)
                    sps = cfg.log_every / dt
                    avg_sps = self.total_step / total_dt
                    eta_min = (cfg.num_frames - self.total_step) / max(avg_sps, 1e-9) / 60.0
                    print(
                        f"[env={cfg.env} seed={cfg.seed}] step {self.total_step}/{cfg.num_frames} "
                        f"({sps:.2f} step/s, avg {avg_sps:.2f}, ETA {eta_min:.1f} min)"
                    )
                    t_last = now

                # episode boundary (manager resets on done)
                if done:
                    scores.append(score)
                    viol_counts.append(viol_cnt)
                    grid_loss_sums.append(gl_sum)
                    pf_fails.append(pf_fail_ep)
                    scores0.append(score0)
                    viol_counts0.append(viol_cnt0)
                    grid_loss_sums0.append(gl_sum0)
                    pf_fails0.append(pf_fail_ep0)

                    if writer is not None:
                        writer.add_scalar("train/episode_score", float(score), self.total_step)
                        writer.add_scalar("train/episode_violation_count", float(viol_cnt), self.total_step)
                        writer.add_scalar("train/episode_grid_loss_sum", float(gl_sum), self.total_step)
                        writer.add_scalar("train/pf_fail_steps", float(pf_fail_ep), self.total_step)

                        writer.add_scalar("train_action0/episode_score", float(score0), self.total_step)
                        writer.add_scalar("train_action0/episode_violation_count", float(viol_cnt0), self.total_step)
                        writer.add_scalar("train_action0/episode_grid_loss_sum", float(gl_sum0), self.total_step)
                        writer.add_scalar("train_action0/pf_fail_steps", float(pf_fail_ep0), self.total_step)

                    score = 0.0
                    viol_cnt = 0.0
                    gl_sum = 0.0
                    pf_fail_ep = 0.0
                    score0 = 0.0
                    viol_cnt0 = 0.0
                    gl_sum0 = 0.0
                    pf_fail_ep0 = 0.0

            # PPO update
            obs = torch.FloatTensor(np.asarray(obs_buf)).to(self.device)
            act = torch.FloatTensor(np.asarray(act_buf)).to(self.device)
            old_logp = torch.FloatTensor(np.asarray(logp_buf)).to(self.device)
            vals = torch.FloatTensor(np.asarray(val_buf)).to(self.device)
            rews = torch.FloatTensor(np.asarray(rew_buf)).to(self.device)
            dones = torch.FloatTensor(np.asarray(done_buf)).to(self.device)

            with torch.no_grad():
                v_last = self.ac._v(torch.FloatTensor(self.state_aug).to(self.device)).detach()
                adv = torch.zeros_like(rews)
                last_gae = 0.0
                for t in reversed(range(len(rews))):
                    if t == len(rews) - 1:
                        next_nonterminal = 1.0 - dones[t]
                        next_values = v_last
                    else:
                        next_nonterminal = 1.0 - dones[t]
                        next_values = vals[t + 1]
                    delta = rews[t] + cfg.gamma * next_values * next_nonterminal - vals[t]
                    last_gae = delta + cfg.gamma * cfg.lam * next_nonterminal * last_gae
                    adv[t] = last_gae
                ret = adv + vals
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            n = len(rews)
            idx = np.arange(n)
            pi_losses_epoch, v_losses_epoch, ent_epoch = [], [], []
            last_kl = 0.0

            for it in range(cfg.train_iters):
                np.random.shuffle(idx)
                for start in range(0, n, cfg.minibatch_size):
                    mb = idx[start : start + cfg.minibatch_size]
                    mb_obs = obs[mb]
                    mb_act = act[mb]
                    mb_old_logp = old_logp[mb]
                    mb_adv = adv[mb]
                    mb_ret = ret[mb]

                    logp, entropy, v_pred = self.ac.logp_and_v(mb_obs, mb_act)
                    ratio = torch.exp(logp - mb_old_logp)
                    clip_adv = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * mb_adv
                    pi_loss = -(torch.min(ratio * mb_adv, clip_adv)).mean()
                    if cfg.entropy_coef != 0.0:
                        pi_loss = pi_loss - cfg.entropy_coef * entropy.mean()
                    v_loss = F.mse_loss(v_pred, mb_ret)
                    approx_kl = (mb_old_logp - logp).mean()
                    last_kl = float(approx_kl.detach().cpu().item())

                    self.pi_opt.zero_grad()
                    pi_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.pi_opt.step()

                    self.vf_opt.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.vf_opt.step()

                    pi_losses_epoch.append(float(pi_loss.detach().cpu().item()))
                    v_losses_epoch.append(float(v_loss.detach().cpu().item()))
                    ent_epoch.append(float(entropy.mean().detach().cpu().item()))

                if last_kl > cfg.target_kl:
                    break

            actor_losses.append(float(np.mean(pi_losses_epoch)) if pi_losses_epoch else 0.0)
            value_losses.append(float(np.mean(v_losses_epoch)) if v_losses_epoch else 0.0)
            entropies.append(float(np.mean(ent_epoch)) if ent_epoch else 0.0)

            if writer is not None:
                writer.add_scalar("loss/policy", actor_losses[-1], self.total_step)
                writer.add_scalar("loss/value", value_losses[-1], self.total_step)
                writer.add_scalar("loss/entropy", entropies[-1], self.total_step)
                writer.add_scalar("loss/approx_kl", float(last_kl), self.total_step)

            self.save_models()

        self._write_csv_and_figures(
            scores,
            viol_counts,
            grid_loss_sums,
            scores0,
            viol_counts0,
            grid_loss_sums0,
            pf_fails,
            pf_fails0,
            actor_losses,
            value_losses,
            entropies,
        )

        if writer is not None:
            writer.flush()
            writer.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO conditioned on varying theta (capacity) for robust control.")
    p.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_frames", type=int, default=96 * 300)
    p.add_argument("--steps_per_update", type=int, default=2048)
    p.add_argument("--minibatch_size", type=int, default=256)
    p.add_argument("--train_iters", type=int, default=10)
    p.add_argument("--pi_lr", type=float, default=3e-4)
    p.add_argument("--vf_lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)

    p.add_argument("--reward_v_weight", type=float, default=200.0)
    p.add_argument("--vmin", type=float, default=0.95)
    p.add_argument("--vmax", type=float, default=1.05)
    p.add_argument("--pf_fail_penalty", type=float, default=1000.0)

    p.add_argument("--log_std_init", type=float, default=-0.5)
    p.add_argument("--log_std_final", type=float, default=-2.0)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=200)

    # theta
    p.add_argument("--enable_pq_curve", action="store_true")
    p.add_argument("--blind_theta", action="store_true", help="Do not append theta to observation (theta-blind baseline).")
    p.add_argument("--pv_s_scale_min", type=float, default=0.95)
    p.add_argument("--pv_s_scale_max", type=float, default=1.05)
    p.add_argument("--svc_q_scale_min", type=float, default=0.95)
    p.add_argument("--svc_q_scale_max", type=float, default=1.05)
    p.add_argument("--cap_total_min", type=float, default=0.0)
    p.add_argument("--cap_total_max", type=float, default=0.5)
    p.add_argument("--cap_buses", type=str, default="", help="Comma-separated cap buses, cap_total is equally distributed.")
    p.add_argument("--episode_len", type=int, default=96)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cap_buses = [int(x) for x in args.cap_buses.split(",") if x.strip() != ""] if args.cap_buses.strip() else None
    cfg = PPOConfig(
        env=args.env,
        seed=args.seed,
        gamma=args.gamma,
        run_name=args.run_name,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        train_iters=args.train_iters,
        minibatch_size=args.minibatch_size,
        steps_per_update=args.steps_per_update,
        num_frames=args.num_frames,
        reward_v_weight=args.reward_v_weight,
        vmin=args.vmin,
        vmax=args.vmax,
        pf_fail_penalty=args.pf_fail_penalty,
        log_std_init=args.log_std_init,
        log_std_final=args.log_std_final,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        log_every=args.log_every,
        enable_pq_curve=bool(args.enable_pq_curve),
        blind_theta=bool(args.blind_theta),
        pv_s_scale_min=args.pv_s_scale_min,
        pv_s_scale_max=args.pv_s_scale_max,
        svc_q_scale_min=args.svc_q_scale_min,
        svc_q_scale_max=args.svc_q_scale_max,
        cap_total_min=args.cap_total_min,
        cap_total_max=args.cap_total_max,
        cap_buses=cap_buses,
        episode_len=args.episode_len,
    )
    set_seed(cfg.seed)
    trainer = PPOTrainerTheta(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
