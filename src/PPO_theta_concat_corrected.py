"""Transition-corrected scalarised Concat PPO baseline for CRDC-PPO."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Env
from PPO_theta_crdc import CRDCThetaEnvManager, compute_gae, normalise_theta, set_seed, standardise
from PPO_theta_robust_film_curriculum import PPOConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class ConcatScalarActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        theta_min: Iterable[float],
        theta_max: Iterable[float],
        hidden: tuple[int, int] = (256, 256),
        log_std_init: float = -2.0,
        condition_on_theta: bool = True,
    ):
        super().__init__()
        h1, h2 = hidden
        self.register_buffer("theta_min", torch.tensor(list(theta_min), dtype=torch.float32))
        self.register_buffer("theta_max", torch.tensor(list(theta_max), dtype=torch.float32))
        self.condition_on_theta = bool(condition_on_theta)
        input_dim = obs_dim + (3 if self.condition_on_theta else 0)
        self.pi_fc1 = nn.Linear(input_dim, h1)
        self.pi_fc2 = nn.Linear(h1, h2)
        self.pi_mu = nn.Linear(h2, act_dim)
        self.pi_log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.v_fc1 = nn.Linear(input_dim, h1)
        self.v_fc2 = nn.Linear(h1, h2)
        self.v_out = nn.Linear(h2, 1)

    def _features(self, obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if not self.condition_on_theta:
            return obs
        norm = normalise_theta(theta, self.theta_min, self.theta_max)
        return torch.cat((obs, norm), dim=-1)

    def _pi(self, obs: torch.Tensor, theta: torch.Tensor):
        x = F.relu(self.pi_fc1(self._features(obs, theta)))
        x = F.relu(self.pi_fc2(x))
        return torch.tanh(self.pi_mu(x)), torch.clamp(self.pi_log_std, -20.0, 2.0)

    def _v(self, obs: torch.Tensor, theta: torch.Tensor):
        x = F.relu(self.v_fc1(self._features(obs, theta)))
        x = F.relu(self.v_fc2(x))
        return self.v_out(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, theta: torch.Tensor):
        mu, log_std = self._pi(obs, theta)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        latent = dist.rsample()
        action = torch.tanh(latent)
        logp = dist.log_prob(latent) - torch.log(1.0 - action.pow(2) + 1e-7)
        return action.cpu().numpy(), mu.cpu().numpy(), float(logp.sum(-1)), float(self._v(obs, theta))

    def logp_entropy_value(self, obs: torch.Tensor, theta: torch.Tensor, action: torch.Tensor):
        mu, log_std = self._pi(obs, theta)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        bounded = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        latent = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        logp = dist.log_prob(latent) - torch.log(1.0 - bounded.pow(2) + 1e-7)
        return logp.sum(-1), dist.entropy().sum(-1), self._v(obs, theta)

    def policy_parameters(self):
        for module in (self.pi_fc1, self.pi_fc2, self.pi_mu):
            yield from module.parameters()
        yield self.pi_log_std

    def value_parameters(self):
        for module in (self.v_fc1, self.v_fc2, self.v_out):
            yield from module.parameters()


class BlindScalarActorCritic(ConcatScalarActorCritic):
    """Matched scalar PPO baseline that cannot observe capacity parameters."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        theta_min: Iterable[float],
        theta_max: Iterable[float],
        hidden: tuple[int, int] = (256, 256),
        log_std_init: float = -2.0,
    ):
        super().__init__(
            obs_dim,
            act_dim,
            theta_min,
            theta_max,
            hidden=hidden,
            log_std_init=log_std_init,
            condition_on_theta=False,
        )


@dataclass
class CorrectedConcatConfig(PPOConfig):
    actor_arch: str = "concat_scalar_corrected"
    stage1_frac: float = 0.0
    stage2_frac: float = 0.0
    pv_s_scale_min: float = 0.7
    pv_s_scale_max: float = 1.5
    svc_q_scale_min: float = 0.7
    svc_q_scale_max: float = 1.5
    cap_total_min: float = 0.0
    cap_total_max: float = 1.0
    log_std_init: float = -2.0
    log_std_final: float = -3.0
    reward_v_weight: float = 200.0
    blind_theta: bool = False


class CorrectedConcatTrainer:
    def __init__(self, cfg: CorrectedConcatConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_pu = np.load(Env.DATA_DIR / "load96.npy")
        gene_pu = np.load(Env.DATA_DIR / "gen96.npy")
        if cfg.env == 33:
            id_iber, id_svc = [17, 21, 24], [32]
        elif cfg.env == 69:
            id_iber, id_svc = [5, 23, 44, 57], [13]
        else:
            id_iber, id_svc = [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]
        self.mgr = CRDCThetaEnvManager(cfg, load_pu, gene_pu, id_iber, id_svc)
        assert self.mgr.env is not None
        actor_class = BlindScalarActorCritic if cfg.blind_theta else ConcatScalarActorCritic
        self.ac = actor_class(
            len(self.mgr.env.observation_space),
            len(self.mgr.env.action_space),
            [cfg.pv_s_scale_min, cfg.svc_q_scale_min, cfg.cap_total_min],
            [cfg.pv_s_scale_max, cfg.svc_q_scale_max, cfg.cap_total_max],
            log_std_init=cfg.log_std_init,
        ).to(self.device)
        self.pi_opt = optim.Adam(self.ac.policy_parameters(), lr=cfg.pi_lr)
        self.vf_opt = optim.Adam(self.ac.value_parameters(), lr=cfg.vf_lr)
        run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        run_root = "BLIND_SCALAR_CORRECTED" if cfg.blind_theta else "CONCAT_SCALAR_CORRECTED"
        self.run_dir = Path("runs") / run_root / f"env{cfg.env}_seed{cfg.seed}_gamma{cfg.gamma}" / run_name
        self.csv_dir, self.model_dir, self.tb_dir = self.run_dir / "csv", self.run_dir / "models", self.run_dir / "tb"
        for directory in (self.csv_dir, self.model_dir, self.tb_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        self.writer = SummaryWriter(str(self.tb_dir)) if SummaryWriter is not None else None
        self.total_step = 0
        self.update_index = 0
        self.episode_rows: list[dict] = []
        self.update_rows: list[dict] = []

    def _group_id(self, theta: np.ndarray) -> int:
        midpoint = np.asarray([
            (self.cfg.pv_s_scale_min + self.cfg.pv_s_scale_max) / 2,
            (self.cfg.svc_q_scale_min + self.cfg.svc_q_scale_max) / 2,
            (self.cfg.cap_total_min + self.cfg.cap_total_max) / 2,
        ])
        bits = np.asarray(theta) >= midpoint
        return int(bits[0] * 4 + bits[1] * 2 + bits[2])

    def _anneal_log_std(self):
        fraction = min(self.total_step / max(self.cfg.num_frames, 1), 1.0)
        value = self.cfg.log_std_init + fraction * (self.cfg.log_std_final - self.cfg.log_std_init)
        with torch.no_grad():
            self.ac.pi_log_std.fill_(float(value))

    def _save(self):
        payload = {
            "algorithm": (
                "Blind-Scalar-PPO-Corrected"
                if self.cfg.blind_theta
                else "Concat-Scalar-PPO-Corrected"
            ),
            "cfg": asdict(self.cfg),
            "ac": self.ac.state_dict(),
            "total_step": self.total_step,
            "update_index": self.update_index,
        }
        torch.save(payload, self.model_dir / "checkpoint.pth")
        torch.save(self.ac.state_dict(), self.model_dir / "actor.pth")
        pd.DataFrame(self.episode_rows).to_csv(self.csv_dir / "episodes.csv", index=False)
        pd.DataFrame(self.update_rows).to_csv(self.csv_dir / "updates.csv", index=False)

    def train(self):
        cfg = self.cfg
        episode_loss_reward = 0.0
        episode_violation_steps = 0
        episode_pf_fail = 0
        episode_theta = np.asarray(self.mgr.theta).copy()
        start_time = time.time()
        while self.total_step < cfg.num_frames:
            self._anneal_log_std()
            buffers = {key: [] for key in ("obs", "theta", "group", "act", "logp", "value", "reward", "done")}
            steps = min(cfg.steps_per_update, cfg.num_frames - self.total_step)
            for _ in range(int(steps)):
                # Action, old_logp and buffer observation use this same pre-step state.
                obs = np.asarray(self.mgr.state, dtype=np.float32).copy()
                theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                group = self._group_id(theta)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                theta_t = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
                action, mean_action, logp, value = self.ac.act(obs_t, theta_t)
                _next_obs, reward_vec, _reward_det, done, info = self.mgr.step(action, mean_action)
                scalar_reward = float(reward_vec[0] + cfg.reward_v_weight * reward_vec[1])
                for key, value_item in (
                    ("obs", obs), ("theta", theta), ("group", group), ("act", action),
                    ("logp", logp), ("value", value), ("reward", scalar_reward), ("done", float(done)),
                ):
                    buffers[key].append(value_item)
                self.total_step += 1
                episode_loss_reward += float(reward_vec[0])
                episode_violation_steps += int(info.get("vflag_s", 0.0))
                episode_pf_fail += int(info.get("pf_fail_s", 0.0))
                if done:
                    self.episode_rows.append({
                        "episode": len(self.episode_rows), "total_step": self.total_step,
                        "group_id": self._group_id(episode_theta),
                        "theta_pv": float(episode_theta[0]), "theta_svc": float(episode_theta[1]),
                        "theta_cap": float(episode_theta[2]), "line_loss_mwh": -0.25 * episode_loss_reward,
                        "violation_steps": episode_violation_steps,
                        "event_day": int(bool(episode_violation_steps or episode_pf_fail)),
                        "pf_fail_steps": episode_pf_fail,
                    })
                    episode_loss_reward = 0.0
                    episode_violation_steps = 0
                    episode_pf_fail = 0
                    episode_theta = np.asarray(self.mgr.theta).copy()

            tensors = {key: torch.as_tensor(np.asarray(value), device=self.device) for key, value in buffers.items()}
            obs, theta = tensors["obs"].float(), tensors["theta"].float()
            groups, actions = tensors["group"].long(), tensors["act"].float()
            old_logp, values = tensors["logp"].float(), tensors["value"].float()
            rewards, dones = tensors["reward"].float(), tensors["done"].float()
            with torch.no_grad():
                last_value = self.ac._v(
                    torch.as_tensor(self.mgr.state, dtype=torch.float32, device=self.device),
                    torch.as_tensor(self.mgr.theta, dtype=torch.float32, device=self.device),
                )
                advantage, returns = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.lam)
                advantage = standardise(advantage)

            indices = np.arange(len(rewards))
            policy_losses, value_losses, entropies = [], [], []
            last_kl = 0.0
            for _iteration in range(cfg.train_iters):
                np.random.shuffle(indices)
                for start in range(0, len(indices), cfg.minibatch_size):
                    mb = indices[start : start + cfg.minibatch_size]
                    logp, entropy, predicted = self.ac.logp_entropy_value(obs[mb], theta[mb], actions[mb])
                    ratio = torch.exp(logp - old_logp[mb])
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                    surrogate = torch.minimum(ratio * advantage[mb], clipped * advantage[mb])
                    mb_groups = groups[mb]
                    group_terms = [surrogate[mb_groups == group].mean() for group in range(8) if torch.any(mb_groups == group)]
                    policy_loss = -torch.stack(group_terms).mean()
                    if cfg.entropy_coef:
                        policy_loss -= cfg.entropy_coef * entropy.mean()
                    self.pi_opt.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.ac.policy_parameters()), cfg.max_grad_norm)
                    self.pi_opt.step()
                    value_loss = F.mse_loss(predicted, returns[mb])
                    self.vf_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.ac.value_parameters()), cfg.max_grad_norm)
                    self.vf_opt.step()
                    last_kl = float((old_logp[mb] - logp).mean().detach().cpu())
                    policy_losses.append(float(policy_loss.detach().cpu()))
                    value_losses.append(float(value_loss.detach().cpu()))
                    entropies.append(float(entropy.mean().detach().cpu()))
                if last_kl > cfg.target_kl:
                    break
            self.update_index += 1
            row = {
                "update": self.update_index, "total_step": self.total_step,
                "policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)), "approx_kl": last_kl,
            }
            self.update_rows.append(row)
            self._save()
            if self.writer is not None:
                for key, value in row.items():
                    if key not in ("update", "total_step"):
                        self.writer.add_scalar(f"update/{key}", value, self.total_step)
            elapsed = max(time.time() - start_time, 1e-9)
            eta = (cfg.num_frames - self.total_step) / max(self.total_step / elapsed, 1e-9) / 60.0
            label = "Blind" if cfg.blind_theta else "Concat"
            print(
                f"[Corrected {label} seed={cfg.seed}] {self.total_step}/{cfg.num_frames} "
                f"update={self.update_index} ETA={eta:.1f} min",
                flush=True,
            )
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Transition-corrected scalar Concat PPO")
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--num_frames", type=int, default=28800)
    parser.add_argument("--steps_per_update", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--reward_v_weight", type=float, default=200.0)
    parser.add_argument("--log_std_init", type=float, default=-2.0)
    parser.add_argument("--log_std_final", type=float, default=-3.0)
    parser.add_argument("--pv_s_scale_min", type=float, default=0.7)
    parser.add_argument("--pv_s_scale_max", type=float, default=1.5)
    parser.add_argument("--svc_q_scale_min", type=float, default=0.7)
    parser.add_argument("--svc_q_scale_max", type=float, default=1.5)
    parser.add_argument("--cap_total_min", type=float, default=0.0)
    parser.add_argument("--cap_total_max", type=float, default=1.0)
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--episode_len", type=int, default=96)
    parser.add_argument("--enable_pq_curve", action="store_true")
    parser.add_argument("--blind_theta", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = CorrectedConcatConfig(
        env=args.env, seed=args.seed, gamma=args.gamma, run_name=args.run_name,
        num_frames=args.num_frames, steps_per_update=args.steps_per_update,
        minibatch_size=args.minibatch_size, train_iters=args.train_iters,
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, lam=args.lam, clip_ratio=args.clip_ratio,
        target_kl=args.target_kl, reward_v_weight=args.reward_v_weight,
        log_std_init=args.log_std_init, log_std_final=args.log_std_final,
        pv_s_scale_min=args.pv_s_scale_min, pv_s_scale_max=args.pv_s_scale_max,
        svc_q_scale_min=args.svc_q_scale_min, svc_q_scale_max=args.svc_q_scale_max,
        cap_total_min=args.cap_total_min, cap_total_max=args.cap_total_max,
        cap_buses=[int(value) for value in args.cap_buses.split(",") if value.strip()],
        episode_len=args.episode_len, enable_pq_curve=bool(args.enable_pq_curve),
        blind_theta=bool(args.blind_theta),
        actor_arch=("blind_scalar_corrected" if args.blind_theta else "concat_scalar_corrected"),
    )
    set_seed(cfg.seed)
    CorrectedConcatTrainer(cfg).train()


if __name__ == "__main__":
    main()
