"""Capacity-Region Directional Constrained PPO for Volt-VAR control.

This experimental trainer keeps line-loss reward, under-voltage cost and
over-voltage cost separate throughout critic fitting and policy optimisation.
Capacity regions receive independent PID Lagrange multipliers so easy hardware
configurations cannot mask unsafe behaviour in constrained regions.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Env
from PPO_theta_robust_film_curriculum import PPOConfig, ThetaEnvManager

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


DIRECTION_UNDER = 0
DIRECTION_OVER = 1
DIRECTION_NAMES = ("under", "over")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalise_theta(
    theta: np.ndarray | torch.Tensor,
    theta_min: np.ndarray | torch.Tensor,
    theta_max: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    scale = theta_max - theta_min
    if isinstance(theta, torch.Tensor):
        return torch.clamp((theta - theta_min) / scale, 0.0, 1.0)
    return np.clip((theta - theta_min) / scale, 0.0, 1.0)


def capacity_group_id(
    theta: Iterable[float],
    theta_min: Iterable[float],
    theta_max: Iterable[float],
) -> int:
    values = np.asarray(list(theta), dtype=np.float64)
    lower = np.asarray(list(theta_min), dtype=np.float64)
    upper = np.asarray(list(theta_max), dtype=np.float64)
    norm = normalise_theta(values, lower, upper)
    bits = (np.asarray(norm) >= 0.5).astype(np.int64)
    return int(bits[0] * 4 + bits[1] * 2 + bits[2])


def directional_voltage_costs(
    vm_pu: np.ndarray,
    vmin: float,
    vmax: float,
    severity_scale: float,
    pf_failed: bool = False,
) -> tuple[float, float, int, int]:
    if pf_failed:
        return 1.0, 1.0, 1, 1
    vm = np.asarray(vm_pu, dtype=np.float64)
    under_excess = np.maximum(0.0, vmin - vm)
    over_excess = np.maximum(0.0, vm - vmax)
    under_event = int(np.any(under_excess > 0.0))
    over_event = int(np.any(over_excess > 0.0))
    under_cost = under_event + severity_scale * float(np.square(under_excess).sum())
    over_cost = over_event + severity_scale * float(np.square(over_excess).sum())
    return under_cost, over_cost, under_event, over_event


class StratifiedCapacitySampler:
    """Uniformly sample within all eight capacity half-boxes in shuffled cycles."""

    def __init__(self, cfg: PPOConfig, seed: int):
        self.lower = np.asarray(
            [cfg.pv_s_scale_min, cfg.svc_q_scale_min, cfg.cap_total_min], dtype=np.float64
        )
        self.upper = np.asarray(
            [cfg.pv_s_scale_max, cfg.svc_q_scale_max, cfg.cap_total_max], dtype=np.float64
        )
        if np.any(self.upper <= self.lower):
            raise ValueError("Every capacity upper bound must exceed its lower bound")
        self.midpoint = (self.lower + self.upper) / 2.0
        self.rng = np.random.default_rng(int(seed) + 9187)
        self._cycle = -1
        self._order = np.arange(8, dtype=np.int64)
        self.last_group_id = 0
        self.last_sample_kind = "group_0"

    def stage(self, episode_idx: int) -> int:
        del episode_idx
        return 3

    def sample(self, episode_idx: int) -> np.ndarray:
        cycle = int(episode_idx) // 8
        position = int(episode_idx) % 8
        if cycle != self._cycle:
            self._order = self.rng.permutation(8)
            self._cycle = cycle
        group_id = int(self._order[position])
        bits = np.asarray([(group_id >> 2) & 1, (group_id >> 1) & 1, group_id & 1])
        low = np.where(bits == 0, self.lower, self.midpoint)
        high = np.where(bits == 0, self.midpoint, self.upper)
        theta = self.rng.uniform(low, high).astype(np.float32)
        self.last_group_id = group_id
        self.last_sample_kind = f"group_{group_id}"
        return theta


class CRDCThetaEnvManager(ThetaEnvManager):
    """Use the existing physical environment with a balanced capacity sampler."""

    def __init__(
        self,
        cfg: PPOConfig,
        load_pu: np.ndarray,
        gene_pu: np.ndarray,
        id_iber: List[int],
        id_svc: List[int],
    ):
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
        self.episode_idx = 0
        self.sampler = StratifiedCapacitySampler(cfg, cfg.seed)
        self.resample_and_reset()


class ConcatDirectionalActorCritic(nn.Module):
    """Direct capacity conditioning with independent reward and safety critics."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        theta_min: Iterable[float],
        theta_max: Iterable[float],
        hidden: tuple[int, int] = (256, 256),
        log_std_init: float = -2.0,
    ):
        super().__init__()
        h1, h2 = hidden
        self.register_buffer("theta_min", torch.tensor(list(theta_min), dtype=torch.float32))
        self.register_buffer("theta_max", torch.tensor(list(theta_max), dtype=torch.float32))
        input_dim = obs_dim + 3

        self.pi_fc1 = nn.Linear(input_dim, h1)
        self.pi_fc2 = nn.Linear(h1, h2)
        self.pi_mu = nn.Linear(h2, act_dim)
        self.pi_log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        self.vr_fc1 = nn.Linear(input_dim, h1)
        self.vr_fc2 = nn.Linear(h1, h2)
        self.vr_out = nn.Linear(h2, 1)

        self.vu_fc1 = nn.Linear(input_dim, h1)
        self.vu_fc2 = nn.Linear(h1, h2)
        self.vu_out = nn.Linear(h2, 1)

        self.vo_fc1 = nn.Linear(input_dim, h1)
        self.vo_fc2 = nn.Linear(h1, h2)
        self.vo_out = nn.Linear(h2, 1)

    def _features(self, obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        norm = normalise_theta(theta, self.theta_min, self.theta_max)
        return torch.cat((obs, norm), dim=-1)

    @staticmethod
    def _trunk(x: torch.Tensor, fc1: nn.Linear, fc2: nn.Linear) -> torch.Tensor:
        return F.relu(fc2(F.relu(fc1(x))))

    def _pi(self, obs: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._features(obs, theta)
        mu = torch.tanh(self.pi_mu(self._trunk(x, self.pi_fc1, self.pi_fc2)))
        return mu, torch.clamp(self.pi_log_std, -20.0, 2.0)

    def _values(
        self, obs: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._features(obs, theta)
        reward_value = self.vr_out(self._trunk(x, self.vr_fc1, self.vr_fc2)).squeeze(-1)
        under_value = self.vu_out(self._trunk(x, self.vu_fc1, self.vu_fc2)).squeeze(-1)
        over_value = self.vo_out(self._trunk(x, self.vo_fc1, self.vo_fc2)).squeeze(-1)
        return reward_value, under_value, over_value

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, theta: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float, float]]:
        mu, log_std = self._pi(obs, theta)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        latent = dist.rsample()
        action = torch.tanh(latent)
        logp = dist.log_prob(latent) - torch.log(1.0 - action.pow(2) + 1e-7)
        values = self._values(obs, theta)
        return (
            action.cpu().numpy(),
            mu.cpu().numpy(),
            float(logp.sum(-1).item()),
            tuple(float(value.item()) for value in values),
        )

    def logp_entropy_values(
        self, obs: torch.Tensor, theta: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._pi(obs, theta)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        bounded = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        latent = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        logp = dist.log_prob(latent) - torch.log(1.0 - bounded.pow(2) + 1e-7)
        entropy = dist.entropy().sum(-1)
        values = self._values(obs, theta)
        return logp.sum(-1), entropy, *values

    def policy_parameters(self):
        for module in (self.pi_fc1, self.pi_fc2, self.pi_mu):
            yield from module.parameters()
        yield self.pi_log_std

    def value_parameters(self):
        for module in (
            self.vr_fc1,
            self.vr_fc2,
            self.vr_out,
            self.vu_fc1,
            self.vu_fc2,
            self.vu_out,
            self.vo_fc1,
            self.vo_fc2,
            self.vo_out,
        ):
            yield from module.parameters()


class GroupPIDLagrange:
    """PID multiplier controller for group x direction event-rate constraints."""

    def __init__(
        self,
        groups: int,
        target: float,
        kp: float,
        ki: float,
        kd: float,
        ema_beta: float,
        initial: float,
        maximum: float,
    ):
        if groups <= 0 or not 0.0 <= target <= 1.0:
            raise ValueError("Invalid group count or event-rate target")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        self.target = float(target)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.ema_beta = float(ema_beta)
        self.maximum = float(maximum)
        self.ema = np.full((groups, 2), self.target, dtype=np.float64)
        self.previous_ema = self.ema.copy()
        self.integral = np.full((groups, 2), float(initial), dtype=np.float64)
        self.multipliers = np.full((groups, 2), float(initial), dtype=np.float64)

    def update(self, rates: np.ndarray, observed: np.ndarray) -> np.ndarray:
        rates = np.asarray(rates, dtype=np.float64)
        observed = np.asarray(observed, dtype=bool)
        if rates.shape != self.multipliers.shape or observed.shape != self.multipliers.shape:
            raise ValueError("rates and observed must have shape [groups, 2]")
        self.previous_ema = self.ema.copy()
        self.ema[observed] = (
            self.ema_beta * self.ema[observed] + (1.0 - self.ema_beta) * rates[observed]
        )
        error = self.ema - self.target
        self.integral[observed] = np.clip(
            self.integral[observed] + self.ki * error[observed], 0.0, self.maximum
        )
        derivative = self.ema - self.previous_ema
        proposal = self.kp * error + self.integral + self.kd * derivative
        self.multipliers[observed] = np.clip(proposal[observed], 0.0, self.maximum)
        return self.multipliers.copy()


@dataclass
class CRDCConfig(PPOConfig):
    actor_arch: str = "concat_directional"
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

    safety_event_target: float = 0.05
    safety_severity_scale: float = 1000.0
    pid_kp: float = 1.0
    pid_ki: float = 0.05
    pid_kd: float = 0.1
    pid_ema_beta: float = 0.8
    lambda_init: float = 1.0
    lambda_max: float = 20.0
    safety_value_coef: float = 1.0


def compute_gae(
    signal: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(signal)
    last_gae = torch.zeros((), dtype=signal.dtype, device=signal.device)
    for index in reversed(range(len(signal))):
        next_value = last_value if index == len(signal) - 1 else values[index + 1]
        nonterminal = 1.0 - dones[index]
        delta = signal[index] + gamma * next_value * nonterminal - values[index]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantage[index] = last_gae
    return advantage, advantage + values


def standardise(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean()) / (value.std(unbiased=False) + 1e-8)


class CRDCPPOTrainer:
    def __init__(self, cfg: CRDCConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_pu = np.load(Env.DATA_DIR / "load96.npy")
        gene_pu = np.load(Env.DATA_DIR / "gen96.npy")
        if cfg.env == 33:
            self.id_iber, self.id_svc = [17, 21, 24], [32]
        elif cfg.env == 69:
            self.id_iber, self.id_svc = [5, 23, 44, 57], [13]
        else:
            self.id_iber, self.id_svc = [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]

        self.mgr = CRDCThetaEnvManager(cfg, load_pu, gene_pu, self.id_iber, self.id_svc)
        assert self.mgr.env is not None
        self.obs_dim = int(self.mgr.env.observation_space.shape[0])
        self.act_dim = int(self.mgr.env.action_space.shape[0])
        self.n_bus = int(len(self.mgr.env.model.bus))
        theta_min = (cfg.pv_s_scale_min, cfg.svc_q_scale_min, cfg.cap_total_min)
        theta_max = (cfg.pv_s_scale_max, cfg.svc_q_scale_max, cfg.cap_total_max)
        self.ac = ConcatDirectionalActorCritic(
            self.obs_dim,
            self.act_dim,
            theta_min,
            theta_max,
            log_std_init=cfg.log_std_init,
        ).to(self.device)
        self.pi_opt = optim.Adam(self.ac.policy_parameters(), lr=cfg.pi_lr)
        self.vf_opt = optim.Adam(self.ac.value_parameters(), lr=cfg.vf_lr)
        self.pid = GroupPIDLagrange(
            8,
            cfg.safety_event_target,
            cfg.pid_kp,
            cfg.pid_ki,
            cfg.pid_kd,
            cfg.pid_ema_beta,
            cfg.lambda_init,
            cfg.lambda_max,
        )

        run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = (
            Path("runs")
            / "CRDC_PPO"
            / f"env{cfg.env}_seed{cfg.seed}_gamma{cfg.gamma}"
            / run_name
        )
        self.csv_dir = self.run_dir / "csv"
        self.model_dir = self.run_dir / "models"
        self.tb_dir = self.run_dir / "tb"
        for directory in (self.csv_dir, self.model_dir, self.tb_dir):
            directory.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(cfg), handle, indent=2)
        self.writer = SummaryWriter(str(self.tb_dir)) if SummaryWriter is not None else None
        self.total_step = 0
        self.update_index = 0
        self.episode_rows: list[dict] = []
        self.update_rows: list[dict] = []

    def _theta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(
                [self.cfg.pv_s_scale_min, self.cfg.svc_q_scale_min, self.cfg.cap_total_min]
            ),
            np.asarray(
                [self.cfg.pv_s_scale_max, self.cfg.svc_q_scale_max, self.cfg.cap_total_max]
            ),
        )

    def _group_id(self, theta: np.ndarray) -> int:
        lower, upper = self._theta_bounds()
        return capacity_group_id(theta, lower, upper)

    def _anneal_log_std(self) -> None:
        fraction = min(self.total_step / max(self.cfg.num_frames, 1), 1.0)
        target = self.cfg.log_std_init + fraction * (
            self.cfg.log_std_final - self.cfg.log_std_init
        )
        with torch.no_grad():
            self.ac.pi_log_std.fill_(float(target))

    def _write_outputs(self) -> None:
        pd.DataFrame(self.episode_rows).to_csv(self.csv_dir / "episodes.csv", index=False)
        pd.DataFrame(self.update_rows).to_csv(self.csv_dir / "updates.csv", index=False)

    def save_checkpoint(self) -> None:
        torch.save(
            {
                "algorithm": "CRDC-PPO",
                "cfg": asdict(self.cfg),
                "ac": self.ac.state_dict(),
                "pid_multipliers": self.pid.multipliers,
                "pid_ema": self.pid.ema,
                "total_step": self.total_step,
                "update_index": self.update_index,
            },
            self.model_dir / "checkpoint.pth",
        )

    def train(self) -> None:
        cfg = self.cfg
        episode_reward = 0.0
        episode_under_steps = 0
        episode_over_steps = 0
        episode_under_event = 0
        episode_over_event = 0
        episode_pf_fail = 0
        episode_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
        episode_group = self._group_id(episode_theta)
        start_time = time.time()

        while self.total_step < cfg.num_frames:
            self._anneal_log_std()
            buffers = {key: [] for key in (
                "obs", "theta", "group", "act", "logp", "vr", "vu", "vo",
                "reward", "under", "over", "done"
            )}
            batch_events = [[[] for _ in DIRECTION_NAMES] for _ in range(8)]
            steps_this_update = min(cfg.steps_per_update, cfg.num_frames - self.total_step)

            for _ in range(int(steps_this_update)):
                # Store the exact pre-step state used to generate action and old_logp.
                current_obs = np.asarray(self.mgr.state, dtype=np.float32).copy()
                current_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                current_group = self._group_id(current_theta)
                obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device)
                theta_tensor = torch.as_tensor(current_theta, dtype=torch.float32, device=self.device)
                action, action_mean, logp, values = self.ac.act(obs_tensor, theta_tensor)
                next_obs, reward_vec, _reward_det, done, info = self.mgr.step(action, action_mean)
                pf_failed = bool(info.get("pf_fail_s", 0.0))
                under_cost, over_cost, under_flag, over_flag = directional_voltage_costs(
                    np.asarray(next_obs[: self.n_bus]),
                    cfg.vmin,
                    cfg.vmax,
                    cfg.safety_severity_scale,
                    pf_failed,
                )

                buffers["obs"].append(current_obs)
                buffers["theta"].append(current_theta)
                buffers["group"].append(current_group)
                buffers["act"].append(np.asarray(action, dtype=np.float32))
                buffers["logp"].append(logp)
                buffers["vr"].append(values[0])
                buffers["vu"].append(values[1])
                buffers["vo"].append(values[2])
                buffers["reward"].append(float(reward_vec[0]))
                buffers["under"].append(under_cost)
                buffers["over"].append(over_cost)
                buffers["done"].append(float(done))

                self.total_step += 1
                episode_reward += float(reward_vec[0])
                episode_under_steps += under_flag
                episode_over_steps += over_flag
                episode_under_event = max(episode_under_event, under_flag)
                episode_over_event = max(episode_over_event, over_flag)
                episode_pf_fail += int(pf_failed)

                if done:
                    batch_events[episode_group][DIRECTION_UNDER].append(episode_under_event)
                    batch_events[episode_group][DIRECTION_OVER].append(episode_over_event)
                    self.episode_rows.append(
                        {
                            "episode": len(self.episode_rows),
                            "total_step": self.total_step,
                            "group_id": episode_group,
                            "theta_pv": float(episode_theta[0]),
                            "theta_svc": float(episode_theta[1]),
                            "theta_cap": float(episode_theta[2]),
                            "line_loss_mwh": -0.25 * episode_reward,
                            "under_steps": episode_under_steps,
                            "over_steps": episode_over_steps,
                            "under_event_day": episode_under_event,
                            "over_event_day": episode_over_event,
                            "pf_fail_steps": episode_pf_fail,
                        }
                    )
                    episode_reward = 0.0
                    episode_under_steps = 0
                    episode_over_steps = 0
                    episode_under_event = 0
                    episode_over_event = 0
                    episode_pf_fail = 0
                    episode_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                    episode_group = self._group_id(episode_theta)

            tensors = {
                key: torch.as_tensor(np.asarray(value), device=self.device)
                for key, value in buffers.items()
            }
            obs = tensors["obs"].float()
            theta = tensors["theta"].float()
            groups = tensors["group"].long()
            actions = tensors["act"].float()
            old_logp = tensors["logp"].float()
            dones = tensors["done"].float()
            values_r, values_u, values_o = tensors["vr"].float(), tensors["vu"].float(), tensors["vo"].float()
            rewards = tensors["reward"].float()
            costs_u, costs_o = tensors["under"].float(), tensors["over"].float()

            with torch.no_grad():
                last_obs = torch.as_tensor(self.mgr.state, dtype=torch.float32, device=self.device)
                last_theta = torch.as_tensor(self.mgr.theta, dtype=torch.float32, device=self.device)
                last_r, last_u, last_o = self.ac._values(last_obs, last_theta)
                adv_r, ret_r = compute_gae(rewards, values_r, dones, last_r, cfg.gamma, cfg.lam)
                adv_u, ret_u = compute_gae(costs_u, values_u, dones, last_u, cfg.gamma, cfg.lam)
                adv_o, ret_o = compute_gae(costs_o, values_o, dones, last_o, cfg.gamma, cfg.lam)
                adv_r, adv_u, adv_o = standardise(adv_r), standardise(adv_u), standardise(adv_o)

            rates = np.zeros((8, 2), dtype=np.float64)
            observed = np.zeros((8, 2), dtype=bool)
            for group_id in range(8):
                for direction in range(2):
                    events = batch_events[group_id][direction]
                    if events:
                        rates[group_id, direction] = float(np.mean(events))
                        observed[group_id, direction] = True
            multipliers = self.pid.update(rates, observed)
            multiplier_tensor = torch.as_tensor(multipliers, dtype=torch.float32, device=self.device)
            combined_adv = (
                adv_r
                - multiplier_tensor[groups, DIRECTION_UNDER] * adv_u
                - multiplier_tensor[groups, DIRECTION_OVER] * adv_o
            )

            indices = np.arange(len(rewards))
            policy_losses: list[float] = []
            reward_value_losses: list[float] = []
            under_value_losses: list[float] = []
            over_value_losses: list[float] = []
            entropies: list[float] = []
            last_kl = 0.0
            for _iteration in range(cfg.train_iters):
                np.random.shuffle(indices)
                for start in range(0, len(indices), cfg.minibatch_size):
                    mb = indices[start : start + cfg.minibatch_size]
                    mb_group = groups[mb]
                    logp, entropy, pred_r, pred_u, pred_o = self.ac.logp_entropy_values(
                        obs[mb], theta[mb], actions[mb]
                    )
                    ratio = torch.exp(logp - old_logp[mb])
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                    surrogate = torch.minimum(ratio * combined_adv[mb], clipped * combined_adv[mb])
                    group_objectives = [
                        surrogate[mb_group == group_id].mean()
                        for group_id in range(8)
                        if torch.any(mb_group == group_id)
                    ]
                    policy_loss = -torch.stack(group_objectives).mean()
                    if cfg.entropy_coef:
                        policy_loss -= cfg.entropy_coef * entropy.mean()

                    self.pi_opt.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.ac.policy_parameters()), cfg.max_grad_norm)
                    self.pi_opt.step()

                    loss_r = F.mse_loss(pred_r, ret_r[mb])
                    loss_u = F.mse_loss(pred_u, ret_u[mb])
                    loss_o = F.mse_loss(pred_o, ret_o[mb])
                    value_loss = loss_r + cfg.safety_value_coef * (loss_u + loss_o)
                    self.vf_opt.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.ac.value_parameters()), cfg.max_grad_norm)
                    self.vf_opt.step()

                    last_kl = float((old_logp[mb] - logp).mean().detach().cpu())
                    policy_losses.append(float(policy_loss.detach().cpu()))
                    reward_value_losses.append(float(loss_r.detach().cpu()))
                    under_value_losses.append(float(loss_u.detach().cpu()))
                    over_value_losses.append(float(loss_o.detach().cpu()))
                    entropies.append(float(entropy.mean().detach().cpu()))
                if last_kl > cfg.target_kl:
                    break

            self.update_index += 1
            row = {
                "update": self.update_index,
                "total_step": self.total_step,
                "policy_loss": float(np.mean(policy_losses)),
                "reward_value_loss": float(np.mean(reward_value_losses)),
                "under_value_loss": float(np.mean(under_value_losses)),
                "over_value_loss": float(np.mean(over_value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": last_kl,
            }
            for group_id in range(8):
                for direction, name in enumerate(DIRECTION_NAMES):
                    row[f"rate_g{group_id}_{name}"] = (
                        rates[group_id, direction] if observed[group_id, direction] else np.nan
                    )
                    row[f"lambda_g{group_id}_{name}"] = multipliers[group_id, direction]
            self.update_rows.append(row)
            self._write_outputs()
            self.save_checkpoint()

            if self.writer is not None:
                self.writer.add_scalar("loss/policy", row["policy_loss"], self.total_step)
                self.writer.add_scalar("loss/value_reward", row["reward_value_loss"], self.total_step)
                self.writer.add_scalar("loss/value_under", row["under_value_loss"], self.total_step)
                self.writer.add_scalar("loss/value_over", row["over_value_loss"], self.total_step)
                for group_id in range(8):
                    for direction, name in enumerate(DIRECTION_NAMES):
                        self.writer.add_scalar(
                            f"lambda/group_{group_id}_{name}",
                            multipliers[group_id, direction],
                            self.total_step,
                        )

            elapsed = max(time.time() - start_time, 1e-9)
            eta_minutes = (cfg.num_frames - self.total_step) / max(self.total_step / elapsed, 1e-9) / 60.0
            print(
                f"[CRDC env={cfg.env} seed={cfg.seed}] {self.total_step}/{cfg.num_frames} "
                f"update={self.update_index} ETA={eta_minutes:.1f} min "
                f"max_lambda={multipliers.max():.3f}"
            )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capacity-region directional constrained PPO")
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=96 * 300)
    parser.add_argument("--steps_per_update", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--train_iters", type=int, default=10)
    parser.add_argument("--pi_lr", type=float, default=3e-4)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--log_std_init", type=float, default=-2.0)
    parser.add_argument("--log_std_final", type=float, default=-3.0)
    parser.add_argument("--vmin", type=float, default=0.95)
    parser.add_argument("--vmax", type=float, default=1.05)
    parser.add_argument("--pf_fail_penalty", type=float, default=1000.0)
    parser.add_argument("--safety_event_target", type=float, default=0.05)
    parser.add_argument("--safety_severity_scale", type=float, default=1000.0)
    parser.add_argument("--pid_kp", type=float, default=1.0)
    parser.add_argument("--pid_ki", type=float, default=0.05)
    parser.add_argument("--pid_kd", type=float, default=0.1)
    parser.add_argument("--pid_ema_beta", type=float, default=0.8)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=20.0)
    parser.add_argument("--pv_s_scale_min", type=float, default=0.7)
    parser.add_argument("--pv_s_scale_max", type=float, default=1.5)
    parser.add_argument("--svc_q_scale_min", type=float, default=0.7)
    parser.add_argument("--svc_q_scale_max", type=float, default=1.5)
    parser.add_argument("--cap_total_min", type=float, default=0.0)
    parser.add_argument("--cap_total_max", type=float, default=1.0)
    parser.add_argument("--cap_buses", type=str, default="20,8")
    parser.add_argument("--episode_len", type=int, default=96)
    parser.add_argument("--enable_pq_curve", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cap_buses = [int(value) for value in args.cap_buses.split(",") if value.strip()]
    cfg = CRDCConfig(
        env=args.env,
        seed=args.seed,
        gamma=args.gamma,
        run_name=args.run_name,
        num_frames=args.num_frames,
        steps_per_update=args.steps_per_update,
        minibatch_size=args.minibatch_size,
        train_iters=args.train_iters,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        entropy_coef=args.entropy_coef,
        log_std_init=args.log_std_init,
        log_std_final=args.log_std_final,
        vmin=args.vmin,
        vmax=args.vmax,
        pf_fail_penalty=args.pf_fail_penalty,
        safety_event_target=args.safety_event_target,
        safety_severity_scale=args.safety_severity_scale,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
        pid_ema_beta=args.pid_ema_beta,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        pv_s_scale_min=args.pv_s_scale_min,
        pv_s_scale_max=args.pv_s_scale_max,
        svc_q_scale_min=args.svc_q_scale_min,
        svc_q_scale_max=args.svc_q_scale_max,
        cap_total_min=args.cap_total_min,
        cap_total_max=args.cap_total_max,
        cap_buses=cap_buses,
        episode_len=args.episode_len,
        enable_pq_curve=bool(args.enable_pq_curve),
    )
    set_seed(cfg.seed)
    CRDCPPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
