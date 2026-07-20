"""
PPO with FiLM-modulated conditioning on capacity parameters theta: learn pi(a | s, theta).

Innovations vs naive concat:
  1) FiLM (Feature-wise Linear Modulation): theta produces per-feature (gamma, beta) that modulate hidden activations.
  2) Curriculum Domain Randomization (CDR):
      - stage1: theta fixed to nominal (1.0, 1.0, cap0) so the agent learns basic control.
      - stage2: theta small perturbation (e.g., 0.9~1.1).
      - stage3: theta full randomization (e.g., 0.5~2.0).

Outputs (aligned style):
  runs/PPO_THETA_FILM/env{env}_seed{seed}_gamma{gamma}/<run_name>/
    tb/ csv/ figures/ models/
"""

from __future__ import annotations

import os
import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

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


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if str(x).strip() != ""]


def capacity_minimum_margin(
    theta: np.ndarray,
    theta_min: Iterable[float],
    theta_max: Iterable[float],
) -> float:
    """Return the clipped minimum normalised margin used by Scheme E."""
    theta_arr = np.asarray(theta, dtype=np.float32)
    lower = np.asarray(list(theta_min), dtype=np.float32)
    upper = np.asarray(list(theta_max), dtype=np.float32)
    if theta_arr.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("theta and its bounds must each contain three values")
    if np.any(upper <= lower):
        raise ValueError("Each theta_max value must be greater than theta_min")
    normalised = np.clip((theta_arr - lower) / (upper - lower), 0.0, 1.0)
    return float(np.min(normalised))


def margin_weighted_voltage_penalty(base_weight: float, margin_gain: float, minimum_margin: float) -> float:
    """Compute w_v(theta) = w_v * [1 + gain * (1 - margin)]."""
    if margin_gain < 0.0:
        raise ValueError("margin_gain must be non-negative")
    margin = float(np.clip(minimum_margin, 0.0, 1.0))
    return float(base_weight) * (1.0 + float(margin_gain) * (1.0 - margin))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: x <- x * gamma(theta) + beta(theta)."""

    def __init__(self, theta_dim: int, feat_dim: int):
        super().__init__()
        self.scale = nn.Linear(theta_dim, feat_dim)
        self.shift = nn.Linear(theta_dim, feat_dim)

        # initialize near-identity modulation
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        gamma = self.scale(theta)
        beta = self.shift(theta)
        return x * (1.0 + gamma) + beta


class FiLMActorCritic(nn.Module):
    """
    Actor-Critic where obs features are modulated by theta using FiLM.
    Policy remains tanh-Gaussian (same math as PPO_theta_robust.py), but theta is not concatenated.
    """

    def __init__(self, obs_dim: int, act_dim: int, theta_dim: int = 3, hidden: tuple[int, int] = (256, 256), log_std_init: float = -0.5):
        super().__init__()
        h1, h2 = hidden

        # policy trunk
        self.pi_fc1 = nn.Linear(obs_dim, h1)
        self.pi_film1 = FiLM(theta_dim, h1)
        self.pi_fc2 = nn.Linear(h1, h2)
        self.pi_film2 = FiLM(theta_dim, h2)
        self.pi_mu = nn.Linear(h2, act_dim)
        self.pi_log_std = nn.Parameter(torch.ones(act_dim) * float(log_std_init))

        # value trunk
        self.v_fc1 = nn.Linear(obs_dim, h1)
        self.v_film1 = FiLM(theta_dim, h1)
        self.v_fc2 = nn.Linear(h1, h2)
        self.v_film2 = FiLM(theta_dim, h2)
        self.v_out = nn.Linear(h2, 1)

    def _pi(self, obs: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.pi_fc1(obs))
        x = self.pi_film1(x, theta)
        x = F.relu(x)
        x = F.relu(self.pi_fc2(x))
        x = self.pi_film2(x, theta)
        x = F.relu(x)
        mu = torch.tanh(self.pi_mu(x))
        log_std = torch.clamp(self.pi_log_std, -20, 2)
        return mu, log_std

    def _v(self, obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.v_fc1(obs))
        x = self.v_film1(x, theta)
        x = F.relu(x)
        x = F.relu(self.v_fc2(x))
        x = self.v_film2(x, theta)
        x = F.relu(x)
        return self.v_out(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, theta: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, float, float]:
        mu, log_std = self._pi(obs, theta)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-7)
        logp = logp.sum(-1).item()
        v = self._v(obs, theta).item()
        return a.cpu().numpy(), mu.cpu().numpy(), logp, v

    def logp_and_v(self, obs: torch.Tensor, theta: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self._pi(obs, theta)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        eps = 1e-6
        a = torch.clamp(act, -1 + eps, 1 - eps)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        logp = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-7)
        logp = logp.sum(-1)
        entropy = dist.entropy().sum(-1)
        v = self._v(obs, theta)
        return logp, entropy, v

    def policy_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.pi_fc1, self.pi_film1, self.pi_fc2, self.pi_film2, self.pi_mu)
        for module in modules:
            yield from module.parameters()
        yield self.pi_log_std

    def value_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.v_fc1, self.v_film1, self.v_fc2, self.v_film2, self.v_out)
        for module in modules:
            yield from module.parameters()


class ThetaConditionFeatures(nn.Module):
    """Explicit capacity geometry for the actor-side residual pathway."""

    feature_dim = 8

    def __init__(self, theta_min: Iterable[float], theta_max: Iterable[float]):
        super().__init__()
        theta_min_t = torch.as_tensor(list(theta_min), dtype=torch.float32)
        theta_max_t = torch.as_tensor(list(theta_max), dtype=torch.float32)
        if theta_min_t.shape != (3,) or theta_max_t.shape != (3,):
            raise ValueError("theta_min and theta_max must each contain three values")
        if torch.any(theta_max_t <= theta_min_t):
            raise ValueError("Each theta_max value must be greater than theta_min")
        self.register_buffer("theta_min", theta_min_t)
        self.register_buffer("theta_max", theta_max_t)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        normalised = (theta - self.theta_min) / (self.theta_max - self.theta_min)
        normalised = torch.clamp(normalised, 0.0, 1.0)
        minimum_margin = normalised.min(dim=-1, keepdim=True).values
        double_low = ((1.0 - normalised[..., 0]) * (1.0 - normalised[..., 1])).unsqueeze(-1)
        # 3 normalised capacities + 1 global margin + 1 PV/SVC interaction + 3 raw capacities.
        return torch.cat((normalised, minimum_margin, double_low, theta), dim=-1)


class HybridFiLMActorCritic(FiLMActorCritic):
    """FiLM policy with zero-initialised direct capacity residuals.

    Only the actor is changed. The critic remains the original FiLM critic so
    the v1 experiment isolates actor-side conditional expressivity.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        theta_dim: int = 3,
        hidden: tuple[int, int] = (256, 256),
        log_std_init: float = -0.5,
        theta_min: Iterable[float] = (0.7, 0.7, 0.0),
        theta_max: Iterable[float] = (1.5, 1.5, 1.0),
    ):
        if theta_dim != 3:
            raise ValueError("Hybrid residual features currently require theta_dim=3")
        super().__init__(obs_dim, act_dim, theta_dim, hidden, log_std_init)
        h1, h2 = hidden
        self.theta_features = ThetaConditionFeatures(theta_min, theta_max)
        feature_dim = self.theta_features.feature_dim
        self.pi_res1 = nn.Linear(feature_dim, h1, bias=False)
        self.pi_res2 = nn.Linear(feature_dim, h2, bias=False)
        self.pi_res_out = nn.Linear(feature_dim, act_dim, bias=False)
        for module in (self.pi_res1, self.pi_res2, self.pi_res_out):
            nn.init.zeros_(module.weight)

    def _actor_terms(
        self, obs: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        phi = self.theta_features(theta)

        hidden1 = F.relu(self.pi_fc1(obs))
        film1 = self.pi_film1(hidden1, theta)
        residual1 = self.pi_res1(phi)
        hidden1_out = F.relu(film1 + residual1)

        hidden2 = F.relu(self.pi_fc2(hidden1_out))
        film2 = self.pi_film2(hidden2, theta)
        residual2 = self.pi_res2(phi)
        hidden2_out = F.relu(film2 + residual2)

        base_logits = self.pi_mu(hidden2_out)
        output_residual = self.pi_res_out(phi)
        logits = base_logits + output_residual
        terms = {
            "film1": film1,
            "residual1": residual1,
            "film2": film2,
            "residual2": residual2,
            "base_logits": base_logits,
            "output_residual": output_residual,
        }
        return logits, terms

    def _pi(self, obs: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self._actor_terms(obs, theta)
        mu = torch.tanh(logits)
        log_std = torch.clamp(self.pi_log_std, -20, 2)
        return mu, log_std

    def actor_conditioning_diagnostics(self, obs: torch.Tensor, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        _, terms = self._actor_terms(obs, theta)

        def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
            return numerator.norm(dim=-1) / denominator.norm(dim=-1).clamp_min(1e-8)

        return {
            "residual_ratio_layer1": ratio(terms["residual1"], terms["film1"]),
            "residual_ratio_layer2": ratio(terms["residual2"], terms["film2"]),
            "residual_ratio_output": ratio(terms["output_residual"], terms["base_logits"]),
            "residual_norm_layer1": terms["residual1"].norm(dim=-1),
            "residual_norm_layer2": terms["residual2"].norm(dim=-1),
            "residual_norm_output": terms["output_residual"].norm(dim=-1),
        }

    def policy_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().policy_parameters()
        for module in (self.pi_res1, self.pi_res2, self.pi_res_out):
            yield from module.parameters()


def build_film_actor_from_state_dict(
    obs_dim: int,
    act_dim: int,
    state_dict: dict[str, torch.Tensor],
) -> FiLMActorCritic:
    """Rebuild legacy FiLM or hybrid-residual actors from checkpoint keys."""
    h1 = int(state_dict["pi_fc1.weight"].shape[0])
    h2 = int(state_dict["pi_fc2.weight"].shape[0])
    theta_dim = int(state_dict["pi_film1.scale.weight"].shape[1])
    if "pi_res1.weight" in state_dict:
        if "theta_features.theta_min" not in state_dict or "theta_features.theta_max" not in state_dict:
            raise RuntimeError("Hybrid checkpoint is missing saved theta bounds")
        actor: FiLMActorCritic = HybridFiLMActorCritic(
            obs_dim,
            act_dim,
            theta_dim=theta_dim,
            hidden=(h1, h2),
            theta_min=state_dict["theta_features.theta_min"].detach().cpu().tolist(),
            theta_max=state_dict["theta_features.theta_max"].detach().cpu().tolist(),
        )
    else:
        actor = FiLMActorCritic(obs_dim, act_dim, theta_dim=theta_dim, hidden=(h1, h2))
    actor.load_state_dict(state_dict)
    return actor


def extract_actor_state_dict(payload: object) -> dict[str, torch.Tensor]:
    """Normalise the checkpoint formats used by the repository."""
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    for key in ("ac", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload  # A plain state_dict is itself a dictionary.


@dataclass
class PPOConfig:
    env: int = 33
    seed: int = 42
    gamma: float = 0.9
    run_name: str = ""
    actor_arch: str = "film"

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
    reward_v_weight: float = 200.0
    reward_v_margin_gain: float = 0.0
    vmin: float = 0.95
    vmax: float = 1.05
    pf_fail_penalty: float = 1000.0

    # exploration anneal
    log_std_init: float = -0.5
    log_std_final: float = -2.0
    entropy_coef: float = 0.0

    # theta curriculum DR
    enable_pq_curve: bool = True
    cap_buses: List[int] | None = None
    cap0: float = 0.0  # stage1 fixed cap_total

    stage1_frac: float = 0.2  # fixed theta
    stage2_frac: float = 0.6  # small perturbation until this fraction, then stage3

    # stage2 (mid) ranges
    pv_s_mid_min: float = 0.9
    pv_s_mid_max: float = 1.1
    svc_q_mid_min: float = 0.9
    svc_q_mid_max: float = 1.1
    cap_mid_min: float = 0.0
    cap_mid_max: float = 0.5

    # stage3 (final) ranges
    # More realistic device capacity uncertainty ranges for power systems.
    # Default to 0.8~1.2; you can still override to 0.7~1.5 if desired.
    pv_s_scale_min: float = 0.8
    pv_s_scale_max: float = 1.2
    svc_q_scale_min: float = 0.8
    svc_q_scale_max: float = 1.2
    cap_total_min: float = 0.0
    cap_total_max: float = 1.0

    # Optional Stage-3 boundary mixture. A zero probability preserves the
    # original uniform sampler and its random-number sequence.
    stage3_corner_prob: float = 0.0
    stage3_corner_mode: str = "pv_svc_corners"

    episode_len: int = 96

    # misc
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    log_every: int = 200


class ThetaCurriculumSampler:
    CORNER_MODES = {"all_vertices", "pv_svc_corners", "critical_edge"}

    def __init__(self, cfg: PPOConfig, total_episodes: int):
        self.cfg = cfg
        self.total_episodes = max(1, int(total_episodes))
        self.last_sample_kind = "uninitialised"
        prob = float(cfg.stage3_corner_prob)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"stage3_corner_prob must be in [0, 1], got {prob}")
        if cfg.stage3_corner_mode not in self.CORNER_MODES:
            choices = ", ".join(sorted(self.CORNER_MODES))
            raise ValueError(f"stage3_corner_mode must be one of {{{choices}}}")

    def stage(self, episode_idx: int) -> int:
        cfg = self.cfg
        frac = min(max(float(episode_idx) / float(self.total_episodes), 0.0), 1.0)
        if frac < float(cfg.stage1_frac):
            return 1
        if frac < float(cfg.stage2_frac):
            return 2
        return 3

    def sample(self, episode_idx: int) -> np.ndarray:
        cfg = self.cfg
        st = self.stage(episode_idx)
        if st == 1:
            self.last_sample_kind = "stage1_fixed"
            return np.asarray([1.0, 1.0, float(cfg.cap0)], dtype=np.float32)
        if st == 2:
            self.last_sample_kind = "stage2_uniform"
            pv = np.random.uniform(cfg.pv_s_mid_min, cfg.pv_s_mid_max)
            svc = np.random.uniform(cfg.svc_q_mid_min, cfg.svc_q_mid_max)
            cap = np.random.uniform(cfg.cap_mid_min, cfg.cap_mid_max)
            return np.asarray([pv, svc, cap], dtype=np.float32)

        prob = float(cfg.stage3_corner_prob)
        if prob > 0.0 and np.random.random() < prob:
            mode = cfg.stage3_corner_mode
            if mode == "critical_edge":
                pv = float(cfg.pv_s_scale_min)
                svc = float(cfg.svc_q_scale_min)
                cap = np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
            else:
                pv = np.random.choice([cfg.pv_s_scale_min, cfg.pv_s_scale_max])
                svc = np.random.choice([cfg.svc_q_scale_min, cfg.svc_q_scale_max])
                if mode == "all_vertices":
                    cap = np.random.choice([cfg.cap_total_min, cfg.cap_total_max])
                else:
                    cap = np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
            self.last_sample_kind = f"stage3_{mode}"
            return np.asarray([pv, svc, cap], dtype=np.float32)

        self.last_sample_kind = "stage3_uniform"
        pv = np.random.uniform(cfg.pv_s_scale_min, cfg.pv_s_scale_max)
        svc = np.random.uniform(cfg.svc_q_scale_min, cfg.svc_q_scale_max)
        cap = np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
        return np.asarray([pv, svc, cap], dtype=np.float32)


class ThetaEnvManager:
    def __init__(self, cfg: PPOConfig, load_pu: np.ndarray, gene_pu: np.ndarray, id_iber: List[int], id_svc: List[int], total_episodes: int):
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
        self.sampler = ThetaCurriculumSampler(cfg, total_episodes=total_episodes)
        self.resample_and_reset()

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

    def stage(self) -> int:
        return int(self.sampler.stage(self.episode_idx))

    def resample_and_reset(self) -> None:
        self.theta = self.sampler.sample(self.episode_idx)
        self.sample_kind = self.sampler.last_sample_kind
        self.is_corner_sample = self.sample_kind.startswith("stage3_") and self.sample_kind != "stage3_uniform"
        self.env = self._build_env(self.theta)
        self.env_det = self._build_env(self.theta)
        self.state = self.env.reset()
        self.state_det = self.env_det.reset()
        self.episode_step = 0

    def step(self, action: np.ndarray, action_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """
        Returns:
          next_obs (sampled env), reward_vec_sampled, reward_vec_det, done, info
        done is True at episode boundary or PF fail.
        """
        cfg = self.cfg
        assert self.env is not None
        assert self.env_det is not None
        assert self.state is not None
        assert self.state_det is not None

        info = {
            "pf_fail_s": 0.0,
            "pf_fail_d": 0.0,
            "stage": float(self.stage()),
            "sample_kind": self.sample_kind,
            "corner_sample": float(self.is_corner_sample),
            "theta_pv": float(self.theta[0]),
            "theta_svc": float(self.theta[1]),
            "theta_cap": float(self.theta[2]),
        }
        done = False

        # deterministic mirror step
        try:
            next_state_d, reward_vec_d, _done_d, violation_d, violM_d, violN_d, vM_d, vN_d, grid_loss_d, new_state_d = self.env_det.step_model(action_mean)
        except Exception:
            info["pf_fail_d"] = 1.0
            reward_vec_d = np.asarray([-cfg.pf_fail_penalty, -cfg.pf_fail_penalty], dtype=np.float32)
            next_state_d = self.state_det
            new_state_d = self.state_det
            done = True

        # sampled step
        try:
            next_state, reward_vec, _done, violation, violM, violN, vM, vN, grid_loss, new_state = self.env.step_model(action)
        except Exception:
            info["pf_fail_s"] = 1.0
            reward_vec = np.asarray([-cfg.pf_fail_penalty, -cfg.pf_fail_penalty], dtype=np.float32)
            next_state = self.state
            new_state = self.state
            done = True

        # update internal states
        self.state = new_state
        self.state_det = new_state_d
        self.episode_step += 1

        if self.episode_step >= cfg.episode_len:
            done = True

        # per-step violation flags for logging
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

        # when done, advance curriculum episode counter and resample
        if done:
            self.episode_idx += 1
            self.resample_and_reset()

        return next_state, reward_vec, reward_vec_d, done, info


class PPOTrainerFiLMTheta:
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

        total_episodes = int(np.ceil(float(cfg.num_frames) / float(max(cfg.episode_len, 1))))
        self.mgr = ThetaEnvManager(cfg, load_pu, gene_pu, id_iber, id_svc, total_episodes=total_episodes)

        # infer dims from base env
        env0 = Env.grid_case(cfg.env, load_pu, gene_pu, id_iber, id_svc)
        self.obs_dim = int(env0.observation_space.shape[0])
        self.act_dim = int(env0.action_space.shape[0])
        self.theta_dim = 3

        if cfg.actor_arch == "hybrid_residual":
            self.ac = HybridFiLMActorCritic(
                self.obs_dim,
                self.act_dim,
                theta_dim=self.theta_dim,
                log_std_init=cfg.log_std_init,
                theta_min=(cfg.pv_s_scale_min, cfg.svc_q_scale_min, cfg.cap_total_min),
                theta_max=(cfg.pv_s_scale_max, cfg.svc_q_scale_max, cfg.cap_total_max),
            ).to(self.device)
        elif cfg.actor_arch == "film":
            self.ac = FiLMActorCritic(
                self.obs_dim, self.act_dim, theta_dim=self.theta_dim, log_std_init=cfg.log_std_init
            ).to(self.device)
        else:
            raise ValueError(f"Unknown actor_arch: {cfg.actor_arch}")
        self.pi_opt = optim.Adam(self.ac.policy_parameters(), lr=cfg.pi_lr)
        self.vf_opt = optim.Adam(self.ac.value_parameters(), lr=cfg.vf_lr)

        gamma_str = str(cfg.gamma)
        group_dir = Path("runs") / "PPO_THETA_FILM" / f"env{cfg.env}_seed{cfg.seed}_gamma{gamma_str}"
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
        self.state = self.mgr.state  # type: ignore
        self.theta = self.mgr.theta

    def _anneal_log_std(self) -> None:
        cfg = self.cfg
        frac = min(max(self.total_step / max(cfg.num_frames, 1), 0.0), 1.0)
        target = cfg.log_std_init + frac * (cfg.log_std_final - cfg.log_std_init)
        with torch.no_grad():
            self.ac.pi_log_std.data.fill_(float(target))

    def _minimum_margin(self, theta: np.ndarray) -> float:
        cfg = self.cfg
        return capacity_minimum_margin(
            theta,
            (cfg.pv_s_scale_min, cfg.svc_q_scale_min, cfg.cap_total_min),
            (cfg.pv_s_scale_max, cfg.svc_q_scale_max, cfg.cap_total_max),
        )

    def _effective_reward_v_weight(self, theta: np.ndarray) -> float:
        return margin_weighted_voltage_penalty(
            self.cfg.reward_v_weight,
            self.cfg.reward_v_margin_gain,
            self._minimum_margin(theta),
        )

    def _scalar_reward(self, reward_vec: np.ndarray, theta: np.ndarray) -> float:
        return float(reward_vec[0] + self._effective_reward_v_weight(theta) * reward_vec[1])

    def save_models(self) -> None:
        torch.save(self.ac.state_dict(), str(self.model_dir / "actor.pth"))
        torch.save({"cfg": self.cfg.__dict__, "ac": self.ac.state_dict(), "total_step": self.total_step}, str(self.model_dir / "checkpoint.pth"))

    def _write_csv_and_figures(
        self,
        scores,
        viol_counts,
        grid_loss_sums,
        pf_fails,
        scores0,
        viol_counts0,
        grid_loss_sums0,
        pf_fails0,
        actor_losses,
        value_losses,
        entropies,
        stages,
        sample_kinds,
        corner_samples,
        theta_pv,
        theta_svc,
        theta_cap,
        theta_min_margin,
        reward_v_weights,
    ) -> None:
        import pandas as pd

        pd.DataFrame(
            {
                "scores": scores,
                "violation_sum_s": viol_counts,
                "grid_loss_sum_s": grid_loss_sums,
                "pf_fail_sum_s": pf_fails,
                "stage": stages,
                "sample_kind": sample_kinds,
                "corner_sample": corner_samples,
                "theta_pv": theta_pv,
                "theta_svc": theta_svc,
                "theta_cap": theta_cap,
                "theta_min_margin": theta_min_margin,
                "reward_v_weight_effective": reward_v_weights,
            }
        ).to_csv(self.csv_dir / "train.csv", index=False)
        pd.DataFrame(
            {
                "scorest": scores0,
                "violation_sum_st": viol_counts0,
                "grid_loss_sum_st": grid_loss_sums0,
                "pf_fail_sum_st": pf_fails0,
                "stage": stages,
                "sample_kind": sample_kinds,
                "corner_sample": corner_samples,
                "theta_pv": theta_pv,
                "theta_svc": theta_svc,
                "theta_cap": theta_cap,
                "theta_min_margin": theta_min_margin,
                "reward_v_weight_effective": reward_v_weights,
            }
        ).to_csv(self.csv_dir / "traintest.csv", index=False)
        pd.DataFrame({"actor_losses": actor_losses, "value_losses": value_losses, "entropy": entropies}).to_csv(self.csv_dir / "trainloss.csv", index=False)

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

        scores, viol_counts, grid_loss_sums, pf_fails = [], [], [], []
        scores0, viol_counts0, grid_loss_sums0, pf_fails0 = [], [], [], []
        stages, sample_kinds, corner_samples = [], [], []
        theta_pv, theta_svc, theta_cap = [], [], []
        theta_min_margin, reward_v_weights = [], []
        actor_losses, value_losses, entropies = [], [], []

        score = 0.0
        viol_cnt = 0.0
        gl_sum = 0.0
        pf_fail_ep = 0.0
        score0 = 0.0
        viol_cnt0 = 0.0
        gl_sum0 = 0.0
        pf_fail_ep0 = 0.0

        obs_buf, theta_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []

        t_start = time.time()
        t_last = t_start

        if writer is not None:
            writer.add_text("meta/algo", "PPO_theta_robust_FiLM_curriculum")
            writer.add_text("meta/theta", "theta=[pv_s_scale, svc_q_scale, cap_total_mvar] (FiLM-modulated)")
            writer.add_scalar("meta/reward_v_weight", float(cfg.reward_v_weight), 0)
            writer.add_scalar("meta/reward_v_margin_gain", float(cfg.reward_v_margin_gain), 0)
            writer.add_text(
                "meta/reward_v_weight_formula",
                "base_weight * [1 + margin_gain * (1 - minimum_normalised_theta_margin)]",
            )
            writer.add_scalar("meta/stage1_frac", float(cfg.stage1_frac), 0)
            writer.add_scalar("meta/stage2_frac", float(cfg.stage2_frac), 0)
            writer.add_scalar("meta/stage3_corner_prob", float(cfg.stage3_corner_prob), 0)
            writer.add_text("meta/stage3_corner_mode", str(cfg.stage3_corner_mode))

        while self.total_step < cfg.num_frames:
            self._anneal_log_std()

            obs_buf.clear(); theta_buf.clear(); act_buf.clear(); logp_buf.clear(); val_buf.clear(); rew_buf.clear(); done_buf.clear()
            steps_this = min(cfg.steps_per_update, cfg.num_frames - self.total_step)

            for _ in range(int(steps_this)):
                self.total_step += 1

                obs_t = torch.FloatTensor(np.asarray(self.mgr.state, dtype=np.float32)).to(self.device)  # type: ignore
                theta_np = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                theta_t = torch.FloatTensor(theta_np).to(self.device)
                a_sample, a_mean, logp, v = self.ac.act(obs_t, theta_t)

                next_obs, reward_vec, reward_vec0, done, info = self.mgr.step(a_sample, a_mean)
                r_scalar = self._scalar_reward(reward_vec, theta_np)
                r_scalar0 = self._scalar_reward(reward_vec0, theta_np)

                score += float(r_scalar)
                score0 += float(r_scalar0)
                viol_cnt += float(info.get("vflag_s", 0.0))
                viol_cnt0 += float(info.get("vflag_d", 0.0))
                gl_sum += float(reward_vec[0])
                gl_sum0 += float(reward_vec0[0])
                pf_fail_ep += float(info.get("pf_fail_s", 0.0))
                pf_fail_ep0 += float(info.get("pf_fail_d", 0.0))

                # buffers (store current state/theta used for action)
                obs_buf.append(np.asarray(self.mgr.state, dtype=np.float32).copy())  # type: ignore
                theta_buf.append(theta_np)
                act_buf.append(np.asarray(a_sample, dtype=np.float32).copy())
                logp_buf.append(float(logp))
                val_buf.append(float(v))
                rew_buf.append(float(r_scalar))
                done_buf.append(1.0 if done else 0.0)

                if writer is not None:
                    writer.add_scalar("step/reward_p", float(reward_vec[0]), self.total_step)
                    writer.add_scalar("step/reward_v", float(reward_vec[1]), self.total_step)
                    writer.add_scalar("step/reward_scalar", float(r_scalar), self.total_step)
                    writer.add_scalar(
                        "step/reward_v_weight_effective",
                        self._effective_reward_v_weight(theta_np),
                        self.total_step,
                    )
                    writer.add_scalar("step/pf_fail", float(info.get("pf_fail_s", 0.0)), self.total_step)
                    writer.add_scalar("step/pf_fail_action0", float(info.get("pf_fail_d", 0.0)), self.total_step)
                    writer.add_scalar("theta/pv_s_scale", float(self.mgr.theta[0]), self.total_step)
                    writer.add_scalar("theta/svc_q_scale", float(self.mgr.theta[1]), self.total_step)
                    writer.add_scalar("theta/cap_total_mvar", float(self.mgr.theta[2]), self.total_step)
                    writer.add_scalar("curriculum/stage", float(info.get("stage", 0.0)), self.total_step)
                    writer.add_scalar("curriculum/corner_sample", float(info.get("corner_sample", 0.0)), self.total_step)

                if cfg.log_every > 0 and self.total_step % cfg.log_every == 0:
                    now = time.time()
                    dt = max(now - t_last, 1e-9)
                    total_dt = max(now - t_start, 1e-9)
                    sps = cfg.log_every / dt
                    avg_sps = self.total_step / total_dt
                    eta_min = (cfg.num_frames - self.total_step) / max(avg_sps, 1e-9) / 60.0
                    print(
                        f"[env={cfg.env} seed={cfg.seed}] step {self.total_step}/{cfg.num_frames} "
                        f"({sps:.2f} step/s, avg {avg_sps:.2f}, ETA {eta_min:.1f} min) | stage={int(self.mgr.stage())}"
                    )
                    t_last = now

                if done:
                    scores.append(score); viol_counts.append(viol_cnt); grid_loss_sums.append(gl_sum); pf_fails.append(pf_fail_ep)
                    scores0.append(score0); viol_counts0.append(viol_cnt0); grid_loss_sums0.append(gl_sum0); pf_fails0.append(pf_fail_ep0)
                    stages.append(int(info.get("stage", self.mgr.stage())))
                    sample_kinds.append(str(info.get("sample_kind", "unknown")))
                    corner_samples.append(int(info.get("corner_sample", 0.0)))
                    theta_pv.append(float(info.get("theta_pv", np.nan)))
                    theta_svc.append(float(info.get("theta_svc", np.nan)))
                    theta_cap.append(float(info.get("theta_cap", np.nan)))
                    theta_min_margin.append(self._minimum_margin(theta_np))
                    reward_v_weights.append(self._effective_reward_v_weight(theta_np))

                    if writer is not None:
                        writer.add_scalar("train/episode_score", float(score), self.total_step)
                        writer.add_scalar("train/episode_violation_count", float(viol_cnt), self.total_step)
                        writer.add_scalar("train/episode_grid_loss_sum", float(gl_sum), self.total_step)
                        writer.add_scalar("train/pf_fail_steps", float(pf_fail_ep), self.total_step)
                        writer.add_scalar("train_action0/episode_score", float(score0), self.total_step)
                        writer.add_scalar("train_action0/episode_violation_count", float(viol_cnt0), self.total_step)
                        writer.add_scalar("train_action0/episode_grid_loss_sum", float(gl_sum0), self.total_step)
                        writer.add_scalar("train_action0/pf_fail_steps", float(pf_fail_ep0), self.total_step)

                    score = 0.0; viol_cnt = 0.0; gl_sum = 0.0; pf_fail_ep = 0.0
                    score0 = 0.0; viol_cnt0 = 0.0; gl_sum0 = 0.0; pf_fail_ep0 = 0.0

            # PPO update
            obs = torch.FloatTensor(np.asarray(obs_buf)).to(self.device)
            theta = torch.FloatTensor(np.asarray(theta_buf)).to(self.device)
            act = torch.FloatTensor(np.asarray(act_buf)).to(self.device)
            old_logp = torch.FloatTensor(np.asarray(logp_buf)).to(self.device)
            vals = torch.FloatTensor(np.asarray(val_buf)).to(self.device)
            rews = torch.FloatTensor(np.asarray(rew_buf)).to(self.device)
            dones = torch.FloatTensor(np.asarray(done_buf)).to(self.device)

            with torch.no_grad():
                v_last = self.ac._v(
                    torch.FloatTensor(np.asarray(self.mgr.state, dtype=np.float32)).to(self.device),  # type: ignore
                    torch.FloatTensor(np.asarray(self.mgr.theta, dtype=np.float32)).to(self.device),
                ).detach()
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

            for _it in range(cfg.train_iters):
                np.random.shuffle(idx)
                for start in range(0, n, cfg.minibatch_size):
                    mb = idx[start : start + cfg.minibatch_size]
                    mb_obs = obs[mb]
                    mb_theta = theta[mb]
                    mb_act = act[mb]
                    mb_old_logp = old_logp[mb]
                    mb_adv = adv[mb]
                    mb_ret = ret[mb]

                    logp, entropy, v_pred = self.ac.logp_and_v(mb_obs, mb_theta, mb_act)
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
            scores, viol_counts, grid_loss_sums, pf_fails,
            scores0, viol_counts0, grid_loss_sums0, pf_fails0,
            actor_losses, value_losses, entropies, stages,
            sample_kinds, corner_samples, theta_pv, theta_svc, theta_cap,
            theta_min_margin, reward_v_weights
        )

        if writer is not None:
            writer.flush()
            writer.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO(theta) with FiLM conditioning + curriculum domain randomization.")
    p.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--run_name", type=str, default="")
    p.add_argument(
        "--actor_arch",
        type=str,
        default="film",
        choices=["film", "hybrid_residual"],
        help="Actor conditioning architecture; hybrid_residual is scheme C v1 without LayerNorm.",
    )
    p.add_argument("--num_frames", type=int, default=96 * 300)
    p.add_argument("--steps_per_update", type=int, default=2048)
    p.add_argument("--minibatch_size", type=int, default=256)
    p.add_argument("--train_iters", type=int, default=10)
    p.add_argument("--pi_lr", type=float, default=3e-4)
    p.add_argument("--vf_lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)

    p.add_argument("--reward_v_weight", type=float, default=200.0)
    p.add_argument(
        "--reward_v_margin_gain",
        type=float,
        default=0.0,
        help="Scheme E gain in w_v(theta)=base*[1+gain*(1-minimum_margin)]; zero preserves legacy training.",
    )
    p.add_argument("--vmin", type=float, default=0.95)
    p.add_argument("--vmax", type=float, default=1.05)
    p.add_argument("--pf_fail_penalty", type=float, default=1000.0)

    p.add_argument("--log_std_init", type=float, default=-0.5)
    p.add_argument("--log_std_final", type=float, default=-2.0)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--log_every", type=int, default=200)

    # theta + curriculum
    p.add_argument("--enable_pq_curve", action="store_true")
    p.add_argument("--cap_buses", type=str, default="", help="Comma-separated cap buses, cap_total is equally distributed.")
    p.add_argument("--cap0", type=float, default=0.0, help="Stage1 fixed cap_total_mvar.")
    p.add_argument("--stage1_frac", type=float, default=0.2, help="Fraction of training episodes in stage1 (fixed theta).")
    p.add_argument("--stage2_frac", type=float, default=0.6, help="Cumulative fraction until end of stage2 (small perturb).")

    p.add_argument("--pv_s_mid_min", type=float, default=0.9)
    p.add_argument("--pv_s_mid_max", type=float, default=1.1)
    p.add_argument("--svc_q_mid_min", type=float, default=0.9)
    p.add_argument("--svc_q_mid_max", type=float, default=1.1)
    p.add_argument("--cap_mid_min", type=float, default=0.0)
    p.add_argument("--cap_mid_max", type=float, default=0.5)

    # stage3 defaults: realistic uncertainty (0.8~1.2)
    p.add_argument("--pv_s_scale_min", type=float, default=0.8)
    p.add_argument("--pv_s_scale_max", type=float, default=1.2)
    p.add_argument("--svc_q_scale_min", type=float, default=0.8)
    p.add_argument("--svc_q_scale_max", type=float, default=1.2)
    p.add_argument("--cap_total_min", type=float, default=0.0)
    p.add_argument("--cap_total_max", type=float, default=1.0)
    p.add_argument(
        "--stage3_corner_prob",
        type=float,
        default=0.0,
        help="Probability of replacing a Stage-3 uniform draw with a boundary-focused draw.",
    )
    p.add_argument(
        "--stage3_corner_mode",
        type=str,
        default="pv_svc_corners",
        choices=sorted(ThetaCurriculumSampler.CORNER_MODES),
        help="Boundary draw: all 3-D vertices, PV/SVC corners with continuous cap, or the double-low critical edge.",
    )
    p.add_argument("--episode_len", type=int, default=96)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cap_buses = _parse_int_list(args.cap_buses) if str(args.cap_buses).strip() else None
    cfg = PPOConfig(
        env=args.env,
        seed=args.seed,
        gamma=args.gamma,
        run_name=args.run_name,
        actor_arch=str(args.actor_arch),
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        train_iters=args.train_iters,
        minibatch_size=args.minibatch_size,
        steps_per_update=args.steps_per_update,
        num_frames=args.num_frames,
        reward_v_weight=args.reward_v_weight,
        reward_v_margin_gain=float(args.reward_v_margin_gain),
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
        cap_buses=cap_buses,
        cap0=float(args.cap0),
        stage1_frac=float(args.stage1_frac),
        stage2_frac=float(args.stage2_frac),
        pv_s_mid_min=float(args.pv_s_mid_min),
        pv_s_mid_max=float(args.pv_s_mid_max),
        svc_q_mid_min=float(args.svc_q_mid_min),
        svc_q_mid_max=float(args.svc_q_mid_max),
        cap_mid_min=float(args.cap_mid_min),
        cap_mid_max=float(args.cap_mid_max),
        pv_s_scale_min=float(args.pv_s_scale_min),
        pv_s_scale_max=float(args.pv_s_scale_max),
        svc_q_scale_min=float(args.svc_q_scale_min),
        svc_q_scale_max=float(args.svc_q_scale_max),
        cap_total_min=float(args.cap_total_min),
        cap_total_max=float(args.cap_total_max),
        stage3_corner_prob=float(args.stage3_corner_prob),
        stage3_corner_mode=str(args.stage3_corner_mode),
        episode_len=int(args.episode_len),
    )
    set_seed(cfg.seed)
    trainer = PPOTrainerFiLMTheta(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
