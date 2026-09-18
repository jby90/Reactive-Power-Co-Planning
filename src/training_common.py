"""Shared configuration and environment management for capacity-conditioned PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

import Env


def relu_violation_sums(
    vm: np.ndarray, vmin: float, vmax: float
) -> tuple[float, float, int]:
    over = float(np.maximum(0.0, vm - vmax).sum())
    under = float(np.maximum(0.0, vmin - vm).sum())
    return over, under, int((over + under) > 0.0)


@dataclass
class PPOConfig:
    env: int = 33
    seed: int = 42
    gamma: float = 0.9
    run_name: str = ""
    actor_arch: str = "capacity_conditioned"

    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_iters: int = 10
    minibatch_size: int = 256
    steps_per_update: int = 2048
    num_frames: int = 96 * 300

    reward_v_weight: float = 200.0
    reward_v_margin_gain: float = 0.0
    vmin: float = 0.95
    vmax: float = 1.05
    pf_fail_penalty: float = 1000.0

    log_std_init: float = -0.5
    log_std_final: float = -2.0
    entropy_coef: float = 0.0

    enable_pq_curve: bool = True
    cap_buses: List[int] | None = None
    cap0: float = 0.0
    stage1_frac: float = 0.2
    stage2_frac: float = 0.6
    pv_s_mid_min: float = 0.9
    pv_s_mid_max: float = 1.1
    svc_q_mid_min: float = 0.9
    svc_q_mid_max: float = 1.1
    cap_mid_min: float = 0.0
    cap_mid_max: float = 0.5
    pv_s_scale_min: float = 0.8
    pv_s_scale_max: float = 1.2
    svc_q_scale_min: float = 0.8
    svc_q_scale_max: float = 1.2
    cap_total_min: float = 0.0
    cap_total_max: float = 1.0
    stage3_corner_prob: float = 0.0
    stage3_corner_mode: str = "pv_svc_corners"

    episode_len: int = 96
    training_days_csv: str = ""
    training_day_seed: int = 0
    action_parameterization: str = "relative"
    svc_absorption_ratio: float = 0.0
    load_profile_path: str = "load96.npy"
    generation_profile_path: str = "gen96.npy"

    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    log_every: int = 200


class ThetaCurriculumSampler:
    CORNER_MODES = {"all_vertices", "pv_svc_corners", "critical_edge"}

    def __init__(self, cfg: PPOConfig, total_episodes: int):
        self.cfg = cfg
        self.total_episodes = max(1, int(total_episodes))
        self.last_sample_kind = "uninitialised"
        if not 0.0 <= float(cfg.stage3_corner_prob) <= 1.0:
            raise ValueError("stage3_corner_prob must be in [0, 1]")
        if cfg.stage3_corner_mode not in self.CORNER_MODES:
            raise ValueError("unsupported stage3_corner_mode")

    def stage(self, episode_idx: int) -> int:
        fraction = min(max(float(episode_idx) / self.total_episodes, 0.0), 1.0)
        if fraction < self.cfg.stage1_frac:
            return 1
        if fraction < self.cfg.stage2_frac:
            return 2
        return 3

    def sample(self, episode_idx: int) -> np.ndarray:
        cfg = self.cfg
        stage = self.stage(episode_idx)
        if stage == 1:
            self.last_sample_kind = "stage1_fixed"
            return np.asarray([1.0, 1.0, cfg.cap0], dtype=np.float32)
        if stage == 2:
            self.last_sample_kind = "stage2_uniform"
            return np.asarray(
                [
                    np.random.uniform(cfg.pv_s_mid_min, cfg.pv_s_mid_max),
                    np.random.uniform(cfg.svc_q_mid_min, cfg.svc_q_mid_max),
                    np.random.uniform(cfg.cap_mid_min, cfg.cap_mid_max),
                ],
                dtype=np.float32,
            )

        if cfg.stage3_corner_prob > 0.0 and np.random.random() < cfg.stage3_corner_prob:
            mode = cfg.stage3_corner_mode
            if mode == "critical_edge":
                pv = cfg.pv_s_scale_min
                svc = cfg.svc_q_scale_min
                cap = np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
            else:
                pv = np.random.choice([cfg.pv_s_scale_min, cfg.pv_s_scale_max])
                svc = np.random.choice([cfg.svc_q_scale_min, cfg.svc_q_scale_max])
                cap = (
                    np.random.choice([cfg.cap_total_min, cfg.cap_total_max])
                    if mode == "all_vertices"
                    else np.random.uniform(cfg.cap_total_min, cfg.cap_total_max)
                )
            self.last_sample_kind = f"stage3_{mode}"
            return np.asarray([pv, svc, cap], dtype=np.float32)

        self.last_sample_kind = "stage3_uniform"
        return np.asarray(
            [
                np.random.uniform(cfg.pv_s_scale_min, cfg.pv_s_scale_max),
                np.random.uniform(cfg.svc_q_scale_min, cfg.svc_q_scale_max),
                np.random.uniform(cfg.cap_total_min, cfg.cap_total_max),
            ],
            dtype=np.float32,
        )


class ThetaEnvManager:
    def __init__(
        self,
        cfg: PPOConfig,
        load_pu: np.ndarray,
        gene_pu: np.ndarray,
        id_iber: List[int],
        id_svc: List[int],
        total_episodes: int,
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
        self.sampler = ThetaCurriculumSampler(cfg, total_episodes)
        self.resample_and_reset()

    def _build_env(self, theta: np.ndarray):
        cap_buses = list(self.cfg.cap_buses or [])
        cap_total = float(theta[2])
        cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_buses and cap_total > 0 else []
        return Env.grid_case(
            self.cfg.env,
            self.load_pu,
            self.gene_pu,
            self.id_iber,
            self.id_svc,
            enable_pq_curve=bool(self.cfg.enable_pq_curve),
            pv_s_scale=float(theta[0]),
            svc_q_scale=float(theta[1]),
            svc_absorption_ratio=float(self.cfg.svc_absorption_ratio),
            cap_buses=cap_buses or None,
            cap_q_mvar=cap_q or None,
            action_parameterization=str(self.cfg.action_parameterization),
        )

    def stage(self) -> int:
        return self.sampler.stage(self.episode_idx)

    def resample_and_reset(self) -> None:
        self.theta = self.sampler.sample(self.episode_idx)
        self.sample_kind = self.sampler.last_sample_kind
        self.is_corner_sample = self.sample_kind.startswith("stage3_") and self.sample_kind != "stage3_uniform"
        self.env = self._build_env(self.theta)
        self.env_det = self._build_env(self.theta)
        self.state = self.env.reset()
        self.state_det = self.env_det.reset()
        self.episode_step = 0

    def step(
        self, action: np.ndarray, action_mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        cfg = self.cfg
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
        try:
            next_det, reward_det, _, _, _, _, _, _, _, state_det = self.env_det.step_model(action_mean)
        except Exception:
            info["pf_fail_d"] = 1.0
            reward_det = np.asarray([-cfg.pf_fail_penalty] * 2, dtype=np.float32)
            next_det = state_det = self.state_det
            done = True
        try:
            next_obs, reward, _, _, _, _, _, _, _, state = self.env.step_model(action)
        except Exception:
            info["pf_fail_s"] = 1.0
            reward = np.asarray([-cfg.pf_fail_penalty] * 2, dtype=np.float32)
            next_obs = state = self.state
            done = True
        self.state, self.state_det = state, state_det
        self.episode_step += 1
        done = done or self.episode_step >= cfg.episode_len
        for key, obs, env in (("vflag_s", next_obs, self.env), ("vflag_d", next_det, self.env_det)):
            try:
                _, _, flag = relu_violation_sums(
                    np.asarray(obs[: len(env.model.bus)], dtype=float), cfg.vmin, cfg.vmax
                )
            except Exception:
                flag = 1
            info[key] = float(flag)
        if done:
            self.episode_idx += 1
            self.resample_and_reset()
        return next_obs, reward, reward_det, done, info
