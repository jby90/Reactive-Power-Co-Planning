"""Worst-group directional CVaR-PPO for capacity-conditioned Volt-VAR control."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

import Env
from PPO_theta_crdc import (
    CRDCConfig,
    CRDCThetaEnvManager,
    DIRECTION_NAMES,
    DIRECTION_OVER,
    DIRECTION_UNDER,
    ConcatDirectionalActorCritic,
    SummaryWriter,
    capacity_group_id,
    compute_gae,
    set_seed,
    standardise,
)


def directional_guard_costs(
    vm_pu: np.ndarray,
    vmin: float,
    vmax: float,
    guard_margin: float,
    pf_failed: bool = False,
) -> tuple[float, float, int, int]:
    """Return continuous guard excursions and actual operational-limit events."""
    if pf_failed:
        return 1.0, 1.0, 1, 1
    vm = np.asarray(vm_pu, dtype=np.float64)
    under_cost = float(np.maximum(0.0, vmin + guard_margin - vm).max(initial=0.0))
    over_cost = float(np.maximum(0.0, vm - (vmax - guard_margin)).max(initial=0.0))
    return under_cost, over_cost, int(np.any(vm < vmin)), int(np.any(vm > vmax))


def empirical_cvar(values: list[float] | np.ndarray, alpha: float) -> float:
    """Upper-tail empirical CVaR using at least one order statistic."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        return 0.0
    tail_count = max(1, int(np.ceil((1.0 - alpha) * samples.size)))
    return float(np.sort(samples)[-tail_count:].mean())


class WorstGroupCVaRController:
    """Rolling directional CVaR duals and adversarial capacity-group weights."""

    def __init__(
        self,
        groups: int,
        alpha: float,
        target: float,
        window: int,
        dual_lr: float,
        lambda_init: float,
        lambda_max: float,
        dro_eta: float,
        group_weight_floor: float,
    ):
        if groups <= 0 or target <= 0.0 or window <= 0:
            raise ValueError("groups, target and window must be positive")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if not 0.0 <= group_weight_floor < 1.0 / groups:
            raise ValueError("group_weight_floor must be in [0, 1/groups)")
        self.groups = groups
        self.alpha = float(alpha)
        self.target = float(target)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)
        self.dro_eta = float(dro_eta)
        self.group_weight_floor = float(group_weight_floor)
        self.history = [
            [deque(maxlen=int(window)) for _ in DIRECTION_NAMES] for _ in range(groups)
        ]
        self.multipliers = np.full((groups, 2), float(lambda_init), dtype=np.float64)
        self.group_logits = np.zeros(groups, dtype=np.float64)
        self.group_weights = np.full(groups, 1.0 / groups, dtype=np.float64)

    def add_episode(self, group: int, under_cost: float, over_cost: float) -> None:
        self.history[int(group)][DIRECTION_UNDER].append(float(under_cost))
        self.history[int(group)][DIRECTION_OVER].append(float(over_cost))

    def cvars(self) -> np.ndarray:
        result = np.zeros((self.groups, 2), dtype=np.float64)
        for group in range(self.groups):
            for direction in range(2):
                result[group, direction] = empirical_cvar(
                    list(self.history[group][direction]), self.alpha
                )
        return result

    def update(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cvars = self.cvars()
        normalised_excess = (cvars - self.target) / self.target
        self.multipliers = np.clip(
            self.multipliers + self.dual_lr * normalised_excess,
            0.0,
            self.lambda_max,
        )

        group_excess = normalised_excess.max(axis=1)
        self.group_logits += self.dro_eta * np.clip(group_excess, -2.0, 10.0)
        stable = self.group_logits - self.group_logits.max()
        softmax = np.exp(stable) / np.exp(stable).sum()
        floor = self.group_weight_floor
        self.group_weights = floor + (1.0 - floor * self.groups) * softmax
        return cvars.copy(), self.multipliers.copy(), self.group_weights.copy()


@dataclass
class WGCVARConfig(CRDCConfig):
    actor_arch: str = "concat_worst_group_cvar"
    guard_margin: float = 0.005
    cvar_alpha: float = 0.90
    cvar_target: float = 0.005
    cvar_window: int = 64
    dual_lr: float = 0.25
    dro_eta: float = 0.10
    group_weight_floor: float = 0.02


class WGCVARPPOTrainer:
    def __init__(self, cfg: WGCVARConfig):
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
        self.risk = WorstGroupCVaRController(
            groups=8,
            alpha=cfg.cvar_alpha,
            target=cfg.cvar_target,
            window=cfg.cvar_window,
            dual_lr=cfg.dual_lr,
            lambda_init=cfg.lambda_init,
            lambda_max=cfg.lambda_max,
            dro_eta=cfg.dro_eta,
            group_weight_floor=cfg.group_weight_floor,
        )

        run_name = cfg.run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = (
            Path("runs")
            / "WG_CVAR_PPO"
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
        history = [
            [list(self.risk.history[group][direction]) for direction in range(2)]
            for group in range(8)
        ]
        torch.save(
            {
                "algorithm": "WG-CVaR-PPO",
                "cfg": asdict(self.cfg),
                "ac": self.ac.state_dict(),
                "cvar_history": history,
                "multipliers": self.risk.multipliers,
                "group_weights": self.risk.group_weights,
                "total_step": self.total_step,
                "update_index": self.update_index,
            },
            self.model_dir / "checkpoint.pth",
        )

    def train(self) -> None:
        cfg = self.cfg
        episode_loss_reward = 0.0
        episode_under_guard = 0.0
        episode_over_guard = 0.0
        episode_under_steps = 0
        episode_over_steps = 0
        episode_pf_fail = 0
        episode_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
        episode_group = self._group_id(episode_theta)
        start_time = time.time()

        while self.total_step < cfg.num_frames:
            self._anneal_log_std()
            keys = (
                "obs", "theta", "group", "act", "logp", "vr", "vu", "vo",
                "reward", "under", "over", "done",
            )
            buffers = {key: [] for key in keys}
            steps_this_update = min(cfg.steps_per_update, cfg.num_frames - self.total_step)

            for _ in range(int(steps_this_update)):
                current_obs = np.asarray(self.mgr.state, dtype=np.float32).copy()
                current_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                current_group = self._group_id(current_theta)
                obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device)
                theta_tensor = torch.as_tensor(current_theta, dtype=torch.float32, device=self.device)
                action, action_mean, logp, values = self.ac.act(obs_tensor, theta_tensor)
                next_obs, reward_vec, _reward_det, done, info = self.mgr.step(action, action_mean)
                pf_failed = bool(info.get("pf_fail_s", 0.0))
                under_cost, over_cost, under_flag, over_flag = directional_guard_costs(
                    np.asarray(next_obs[: self.n_bus]),
                    cfg.vmin,
                    cfg.vmax,
                    cfg.guard_margin,
                    pf_failed,
                )

                for key, item in (
                    ("obs", current_obs), ("theta", current_theta), ("group", current_group),
                    ("act", np.asarray(action, dtype=np.float32)), ("logp", logp),
                    ("vr", values[0]), ("vu", values[1]), ("vo", values[2]),
                    ("reward", float(reward_vec[0])), ("under", under_cost),
                    ("over", over_cost), ("done", float(done)),
                ):
                    buffers[key].append(item)

                self.total_step += 1
                episode_loss_reward += float(reward_vec[0])
                episode_under_guard = max(episode_under_guard, under_cost)
                episode_over_guard = max(episode_over_guard, over_cost)
                episode_under_steps += under_flag
                episode_over_steps += over_flag
                episode_pf_fail += int(pf_failed)

                if done:
                    self.risk.add_episode(
                        episode_group, episode_under_guard, episode_over_guard
                    )
                    self.episode_rows.append(
                        {
                            "episode": len(self.episode_rows),
                            "total_step": self.total_step,
                            "group_id": episode_group,
                            "theta_pv": float(episode_theta[0]),
                            "theta_svc": float(episode_theta[1]),
                            "theta_cap": float(episode_theta[2]),
                            "line_loss_mwh": -0.25 * episode_loss_reward,
                            "under_guard_excursion": episode_under_guard,
                            "over_guard_excursion": episode_over_guard,
                            "under_steps": episode_under_steps,
                            "over_steps": episode_over_steps,
                            "event_day": int(bool(episode_under_steps or episode_over_steps or episode_pf_fail)),
                            "pf_fail_steps": episode_pf_fail,
                        }
                    )
                    episode_loss_reward = 0.0
                    episode_under_guard = 0.0
                    episode_over_guard = 0.0
                    episode_under_steps = 0
                    episode_over_steps = 0
                    episode_pf_fail = 0
                    episode_theta = np.asarray(self.mgr.theta, dtype=np.float32).copy()
                    episode_group = self._group_id(episode_theta)

            tensors = {
                key: torch.as_tensor(np.asarray(value), device=self.device)
                for key, value in buffers.items()
            }
            obs, theta = tensors["obs"].float(), tensors["theta"].float()
            groups, actions = tensors["group"].long(), tensors["act"].float()
            old_logp = tensors["logp"].float()
            values_r = tensors["vr"].float()
            values_u = tensors["vu"].float()
            values_o = tensors["vo"].float()
            rewards = tensors["reward"].float()
            costs_u = tensors["under"].float()
            costs_o = tensors["over"].float()
            dones = tensors["done"].float()

            with torch.no_grad():
                last_obs = torch.as_tensor(self.mgr.state, dtype=torch.float32, device=self.device)
                last_theta = torch.as_tensor(self.mgr.theta, dtype=torch.float32, device=self.device)
                last_r, last_u, last_o = self.ac._values(last_obs, last_theta)
                adv_r, ret_r = compute_gae(rewards, values_r, dones, last_r, cfg.gamma, cfg.lam)
                adv_u, ret_u = compute_gae(costs_u, values_u, dones, last_u, cfg.gamma, cfg.lam)
                adv_o, ret_o = compute_gae(costs_o, values_o, dones, last_o, cfg.gamma, cfg.lam)
                adv_r, adv_u, adv_o = standardise(adv_r), standardise(adv_u), standardise(adv_o)

            cvars, multipliers, group_weights = self.risk.update()
            multiplier_tensor = torch.as_tensor(multipliers, dtype=torch.float32, device=self.device)
            group_weight_tensor = torch.as_tensor(group_weights, dtype=torch.float32, device=self.device)
            combined_adv = (
                adv_r
                - multiplier_tensor[groups, DIRECTION_UNDER] * adv_u
                - multiplier_tensor[groups, DIRECTION_OVER] * adv_o
            )

            indices = np.arange(len(rewards))
            metric_rows = {name: [] for name in (
                "policy", "value_r", "value_u", "value_o", "entropy",
            )}
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
                    present = [group for group in range(8) if torch.any(mb_group == group)]
                    weights = group_weight_tensor[present]
                    weights = weights / weights.sum()
                    objectives = torch.stack(
                        [surrogate[mb_group == group].mean() for group in present]
                    )
                    policy_loss = -(weights * objectives).sum()
                    if cfg.entropy_coef:
                        policy_loss -= cfg.entropy_coef * entropy.mean()
                    self.pi_opt.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.ac.policy_parameters()), cfg.max_grad_norm
                    )
                    self.pi_opt.step()

                    loss_r = F.mse_loss(pred_r, ret_r[mb])
                    loss_u = F.mse_loss(pred_u, ret_u[mb])
                    loss_o = F.mse_loss(pred_o, ret_o[mb])
                    value_loss = loss_r + cfg.safety_value_coef * (loss_u + loss_o)
                    self.vf_opt.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.ac.value_parameters()), cfg.max_grad_norm
                    )
                    self.vf_opt.step()

                    last_kl = float((old_logp[mb] - logp).mean().detach().cpu())
                    metric_rows["policy"].append(float(policy_loss.detach().cpu()))
                    metric_rows["value_r"].append(float(loss_r.detach().cpu()))
                    metric_rows["value_u"].append(float(loss_u.detach().cpu()))
                    metric_rows["value_o"].append(float(loss_o.detach().cpu()))
                    metric_rows["entropy"].append(float(entropy.mean().detach().cpu()))
                if last_kl > cfg.target_kl:
                    break

            self.update_index += 1
            row = {
                "update": self.update_index,
                "total_step": self.total_step,
                "policy_loss": float(np.mean(metric_rows["policy"])),
                "reward_value_loss": float(np.mean(metric_rows["value_r"])),
                "under_value_loss": float(np.mean(metric_rows["value_u"])),
                "over_value_loss": float(np.mean(metric_rows["value_o"])),
                "entropy": float(np.mean(metric_rows["entropy"])),
                "approx_kl": last_kl,
            }
            for group in range(8):
                row[f"group_weight_g{group}"] = group_weights[group]
                for direction, name in enumerate(DIRECTION_NAMES):
                    row[f"cvar_g{group}_{name}"] = cvars[group, direction]
                    row[f"lambda_g{group}_{name}"] = multipliers[group, direction]
                    row[f"history_n_g{group}_{name}"] = len(self.risk.history[group][direction])
            self.update_rows.append(row)
            self._write_outputs()
            self.save_checkpoint()

            if self.writer is not None:
                self.writer.add_scalar("loss/policy", row["policy_loss"], self.total_step)
                self.writer.add_scalar("risk/max_cvar", float(cvars.max()), self.total_step)
                self.writer.add_scalar("risk/max_lambda", float(multipliers.max()), self.total_step)
                for group in range(8):
                    self.writer.add_scalar(
                        f"group_weight/g{group}", group_weights[group], self.total_step
                    )

            elapsed = max(time.time() - start_time, 1e-9)
            eta_minutes = (
                (cfg.num_frames - self.total_step)
                / max(self.total_step / elapsed, 1e-9)
                / 60.0
            )
            print(
                f"[WG-CVaR env={cfg.env} seed={cfg.seed}] "
                f"{self.total_step}/{cfg.num_frames} update={self.update_index} "
                f"ETA={eta_minutes:.1f} min max_cvar={cvars.max():.5f} "
                f"max_lambda={multipliers.max():.3f}",
                flush=True,
            )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Worst-group directional CVaR-PPO")
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
    parser.add_argument("--guard_margin", type=float, default=0.005)
    parser.add_argument("--cvar_alpha", type=float, default=0.90)
    parser.add_argument("--cvar_target", type=float, default=0.005)
    parser.add_argument("--cvar_window", type=int, default=64)
    parser.add_argument("--dual_lr", type=float, default=0.25)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=20.0)
    parser.add_argument("--dro_eta", type=float, default=0.10)
    parser.add_argument("--group_weight_floor", type=float, default=0.02)
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
    cfg = WGCVARConfig(
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
        guard_margin=args.guard_margin,
        cvar_alpha=args.cvar_alpha,
        cvar_target=args.cvar_target,
        cvar_window=args.cvar_window,
        dual_lr=args.dual_lr,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        dro_eta=args.dro_eta,
        group_weight_floor=args.group_weight_floor,
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
    WGCVARPPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
