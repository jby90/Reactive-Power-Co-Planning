"""Capacity-conditioned deterministic actor used by the VMOD study."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalise_theta(
    theta: np.ndarray | torch.Tensor,
    theta_min: np.ndarray | torch.Tensor,
    theta_max: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    scale = theta_max - theta_min
    if isinstance(theta, torch.Tensor):
        return torch.clamp((theta - theta_min) / scale, 0.0, 1.0)
    return np.clip((theta - theta_min) / scale, 0.0, 1.0)


class CapacityConditionedActor(nn.Module):
    """Two-layer actor with physical-capacity context and compatible state keys."""

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
        # Kept outside state_dict so historical actor checkpoints remain compatible.
        self.observation_mean = torch.zeros(obs_dim, dtype=torch.float32)
        self.observation_scale = torch.ones(obs_dim, dtype=torch.float32)
        self.observation_mask = torch.ones(obs_dim, dtype=torch.float32)

        self.pi_fc1 = nn.Linear(input_dim, h1)
        self.pi_fc2 = nn.Linear(h1, h2)
        self.pi_mu = nn.Linear(h2, act_dim)
        self.pi_log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        # Critic layers are retained to preserve the existing checkpoint schema.
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
        normalised = normalise_theta(theta, self.theta_min, self.theta_max)
        mean = self.observation_mean.to(device=obs.device, dtype=obs.dtype)
        scale = self.observation_scale.to(device=obs.device, dtype=obs.dtype)
        mask = self.observation_mask.to(device=obs.device, dtype=obs.dtype)
        transformed_obs = ((obs - mean) / scale) * mask
        return torch.cat((transformed_obs, normalised), dim=-1)

    def configure_observation_transform(
        self,
        mean: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> None:
        """Install a training-fitted observation transform without changing topology."""
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
        expected = self.observation_mean.shape
        if (
            mean_tensor.shape != expected
            or scale_tensor.shape != expected
            or mask_tensor.shape != expected
        ):
            raise ValueError(
                "Observation transform shape mismatch: "
                f"expected {tuple(expected)}, got mean={tuple(mean_tensor.shape)}, "
                f"scale={tuple(scale_tensor.shape)}, mask={tuple(mask_tensor.shape)}"
            )
        if torch.any(scale_tensor <= 0):
            raise ValueError("Observation scales must be strictly positive")
        self.observation_mean = mean_tensor.clone()
        self.observation_scale = scale_tensor.clone()
        self.observation_mask = mask_tensor.clone()

    @staticmethod
    def _trunk(x: torch.Tensor, fc1: nn.Linear, fc2: nn.Linear) -> torch.Tensor:
        return F.relu(fc2(F.relu(fc1(x))))

    def _pi(self, obs: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._features(obs, theta)
        mean = torch.tanh(self.pi_mu(self._trunk(features, self.pi_fc1, self.pi_fc2)))
        return mean, torch.clamp(self.pi_log_std, -20.0, 2.0)

    def policy_parameters(self):
        for module in (self.pi_fc1, self.pi_fc2, self.pi_mu):
            yield from module.parameters()
        yield self.pi_log_std


# Keep the historical class name as a checkpoint/API compatibility alias.
ConcatDirectionalActorCritic = CapacityConditionedActor


def load_policy_initialisation(model: nn.Module, checkpoint_path: str) -> None:
    """Load exactly the policy tensors from an OPF-distillation checkpoint."""
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload.get("policy_state_dict", payload)
    observation_transform = payload.get("observation_transform")
    if observation_transform is not None:
        if not hasattr(model, "configure_observation_transform"):
            raise ValueError(
                "Checkpoint contains an observation transform but the target model "
                "does not support it"
            )
        model.configure_observation_transform(
            observation_transform["mean"],
            observation_transform["scale"],
            observation_transform["mask"],
        )
    policy_prefixes = ("pi_fc1.", "pi_fc2.", "pi_mu.", "pi_log_std")
    selected = {
        key: value for key, value in source.items() if key.startswith(policy_prefixes)
    }
    expected = {
        key for key in model.state_dict() if key.startswith(policy_prefixes)
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(
            f"Policy initialisation mismatch; missing={missing}, extra={extra}"
        )
    model.load_state_dict(selected, strict=False)
