"""Deterministic held-out evaluation for a CRDC-PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import Env
from PPO_theta_crdc import ConcatDirectionalActorCritic
from PPO_theta_concat_corrected import BlindScalarActorCritic, ConcatScalarActorCritic


STEPS_PER_DAY = 96
DT_HOURS = 0.25


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def device_ids(env_id: int) -> tuple[list[int], list[int]]:
    if env_id == 33:
        return [17, 21, 24], [32]
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    if env_id == 118:
        return [33, 50, 53, 68, 74, 97, 107, 111], [44, 104]
    raise ValueError(f"Unsupported environment: {env_id}")


def load_day_indices(path: str | Path) -> list[int]:
    """Load held-out day indices from the released CSV or compatible JSON."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
        if "day_index" not in table.columns:
            raise ValueError(f"Day table must contain a day_index column: {source}")
        return [int(day) for day in table["day_index"].tolist()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("final_day_indices", "day_indices"):
        if key in payload:
            return [int(day) for day in payload[key]]
    raise ValueError(f"Day metadata must define final_day_indices or day_indices: {source}")


def _resolve_profile_path(value: str, checkpoint_path: Path | None) -> Path:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / path)
        if checkpoint_path is not None:
            candidates.append(checkpoint_path.resolve().parent / path)
    # Released checkpoints retain their provenance path, but a clean clone keeps
    # the same arrays under data/inputs. Fall back by filename when the recorded
    # absolute path is not available on the current machine.
    module_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            module_dir / "data" / "inputs" / path.name,
            module_dir.parent / "data" / "inputs" / path.name,
            Path.cwd() / "data" / "inputs" / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Profile array not found: {value}")


def load_profiles_from_cfg(
    cfg: dict, checkpoint_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact profiles recorded by a checkpoint, with legacy fallback."""
    load_value = cfg.get("load_profile_path")
    generation_value = cfg.get("generation_profile_path")
    if bool(load_value) != bool(generation_value):
        raise ValueError(
            "Checkpoint must define both load_profile_path and generation_profile_path"
        )
    if load_value:
        load_path = _resolve_profile_path(str(load_value), checkpoint_path)
        generation_path = _resolve_profile_path(str(generation_value), checkpoint_path)
    else:
        root = Path(__file__).resolve().parent
        load_path = root / "load96.npy"
        generation_path = root / "gen96.npy"
    load_pu = np.load(load_path)
    gene_pu = np.load(generation_path)
    if load_pu.ndim not in (1, 2) or gene_pu.ndim not in (1, 2):
        raise ValueError("Profile arrays must be one- or two-dimensional")
    if len(load_pu) != len(gene_pu):
        raise ValueError("Load and generation profiles must have equal time lengths")
    if len(load_pu) < STEPS_PER_DAY:
        raise ValueError("Profile arrays must contain at least one 96-step day")
    return load_pu, gene_pu


def build_env(
    env_id: int,
    theta: np.ndarray,
    cap_buses: list[int],
    pv_generation_scale: float,
    action_parameterization: str = "relative",
    load_pu: np.ndarray | None = None,
    gene_pu: np.ndarray | None = None,
    svc_absorption_ratio: float = 0.0,
) -> Env.grid_case:
    if load_pu is None or gene_pu is None:
        load_pu, gene_pu = load_profiles_from_cfg({})
    id_iber, id_svc = device_ids(env_id)
    cap_total = float(theta[2])
    cap_q = [cap_total / len(cap_buses)] * len(cap_buses) if cap_total > 0 and cap_buses else []
    return Env.grid_case(
        env_id,
        load_pu,
        gene_pu,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=float(theta[0]),
        pv_generation_scale=float(pv_generation_scale),
        svc_q_scale=float(theta[1]),
        svc_absorption_ratio=float(svc_absorption_ratio),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization=action_parameterization,
    )


def load_actor(
    run_dir: Path, obs_dim: int, act_dim: int, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = run_dir / "models" / "checkpoint.pth"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    actor_class = {
        "CRDC-PPO": ConcatDirectionalActorCritic,
        "WG-CVaR-PPO": ConcatDirectionalActorCritic,
        "Concat-Scalar-PPO-Corrected": ConcatScalarActorCritic,
        "Blind-Scalar-PPO-Corrected": BlindScalarActorCritic,
    }.get(payload.get("algorithm"))
    if actor_class is None:
        raise ValueError(f"Unsupported checkpoint algorithm: {payload.get('algorithm')}")
    actor = actor_class(
        obs_dim,
        act_dim,
        [cfg["pv_s_scale_min"], cfg["svc_q_scale_min"], cfg["cap_total_min"]],
        [cfg["pv_s_scale_max"], cfg["svc_q_scale_max"], cfg["cap_total_max"]],
        log_std_init=cfg["log_std_init"],
    ).to(device)
    actor.load_state_dict(payload["ac"])
    actor.eval()
    return actor, payload


def evaluate_day(
    env: Env.grid_case,
    actor: torch.nn.Module,
    theta: np.ndarray,
    day: int,
    device: torch.device,
) -> dict:
    obs = env.reset_at_step(int(day) * STEPS_PER_DAY)
    under_steps = 0
    over_steps = 0
    pf_fail = 0
    line_loss_mwh = 0.0
    minimum_voltage = np.inf
    maximum_voltage = -np.inf
    for _ in range(STEPS_PER_DAY):
        with torch.no_grad():
            action, _ = actor._pi(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
                torch.as_tensor(theta, dtype=torch.float32, device=device),
            )
        try:
            next_obs, _, _, _, _, _, vmax, vmin, grid_loss, new_state = env.step_model(
                action.detach().cpu().numpy()
            )
        except Exception:
            pf_fail = 1
            break
        voltage = np.asarray(next_obs[: len(env.model.bus)], dtype=float)
        under_steps += int(np.any(voltage < 0.95))
        over_steps += int(np.any(voltage > 1.05))
        minimum_voltage = min(minimum_voltage, float(vmin))
        maximum_voltage = max(maximum_voltage, float(vmax))
        line_loss_mwh += max(0.0, float(-grid_loss)) * DT_HOURS
        obs = new_state
    return {
        "day": int(day),
        "event": int(bool(under_steps or over_steps or pf_fail)),
        "under_event": int(bool(under_steps or pf_fail)),
        "over_event": int(bool(over_steps or pf_fail)),
        "under_steps": under_steps,
        "over_steps": over_steps,
        "pf_fail": pf_fail,
        "minimum_voltage_pu": minimum_voltage,
        "maximum_voltage_pu": maximum_voltage,
        "line_loss_mwh": line_loss_mwh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRDC-PPO on fixed held-out days")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--theta", default="0.7,0.7,0.25")
    parser.add_argument("--days_metadata", required=True)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--cap_buses", default="20,8")
    parser.add_argument("--pv_generation_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    theta = np.asarray(parse_floats(args.theta), dtype=np.float32)
    if theta.shape != (3,):
        parser.error("--theta must contain exactly three values")
    days = load_day_indices(args.days_metadata)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_payload = torch.load(
        Path(args.run_dir) / "models" / "checkpoint.pth",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_path = Path(args.run_dir) / "models" / "checkpoint.pth"
    checkpoint_cfg = checkpoint_payload.get("cfg", {})
    load_pu, gene_pu = load_profiles_from_cfg(checkpoint_cfg, checkpoint_path)
    env = build_env(
        args.env,
        theta,
        parse_ints(args.cap_buses),
        args.pv_generation_scale,
        str(checkpoint_cfg.get("action_parameterization", "relative")),
        load_pu,
        gene_pu,
        float(checkpoint_cfg.get("svc_absorption_ratio", 0.0)),
    )
    actor, checkpoint = load_actor(
        Path(args.run_dir), len(env.observation_space), len(env.action_space), device
    )
    rows = [evaluate_day(env, actor, theta, day, device) for day in days]
    daily = pd.DataFrame(rows)
    summary = {
        "algorithm": checkpoint["algorithm"],
        "checkpoint_step": int(checkpoint["total_step"]),
        "theta": theta.tolist(),
        "evaluated_days": len(daily),
        "event_days": int(daily["event"].sum()),
        "under_event_days": int(daily["under_event"].sum()),
        "over_event_days": int(daily["over_event"].sum()),
        "pf_fail_days": int(daily["pf_fail"].sum()),
        "mean_daily_line_loss_mwh": float(daily["line_loss_mwh"].mean()),
        "worst_minimum_voltage_pu": float(daily["minimum_voltage_pu"].min()),
        "worst_maximum_voltage_pu": float(daily["maximum_voltage_pu"].max()),
        "day_indices": days,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "daily.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
