"""Convert training-day AC-OPF trajectories into actor imitation examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import Env
from vmod_evaluation import action_reference_scales, device_ids


@lru_cache(maxsize=8)
def _load_arrays(load_path: str, generation_path: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(load_path), np.load(generation_path)


def _process_job(payload: tuple) -> dict | None:
    (
        job,
        q_targets,
        load_path,
        generation_path,
        env_id,
        svc_absorption_ratio,
    ) = payload
    if q_targets is None:
        return None
    load, generation = _load_arrays(load_path, generation_path)
    id_iber, id_svc = device_ids(env_id)
    reference_pv_s_scale, reference_svc_q_scale = action_reference_scales(env_id)
    cap_total = float(job["cap_total_mvar"])
    cap_buses = [20, 8] if env_id == 33 else []
    cap_q = (
        [cap_total / len(cap_buses)] * len(cap_buses)
        if cap_total > 0 and cap_buses
        else []
    )
    env = Env.grid_case(
        env_id,
        load,
        generation,
        id_iber,
        id_svc,
        enable_pq_curve=True,
        pv_s_scale=float(job["pv_s_scale"]),
        svc_q_scale=float(job["svc_q_scale"]),
        svc_absorption_ratio=float(svc_absorption_ratio),
        cap_buses=cap_buses or None,
        cap_q_mvar=cap_q or None,
        action_parameterization="reference_mvar",
        reference_pv_s_scale=reference_pv_s_scale,
        reference_svc_q_scale=reference_svc_q_scale,
    )
    observation = env.reset_at_step(int(job["t0"]))
    theta = np.asarray(
        [job["pv_s_scale"], job["svc_q_scale"], job["cap_total_mvar"]],
        dtype=np.float32,
    )
    observations = []
    actions = []
    errors = []
    for q_target in q_targets:
        action_target = env.physical_q_to_action(q_target)
        represented_q = env.action_clip(action_target)
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action_target.astype(np.float32))
        errors.append(float(np.max(np.abs(represented_q - q_target))))
        observation = env.step_model(action_target)[-1]
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "thetas": np.repeat(theta[None, :], len(q_targets), axis=0),
        "target_actions": np.asarray(actions, dtype=np.float32),
        "day_indices": np.full(len(q_targets), int(job["day_index"]), dtype=np.int32),
        "candidate_ids": np.full(
            len(q_targets), int(job["candidate_id"]), dtype=np.int16
        ),
        "representation_errors": np.asarray(errors, dtype=float),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--opf_steps_csv", type=Path, required=True)
    parser.add_argument("--load_profile", type=Path, required=True)
    parser.add_argument("--generation_profile", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--env", type=int, default=33, choices=[33, 69, 118])
    parser.add_argument("--svc_absorption_ratio", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    jobs = pd.read_csv(args.jobs_csv).sort_values(["scenario_index", "candidate_id"])
    steps = pd.read_csv(args.opf_steps_csv)
    q_columns = sorted(
        (column for column in steps if column.startswith("q_mvar_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    payloads = []
    for job in jobs.itertuples(index=False):
        trajectory = steps[
            (steps.candidate_id == job.candidate_id)
            & (steps.scenario_index == job.scenario_index)
        ].sort_values("step")
        if len(trajectory) != 96 or (trajectory.solver.astype(str) == "failed").any():
            q_targets = None
        else:
            q_targets = trajectory[q_columns].to_numpy(dtype=float)
        payloads.append(
            (
                job._asdict(),
                q_targets,
                str(args.load_profile.resolve()),
                str(args.generation_profile.resolve()),
                args.env,
                args.svc_absorption_ratio,
            )
        )

    if args.workers == 1:
        results = [_process_job(payload) for payload in payloads]
    else:
        # Python 3.11 on Windows can deadlock while replacing workers after
        # ``max_tasks_per_child`` is reached. The bounded 360-job study does not
        # need worker recycling, and a stable pool preserves deterministic job
        # ordering through ``executor.map``.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_process_job, payloads, chunksize=1))
    valid_results = [result for result in results if result is not None]
    excluded_jobs = len(results) - len(valid_results)
    if not valid_results:
        raise RuntimeError("No converged OPF trajectories were available")
    observations = np.concatenate(
        [result["observations"] for result in valid_results], axis=0
    )
    thetas = np.concatenate([result["thetas"] for result in valid_results], axis=0)
    target_actions = np.concatenate(
        [result["target_actions"] for result in valid_results], axis=0
    )
    day_indices = np.concatenate(
        [result["day_indices"] for result in valid_results], axis=0
    )
    candidate_ids = np.concatenate(
        [result["candidate_ids"] for result in valid_results], axis=0
    )
    representation_errors = np.concatenate(
        [result["representation_errors"] for result in valid_results], axis=0
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.out_dir / "opf_imitation_dataset.npz"
    np.savez_compressed(
        dataset,
        observations=observations,
        thetas=thetas,
        target_actions=target_actions,
        day_indices=day_indices,
        candidate_ids=candidate_ids,
    )
    errors = representation_errors
    coverage_rows = []
    for job, result in zip(jobs.itertuples(index=False), results):
        coverage_rows.append(
            {
                "candidate_id": int(job.candidate_id),
                "capacity_label": str(
                    getattr(job, "capacity_label", f"candidate_{job.candidate_id}")
                ),
                "day_index": int(job.day_index),
                "converged_job": result is not None,
                "examples": 0 if result is None else int(len(result["observations"])),
            }
        )
    coverage = (
        pd.DataFrame(coverage_rows)
        .groupby(["candidate_id", "capacity_label"], as_index=False)
        .agg(
            attempted_days=("day_index", "nunique"),
            converged_days=("converged_job", "sum"),
            examples=("examples", "sum"),
        )
        .sort_values("candidate_id")
    )
    coverage["excluded_nonconverged_days"] = (
        coverage.attempted_days - coverage.converged_days
    )
    manifest = {
        "examples": len(observations),
        "observation_dim": int(np.asarray(observations).shape[1]),
        "action_dim": int(np.asarray(target_actions).shape[1]),
        "unique_days": int(np.unique(day_indices).size),
        "unique_candidates": int(np.unique(candidate_ids).size),
        "excluded_nonconverged_jobs": excluded_jobs,
        "candidate_coverage": coverage.to_dict("records"),
        "svc_absorption_ratio": float(args.svc_absorption_ratio),
        "reference_pv_s_scale": action_reference_scales(args.env)[0],
        "reference_svc_q_scale": action_reference_scales(args.env)[1],
        "maximum_q_representation_error_mvar": float(errors.max(initial=0.0)),
        "mean_q_representation_error_mvar": float(errors.mean()) if errors.size else 0.0,
        "jobs_sha256": sha256(args.jobs_csv),
        "opf_steps_sha256": sha256(args.opf_steps_csv),
        "load_profile_sha256": sha256(args.load_profile),
        "generation_profile_sha256": sha256(args.generation_profile),
        "dataset_sha256": sha256(dataset),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
