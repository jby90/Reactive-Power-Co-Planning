"""Run the frozen three-stage capacity-planning and ablation protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
EVALUATOR_SCRIPT = str(ROOT / "src" / "evaluate_capacity_job_table.py")
OUT = ROOT / "reproduced" / "capacity_planning"
LOCKED_DAYS = ROOT / "data" / "results" / "capacity_planning" / "locked_days"
SEEDS = (42, 43, 44)
EPSILON = 0.01
GRID_SIZE = 2000
WORKERS_PER_EVALUATION = 4


def run_dir(family: str, seed: int) -> Path:
    if family == "wg":
        return ROOT / "models" / "wg_cvar" / f"seed{seed}"
    if family == "concat":
        return ROOT / "models" / "concat" / f"seed{seed}"
    raise ValueError(f"Unknown family: {family}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_index(frame: pd.DataFrame) -> pd.Series:
    return 50.0 * frame.pv_s_scale + 50.0 * frame.svc_q_scale + 2.0 * frame.cap_total_mvar


def grid_candidates() -> pd.DataFrame:
    pv_levels = np.linspace(0.7, 1.5, 20)
    svc_levels = np.linspace(0.7, 1.5, 20)
    cap_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    rows = []
    candidate_id = 0
    for cap in cap_levels:
        for pv in pv_levels:
            for svc in svc_levels:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "pv_s_scale": float(pv),
                        "svc_q_scale": float(svc),
                        "cap_total_mvar": float(cap),
                        "selection_role": "grid",
                    }
                )
                candidate_id += 1
    frame = pd.DataFrame(rows)
    if len(frame) != GRID_SIZE:
        raise AssertionError(f"Expected {GRID_SIZE} grid candidates")
    frame["resource_index"] = resource_index(frame)
    return frame


def diagonal_candidates() -> pd.DataFrame:
    fraction = np.linspace(0.0, 1.0, 17)
    frame = pd.DataFrame(
        {
            "candidate_id": np.arange(GRID_SIZE, GRID_SIZE + len(fraction)),
            "pv_s_scale": 0.7 + 0.8 * fraction,
            "svc_q_scale": 0.7 + 0.8 * fraction,
            "cap_total_mvar": fraction,
            "selection_role": "risk_path",
            "path_fraction": fraction,
        }
    )
    frame["resource_index"] = resource_index(frame)
    return frame


def make_jobs(candidates: pd.DataFrame, split: str, path: Path) -> pd.DataFrame:
    days = pd.read_csv(LOCKED_DAYS / f"{split}_days.csv")
    rows = []
    for candidate in candidates.itertuples(index=False):
        for day in days.itertuples(index=False):
            rows.append(
                {
                    "candidate_id": int(candidate.candidate_id),
                    "scenario_index": int(day.scenario_index),
                    "pv_s_scale": float(candidate.pv_s_scale),
                    "svc_q_scale": float(candidate.svc_q_scale),
                    "cap_total_mvar": float(candidate.cap_total_mvar),
                    "t0": int(day.t0),
                }
            )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def evaluate_one(
    stage: str, jobs_csv: Path, family: str, projected: bool, seed: int
) -> Path:
    variant = f"{family}_{'projected' if projected else 'raw'}"
    out_dir = OUT / stage / variant / f"seed{seed}"
    if (out_dir / "episodes.csv").exists() and (out_dir / "summary.json").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        EVALUATOR_SCRIPT,
        "--run_dir",
        str(run_dir(family, seed)),
        "--jobs_csv",
        str(jobs_csv),
        "--workers",
        str(WORKERS_PER_EVALUATION),
        "--out_dir",
        str(out_dir),
    ]
    if projected:
        command.append("--projected")
    with (log_dir / f"{stage}_{variant}_seed{seed}.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return out_dir


def evaluate_seeds(
    stage: str, jobs_csv: Path, family: str = "wg", projected: bool = True
) -> None:
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(evaluate_one, stage, jobs_csv, family, projected, seed)
            for seed in SEEDS
        ]
        for future in futures:
            future.result()


def aggregate_stage(stage: str, candidates: pd.DataFrame, variant: str) -> pd.DataFrame:
    seed_rows = []
    for seed in SEEDS:
        path = OUT / stage / variant / f"seed{seed}" / "episodes.csv"
        daily = pd.read_csv(path)
        grouped = daily.groupby("candidate_id", as_index=False).agg(
            risk=("risk", "mean"),
            pf_fail_rate=("pf_fail", "mean"),
            mean_daily_loss_mwh=("daily_loss_mwh", "mean"),
            mean_intervention_rate=("intervention_rate", "mean"),
            projection_failure_steps=("projection_failures", "sum"),
            worst_vmin=("vmin", "min"),
            worst_vmax=("vmax", "max"),
            raw_worst_vmin=("raw_vmin", "min"),
            raw_worst_vmax=("raw_vmax", "max"),
        )
        grouped["seed"] = seed
        seed_rows.append(grouped)
    by_seed = pd.concat(seed_rows, ignore_index=True).merge(
        candidates, on="candidate_id", how="left", validate="many_to_one"
    )
    by_seed.to_csv(OUT / stage / f"{variant}_by_seed.csv", index=False)
    summary = by_seed.groupby("candidate_id", as_index=False).agg(
        worst_seed_risk=("risk", "max"),
        mean_seed_risk=("risk", "mean"),
        worst_seed_pf_fail_rate=("pf_fail_rate", "max"),
        mean_daily_loss_mwh=("mean_daily_loss_mwh", "mean"),
        maximum_seed_loss_mwh=("mean_daily_loss_mwh", "max"),
        mean_intervention_rate=("mean_intervention_rate", "mean"),
        projection_failure_steps=("projection_failure_steps", "sum"),
        worst_vmin=("worst_vmin", "min"),
        worst_vmax=("worst_vmax", "max"),
        raw_worst_vmin=("raw_worst_vmin", "min"),
        raw_worst_vmax=("raw_worst_vmax", "max"),
    )
    summary = summary.merge(candidates, on="candidate_id", how="left", validate="one_to_one")
    summary["robust_feasible"] = (
        (summary.worst_seed_risk <= EPSILON + 1e-12)
        & (summary.worst_seed_pf_fail_rate <= EPSILON + 1e-12)
        & (summary.projection_failure_steps == 0)
    ).astype(int)
    summary.to_csv(OUT / stage / f"{variant}_candidate_summary.csv", index=False)
    return summary


def select_refine(screen: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    zero_event = screen[
        (screen.worst_seed_risk == 0.0)
        & (screen.worst_seed_pf_fail_rate == 0.0)
        & (screen.projection_failure_steps == 0)
    ].sort_values(["resource_index", "mean_daily_loss_mwh"]).head(300)
    lowest_risk = screen.sort_values(
        ["worst_seed_risk", "worst_seed_pf_fail_rate", "resource_index", "mean_daily_loss_mwh"]
    ).head(100)
    ids = pd.concat([zero_event[["candidate_id"]], lowest_risk[["candidate_id"]]]).drop_duplicates()
    selected = grid[grid.candidate_id.isin(ids.candidate_id)].copy()
    selected["selection_role"] = "refine_candidate"
    return selected.sort_values("candidate_id")


def select_confirm(refine: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    feasible = refine[refine.robust_feasible == 1].sort_values(
        ["resource_index", "mean_daily_loss_mwh"]
    ).head(20)
    selected = grid[grid.candidate_id.isin(feasible.candidate_id)].copy()
    selected["selection_role"] = "optimiser_candidate"
    path = diagonal_candidates()
    return pd.concat([selected, path], ignore_index=True).drop_duplicates(
        subset=["candidate_id"], keep="first"
    )


def choose_optimum(confirm: pd.DataFrame) -> pd.DataFrame:
    feasible = confirm[
        (confirm.robust_feasible == 1) & (confirm.candidate_id < GRID_SIZE)
    ].sort_values(["resource_index", "mean_daily_loss_mwh"])
    optimum = feasible.head(1).copy()
    optimum.to_csv(OUT / "final_optimum.csv", index=False)
    return optimum


def run_ablation(optimum: pd.DataFrame, confirm_jobs: pd.DataFrame) -> None:
    if optimum.empty:
        return
    optimum_id = int(optimum.iloc[0].candidate_id)
    high_id = GRID_SIZE + 16
    selected_jobs = confirm_jobs[confirm_jobs.candidate_id.isin([optimum_id, high_id])].copy()
    jobs_path = OUT / "ablation" / "jobs.csv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    selected_jobs.to_csv(jobs_path, index=False)
    tasks = [
        (family, projected, seed)
        for family in ("wg", "concat")
        for projected in (False, True)
        for seed in SEEDS
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(evaluate_one, "ablation", jobs_path, family, projected, seed)
            for family, projected, seed in tasks
        ]
        for future in futures:
            future.result()

    candidates = pd.concat(
        [
            optimum[["candidate_id", "pv_s_scale", "svc_q_scale", "cap_total_mvar", "selection_role", "resource_index"]],
            diagonal_candidates().query("candidate_id == @high_id"),
        ],
        ignore_index=True,
    )
    tables = []
    for family in ("wg", "concat"):
        for projected in (False, True):
            variant = f"{family}_{'projected' if projected else 'raw'}"
            table = aggregate_stage("ablation", candidates, variant)
            table["variant"] = variant
            tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(OUT / "ablation" / "comparison.csv", index=False)


def run_raw_risk_path(confirm_jobs: pd.DataFrame) -> None:
    path_ids = set(diagonal_candidates().candidate_id.tolist())
    jobs = confirm_jobs[confirm_jobs.candidate_id.isin(path_ids)].copy()
    jobs_path = OUT / "risk_path_raw" / "jobs.csv"
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(jobs_path, index=False)
    evaluate_seeds("risk_path_raw", jobs_path, family="wg", projected=False)
    aggregate_stage("risk_path_raw", diagonal_candidates(), "wg_raw")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = grid_candidates()
    grid.to_csv(OUT / "grid_candidates.csv", index=False)

    screen_jobs_path = OUT / "screen" / "jobs.csv"
    screen_jobs = make_jobs(grid, "screen", screen_jobs_path)
    evaluate_seeds("screen", screen_jobs_path)
    screen = aggregate_stage("screen", grid, "wg_projected")

    refine_candidates = select_refine(screen, grid)
    (OUT / "refine").mkdir(parents=True, exist_ok=True)
    refine_candidates.to_csv(OUT / "refine" / "selected_candidates.csv", index=False)
    refine_jobs_path = OUT / "refine" / "jobs.csv"
    make_jobs(refine_candidates, "refine", refine_jobs_path)
    evaluate_seeds("refine", refine_jobs_path)
    refine = aggregate_stage("refine", refine_candidates, "wg_projected")

    confirm_candidates = select_confirm(refine, grid)
    (OUT / "confirm").mkdir(parents=True, exist_ok=True)
    confirm_candidates.to_csv(OUT / "confirm" / "selected_candidates.csv", index=False)
    confirm_jobs_path = OUT / "confirm" / "jobs.csv"
    confirm_jobs = make_jobs(confirm_candidates, "confirm", confirm_jobs_path)
    evaluate_seeds("confirm", confirm_jobs_path)
    confirm = aggregate_stage("confirm", confirm_candidates, "wg_projected")
    optimum = choose_optimum(confirm)

    run_raw_risk_path(confirm_jobs)
    run_ablation(optimum, confirm_jobs)
    metadata = {
        "protocol": "experiments/final_capacity_planning_protocol.md",
        "grid_candidates": len(grid),
        "refine_candidates": len(refine_candidates),
        "confirm_candidates": len(confirm_candidates),
        "epsilon": EPSILON,
        "training_seeds": list(SEEDS),
        "jobs_sha256": {
            "screen": sha256(screen_jobs_path),
            "refine": sha256(refine_jobs_path),
            "confirm": sha256(confirm_jobs_path),
        },
        "optimum_found": not optimum.empty,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
