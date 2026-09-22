"""Tune and evaluate a fixed-sensitivity online optimisation baseline."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from vmod_current_study import (
    DEVELOPMENT_PROFILES_33,
    DEVELOPMENT_PROTOCOL_33,
    EXTERNAL_PROFILES_33,
    EXTERNAL_PROTOCOL_33,
    main_run_dir,
    read_selected_margin,
)
from vmod_evaluation import build_env, load_day_indices
from vmod_sensitivity_qp import (
    SensitivityQPConfig,
    SensitivityQPController,
    estimate_voltage_sensitivity,
    evaluate_controller_day,
    representative_profile_step,
)


ROOT = Path(__file__).resolve().parent


def load_capacity_rows(candidate_ids: tuple[int, ...]) -> list[dict]:
    frame = pd.read_csv(EXTERNAL_PROTOCOL_33 / "capacity_path.csv").set_index(
        "path_index", drop=False
    )
    return [frame.loc[candidate].to_dict() for candidate in candidate_ids]


def theta_from_row(row: dict) -> list[float]:
    return [
        float(row["pv_s_scale"]),
        float(row["svc_q_scale"]),
        float(row["cap_total_mvar"]),
    ]


def evaluate_payload(payload: tuple) -> dict:
    (
        theta,
        days,
        profile_dir,
        sensitivity,
        config_dict,
        label,
    ) = payload
    profile_root = Path(profile_dir)
    load = np.load(profile_root / "load_15min_366d.npy")
    generation = np.load(profile_root / "pv_15min_366d.npy")
    env = build_env(
        33,
        np.asarray(theta, dtype=float),
        [20, 8],
        1.0,
        "reference_mvar",
        load,
        generation,
        1.0,
    )
    config = SensitivityQPConfig(**config_dict)
    controller = SensitivityQPController(np.asarray(sensitivity), config)
    rows = [evaluate_controller_day(env, controller, int(day)) for day in days]
    frame = pd.DataFrame(rows)
    return {
        "label": label,
        "theta": list(map(float, theta)),
        "config": config.to_dict(),
        "days": int(len(frame)),
        "event_days": int(frame.event.sum()),
        "pf_fail_days": int(frame.pf_fail.sum()),
        "solver_failures": int(frame.solver_failures.sum()),
        "mean_daily_loss_mwh": float(frame.line_loss_mwh.mean()),
        "mean_daily_absolute_reactive_throughput_mvarh": float(
            frame.absolute_reactive_throughput_mvarh.mean()
        ),
        "worst_minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "worst_maximum_voltage_pu": float(frame.maximum_voltage_pu.max()),
        "mean_online_optimisation_ms": float(
            frame.mean_online_optimisation_ms.mean()
        ),
        "daily_rows": rows,
    }


def config_grid() -> list[SensitivityQPConfig]:
    return [
        SensitivityQPConfig(
            band_weight=band,
            centering_weight=centering,
            reactive_weight=reactive,
        )
        for band in (1_000.0, 10_000.0)
        for centering in (0.0, 1.0)
        for reactive in (0.01, 0.1, 1.0)
    ]


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "daily_rows"}


def write_result(result: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["daily_rows"]).to_csv(destination / "daily.csv", index=False)
    (destination / "summary.json").write_text(
        json.dumps(compact(result), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_ids", default="2,3,4")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    candidate_ids = tuple(int(item) for item in args.candidate_ids.split(","))
    rows = load_capacity_rows(candidate_ids)
    selected = next(row for row in rows if int(row["path_index"]) == 3)
    selected_theta = theta_from_row(selected)
    path_run = main_run_dir(read_selected_margin())
    output = path_run / "sensitivity_qp_baseline"
    output.mkdir(parents=True, exist_ok=True)

    fitting_days = load_day_indices(DEVELOPMENT_PROTOCOL_33 / "fitting_days.csv")
    development_load = np.load(DEVELOPMENT_PROFILES_33 / "load_15min_366d.npy")
    development_generation = np.load(
        DEVELOPMENT_PROFILES_33 / "pv_15min_366d.npy"
    )
    representative_step = representative_profile_step(
        development_load, development_generation, fitting_days
    )
    sensitivity_env = build_env(
        33,
        np.asarray(selected_theta),
        [20, 8],
        1.0,
        "reference_mvar",
        development_load,
        development_generation,
        1.0,
    )
    sensitivity, sensitivity_manifest = estimate_voltage_sensitivity(
        sensitivity_env, representative_step
    )
    np.save(output / "voltage_q_sensitivity.npy", sensitivity)
    (output / "sensitivity_manifest.json").write_text(
        json.dumps(sensitivity_manifest, indent=2), encoding="utf-8"
    )

    frozen_config_path = output / "frozen_config.json"
    if frozen_config_path.exists():
        frozen_config = SensitivityQPConfig(
            **json.loads(frozen_config_path.read_text(encoding="utf-8"))["config"]
        )
    else:
        grid = config_grid()
        payloads = [
            (
                selected_theta,
                fitting_days,
                str(DEVELOPMENT_PROFILES_33),
                sensitivity,
                config.to_dict(),
                f"config_{index:02d}",
            )
            for index, config in enumerate(grid)
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            tuning_results = list(executor.map(evaluate_payload, payloads, chunksize=1))
        tuning_table = pd.DataFrame(compact(result) for result in tuning_results)
        tuning_table["config"] = tuning_table.config.map(json.dumps)
        tuning_table.to_csv(output / "development_tuning.csv", index=False)
        best = min(
            tuning_results,
            key=lambda item: (
                item["pf_fail_days"],
                item["event_days"],
                item["solver_failures"],
                item["mean_daily_loss_mwh"],
            ),
        )
        frozen_config = SensitivityQPConfig(**best["config"])
        frozen_config_path.write_text(
            json.dumps(
                {
                    "selection_rule": (
                        "lexicographic development-only minimisation of AC failures, "
                        "event-days, optimiser failures, and mean daily line loss"
                    ),
                    "representative_step": representative_step,
                    "config": frozen_config.to_dict(),
                    "development_result": compact(best),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    evaluation_payloads = []
    for row in rows:
        for split in ("selection", "confirmation"):
            days = load_day_indices(EXTERNAL_PROTOCOL_33 / f"{split}_days.csv")
            evaluation_payloads.append(
                (
                    theta_from_row(row),
                    days,
                    str(EXTERNAL_PROFILES_33),
                    sensitivity,
                    frozen_config.to_dict(),
                    f"{row['capacity_label']}::{split}",
                )
            )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate_payload, evaluation_payloads, chunksize=1))
    summaries = []
    for result in results:
        label, split = result["label"].split("::")
        destination = output / label / split
        write_result(result, destination)
        summaries.append({"capacity_label": label, "split": split, **compact(result)})
    pd.DataFrame(summaries).drop(columns=["config"]).to_csv(
        output / "path_summary.csv", index=False
    )
    final = {
        "candidate_ids": list(candidate_ids),
        "python_executable_name": Path(sys.executable).name,
        "frozen_config": frozen_config.to_dict(),
        "sensitivity_manifest": sensitivity_manifest,
        "results": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
