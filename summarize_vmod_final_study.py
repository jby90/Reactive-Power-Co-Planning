"""Assemble final VMOD evidence and enforce the pre-specified claim gates."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from vmod_statistics import clopper_pearson_upper, mean_t_interval
from vmod_current_study import (
    DEVELOPMENT_MARGIN_ROOT,
    FINAL_ROOT,
    PATH_MARGIN_SUMMARY,
    env69_run_dir,
    main_run_dir,
    read_selected_margin,
)


ROOT = Path(__file__).resolve().parent
MARGIN_ROOT = DEVELOPMENT_MARGIN_ROOT
OUT = FINAL_ROOT
MANUSCRIPT_RESULTS = Path(
    os.environ.get(
        "VMOD_RESULTS_TEX",
        str(
            ROOT
            / "outputs"
            / "."
            / "."
            / "vmod_results.tex"
        ),
    )
)
MIN_TEACHER_COVERAGE_FRACTION = 0.8


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for(paths: list[Path]) -> None:
    while not all(path.exists() for path in paths):
        time.sleep(60)


def comparison(summary: dict, baseline: str, split: str) -> dict:
    return next(
        row
        for row in summary["comparisons"]
        if row["baseline"] == baseline and row["split"] == split
    )


def tex_number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def tex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{} ").replace("_", "\\_")


def teacher_coverage_gate(row: dict) -> tuple[bool, float]:
    attempted = int(row["attempted_days"])
    converged = int(row["converged_days"])
    fraction = 0.0 if attempted == 0 else converged / attempted
    return bool(fraction >= MIN_TEACHER_COVERAGE_FRACTION), float(fraction)


def confirmed_boundary_index(boundary: dict) -> int | None:
    if not boundary.get("boundary_confirmed", False):
        return None
    selected = next(
        (row for row in boundary.get("points", []) if row.get("role") == "selected"),
        None,
    )
    return None if selected is None else int(selected["path_index"])


def validate_confirmation_statistics(
    boundary: dict, expected_seed_count: int = 5
) -> None:
    """Reject stale or internally inconsistent finite-sample evidence."""
    confirmation_days = int(boundary["confirmation_days_per_seed"])
    selected = next(
        row for row in boundary["points"] if row["role"] == "selected"
    )
    seed_rows = selected["seed_summaries"]
    if len(seed_rows) != expected_seed_count:
        raise ValueError(
            f"Expected {expected_seed_count} selected-point seeds, got {len(seed_rows)}"
        )
    for row in seed_rows:
        trials = int(row["evaluated_days"])
        events = int(row["event_days"])
        if trials != confirmation_days:
            raise ValueError(
                "Selected-point confirmation-day mismatch: "
                f"expected {confirmation_days}, got {trials} for seed {row.get('seed')}"
            )
        expected = clopper_pearson_upper(events, trials)
        reported = float(row["one_sided_clopper_pearson_upper_95"])
        if abs(expected - reported) > 1e-12:
            raise ValueError(
                "Stale or incorrect Clopper-Pearson upper bound for seed "
                f"{row.get('seed')}: expected {expected}, got {reported}"
            )


def validate_conditioning_ablation(
    conditioning: dict, confirmation_days: int, expected_seed_count: int = 5
) -> None:
    """Reject selection-set or stale conditioning-ablation summaries."""
    if conditioning.get("primary_inference_split") != "confirmation":
        raise ValueError(
            "Conditioning-ablation inference must use the confirmation split"
        )
    rows = conditioning.get("seed_contrasts", [])
    if len(rows) != expected_seed_count:
        raise ValueError(
            f"Expected {expected_seed_count} conditioning contrasts, got {len(rows)}"
        )
    for row in rows:
        if row.get("split") != "confirmation":
            raise ValueError("Conditioning contrast is not labelled confirmation")
        if int(row.get("paired_days", -1)) != int(confirmation_days):
            raise ValueError(
                "Conditioning-ablation paired-day count does not match the "
                "external confirmation split"
            )


def paired_boundary_outcomes(
    vmod_daily: pd.DataFrame,
    conventional_daily: pd.DataFrame,
    method: str = "droop",
) -> dict:
    baseline = conventional_daily.loc[
        conventional_daily.method == method,
        ["day", "line_loss_mwh", "absolute_reactive_throughput_mvarh"],
    ].rename(
        columns={
            "line_loss_mwh": "baseline_loss_mwh",
            "absolute_reactive_throughput_mvarh": "baseline_throughput_mvarh",
        }
    )
    seed_rows = []
    for seed, group in vmod_daily.groupby("seed", sort=True):
        paired = group.merge(baseline, on="day", validate="one_to_one")
        if len(paired) != len(group) or len(paired) != len(baseline):
            raise ValueError("Boundary comparison requires exactly matched days")
        seed_rows.append(
            {
                "seed": int(seed),
                "paired_days": int(len(paired)),
                "vmod_mean_loss_mwh": float(paired.line_loss_mwh.mean()),
                "droop_mean_loss_mwh": float(paired.baseline_loss_mwh.mean()),
                "vmod_minus_droop_loss_mwh": float(
                    (paired.line_loss_mwh - paired.baseline_loss_mwh).mean()
                ),
                "vmod_minus_droop_throughput_mvarh": float(
                    (
                        paired.absolute_reactive_throughput_mvarh
                        - paired.baseline_throughput_mvarh
                    ).mean()
                ),
            }
        )
    frame = pd.DataFrame(seed_rows)
    loss_mean, loss_low, loss_high = mean_t_interval(
        frame.vmod_minus_droop_loss_mwh.to_numpy()
    )
    throughput_mean, throughput_low, throughput_high = mean_t_interval(
        frame.vmod_minus_droop_throughput_mvarh.to_numpy()
    )
    droop_loss = float(frame.droop_mean_loss_mwh.mean())
    return {
        "method": method,
        "n_training_seeds": int(len(frame)),
        "paired_days_per_seed": int(frame.paired_days.min()),
        "vmod_minus_droop_loss_mean_mwh": loss_mean,
        "loss_seed_sd_mwh": float(frame.vmod_minus_droop_loss_mwh.std(ddof=1)),
        "loss_ci95_low_mwh": loss_low,
        "loss_ci95_high_mwh": loss_high,
        "loss_change_percent_of_droop": 100.0 * loss_mean / droop_loss,
        "vmod_minus_droop_throughput_mean_mvarh": throughput_mean,
        "throughput_seed_sd_mvarh": float(
            frame.vmod_minus_droop_throughput_mvarh.std(ddof=1)
        ),
        "throughput_ci95_low_mvarh": throughput_low,
        "throughput_ci95_high_mvarh": throughput_high,
        "seed_values": seed_rows,
    }


def offline_computation_summary(
    teacher_episodes_path: Path,
    teacher_steps_path: Path,
    policy_summary_paths: list[Path],
    dataset_manifest: dict,
) -> dict:
    teacher = pd.read_csv(teacher_episodes_path)
    teacher_steps = pd.read_csv(teacher_steps_path)
    if teacher.empty or "solve_total_seconds" not in teacher:
        raise ValueError("Teacher episodes do not contain offline solve timings")
    if teacher_steps.empty or "solver" not in teacher_steps:
        raise ValueError("Teacher steps do not contain solver provenance")
    policies = [read_json(path) for path in policy_summary_paths]
    if len(policies) != 5:
        raise ValueError("Offline-cost reporting requires five policy summaries")
    walls = pd.Series(
        [float(row["training_wall_seconds"]) for row in policies], dtype=float
    )
    throughputs = pd.Series(
        [float(row["training_example_updates_per_second"]) for row in policies],
        dtype=float,
    )
    realised_training_days = [
        int(row["training_unique_days_with_examples"]) for row in policies
    ]
    realised_validation_days = [
        int(row["validation_unique_days_with_examples"]) for row in policies
    ]
    if len(set(realised_training_days)) != 1 or len(set(realised_validation_days)) != 1:
        raise ValueError("Every student must use the same realised day split")
    solver_counts = {
        str(key): int(value)
        for key, value in teacher_steps.solver.value_counts(dropna=False).items()
    }
    return {
        "teacher_capacity_day_jobs": int(len(teacher)),
        "teacher_accumulated_solver_seconds": float(
            teacher.solve_total_seconds.sum()
        ),
        "teacher_mean_solver_seconds_per_capacity_day": float(
            teacher.solve_total_seconds.mean()
        ),
        "teacher_p95_solver_seconds_per_capacity_day": float(
            teacher.solve_total_seconds.quantile(0.95)
        ),
        "teacher_parallel_workers": 4,
        "teacher_solver_provenance": {
            "total_step_records": int(len(teacher_steps)),
            "pandapower_ac_opf_steps": solver_counts.get("pandapower_ac_opf", 0),
            "inherited_nested_dispatch_steps": solver_counts.get(
                "inherited_nested_dispatch", 0
            ),
            "direct_slsqp_fallback_steps": solver_counts.get("direct_slsqp_ac_pf", 0),
            "failed_steps": solver_counts.get("failed", 0),
        },
        "student_training_seeds": 5,
        "student_training_wall_seconds_per_seed": walls.tolist(),
        "student_training_wall_seconds_median": float(walls.median()),
        "student_training_wall_seconds_range": [float(walls.min()), float(walls.max())],
        "student_update_throughput_examples_per_second_median": float(
            throughputs.median()
        ),
        "trainable_policy_parameters": int(policies[0]["trainable_policy_parameters"]),
        "training_device_names": sorted(
            {str(row.get("cuda_device_name", row.get("device", "unknown"))) for row in policies}
        ),
        "imitation_examples": int(dataset_manifest["examples"]),
        "realised_unique_fitting_days": int(dataset_manifest["unique_days"]),
        "realised_training_days_per_student": realised_training_days[0],
        "realised_validation_days_per_student": realised_validation_days[0],
    }


def margin_calibration_computation_summary(margin_root: Path) -> dict:
    candidates = sorted(
        path for path in margin_root.glob("margin_*") if path.is_dir()
    )
    if len(candidates) != 4:
        raise ValueError("Full margin-cost reporting requires four candidate directories")
    teacher_frames = []
    policy_rows = []
    for candidate in candidates:
        episodes = candidate / "teacher" / "episodes.csv"
        if not episodes.is_file():
            raise FileNotFoundError(episodes)
        frame = pd.read_csv(episodes)
        if frame.empty or "solve_total_seconds" not in frame:
            raise ValueError(f"Missing teacher timings in {episodes}")
        teacher_frames.append(frame)
        summaries = sorted((candidate / "policies").glob("summary_seed*.json"))
        if len(summaries) != 5:
            raise ValueError(f"Expected five policy summaries in {candidate}")
        policy_rows.extend(read_json(path) for path in summaries)
    teacher = pd.concat(teacher_frames, ignore_index=True)
    walls = [float(row["training_wall_seconds"]) for row in policy_rows]
    return {
        "margin_candidates": len(candidates),
        "teacher_capacity_day_jobs": int(len(teacher)),
        "teacher_accumulated_solver_seconds": float(
            teacher.solve_total_seconds.sum()
        ),
        "student_fits": len(policy_rows),
        "student_accumulated_training_seconds": float(sum(walls)),
        "student_training_seconds_median_per_fit": float(pd.Series(walls).median()),
    }


def portability_training_summary(
    policy_summary_paths: list[Path], dataset_manifest: dict
) -> dict:
    """Audit the realised frozen split for the second-feeder portability fit."""
    if len(policy_summary_paths) != 5 or not all(
        path.is_file() for path in policy_summary_paths
    ):
        raise ValueError("Second-feeder reporting requires five policy summaries")
    rows = [read_json(path) for path in policy_summary_paths]
    requested = {tuple(map(int, row["requested_validation_days"])) for row in rows}
    realised = {tuple(map(int, row["validation_days"])) for row in rows}
    missing = {tuple(map(int, row["missing_validation_days"])) for row in rows}
    training_counts = {
        int(row["training_unique_days_with_examples"]) for row in rows
    }
    validation_counts = {
        int(row["validation_unique_days_with_examples"]) for row in rows
    }
    if any(
        len(values) != 1
        for values in (
            requested,
            realised,
            missing,
            training_counts,
            validation_counts,
        )
    ):
        raise ValueError("Second-feeder actors did not use an identical frozen split")
    requested_days = list(next(iter(requested)))
    realised_days = list(next(iter(realised)))
    missing_days = list(next(iter(missing)))
    if sorted(set(requested_days) - set(realised_days)) != sorted(missing_days):
        raise ValueError("Second-feeder missing-validation accounting is inconsistent")
    return {
        "dataset_unique_days_with_examples": int(dataset_manifest["unique_days"]),
        "initial_complete_teacher_days": int(
            dataset_manifest.get("initial_complete_teacher_days", dataset_manifest["unique_days"])
        ),
        "counterexample_teacher_days": int(
            dataset_manifest.get("counterexample_teacher_days", 0)
        ),
        "requested_validation_days": requested_days,
        "realised_validation_days": realised_days,
        "missing_validation_days": missing_days,
        "realised_training_days_per_seed": next(iter(training_counts)),
        "realised_validation_days_per_seed": next(iter(validation_counts)),
        "missing_days_reassigned_to_training": False,
    }


CORE_STORY_GATES = (
    "selection_feasibility_is_monotone_above_boundary",
    "main_adjacent_boundary_confirmed",
    "per_seed_finite_sample_upper_below_one_percent",
    "vmod_capacity_no_greater_than_confirmed_pilot_droop",
    "vmod_has_lower_paired_loss_than_pilot_droop",
    "margin_calibration_completed_without_data_leakage",
    "capacity_conditioning_not_worse_for_confirmation_events",
    "selected_point_has_adequate_teacher_trajectory_support",
    "ac_nodal_balance_verified",
)


def core_story_supported(gates: dict[str, bool]) -> bool:
    return bool(all(gates[name] for name in CORE_STORY_GATES))


def seedwise_noninferior_with_improvement(
    rows: list[dict], proposed_field: str, baseline_field: str
) -> tuple[bool, bool]:
    noninferior = bool(
        rows
        and all(
            int(row[proposed_field]) <= int(row[baseline_field]) for row in rows
        )
    )
    improved = bool(
        noninferior
        and any(
            int(row[proposed_field]) < int(row[baseline_field]) for row in rows
        )
    )
    return noninferior, improved


def write_manuscript_results(
    summary: dict,
    baselines: dict,
    conditioning: dict,
    power: dict,
    runtime: dict,
    scalability: dict,
    env69: dict,
) -> None:
    selected = summary["selected_capacity"]
    boundary_source = read_json(Path(summary["source_files"]["boundary"]))
    confirmation_days = int(boundary_source["confirmation_days_per_seed"])
    rejected = next(
        row
        for row in boundary_source["points"]
        if row["role"] == "adjacent_rejected"
    )
    claims = summary["permitted_claims"]
    capacity = summary["capacity_comparisons"]
    selected_seed_rows = selected["seed_summaries"]
    event_counts = [int(row["event_days"]) for row in selected_seed_rows]
    uppers = [
        float(row["one_sided_clopper_pearson_upper_95"])
        for row in selected_seed_rows
    ]
    lower_tail_counts = [
        int(
            row.get(
                "days_with_minimum_voltage_within_0p001_pu_of_lower_limit", 0
            )
        )
        for row in selected_seed_rows
    ]
    upper_tail_counts = [
        int(
            row.get(
                "days_with_maximum_voltage_within_0p001_pu_of_upper_limit", 0
            )
        )
        for row in selected_seed_rows
    ]
    zero = comparison(baselines, "zero_margin_opf_imitation", "confirmation")
    zero_rows = zero["paired_loss"]["seed_values"]
    vmod_events = sum(int(row["vmod_event_days"]) for row in zero_rows)
    zero_events = sum(int(row["baseline_event_days"]) for row in zero_rows)
    droop = capacity.get("pilot_droop", capacity["droop"])
    boundary_efficiency = summary["confirmed_boundary_operational_comparison"]
    conditioning_rows = conditioning["seed_contrasts"]
    shared_conditioning_events = [
        int(row["shared_event_days"]) for row in conditioning_rows
    ]
    fixed_conditioning_events = [
        int(row["fixed_event_days"]) for row in conditioning_rows
    ]

    if claims["core_story_supported"]:
        pilot_index = int(droop["selected_path_index"])
        selected_index = int(selected["path_index"])
        boundary_relation = (
            f"the same P{selected_index:02d} boundary as"
            if selected_index == pilot_index
            else f"a lower P{selected_index:02d} boundary than P{pilot_index:02d} for"
        )
        core = (
            f"On the 33-bus case, VMOD confirmed {boundary_relation} tuned "
            "feeder-wide pilot droop, using "
            f"{tex_number(selected['pv_inverter_nameplate_mva'], 4)}~MVA of PV-inverter "
            f"nameplate and {tex_number(selected['svc_nameplate_mvar'], 3)}~MVAr of SVC. "
            f"Each actor recorded 0/{confirmation_days} event-days (one-sided 95\\% "
            f"upper bound {tex_number(100 * max(uppers), 3)}\\%), while VMOD reduced "
            "paired line loss relative to pilot droop by "
            f"{tex_number(abs(boundary_efficiency['loss_change_percent_of_droop']), 2)}\\% "
            f"with a five-seed VMOD-minus-pilot interval of "
            f"[{tex_number(boundary_efficiency['loss_ci95_low_mwh'], 4)}, "
            f"{tex_number(boundary_efficiency['loss_ci95_high_mwh'], 4)}]~MWh/day, "
            "without online safety projection."
        )
    else:
        core = (
            "The pre-specified joint capacity-and-efficiency claim was not "
            "supported; the results are therefore interpreted as a "
            "controller-specific raw-control feasibility map."
        )

    boundary_text = (
        "The lowest confirmed tested VMOD point used "
        f"{tex_number(selected['pv_inverter_nameplate_mva'], 4)}~MVA of total "
        f"PV-inverter nameplate and {tex_number(selected['svc_nameplate_mvar'], 4)}~MVAr "
        f"of SVC capacity (path index {int(selected['path_index'])}); the adjacent "
        f"lower point at index {int(rejected['path_index'])} failed confirmation."
    )
    selected_margin = float(summary["selected_margin_pu"])
    if selected_margin == 0.0:
        margin_text = (
            "The pre-specified four-candidate calibration selected zero teacher "
            "margin; consequently, no positive-margin safety benefit is claimed."
        )
    else:
        zero_loss = zero["paired_loss"]
        margin_text = (
            f"The selected {tex_number(selected_margin, 3)}~pu teacher margin and "
            f"its zero-margin ablation recorded {vmod_events} and {zero_events} "
            "matched confirmation event-days, respectively. The selected-margin "
            "actors used "
            f"{tex_number(zero_loss['mean_mwh'], 4)}~MWh/day more line-loss energy "
            "than the zero-margin actors, with a five-seed 95\\% interval of "
            f"[{tex_number(zero_loss['seed_level_t_ci95_low_mwh'], 4)}, "
            f"{tex_number(zero_loss['seed_level_t_ci95_high_mwh'], 4)}]~MWh/day."
        )
    if droop["confirmed"]:
        pilot_point = droop.get("selected_point")
        pilot_statistics = ""
        if pilot_point:
            pilot_statistics = (
                f", with {int(pilot_point['event_days'])}/"
                f"{int(pilot_point['evaluated_days'])} event-days, "
                f"{int(pilot_point['pf_fail_days'])} power-flow failures and a "
                "one-sided 95\\% upper bound of "
                f"{tex_number(100 * pilot_point['one_sided_clopper_pearson_upper_95'], 3)}\\%"
            )
        baseline_text = (
            f"Tuned feeder-wide pilot droop was independently confirmed at path index "
            f"{int(droop['selected_path_index'])}{pilot_statistics}. {margin_text}"
        )
    else:
        baseline_text = (
            "The tuned pilot-droop path did not yield an independently confirmed adjacent "
            f"rejected/passed boundary, so no pilot-droop capacity boundary was claimed. {margin_text}"
        )
    if "pilot_droop" in capacity:
        local_droop = capacity.get("droop", {})
        no_control = capacity.get("no_control", {})
        if not local_droop.get("confirmed", False) and not no_control.get(
            "confirmed", False
        ):
            baseline_text += (
                " Device-local droop and no control did not identify qualifying "
                "confirmed boundaries on the tested path."
            )
    conditioning_text = (
        f"On the {confirmation_days}-day confirmation split, the shared capacity-conditioned "
        f"actors and capacity-specific actors recorded per-seed event counts "
        f"{shared_conditioning_events} and {fixed_conditioning_events}, respectively. "
        "The five-seed mean shared-minus-fixed line-loss contrast was "
        f"{tex_number(conditioning['shared_minus_fixed_loss_mean_mwh'], 4)}~MWh/day "
        "with a 95\\% interval of "
        f"[{tex_number(conditioning['seed_level_t_ci95_low_mwh'], 4)}, "
        f"{tex_number(conditioning['seed_level_t_ci95_high_mwh'], 4)}]~MWh/day. "
    )
    if claims["capacity_conditioning_is_noninferior_compression"]:
        conditioning_text += (
            "Thus, at the selected point, sharing one model across the capacity path "
            "did not increase any seed's confirmation event count relative to the "
            "matched-budget capacity-specific fit; this supports non-inferior model "
            "sharing, not superior safety."
        )
    else:
        conditioning_text += (
            "The pre-specified non-inferiority gate was not met, so no benefit of "
            "capacity conditioning is claimed."
        )
    baseline_text = f"{baseline_text} {conditioning_text}"
    statistics = (
        f"The five VMOD actors recorded event counts {event_counts} over "
        f"{confirmation_days} days "
        "per seed; their exact one-sided 95\\% upper bounds ranged from "
        f"{tex_number(100 * min(uppers), 3)}\\% to {tex_number(100 * max(uppers), 3)}\\%. "
        "Across seeds, the numbers of daily minima within 0.001~pu above the "
        f"lower limit were {lower_tail_counts}, and the corresponding daily maxima "
        f"within 0.001~pu below the upper limit were {upper_tail_counts}."
    )
    if claims.get(
        "no_greater_capacity_and_lower_loss_vs_pilot_droop",
        claims.get("smaller_confirmed_capacity_also_reduces_line_loss_vs_droop", False),
    ):
        efficiency_text = (
            f"Across the same {confirmation_days} confirmation days, the VMOD point reduced mean "
            "daily line loss relative to the independently confirmed pilot-droop point by "
            f"{tex_number(abs(boundary_efficiency['loss_change_percent_of_droop']), 2)}\\%; "
            "the five-seed 95\\% interval for VMOD minus pilot droop was "
            f"[{tex_number(boundary_efficiency['loss_ci95_low_mwh'], 4)}, "
            f"{tex_number(boundary_efficiency['loss_ci95_high_mwh'], 4)}]~MWh/day, "
            f"with a seed-level standard deviation of "
            f"{tex_number(boundary_efficiency['loss_seed_sd_mwh'], 4)}~MWh/day."
        )
    elif droop["confirmed"]:
        efficiency_text = (
            "The pre-specified paired-loss gate did not establish lower line loss "
            "at the VMOD boundary than at the confirmed pilot-droop boundary; no "
            "capacity-and-loss efficiency claim is made."
        )
    else:
        efficiency_text = (
            "Because the tuned pilot-droop path did not yield an independently confirmed "
            "capacity boundary, no matched boundary-level capacity-and-loss efficiency "
            "claim is made."
        )
    total = runtime["total_online_path_ms"]
    runtime_text = (
        "The raw online path required "
        f"{tex_number(total['p95'], 3)}~ms at the 95th percentile and "
        f"{tex_number(total['maximum'], 3)}~ms at maximum, including nonlinear AC "
        "plant execution; actor inference and plant time are reported separately."
    )
    scale_118 = next(
        row for row in scalability["environments"] if int(row["environment"]) == 118
    )
    runtime_text += (
        " In a separate architecture-and-solver diagnostic, the 118-bus total path "
        f"required {tex_number(scale_118['total_online_path_ms']['p95'], 3)}~ms at "
        "the 95th percentile. This zero-command timing test does not validate "
        "118-bus control performance or a capacity boundary."
    )
    runtime_hardware = runtime.get("hardware", {})
    runtime_software = runtime.get("software", {})
    platform_name = str(runtime_hardware.get("platform", "")).strip()
    processor_name = str(runtime_hardware.get("processor", "")).strip()
    if platform_name or processor_name:
        machine = processor_name or platform_name
        runtime_text += f" Measurements used {tex_escape(machine)}"
        versions = [
            f"Python {runtime_software.get('python')}"
            if runtime_software.get("python")
            else None,
            f"PyTorch {runtime_software.get('torch')}"
            if runtime_software.get("torch")
            else None,
            f"pandapower {runtime_software.get('pandapower')}"
            if runtime_software.get("pandapower")
            else None,
        ]
        versions = [value for value in versions if value]
        if versions:
            runtime_text += " with " + ", ".join(versions)
        runtime_text += "."
    offline = summary["offline_computation"]
    calibration_cost = offline["full_margin_calibration"]
    offline_text = (
        "For the frozen selected candidate, offline teacher generation accumulated "
        f"{tex_number(offline['teacher_accumulated_solver_seconds'] / 3600.0, 2)} "
        "solver-hours across "
        f"{int(offline['teacher_capacity_day_jobs'])} capacity--day jobs using "
        f"{int(offline['teacher_parallel_workers'])} workers. Of the "
        f"{int(offline['teacher_solver_provenance']['total_step_records']):,} teacher step records, "
        f"{int(offline['teacher_solver_provenance']['pandapower_ac_opf_steps']):,} retained a feasible "
        "pandapower AC-OPF result and "
        f"{int(offline['teacher_solver_provenance']['inherited_nested_dispatch_steps']):,} retained the "
        "feasible dispatch inherited from the adjacent lower-capacity point; solver provenance is "
        "therefore reported rather than treating every record as a certified global optimum. The five student fits "
        f"used {int(offline['realised_unique_fitting_days'])} of 40 allocated fitting days with complete examples "
        f"({int(offline['realised_training_days_per_student'])} parameter-training and "
        f"{int(offline['realised_validation_days_per_student'])} early-stopping days per student) and required a median "
        f"{tex_number(offline['student_training_wall_seconds_median'], 1)}~s per seed "
        "on "
        f"{tex_escape(', '.join(offline['training_device_names']))}; the actor contains "
        f"{int(offline['trainable_policy_parameters']):,} trainable policy parameters. "
        "The complete four-candidate margin-calibration procedure accumulated "
        f"{tex_number(calibration_cost['teacher_accumulated_solver_seconds'] / 3600.0, 2)} "
        f"teacher solver-hours and {tex_number(calibration_cost['student_accumulated_training_seconds'] / 3600.0, 2)} "
        f"student-training hours across {int(calibration_cost['student_fits'])} fits."
    )
    env69_index = confirmed_boundary_index(env69)
    portability = (
        "After feeder-specific adaptation, the protocol confirmed a 69-bus "
        "rejected/passed boundary at path "
        f"index {int(env69_index)} on an untouched annual window."
        if env69_index is not None
        else "No rejected/passed capacity boundary was confirmed on the 69-bus case."
    )
    env69_adaptation = env69.get("_adaptation_evidence", {})
    if env69_adaptation:
        initial = env69_adaptation["initial_protocol_result"]
        first_external = env69_adaptation["first_external_attempt"]
        counterexample = env69_adaptation["counterexample_guided_adaptation"]
        final = env69_adaptation["untouched_2019_confirmation"]
        portability += (
            " Direct reuse of the original student representation first failed: "
            f"the highest development point recorded {int(initial['event_seed_days'])}/"
            f"{int(initial['evaluated_seed_days'])} event seed--days, and the first "
            f"external candidate later recorded {int(first_external['event_seed_days'])}/"
            f"{int(first_external['evaluated_seed_days'])}. The adaptation retained "
            "feeder-specific retraining, standardised the active-power observation "
            "features and added "
            f"{int(counterexample['teacher_examples']):,} AC-OPF examples from "
            f"{int(counterexample['unique_failed_days'])} exposed tail days. On the "
            "subsequent untouched 2019 window, the selected point recorded "
            f"{int(final['selected_event_seed_days'])}/"
            f"{int(final['selected_evaluated_seed_days'])} event seed--days while its "
            f"adjacent lower point recorded {int(final['adjacent_rejected_event_seed_days'])}."
        )
    env69_calibration = env69.get("_calibration_evidence", [])
    env69_dataset = env69.get("_dataset_evidence", {})
    if env69_index is None and env69_calibration:
        highest = max(env69_calibration, key=lambda row: int(row["path_index"]))
        portability += (
            f" All {len(env69_calibration)} development-stage path points retained "
            "voltage-event days; even the highest point, "
            f"{tex_escape(str(highest['capacity_label']))}, recorded "
            f"{int(highest['total_event_days'])}/{int(highest['evaluated_seed_days'])} "
            "seed--days with events. The external selection and confirmation splits "
            "were therefore not opened, and no second-feeder portability claim is made."
        )
    if env69_dataset:
        portability += (
            " The corrected 69-bus action representation had a maximum reactive-power "
            "label error of "
            f"{tex_number(float(env69_dataset['maximum_q_representation_error_mvar']) / 1e-11, 2)}"
            "$\\times10^{-11}$~MVAr."
        )
    portability += (
        f" Separately, the principal 33-bus direct AC nodal-balance audit covered {int(power['audited_steps']):,} "
        f"executed steps with {int(power['power_flow_failures'])} power-flow failures; "
        "the maximum active and reactive residuals were "
        f"{tex_number(power['maximum_active_balance_residual_mw'] / 1e-7, 2)}"
        "$\\times10^{-7}$~MW and "
        f"{tex_number(power['maximum_reactive_balance_residual_mvar'] / 1e-7, 2)}"
        "$\\times10^{-7}$~MVAr, "
        "respectively."
    )
    env69_training = env69.get("_training_evidence")
    if env69_training:
        portability += (
            " On the 69-bus teacher data, "
            f"{int(env69_training['initial_complete_teacher_days'])} initial days and "
            f"{int(env69_training['counterexample_teacher_days'])} adaptation days "
            "supplied complete trajectories. "
            f"The frozen eight-day validation list retained "
            f"{int(env69_training['realised_validation_days_per_seed'])} available "
            "days; "
            f"{len(env69_training['missing_validation_days'])} days without any "
            "complete teacher trajectory were reported and were not reassigned to training."
        )
    content = "\n".join(
        [
            "% Auto-generated by summarize_vmod_final_study.py.",
            f"\\newcommand{{\\VMODCoreResult}}{{{core}}}",
            f"\\newcommand{{\\VMODBoundaryResult}}{{{boundary_text}}}",
            f"\\newcommand{{\\VMODBaselineResult}}{{{baseline_text}}}",
            f"\\newcommand{{\\VMODStatisticsResult}}{{{statistics}}}",
            f"\\newcommand{{\\VMODEfficiencyResult}}{{{efficiency_text}}}",
            f"\\newcommand{{\\VMODRuntimeResult}}{{{runtime_text}}}",
            f"\\newcommand{{\\VMODOfflineCostResult}}{{{offline_text}}}",
            f"\\newcommand{{\\VMODPortabilityResult}}{{{portability}}}",
            "% Additional audit values retained in the machine-readable summary:",
            f"% conditioning shared/fixed events: {sum(row['shared_event_days'] for row in conditioning['seed_contrasts'])}/{sum(row['fixed_event_days'] for row in conditioning['seed_contrasts'])}",
            f"% maximum AC balance residuals: {power['maximum_active_balance_residual_mw']} MW, {power['maximum_reactive_balance_residual_mvar']} MVAr",
            "",
        ]
    )
    MANUSCRIPT_RESULTS.write_text(content, encoding="utf-8")


def main() -> None:
    margin_summary = read_json(PATH_MARGIN_SUMMARY)
    margin = read_selected_margin()
    main_run = main_run_dir(margin)
    env69_run = env69_run_dir(margin)
    paths = {
        "boundary": main_run / "confirmation_boundary_summary.json",
        "selection": main_run / "selection_path_summary.json",
        "conventional": main_run
        / "conventional_capacity_boundaries"
        / "summary.json",
        "baselines": main_run / "final_baselines" / "summary.json",
        "conditioning": main_run / "conditioning_ablation" / "summary.json",
        "shift": main_run / "operational_shift" / "study_summary.json",
        "power": main_run / "power_balance_audit" / "summary.json",
        "runtime": main_run / "runtime_benchmark" / "summary.json",
        "scalability": main_run / "runtime_scalability" / "summary.json",
        "env69": env69_run / "confirmation_boundary_summary.json",
        "env69_dataset": env69_run / "dataset" / "manifest.json",
        "env69_adaptation": env69_run / "adaptation_audit.json",
        "teacher": main_run / "teacher" / "summary.json",
        "dataset": main_run / "dataset" / "manifest.json",
    }
    wait_for(list(paths.values()))
    evidence = {name: read_json(path) for name, path in paths.items()}

    boundary = evidence["boundary"]
    validate_confirmation_statistics(boundary)
    selection = evidence["selection"]
    selected = next(row for row in boundary["points"] if row["role"] == "selected")
    vmod_index = int(selected["path_index"])
    finite_sample_gate = bool(
        boundary["boundary_confirmed"]
        and all(
            float(row["one_sided_clopper_pearson_upper_95"]) < 0.01
            for row in selected["seed_summaries"]
        )
    )

    conventional = evidence["conventional"]
    conventional_rows = {
        row["method"]: row for row in conventional["confirmation"]
    }
    capacity_comparisons = {}
    for method in ("pilot_droop", "droop", "no_control"):
        row = conventional_rows.get(method, {})
        index = row.get("selected_path_index")
        passed = bool(row.get("passed", False))
        capacity_comparisons[method] = {
            "confirmed": passed,
            "selected_path_index": index,
            "selected_point": (
                row.get("points", {}).get("selected") if passed else None
            ),
            "vmod_path_index": vmod_index,
            "vmod_strictly_lower": bool(
                passed and index is not None and vmod_index < int(index)
            ),
            "vmod_no_greater": bool(
                passed and index is not None and vmod_index <= int(index)
            ),
        }

    droop_confirmation = conventional_rows.get("pilot_droop", {})
    if droop_confirmation.get("passed", False):
        droop_point = droop_confirmation["points"]["selected"]
        vmod_daily = pd.read_csv(
            main_run
            / "confirmation"
            / str(selected["capacity_label"])
            / "daily.csv"
        )
        droop_daily = pd.read_csv(
            main_run
            / "conventional_capacity_boundaries"
            / str(droop_point["capacity_label"])
            / "confirmation_refined"
            / "pilot_droop"
            / "daily.csv"
        )
        boundary_operational_comparison = paired_boundary_outcomes(
            vmod_daily, droop_daily, method="pilot_droop"
        )
    else:
        boundary_operational_comparison = {
            "method": "pilot_droop",
            "n_training_seeds": 0,
            "paired_days_per_seed": 0,
            "vmod_minus_droop_loss_mean_mwh": None,
            "loss_ci95_low_mwh": None,
            "loss_ci95_high_mwh": None,
            "loss_change_percent_of_droop": None,
            "loss_seed_sd_mwh": None,
            "seed_values": [],
        }
    paired_loss_gate = bool(
        droop_confirmation.get("passed", False)
        and boundary_operational_comparison["loss_ci95_high_mwh"] is not None
        and boundary_operational_comparison["loss_ci95_high_mwh"] < 0.0
    )

    baselines = evidence["baselines"]
    zero_margin = comparison(
        baselines, "zero_margin_opf_imitation", "confirmation"
    )
    zero_seed_rows = zero_margin["paired_loss"]["seed_values"]
    _, margin_safety_gate = seedwise_noninferior_with_improvement(
        zero_seed_rows, "vmod_event_days", "baseline_event_days"
    )
    selected_margin_row = next(
        row
        for row in margin_summary["candidates"]
        if float(row["margin_pu"]) == float(margin)
    )
    passing_margins = [
        float(row["margin_pu"])
        for row in margin_summary["candidates"]
        if row.get("passed_path_gate", False)
    ]
    margin_calibration_gate = bool(
        margin_summary.get("development_selection_data_used") is False
        and margin_summary.get("external_validation_data_used") is False
        and selected_margin_row.get("passed_path_gate", False)
        and passing_margins
        and float(margin) == min(passing_margins)
    )

    conditioning = evidence["conditioning"]
    validate_conditioning_ablation(
        conditioning, int(boundary["confirmation_days_per_seed"])
    )
    conditioning_rows = conditioning["seed_contrasts"]
    (
        conditioning_compression_gate,
        conditioning_safety_gate,
    ) = seedwise_noninferior_with_improvement(
        conditioning_rows, "shared_event_days", "fixed_event_days"
    )

    power = evidence["power"]
    ac_balance_gate = bool(
        power["power_flow_failures"] == 0
        and power["maximum_active_balance_residual_mw"] < 1e-6
        and power["maximum_reactive_balance_residual_mvar"] < 1e-6
    )
    runtime = evidence["runtime"]
    runtime_gate = bool(runtime["maximum_budget_fraction"] < 0.01)
    scalability = evidence["scalability"]
    scalability_runtime_gate = bool(
        scalability["environments"]
        and all(
            float(row["maximum_budget_fraction"]) < 0.01
            for row in scalability["environments"]
        )
    )
    env69 = evidence["env69"]
    env69["_adaptation_evidence"] = evidence["env69_adaptation"]
    env69_calibration = pd.read_csv(
        env69_run / "calibration_path_summary.csv"
    ).to_dict(orient="records")
    env69_training = portability_training_summary(
        [
            env69_run / "policies" / f"summary_seed{seed}.json"
            for seed in (42, 43, 44, 45, 46)
        ],
        evidence["env69_dataset"],
    )
    portability_gate = bool(confirmed_boundary_index(env69) is not None)
    dataset = evidence["dataset"]
    teacher_episodes_path = paths["teacher"].parent / "episodes.csv"
    teacher_steps_path = paths["teacher"].parent / "steps.csv"
    policy_summary_paths = [
        main_run / "policies" / f"summary_seed{seed}.json"
        for seed in (42, 43, 44, 45, 46)
    ]
    offline_computation = offline_computation_summary(
        teacher_episodes_path, teacher_steps_path, policy_summary_paths, dataset
    )
    offline_computation["full_margin_calibration"] = (
        margin_calibration_computation_summary(MARGIN_ROOT)
    )
    selected_coverage = next(
        row
        for row in dataset["candidate_coverage"]
        if int(row["candidate_id"]) == vmod_index
    )
    (
        selected_training_support_gate,
        selected_coverage_fraction,
    ) = teacher_coverage_gate(
        selected_coverage
    )

    gates = {
        "selection_feasibility_is_monotone_above_boundary": bool(
            selection.get("all_higher_points_pass", False)
        ),
        "main_adjacent_boundary_confirmed": bool(boundary["boundary_confirmed"]),
        "per_seed_finite_sample_upper_below_one_percent": finite_sample_gate,
        "vmod_capacity_strictly_lower_than_confirmed_droop": capacity_comparisons[
            "pilot_droop"
        ]["vmod_strictly_lower"],
        "vmod_capacity_no_greater_than_confirmed_pilot_droop": (
            capacity_comparisons["pilot_droop"]["vmod_no_greater"]
        ),
        "vmod_capacity_strictly_lower_than_confirmed_no_control": (
            capacity_comparisons["no_control"]["vmod_strictly_lower"]
        ),
        "vmod_lower_capacity_also_has_lower_paired_loss_than_droop": (
            capacity_comparisons["pilot_droop"]["vmod_strictly_lower"]
            and paired_loss_gate
        ),
        "vmod_has_lower_paired_loss_than_pilot_droop": paired_loss_gate,
        "margin_calibration_completed_without_data_leakage": (
            margin_calibration_gate
        ),
        "positive_voltage_margin_reduces_confirmation_events": bool(
            float(margin) > 0.0 and margin_safety_gate
        ),
        "capacity_conditioning_reduces_confirmation_events": conditioning_safety_gate,
        "capacity_conditioning_not_worse_for_confirmation_events": (
            conditioning_compression_gate
        ),
        "selected_point_has_adequate_teacher_trajectory_support": (
            selected_training_support_gate
        ),
        "ac_nodal_balance_verified": ac_balance_gate,
        "online_path_below_one_percent_of_control_interval": runtime_gate,
        "timing_diagnostic_below_one_percent_on_33_69_118_bus_cases": (
            scalability_runtime_gate
        ),
        "sixty_nine_bus_boundary_confirmed": portability_gate,
    }
    core_story_gate = core_story_supported(gates)
    claims = {
        "raw_controller_empirically_feasible": finite_sample_gate,
        "controller_reduces_required_capacity_vs_tuned_droop": gates[
            "vmod_capacity_strictly_lower_than_confirmed_droop"
        ],
        "smaller_confirmed_capacity_also_reduces_line_loss_vs_droop": bool(
            gates["vmod_capacity_strictly_lower_than_confirmed_droop"]
            and paired_loss_gate
        ),
        "no_greater_capacity_and_lower_loss_vs_pilot_droop": bool(
            gates["vmod_capacity_no_greater_than_confirmed_pilot_droop"]
            and paired_loss_gate
        ),
        "positive_voltage_margin_is_safety_relevant": bool(
            float(margin) > 0.0 and margin_safety_gate
        ),
        "capacity_conditioning_improves_safety": conditioning_safety_gate,
        "capacity_conditioning_is_noninferior_compression": (
            conditioning_compression_gate
        ),
        "portable_to_second_benchmark": False,
        "replicated_on_second_benchmark_after_feeder_adaptation": portability_gate,
        "formal_or_field_safety_guarantee": False,
        "global_or_economic_capacity_optimum": False,
        "core_story_supported": core_story_gate,
    }
    summary = {
        "selected_margin_pu": margin,
        "selected_capacity": selected,
        "capacity_comparisons": capacity_comparisons,
        "confirmed_boundary_operational_comparison": (
            boundary_operational_comparison
        ),
        "selected_training_coverage": {
            **selected_coverage,
            "coverage_fraction": selected_coverage_fraction,
            "minimum_required_fraction": MIN_TEACHER_COVERAGE_FRACTION,
        },
        "gates": gates,
        "permitted_claims": claims,
        "operational_shift_levels": evidence["shift"]["levels"],
        "teacher_envelope": evidence["teacher"],
        "offline_computation": offline_computation,
        "runtime_scalability_diagnostic": scalability,
        "second_feeder_training": env69_training,
        "source_files": {name: str(path) for name, path in paths.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "claim_gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# VMOD Final Claim-Gate Report",
        "",
        f"Core story supported: **{core_story_gate}**",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in gates.items()
    )
    lines.extend(
        [
            "",
            "## Claim discipline",
            "",
            "- A failed gate removes the corresponding superiority claim; it is not repaired by wording.",
            "- The selected point is the lowest confirmed tested path point, not a global or economic optimum.",
            "- Shift and second-feeder results are empirical scope checks, not formal or field guarantees.",
            "",
        ]
    )
    (OUT / "claim_gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_manuscript_results(
        summary,
        baselines,
        conditioning,
        power,
        runtime,
        scalability,
        {
            **env69,
            "_training_evidence": env69_training,
            "_calibration_evidence": env69_calibration,
            "_dataset_evidence": evidence["env69_dataset"],
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
