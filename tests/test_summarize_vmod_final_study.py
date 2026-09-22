import json
from pathlib import Path

import pandas as pd
import pytest

import summarize_vmod_final_study as final_summary


def _fixtures(tmp_path, core_story_supported=True):
    boundary = tmp_path / "boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "confirmation_days_per_seed": 349,
                "points": [
                    {
                        "role": "adjacent_rejected",
                        "path_index": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selected = {
        "path_index": 3,
        "pv_inverter_nameplate_mva": 4.5279,
        "svc_nameplate_mvar": 0.75,
        "seed_summaries": [
            {
                "seed": seed,
                "evaluated_days": 349,
                "event_days": 0,
                "one_sided_clopper_pearson_upper_95": 0.008547024811548625,
            }
            for seed in range(42, 47)
        ],
    }
    summary = {
        "selected_margin_pu": 0.0,
        "selected_capacity": selected,
        "permitted_claims": {
            "core_story_supported": core_story_supported,
            "smaller_confirmed_capacity_also_reduces_line_loss_vs_droop": (
                core_story_supported
            ),
            "capacity_conditioning_is_noninferior_compression": True,
        },
        "capacity_comparisons": {
            "droop": {
                "confirmed": True,
                "selected_path_index": 5,
            }
        },
        "gates": {
            "main_adjacent_boundary_confirmed": True,
            "vmod_capacity_strictly_lower_than_confirmed_droop": (
                core_story_supported
            ),
        },
        "source_files": {"boundary": str(boundary)},
        "confirmed_boundary_operational_comparison": {
            "loss_change_percent_of_droop": -4.25,
            "loss_seed_sd_mwh": 0.02,
            "loss_ci95_low_mwh": -0.08,
            "loss_ci95_high_mwh": -0.02,
            "vmod_minus_droop_throughput_mean_mvarh": -1.50,
            "throughput_ci95_low_mvarh": -1.75,
            "throughput_ci95_high_mvarh": -1.25,
            "total_seed_day_pairs": 1745,
        },
        "offline_computation": {
            "teacher_accumulated_solver_seconds": 7200.0,
            "teacher_capacity_day_jobs": 360,
            "teacher_parallel_workers": 4,
            "teacher_solver_provenance": {
                "total_step_records": 1000,
                "pandapower_ac_opf_steps": 800,
                "inherited_nested_dispatch_steps": 200,
                "direct_slsqp_fallback_steps": 0,
                "failed_steps": 0,
            },
            "student_training_wall_seconds_median": 20.0,
            "realised_unique_fitting_days": 39,
            "realised_training_days_per_student": 31,
            "realised_validation_days_per_student": 8,
            "training_device_names": ["Test GPU"],
            "trainable_policy_parameters": 94216,
            "full_margin_calibration": {
                "teacher_accumulated_solver_seconds": 28800.0,
                "student_accumulated_training_seconds": 7200.0,
                "student_fits": 20,
            },
        },
    }
    seed_values = [
        {
            "vmod_event_days": 0,
            "baseline_event_days": 2,
        }
        for _ in range(5)
    ]
    baselines = {
        "comparisons": [
            {
                "baseline": "zero_margin_opf_imitation",
                "split": "confirmation",
                "paired_loss": {
                    "mean_mwh": 0.0147,
                    "seed_level_t_ci95_low_mwh": 0.0107,
                    "seed_level_t_ci95_high_mwh": 0.0188,
                    "seed_values": seed_values,
                },
            }
        ]
    }
    conditioning = {
        "shared_minus_fixed_loss_mean_mwh": -0.0012,
        "seed_level_t_ci95_low_mwh": -0.0020,
        "seed_level_t_ci95_high_mwh": -0.0004,
        "seed_contrasts": [
            {"shared_event_days": 0, "fixed_event_days": 1}
            for _ in range(5)
        ]
    }
    power = {
        "audited_steps": 8160,
        "power_flow_failures": 0,
        "maximum_active_balance_residual_mw": 1e-9,
        "maximum_reactive_balance_residual_mvar": 2e-9,
    }
    runtime = {
        "actor_inference_ms": {
            "median": 0.187,
        },
        "total_online_path_ms": {
            "p95": 31.25,
            "maximum": 48.5,
        },
        "hardware": {"platform": "Test OS", "processor": "Test CPU"},
        "software": {
            "python": "3.11.3",
            "torch": "2.9.1",
            "pandapower": "3.1.2",
        },
    }
    scalability = {
        "environments": [
            {
                "environment": 118,
                "total_online_path_ms": {
                    "p95": 64.125,
                    "maximum": 81.0,
                },
            }
        ]
    }
    env69 = {
        "boundary_confirmed": True,
        "points": [{"role": "selected", "path_index": 4}],
    }
    return summary, baselines, conditioning, power, runtime, scalability, env69


def test_write_manuscript_results_for_passing_core_story(tmp_path, monkeypatch):
    output = tmp_path / "vmod_results.tex"
    monkeypatch.setattr(final_summary, "MANUSCRIPT_RESULTS", output)
    final_summary.write_manuscript_results(*_fixtures(tmp_path, True))
    text = output.read_text(encoding="utf-8")

    assert "a lower P03 boundary than P05" in text
    assert "0/349 event-days" in text
    assert "five-seed VMOD-minus-pilot interval" in text
    assert "path index 3" in text
    assert "adjacent lower point at index 2 failed confirmation" in text
    assert "0.855\\%" in text
    assert "over 349 days" in text
    assert "31.250~ms" in text
    assert "Median actor inference required 0.187~ms" in text
    assert "118-bus total path required 64.125~ms" in text
    assert "Measurements used Test CPU with Python 3.11.3" in text
    assert "PyTorch 2.9.1, pandapower 3.1.2" in text
    assert "confirmed a 69-bus rejected/passed boundary at path index 4" in text
    assert "direct AC nodal-balance audit covered 8,160 executed steps" in text
    assert "reduced mean daily line loss" in text
    assert "seed-level standard deviation of 0.0200~MWh/day" in text
    assert "1.50~MVArh/day" in text
    assert "all 1,745 paired seed--day comparisons" in text
    assert "2.00 solver-hours" in text
    assert "teacher generation accumulated 8.00 solver-hours" in text
    assert "20 student fits accumulated 2.00 training hours" in text
    assert "shared capacity-conditioned actors" in text
    assert "[0, 0, 0, 0, 0] and [1, 1, 1, 1, 1]" in text
    assert "no detected selected-point loss penalty" in text


def test_write_manuscript_results_exposes_failed_gate(tmp_path, monkeypatch):
    output = tmp_path / "vmod_results.tex"
    monkeypatch.setattr(final_summary, "MANUSCRIPT_RESULTS", output)
    final_summary.write_manuscript_results(*_fixtures(tmp_path, False))
    text = output.read_text(encoding="utf-8")

    assert "joint capacity-and-efficiency claim was not supported" in text
    assert "vmod\\_capacity" not in text
    assert "controller-specific raw-control feasibility map" in text


def test_write_manuscript_results_labels_equal_pilot_boundary_without_savings(
    tmp_path, monkeypatch
):
    output = tmp_path / "vmod_results.tex"
    monkeypatch.setattr(final_summary, "MANUSCRIPT_RESULTS", output)
    fixtures = list(_fixtures(tmp_path, True))
    summary = fixtures[0]
    summary["capacity_comparisons"]["pilot_droop"] = {
        "confirmed": True,
        "selected_path_index": 3,
        "selected_point": {
            "event_days": 0,
            "evaluated_days": 349,
            "pf_fail_days": 0,
            "one_sided_clopper_pearson_upper_95": 0.008547024811548625,
        },
    }
    summary["capacity_comparisons"]["droop"] = {
        "confirmed": False,
        "selected_path_index": None,
    }
    summary["capacity_comparisons"]["no_control"] = {
        "confirmed": False,
        "selected_path_index": None,
    }
    summary["permitted_claims"][
        "no_greater_capacity_and_lower_loss_vs_pilot_droop"
    ] = True

    final_summary.write_manuscript_results(*fixtures)
    text = output.read_text(encoding="utf-8")

    assert "the same P03 boundary as tuned feeder-wide pilot droop" in text
    assert "0/349 event-days, 0 power-flow failures" in text
    assert "one-sided 95\\% upper bound of 0.855\\%" in text
    assert "Device-local droop and no control did not identify" in text
    assert "hardware saving" not in text


def test_write_manuscript_results_reports_margin_safety_loss_tradeoff(
    tmp_path, monkeypatch
):
    output = tmp_path / "vmod_results.tex"
    monkeypatch.setattr(final_summary, "MANUSCRIPT_RESULTS", output)
    fixtures = list(_fixtures(tmp_path, True))
    fixtures[0]["selected_margin_pu"] = 0.003

    final_summary.write_manuscript_results(*fixtures)
    text = output.read_text(encoding="utf-8")

    assert "0 and 10 matched confirmation event-days" in text
    assert "0.0147~MWh/day more line-loss energy" in text
    assert "[0.0107, 0.0188]~MWh/day" in text


def test_write_manuscript_results_does_not_invent_a_droop_boundary(
    tmp_path, monkeypatch
):
    output = tmp_path / "vmod_results.tex"
    monkeypatch.setattr(final_summary, "MANUSCRIPT_RESULTS", output)
    fixtures = list(_fixtures(tmp_path, False))
    summary = fixtures[0]
    summary["capacity_comparisons"]["droop"] = {
        "confirmed": False,
        "selected_path_index": None,
    }
    summary["permitted_claims"][
        "smaller_confirmed_capacity_also_reduces_line_loss_vs_droop"
    ] = False

    final_summary.write_manuscript_results(*fixtures)
    text = output.read_text(encoding="utf-8")

    assert "did not yield an independently confirmed adjacent" in text
    assert "no matched boundary-level capacity-and-loss efficiency claim" in text
    assert "at the confirmed droop boundary" not in text


def test_teacher_coverage_gate_requires_eighty_percent():
    passed, fraction = final_summary.teacher_coverage_gate(
        {"attempted_days": 40, "converged_days": 32}
    )
    failed, lower_fraction = final_summary.teacher_coverage_gate(
        {"attempted_days": 40, "converged_days": 31}
    )

    assert passed
    assert fraction == 0.8
    assert not failed
    assert lower_fraction == 0.775


def test_conditioning_ablation_must_use_complete_confirmation_split():
    rows = [
        {
            "split": "confirmation",
            "seed": seed,
            "paired_days": 349,
            "shared_event_days": 0,
            "fixed_event_days": 1,
        }
        for seed in range(42, 47)
    ]
    final_summary.validate_conditioning_ablation(
        {"primary_inference_split": "confirmation", "seed_contrasts": rows},
        349,
    )

    with pytest.raises(ValueError, match="confirmation split"):
        final_summary.validate_conditioning_ablation(
            {"primary_inference_split": "selection", "seed_contrasts": rows},
            349,
        )
    rows[0]["paired_days"] = 17
    with pytest.raises(ValueError, match="paired-day count"):
        final_summary.validate_conditioning_ablation(
            {"primary_inference_split": "confirmation", "seed_contrasts": rows},
            349,
        )


def test_teacher_coverage_gate_rejects_empty_attempts():
    passed, fraction = final_summary.teacher_coverage_gate(
        {"attempted_days": 0, "converged_days": 0}
    )

    assert not passed
    assert fraction == 0.0


def test_confirmed_boundary_index_requires_confirmation():
    assert final_summary.confirmed_boundary_index(
        {
            "boundary_confirmed": True,
            "points": [{"role": "selected", "path_index": 5}],
        }
    ) == 5
    assert final_summary.confirmed_boundary_index(
        {
            "boundary_confirmed": False,
            "points": [{"role": "selected", "path_index": 5}],
        }
    ) is None


def test_confirmation_statistics_match_349_day_exact_bounds(tmp_path):
    summary = _fixtures(tmp_path, True)[0]
    boundary_path = summary["source_files"]["boundary"]
    boundary = json.loads(Path(boundary_path).read_text(encoding="utf-8"))
    boundary["points"].append(
        {"role": "selected", **summary["selected_capacity"]}
    )

    final_summary.validate_confirmation_statistics(boundary)


def test_confirmation_statistics_reject_stale_299_day_bound():
    boundary = {
        "confirmation_days_per_seed": 349,
        "points": [
            {
                "role": "selected",
                "seed_summaries": [
                    {
                        "seed": seed,
                        "evaluated_days": 349,
                        "event_days": 0,
                        "one_sided_clopper_pearson_upper_95": 0.009969,
                    }
                    for seed in range(42, 47)
                ],
            }
        ],
    }

    try:
        final_summary.validate_confirmation_statistics(boundary)
    except ValueError as exc:
        assert "Clopper-Pearson" in str(exc)
    else:
        raise AssertionError("The stale 299-day confidence bound was accepted")


def test_paired_boundary_outcomes_uses_seed_level_contrasts():
    vmod = pd.DataFrame(
        [
            {
                "seed": seed,
                "day": day,
                "line_loss_mwh": 0.8 + 0.01 * (seed - 42),
                "absolute_reactive_throughput_mvarh": 5.0,
            }
            for seed in range(42, 47)
            for day in (1, 2, 3)
        ]
    )
    droop = pd.DataFrame(
        [
            {
                "method": "droop",
                "day": day,
                "line_loss_mwh": 1.0,
                "absolute_reactive_throughput_mvarh": 6.0,
            }
            for day in (1, 2, 3)
        ]
    )

    result = final_summary.paired_boundary_outcomes(vmod, droop)

    assert result["n_training_seeds"] == 5
    assert result["paired_days_per_seed"] == 3
    assert result["vmod_minus_droop_loss_mean_mwh"] < 0
    assert result["loss_ci95_high_mwh"] < 0
    assert result["loss_seed_sd_mwh"] > 0
    assert result["total_seed_day_pairs"] == 15
    assert result["loss_better_seed_day_pairs"] == 15
    assert result["throughput_better_seed_day_pairs"] == 15


def test_core_story_requires_every_method_and_evidence_gate():
    gates = {name: True for name in final_summary.CORE_STORY_GATES}
    assert final_summary.core_story_supported(gates)

    for name in final_summary.CORE_STORY_GATES:
        failed = dict(gates)
        failed[name] = False
        assert not final_summary.core_story_supported(failed)


def test_seedwise_gate_does_not_hide_a_seed_reversal():
    rows = [
        {"proposed": 0, "baseline": 2},
        {"proposed": 1, "baseline": 0},
    ]
    noninferior, improved = final_summary.seedwise_noninferior_with_improvement(
        rows, "proposed", "baseline"
    )
    assert not noninferior
    assert not improved

    clean_rows = [
        {"proposed": 0, "baseline": 2},
        {"proposed": 0, "baseline": 0},
    ]
    noninferior, improved = final_summary.seedwise_noninferior_with_improvement(
        clean_rows, "proposed", "baseline"
    )
    assert noninferior
    assert improved


def test_offline_cost_summary_reports_solver_and_seed_costs(tmp_path):
    episodes = tmp_path / "episodes.csv"
    steps = tmp_path / "steps.csv"
    pd.DataFrame(
        {
            "solve_total_seconds": [2.0, 4.0, 6.0],
        }
    ).to_csv(episodes, index=False)
    pd.DataFrame(
        {"solver": ["pandapower_ac_opf", "inherited_nested_dispatch", "failed"]}
    ).to_csv(steps, index=False)
    summaries = []
    for seed, wall in zip(range(42, 47), [10.0, 12.0, 11.0, 14.0, 13.0]):
        path = tmp_path / f"summary_seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "training_wall_seconds": wall,
                    "training_example_updates_per_second": 1000.0 + seed,
                    "training_unique_days_with_examples": 31,
                    "validation_unique_days_with_examples": 8,
                    "trainable_policy_parameters": 94216,
                    "cuda_device_name": "Test GPU",
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)

    result = final_summary.offline_computation_summary(
        episodes,
        steps,
        summaries,
        {"examples": 1234, "unique_days": 39},
    )

    assert result["teacher_capacity_day_jobs"] == 3
    assert result["teacher_accumulated_solver_seconds"] == 12.0
    assert result["teacher_solver_provenance"]["pandapower_ac_opf_steps"] == 1
    assert result["realised_unique_fitting_days"] == 39
    assert result["realised_training_days_per_student"] == 31
    assert result["realised_validation_days_per_student"] == 8
    assert result["teacher_solver_provenance"]["inherited_nested_dispatch_steps"] == 1
    assert result["student_training_wall_seconds_median"] == 12.0
    assert result["imitation_examples"] == 1234


def test_portability_training_summary_reports_missing_frozen_days(tmp_path):
    summaries = []
    for seed in range(42, 47):
        path = tmp_path / f"summary_seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "requested_validation_days": [10, 20, 30],
                    "validation_days": [10, 30],
                    "missing_validation_days": [20],
                    "training_unique_days_with_examples": 29,
                    "validation_unique_days_with_examples": 2,
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)

    result = final_summary.portability_training_summary(
        summaries, {"unique_days": 31}
    )

    assert result["requested_validation_days"] == [10, 20, 30]
    assert result["realised_validation_days"] == [10, 30]
    assert result["missing_validation_days"] == [20]
    assert result["realised_training_days_per_seed"] == 29
    assert result["missing_days_reassigned_to_training"] is False


def test_margin_calibration_cost_reports_all_four_candidates(tmp_path):
    for candidate_index in range(4):
        candidate = tmp_path / f"margin_{candidate_index:03d}"
        teacher = candidate / "teacher"
        policies = candidate / "policies"
        teacher.mkdir(parents=True)
        policies.mkdir(parents=True)
        pd.DataFrame({"solve_total_seconds": [2.0, 3.0]}).to_csv(
            teacher / "episodes.csv", index=False
        )
        for seed in range(42, 47):
            (policies / f"summary_seed{seed}.json").write_text(
                json.dumps({"training_wall_seconds": 10.0}),
                encoding="utf-8",
            )

    result = final_summary.margin_calibration_computation_summary(tmp_path)

    assert result["margin_candidates"] == 4
    assert result["teacher_capacity_day_jobs"] == 8
    assert result["teacher_accumulated_solver_seconds"] == 20.0
    assert result["student_fits"] == 20
    assert result["student_accumulated_training_seconds"] == 200.0
