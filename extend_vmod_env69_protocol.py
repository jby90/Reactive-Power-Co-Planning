"""Create the development-triggered IEEE 69-bus capacity-path extension.

The original nine-point range is extended only when every original point fails
the development calibration.  The extension is fixed before any external
selection output exists: SVC and PV scales are alternately increased by 0.25
from (0.75, 0.75) to the actor-domain endpoint (1.50, 1.50).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_PROTOCOL = ROOT / "runs" / "VMOD_PROTOCOL_MAIN_ENV69_20260920"
SOURCE_RUN = ROOT / "runs" / "VMOD_ENV69_EXTERNAL2018_MARGIN_0p003_20260921"
OUT_PROTOCOL = ROOT / "runs" / "VMOD_PROTOCOL_MAIN_ENV69_EXTENDED_20260921"
EXTERNAL_PROFILE = ROOT / "data" / "vmod" / "profiles69_external2018"
OUT_EXTERNAL_PROTOCOL = (
    ROOT / "runs" / "VMOD_PROTOCOL_EXTERNAL2018_ENV69_EXTENDED_20260921"
)

EXTENSION = (
    (0.75, 1.00),
    (1.00, 1.00),
    (1.00, 1.25),
    (1.25, 1.25),
    (1.25, 1.50),
    (1.50, 1.50),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def pv_scale_token(value: float) -> str:
    return f"{int(round(100 * value)):03d}"


def svc_scale_token(value: float) -> str:
    return f"{int(round(1000 * value)):04d}"


def verify_development_trigger(source_run: Path) -> dict:
    calibration = source_run / "calibration"
    summaries = sorted(calibration.glob("P*/summary.json"))
    if len(summaries) != 9:
        raise RuntimeError("The original nine-point development calibration is incomplete")
    event_totals = []
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("all_seeds_zero_events", False):
            raise RuntimeError(
                "At least one original point passed development calibration; "
                "the pre-specified extension trigger is not met"
            )
        event_totals.append(int(payload["total_event_days"]))
    if (source_run / "selection_path_summary.json").exists():
        raise RuntimeError("External selection summary already exists")
    persisted_selection = list((source_run / "selection").glob("**/*"))
    persisted_selection = [path for path in persisted_selection if path.is_file()]
    if persisted_selection:
        raise RuntimeError("External selection outputs already exist")
    return {
        "original_points": 9,
        "development_total_event_days_by_point": event_totals,
        "all_original_points_failed_development_calibration": True,
        "external_selection_summary_existed": False,
        "persisted_external_selection_output_count": 0,
    }


def build_extended_path(source_protocol: Path) -> pd.DataFrame:
    original = pd.read_csv(source_protocol / "capacity_path.csv").sort_values(
        "path_index"
    )
    if len(original) != 9:
        raise ValueError("Expected the original nine-point path")
    pv_base = float(
        original.loc[original.pv_s_scale > 0, "pv_inverter_nameplate_mva"].iloc[0]
        / original.loc[original.pv_s_scale > 0, "pv_s_scale"].iloc[0]
    )
    svc_row = original.loc[original.svc_q_scale > 0].iloc[0]
    svc_base = float(svc_row.svc_nameplate_mvar / svc_row.svc_q_scale)
    rows = original.to_dict("records")
    for pv_scale, svc_scale in EXTENSION:
        index = len(rows)
        rows.append(
            {
                "capacity_label": (
                    f"P{index:02d}_pv{pv_scale_token(pv_scale)}_svc{svc_scale_token(svc_scale)}"
                ),
                "pv_s_scale": pv_scale,
                "svc_q_scale": svc_scale,
                "cap_total_mvar": 0.0,
                "path_index": index,
                "pv_inverter_nameplate_mva": pv_scale * pv_base,
                "svc_nameplate_mvar": svc_scale * svc_base,
                "fixed_shunt_mvar": 0.0,
            }
        )
    path = pd.DataFrame(rows)
    if not (
        path.pv_s_scale.diff().fillna(0).ge(0).all()
        and path.svc_q_scale.diff().fillna(0).ge(0).all()
    ):
        raise ValueError("Extended path is not componentwise nested")
    if tuple(path.iloc[-1][["pv_s_scale", "svc_q_scale"]]) != (1.5, 1.5):
        raise ValueError("Extended path does not reach the frozen actor-domain endpoint")
    return path


def write_protocols() -> dict:
    trigger = verify_development_trigger(SOURCE_RUN)
    path = build_extended_path(SOURCE_PROTOCOL)
    OUT_PROTOCOL.mkdir(parents=True, exist_ok=True)
    for name in (
        "fitting_days.csv",
        "student_training_days.csv",
        "student_validation_days.csv",
        "calibration_days.csv",
        "selection_days.csv",
        "confirmation_days.csv",
    ):
        shutil.copy2(SOURCE_PROTOCOL / name, OUT_PROTOCOL / name)
    path.to_csv(OUT_PROTOCOL / "capacity_path.csv", index=False)
    fitting = pd.read_csv(OUT_PROTOCOL / "fitting_days.csv")
    jobs = fitting.assign(_key=1).merge(path.assign(_key=1), on="_key").drop(
        columns=["_key", "path_index", "pv_inverter_nameplate_mva", "svc_nameplate_mvar", "fixed_shunt_mvar"]
    )
    jobs = jobs.rename(columns={"capacity_label": "capacity_label"})
    jobs.insert(0, "candidate_id", jobs.groupby("scenario_index").cumcount())
    jobs = jobs[
        [
            "candidate_id",
            "capacity_label",
            "scenario_index",
            "day_index",
            "t0",
            "pv_s_scale",
            "svc_q_scale",
            "cap_total_mvar",
        ]
    ]
    jobs.to_csv(OUT_PROTOCOL / "training_path_jobs.csv", index=False)

    source_meta = json.loads(
        (SOURCE_PROTOCOL / "protocol.json").read_text(encoding="utf-8")
    )
    protocol = {
        **source_meta,
        "protocol": "VMOD 69-bus development-triggered extended capacity path",
        "training_path_job_rows": int(len(jobs)),
        "confirmation_days": 349,
        "capacity_path": [
            {
                "path_index": int(row.path_index),
                "label": row.capacity_label,
                "pv_s_scale": float(row.pv_s_scale),
                "svc_q_scale": float(row.svc_q_scale),
                "cap_total_mvar": float(row.cap_total_mvar),
                "pv_inverter_nameplate_mva": float(row.pv_inverter_nameplate_mva),
                "svc_nameplate_mvar": float(row.svc_nameplate_mvar),
                "fixed_shunt_mvar": float(row.fixed_shunt_mvar),
            }
            for row in path.itertuples(index=False)
        ],
        "second_feeder_path_extension": {
            "trigger": "all original nine points failed development calibration",
            "rule": (
                "alternate +0.25 SVC and PV scale increments from (0.75,0.75) "
                "to the actor-domain endpoint (1.50,1.50)"
            ),
            "external_selection_outputs_observed": False,
            "trigger_evidence": trigger,
        },
        "capacity_path_sha256": sha256(OUT_PROTOCOL / "capacity_path.csv"),
        "training_path_jobs_sha256": sha256(
            OUT_PROTOCOL / "training_path_jobs.csv"
        ),
    }
    (OUT_PROTOCOL / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    OUT_EXTERNAL_PROTOCOL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_PROTOCOL / "capacity_path.csv", OUT_EXTERNAL_PROTOCOL / "capacity_path.csv")
    shutil.copy2(OUT_PROTOCOL / "training_path_jobs.csv", OUT_EXTERNAL_PROTOCOL / "training_path_jobs.csv")
    for name in ("selection_days.csv", "confirmation_days.csv"):
        shutil.copy2(EXTERNAL_PROFILE / name, OUT_EXTERNAL_PROTOCOL / name)
    selection = pd.read_csv(OUT_EXTERNAL_PROTOCOL / "selection_days.csv")
    confirmation = pd.read_csv(OUT_EXTERNAL_PROTOCOL / "confirmation_days.csv")
    external = {
        "role": "post-development external validation protocol",
        "source_protocol": str(OUT_PROTOCOL),
        "selection_days": int(len(selection)),
        "confirmation_days": int(len(confirmation)),
        "selection_days_sha256": sha256(OUT_EXTERNAL_PROTOCOL / "selection_days.csv"),
        "confirmation_days_sha256": sha256(
            OUT_EXTERNAL_PROTOCOL / "confirmation_days.csv"
        ),
        "capacity_path_sha256": sha256(OUT_EXTERNAL_PROTOCOL / "capacity_path.csv"),
        "development_triggered_extension_frozen_before_external_outputs": True,
        "external_selection_results_used_to_define_path": False,
        "external_confirmation_results_used_to_define_path": False,
    }
    (OUT_EXTERNAL_PROTOCOL / "external_validation_protocol.json").write_text(
        json.dumps(external, indent=2), encoding="utf-8"
    )
    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "extend the 69-bus capacity path before external selection output",
        "trigger": trigger,
        "extension_points": [list(values) for values in EXTENSION],
        "source_capacity_path_sha256": sha256(SOURCE_PROTOCOL / "capacity_path.csv"),
        "extended_capacity_path_sha256": sha256(OUT_PROTOCOL / "capacity_path.csv"),
        "external_selection_output_observed": False,
    }
    (OUT_PROTOCOL / "extension_decision_audit.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    return {"protocol": protocol, "external": external, "decision": decision}


if __name__ == "__main__":
    print(json.dumps(write_protocols(), indent=2))
