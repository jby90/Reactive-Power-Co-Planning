"""Prepare a fresh OPSD window for post-development VMOD validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_opsd_revision_profiles import (
    STEPS_PER_DAY,
    build_profiles,
    day_hashes,
    extract_window,
    file_sha256,
    write_split,
)


def split_external_days(
    days: int, seed: int, selection_days: int = 17
) -> dict[str, np.ndarray]:
    if days <= selection_days:
        raise ValueError("External window must contain confirmation days")
    permutation = np.random.default_rng(seed).permutation(days)
    return {
        "selection": np.sort(permutation[:selection_days]),
        "confirmation": np.sort(permutation[selection_days:]),
    }


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_profiles(
    source: Path,
    out_dir: Path,
    start: str,
    days: int,
    split_seed: int,
    load_columns: int,
    pv_columns: int,
    load_reference: float,
    solar_reference: float,
    selection_days: int = 17,
    missing_policy: str = "reject",
) -> dict:
    frame = extract_window(
        source, start=start, days=days, missing_policy=missing_policy
    )
    load, solar, stats = build_profiles(
        frame,
        load_columns=load_columns,
        pv_columns=pv_columns,
        load_reference=load_reference,
        solar_reference=solar_reference,
    )
    hashes = day_hashes(load, solar)
    if len(set(hashes)) != days:
        raise ValueError("External validation contains duplicate joint daily profiles")
    splits = split_external_days(days, split_seed, selection_days=selection_days)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_path = out_dir / "load_15min_366d.npy"
    solar_path = out_dir / "pv_15min_366d.npy"
    np.save(load_path, load)
    np.save(solar_path, solar)
    for name, indices in splits.items():
        write_split(out_dir / f"{name}_days.csv", indices)
    metadata = {
        "role": "post-development external validation",
        "source": "Open Power System Data, Time series package 2020-10-06",
        "source_url": "https://data.open-power-system-data.org/time_series/2020-10-06/",
        "source_doi": "https://doi.org/10.25832/time_series/2020-10-06",
        "source_file_sha256": file_sha256(source),
        "window_start_utc": str(frame.iloc[0, 0]),
        "window_end_utc_inclusive": str(frame.iloc[-1, 0]),
        "days": days,
        "steps_per_day": STEPS_PER_DAY,
        "split_seed": split_seed,
        "selection_days": len(splits["selection"]),
        "confirmation_days": len(splits["confirmation"]),
        "normalisation": (
            "Uses the frozen development-window load and solar references; "
            "no external-validation value exceeded either reference."
        ),
        "missing_data": {
            "policy": frame.attrs.get("missing_policy", "reject"),
            "values_before_repair": frame.attrs.get(
                "missing_values_before_repair", {}
            ),
        },
        "statistics": stats,
        "load_shape": list(load.shape),
        "solar_shape": list(solar.shape),
        "unique_joint_daily_profiles": len(set(hashes)),
        "load_npy_sha256": file_sha256(load_path),
        "solar_npy_sha256": file_sha256(solar_path),
        "selection_days_sha256": sha256_bytes(out_dir / "selection_days.csv"),
        "confirmation_days_sha256": sha256_bytes(out_dir / "confirmation_days.csv"),
    }
    if stats["load_values_above_reference"] or stats["solar_values_above_reference"]:
        raise ValueError("External values exceed the frozen development reference")
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def prepare_protocol(
    profile_dir: Path, source_protocol: Path, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("capacity_path.csv", "training_path_jobs.csv"):
        source = source_protocol / name
        if source.is_file():
            shutil.copy2(source, out_dir / name)
    for name in ("selection_days.csv", "confirmation_days.csv"):
        shutil.copy2(profile_dir / name, out_dir / name)
    selection = pd.read_csv(out_dir / "selection_days.csv")
    confirmation = pd.read_csv(out_dir / "confirmation_days.csv")
    if set(selection.day_index.astype(int)) & set(confirmation.day_index.astype(int)):
        raise ValueError("External selection and confirmation days overlap")
    manifest = {
        "role": "post-development external validation protocol",
        "source_protocol": str(source_protocol),
        "selection_days": len(selection),
        "confirmation_days": len(confirmation),
        "selection_days_sha256": sha256_bytes(out_dir / "selection_days.csv"),
        "confirmation_days_sha256": sha256_bytes(out_dir / "confirmation_days.csv"),
        "capacity_path_sha256": sha256_bytes(out_dir / "capacity_path.csv"),
        "development_data_used_for_external_selection": False,
        "development_data_used_for_external_confirmation": False,
    }
    (out_dir / "external_validation_protocol.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile_out", type=Path, required=True)
    parser.add_argument("--source_protocol", type=Path, required=True)
    parser.add_argument("--protocol_out", type=Path, required=True)
    parser.add_argument("--start", default="2018-01-02T00:00:00Z")
    parser.add_argument("--days", type=int, default=366)
    parser.add_argument("--split_seed", type=int, default=20260921)
    parser.add_argument("--selection_days", type=int, default=17)
    parser.add_argument(
        "--missing_policy",
        choices=("reject", "interpolate_time"),
        default="reject",
    )
    parser.add_argument("--load_columns", type=int, required=True)
    parser.add_argument("--pv_columns", type=int, required=True)
    parser.add_argument("--load_reference", type=float, default=77852.94)
    parser.add_argument("--solar_reference", type=float, default=0.684)
    args = parser.parse_args()
    metadata = prepare_profiles(
        args.source,
        args.profile_out,
        args.start,
        args.days,
        args.split_seed,
        args.load_columns,
        args.pv_columns,
        args.load_reference,
        args.solar_reference,
        args.selection_days,
        args.missing_policy,
    )
    protocol = prepare_protocol(
        args.profile_out, args.source_protocol, args.protocol_out
    )
    print(json.dumps({"profiles": metadata, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
