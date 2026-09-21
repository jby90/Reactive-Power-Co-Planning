"""Prepare traceable 15-minute load/PV profiles from the frozen OPSD release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
SOLAR_COLUMN = "DE_solar_profile"
TIMESTAMP_COLUMN = "utc_timestamp"
STEPS_PER_DAY = 96


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_window(
    source: Path,
    start: str = "2017-01-01T00:00:00Z",
    days: int = 366,
) -> pd.DataFrame:
    frame = pd.read_csv(
        source,
        usecols=[TIMESTAMP_COLUMN, LOAD_COLUMN, SOLAR_COLUMN],
        parse_dates=[TIMESTAMP_COLUMN],
    )
    start_time = pd.Timestamp(start)
    end_time = start_time + pd.Timedelta(days=days)
    frame = frame[
        (frame[TIMESTAMP_COLUMN] >= start_time)
        & (frame[TIMESTAMP_COLUMN] < end_time)
    ].copy()
    frame = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    expected_rows = days * STEPS_PER_DAY
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    expected_time = pd.date_range(start_time, periods=expected_rows, freq="15min")
    if not frame[TIMESTAMP_COLUMN].equals(pd.Series(expected_time, name=TIMESTAMP_COLUMN)):
        raise ValueError("Selected OPSD window is not a complete regular 15-minute series")
    if frame[[LOAD_COLUMN, SOLAR_COLUMN]].isna().any().any():
        raise ValueError("Selected OPSD window contains missing load or solar values")
    if (frame[[LOAD_COLUMN, SOLAR_COLUMN]] < 0).any().any():
        raise ValueError("Selected OPSD window contains negative load or solar values")
    return frame


def build_profiles(
    frame: pd.DataFrame,
    load_columns: int = 32,
    pv_columns: int = 3,
    load_reference: float | None = None,
    solar_reference: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    load_raw = frame[LOAD_COLUMN].to_numpy(dtype=float)
    solar_raw = frame[SOLAR_COLUMN].to_numpy(dtype=float)
    load_reference = float(
        load_raw.max() if load_reference is None else load_reference
    )
    solar_reference = float(
        solar_raw.max() if solar_reference is None else solar_reference
    )
    if load_reference <= 0 or solar_reference <= 0:
        raise ValueError("Normalisation references must be positive")
    load_factor = np.clip(load_raw / load_reference, 0.0, 1.0)
    solar_factor = np.clip(solar_raw / solar_reference, 0.0, 1.0)
    load = np.repeat(load_factor[:, None], load_columns, axis=1)
    solar = np.repeat(solar_factor[:, None], pv_columns, axis=1)
    stats = {
        "load_reference_mw": load_reference,
        "solar_profile_reference": solar_reference,
        "load_values_above_reference": int((load_raw > load_reference).sum()),
        "solar_values_above_reference": int((solar_raw > solar_reference).sum()),
        "load_factor_percentiles": {
            str(q): float(np.quantile(load_factor, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)
        },
        "solar_factor_percentiles": {
            str(q): float(np.quantile(solar_factor, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
    }
    return load, solar, stats


def split_days(days: int, seed: int) -> dict[str, np.ndarray]:
    if days != 366:
        raise ValueError("The frozen split requires exactly 366 days")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(days)
    return {
        "training": np.sort(permutation[:50]),
        "selection": np.sort(permutation[50:67]),
        "confirmation": np.sort(permutation[67:]),
    }


def day_hashes(load: np.ndarray, solar: np.ndarray) -> list[str]:
    hashes = []
    for day in range(len(load) // STEPS_PER_DAY):
        start = day * STEPS_PER_DAY
        stop = start + STEPS_PER_DAY
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(load[start:stop]).tobytes())
        digest.update(np.ascontiguousarray(solar[start:stop]).tobytes())
        hashes.append(digest.hexdigest())
    return hashes


def write_split(path: Path, values: np.ndarray) -> None:
    pd.DataFrame(
        {
            "scenario_index": np.arange(len(values), dtype=int),
            "day_index": values,
            "t0": values * STEPS_PER_DAY,
        }
    ).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split_seed", type=int, default=20260918)
    parser.add_argument("--load_columns", type=int, default=32)
    parser.add_argument("--pv_columns", type=int, default=3)
    args = parser.parse_args()

    frame = extract_window(args.source)
    load, solar, stats = build_profiles(
        frame, load_columns=args.load_columns, pv_columns=args.pv_columns
    )
    hashes = day_hashes(load, solar)
    if len(set(hashes)) != 366:
        raise ValueError("The selected OPSD window contains duplicate joint daily profiles")
    splits = split_days(366, args.split_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_path = args.out_dir / "load_15min_366d.npy"
    solar_path = args.out_dir / "pv_15min_366d.npy"
    np.save(load_path, load)
    np.save(solar_path, solar)
    for name, values in splits.items():
        write_split(args.out_dir / f"{name}_days.csv", values)
    metadata = {
        "source": "Open Power System Data, Time series package 2020-10-06",
        "source_url": "https://data.open-power-system-data.org/time_series/2020-10-06/",
        "source_doi": "https://doi.org/10.25832/time_series/2020-10-06",
        "primary_source": "ENTSO-E Transparency Platform",
        "source_file": str(args.source),
        "source_file_sha256": file_sha256(args.source),
        "window_start_utc": str(frame[TIMESTAMP_COLUMN].iloc[0]),
        "window_end_utc_inclusive": str(frame[TIMESTAMP_COLUMN].iloc[-1]),
        "time_resolution_minutes": 15,
        "days": 366,
        "load_column": LOAD_COLUMN,
        "solar_column": SOLAR_COLUMN,
        "mapping": (
            f"The common normalized temporal load factor is broadcast to {args.load_columns} "
            f"load columns and preserves the feeder's original spatial allocation; the common "
            f"normalized solar factor drives {args.pv_columns} PV units."
        ),
        "normalisation": (
            "Each selected source series is divided by its maximum over the frozen 366-day window."
        ),
        "not_a_field_feeder_claim": (
            "The public German system-level profiles provide traceable temporal shapes; "
            "they are not measurements from either IEEE benchmark feeder."
        ),
        "load_shape": list(load.shape),
        "solar_shape": list(solar.shape),
        "load_npy_sha256": file_sha256(load_path),
        "solar_npy_sha256": file_sha256(solar_path),
        "unique_joint_daily_profiles": len(set(hashes)),
        "split_seed": args.split_seed,
        "split_counts": {name: len(values) for name, values in splits.items()},
        "statistics": stats,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
