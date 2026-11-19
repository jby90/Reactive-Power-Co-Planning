"""Create deterministic node-heterogeneous stress profiles from OPSD factors.

The source profiles retain the public OPSD system-level load and solar factors. This
script adds bounded, autocorrelated, location-specific multiplicative deviations for
a sensitivity test; the generated profiles are synthetic stresses, not measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autocorrelated_deviations(
    length: int, columns: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    innovations = rng.standard_normal((length, columns))
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    innovation_scale = np.sqrt(1.0 - rho * rho)
    for index in range(1, length):
        values[index] = rho * values[index - 1] + innovation_scale * innovations[index]
    values -= values.mean(axis=0, keepdims=True)
    values /= values.std(axis=0, keepdims=True)
    return values


def weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    mean = (values * weights).sum(axis=0, keepdims=True) / denominator
    return values - mean


def mean_off_diagonal_correlation(values: np.ndarray) -> float:
    if values.shape[1] < 2:
        return 1.0
    matrix = np.corrcoef(values.T)
    return float(matrix[np.triu_indices(values.shape[1], 1)].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--load_sigma", type=float, default=0.05)
    parser.add_argument("--pv_sigma", type=float, default=0.10)
    parser.add_argument("--load_rho", type=float, default=0.96)
    parser.add_argument("--pv_rho", type=float, default=0.90)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    load_source = np.load(source / "load_15min_366d.npy")
    pv_source = np.load(source / "pv_15min_366d.npy")
    if load_source.ndim != 2 or pv_source.ndim != 2:
        raise ValueError("Expected pre-expanded two-dimensional source profiles")
    if load_source.shape[0] != pv_source.shape[0]:
        raise ValueError("Load and PV source profiles must have equal time length")

    rng = np.random.default_rng(args.seed)
    load_z = autocorrelated_deviations(
        load_source.shape[0], load_source.shape[1], args.load_rho, rng
    )
    pv_z = autocorrelated_deviations(
        pv_source.shape[0], pv_source.shape[1], args.pv_rho, rng
    )
    load_z = weighted_center(load_z, load_source)
    pv_z = weighted_center(pv_z, pv_source)

    load_multiplier = np.clip(1.0 + args.load_sigma * load_z, 0.80, 1.20)
    pv_multiplier = np.clip(1.0 + args.pv_sigma * pv_z, 0.70, 1.30)
    heterogeneous_load = load_source * load_multiplier
    heterogeneous_pv = pv_source * pv_multiplier

    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_path = args.out_dir / "load_15min_366d.npy"
    pv_path = args.out_dir / "pv_15min_366d.npy"
    np.save(load_path, heterogeneous_load)
    np.save(pv_path, heterogeneous_pv)

    source_load_energy = float(load_source.sum())
    source_pv_energy = float(pv_source.sum())
    metadata = {
        "role": "deterministic synthetic node-heterogeneity sensitivity test",
        "measurement_status": (
            "Not measured nodal data. Bounded autocorrelated location-specific "
            "multipliers are applied to the public OPSD system-level factors."
        ),
        "source_dir": args.source_dir.as_posix(),
        "seed": args.seed,
        "load_multiplier": {
            "sigma": args.load_sigma,
            "ar1_rho": args.load_rho,
            "bounds": [0.80, 1.20],
        },
        "pv_multiplier": {
            "sigma": args.pv_sigma,
            "ar1_rho": args.pv_rho,
            "bounds": [0.70, 1.30],
        },
        "shape": {
            "load": list(heterogeneous_load.shape),
            "pv": list(heterogeneous_pv.shape),
        },
        "mean_off_diagonal_correlation": {
            "source_load": mean_off_diagonal_correlation(load_source),
            "heterogeneous_load": mean_off_diagonal_correlation(heterogeneous_load),
            "source_pv": mean_off_diagonal_correlation(pv_source),
            "heterogeneous_pv": mean_off_diagonal_correlation(heterogeneous_pv),
        },
        "aggregate_energy_ratio": {
            "load": float(heterogeneous_load.sum() / source_load_energy),
            "pv": float(heterogeneous_pv.sum() / source_pv_energy),
        },
        "load_npy_sha256": sha256(load_path),
        "pv_npy_sha256": sha256(pv_path),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
