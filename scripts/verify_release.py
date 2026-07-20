"""Verify the public release and recompute the manuscript's headline values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing release file: {path.relative_to(ROOT)}")
    return path


def verify_models() -> None:
    expected = {
        "wg_cvar": "concat_worst_group_cvar",
        "concat": "concat_scalar_corrected",
        "blind": "blind_scalar_corrected",
    }
    for family, actor_arch in expected.items():
        for seed in (42, 43, 44):
            run_dir = ROOT / "models" / family / f"seed{seed}"
            config = json.loads(require(run_dir / "config.json").read_text(encoding="utf-8"))
            require(run_dir / "models" / "checkpoint.pth")
            require(run_dir / "csv" / "episodes.csv")
            assert config["actor_arch"] == actor_arch, (family, seed, config["actor_arch"])
            assert int(config["seed"]) == seed


def verify_locked_comparison() -> tuple[float, float]:
    base = RESULTS / "locked_controller_evaluation"
    summary = pd.read_csv(require(base / "locked_summary.csv"))
    paired = pd.read_csv(require(base / "locked_paired_comparison.csv"))
    assert len(summary) == 24
    assert len(paired) == 6
    projected = summary[summary.variant.isin(["wg_projected", "concat_projected"])]
    assert len(projected) == 12
    assert np.allclose(projected.risk, 0.0)
    reductions: list[float] = []
    for mode in ("uniform", "stress"):
        rows = projected[projected["mode"] == mode].set_index("variant")
        wg = float(rows.loc["wg_projected", "mean_loss_mw"].mean())
        concat = float(rows.loc["concat_projected", "mean_loss_mw"].mean())
        reductions.append(100.0 * (concat - wg) / concat)
    np.testing.assert_allclose(reductions, [3.74, 4.07], atol=0.01)
    return reductions[0], reductions[1]


def verify_capacity_planning() -> tuple[float, float]:
    base = RESULTS / "capacity_planning"
    grid = pd.read_csv(require(base / "grid_candidates.csv"))
    confirm = pd.read_csv(require(base / "confirm" / "wg_projected_candidate_summary.csv"))
    optimum = pd.read_csv(require(base / "final_optimum.csv")).iloc[0]
    ablation = pd.read_csv(require(base / "ablation" / "comparison.csv"))
    assert len(grid) == 2000
    assert len(confirm) == 37
    np.testing.assert_allclose(
        optimum[["pv_s_scale", "svc_q_scale", "cap_total_mvar"]].astype(float),
        [0.7, 0.7, 0.0],
    )
    selected = ablation[(ablation.variant == "wg_projected")].set_index("selection_role")
    minimum = selected.loc["optimiser_candidate"]
    high = selected.loc["risk_path"]
    resource_reduction = 100.0 * (1.0 - minimum.resource_index / high.resource_index)
    loss_reduction = 100.0 * (1.0 - minimum.mean_daily_loss_mwh / high.mean_daily_loss_mwh)
    np.testing.assert_allclose([resource_reduction, loss_reduction], [53.95, 62.88], atol=0.01)
    assert minimum.worst_seed_risk == high.worst_seed_risk == 0.0
    return float(resource_reduction), float(loss_reduction)


def main() -> None:
    verify_models()
    uniform, stress = verify_locked_comparison()
    resource, loss = verify_capacity_planning()
    print("Release verification passed.")
    print(f"Locked line-loss reductions: uniform={uniform:.2f}%, stress={stress:.2f}%")
    print(f"High-redundancy comparison: resource={resource:.2f}%, line loss={loss:.2f}%")


if __name__ == "__main__":
    main()
