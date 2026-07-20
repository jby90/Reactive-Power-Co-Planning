"""Plot the matched common reward component for the final controller study."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "generated"
SOURCE_DATA = ROOT / "data" / "figure_source" / "fig_training_reward_source.csv"
WINDOW = 25
SEEDS = (42, 43, 44)

METHODS = {
    "WG-CVaR-PPO": {
        "colour": "#C65D2E",
        "linestyle": "-",
        "path": (
            "models/wg_cvar/seed{seed}/csv/episodes.csv"
        ),
    },
    "Concat PPO": {
        "colour": "#3B6F8F",
        "linestyle": (0, (5, 2)),
        "path": (
            "models/concat/seed{seed}/csv/episodes.csv"
        ),
    },
    "Blind PPO": {
        "colour": "#3C9363",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "path": (
            "models/blind/seed{seed}/csv/episodes.csv"
        ),
    },
}


def load_episode_data() -> dict[str, dict[int, pd.DataFrame]]:
    data: dict[str, dict[int, pd.DataFrame]] = {}
    for method, spec in METHODS.items():
        data[method] = {}
        for seed in SEEDS:
            path = ROOT / spec["path"].format(seed=seed)
            frame = pd.read_csv(path).sort_values("episode").reset_index(drop=True)
            if len(frame) != 300:
                raise ValueError(f"Expected 300 episodes in {path}, found {len(frame)}")
            # Each 96-step episode uses 15-minute intervals. The trainer logs
            # E_loss = -0.25 * sum(r_loss), so sum(r_loss) = -4 * E_loss.
            frame["common_loss_return"] = -4.0 * frame["line_loss_mwh"]
            frame["rolling_return"] = frame["common_loss_return"].rolling(
                WINDOW, min_periods=WINDOW
            ).mean()
            data[method][seed] = frame

    for seed in SEEDS:
        proposed = data["WG-CVaR-PPO"][seed]
        columns = ["theta_pv", "theta_svc", "theta_cap"]
        for baseline_name in ("Concat PPO", "Blind PPO"):
            baseline = data[baseline_name][seed]
            if not np.allclose(proposed[columns], baseline[columns], atol=1e-12):
                raise ValueError(
                    f"Capacity samples are not matched for {baseline_name}, seed {seed}"
                )
    return data


def aggregate(data: dict[str, dict[int, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in METHODS:
        stack = np.vstack(
            [data[method][seed]["rolling_return"].to_numpy() for seed in SEEDS]
        )
        for episode in range(stack.shape[1]):
            values = stack[:, episode]
            valid = values[np.isfinite(values)]
            row: dict[str, float | int | str] = {
                "episode": episode + 1,
                "method": method,
                "rolling_mean": float(np.mean(valid)) if len(valid) else np.nan,
                "seed_min": float(np.min(valid)) if len(valid) else np.nan,
                "seed_max": float(np.max(valid)) if len(valid) else np.nan,
            }
            for index, seed in enumerate(SEEDS):
                row[f"seed_{seed}"] = float(values[index])
            rows.append(row)
    return pd.DataFrame(rows)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(summary: pd.DataFrame) -> plt.Figure:
    configure_style()
    fig, ax = plt.subplots(figsize=(7.15, 3.65), constrained_layout=True)

    for method, spec in METHODS.items():
        frame = summary[summary["method"] == method]
        x = frame["episode"].to_numpy(dtype=float)
        mean = frame["rolling_mean"].to_numpy(dtype=float)
        lower = frame["seed_min"].to_numpy(dtype=float)
        upper = frame["seed_max"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color=spec["colour"], alpha=0.15, linewidth=0)
        ax.plot(
            x,
            mean,
            color=spec["colour"],
            linestyle=spec["linestyle"],
            linewidth=2.0,
            label=method,
        )

    # The zoom window is fixed a priori as the final 50 training episodes and
    # is shown only to resolve overlapping converged trajectories.
    zoom_start = 251
    ax.axvspan(zoom_start, 300, color="#777777", alpha=0.035, linewidth=0)
    inset = ax.inset_axes([0.065, 0.57, 0.38, 0.32])
    for method, spec in METHODS.items():
        frame = summary[
            (summary["method"] == method) & (summary["episode"] >= zoom_start)
        ]
        x = frame["episode"].to_numpy(dtype=float)
        mean = frame["rolling_mean"].to_numpy(dtype=float)
        lower = frame["seed_min"].to_numpy(dtype=float)
        upper = frame["seed_max"].to_numpy(dtype=float)
        inset.fill_between(
            x, lower, upper, color=spec["colour"], alpha=0.12, linewidth=0
        )
        inset.plot(
            x,
            mean,
            color=spec["colour"],
            linestyle=spec["linestyle"],
            linewidth=1.25,
        )
    zoom_rows = summary[summary["episode"] >= zoom_start]
    zoom_min = float(zoom_rows["seed_min"].min())
    zoom_max = float(zoom_rows["seed_max"].max())
    zoom_pad = 0.08 * (zoom_max - zoom_min)
    inset.set_xlim(zoom_start, 300)
    inset.set_ylim(zoom_min - zoom_pad, zoom_max + zoom_pad)
    inset.set_xticks([251, 275, 300])
    inset.tick_params(labelsize=6, length=2.5, width=0.6)
    inset.grid(axis="y", color="#DEDEDE", linewidth=0.45)
    inset.set_title("Final 50 episodes", fontsize=6.8, pad=2.5)
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)

    finite_lower = summary["seed_min"].dropna().to_numpy(dtype=float)
    finite_upper = summary["seed_max"].dropna().to_numpy(dtype=float)
    ax.set_ylim(
        np.floor(finite_lower.min() * 2.0) / 2.0 - 0.1,
        np.ceil(finite_upper.max() * 2.0) / 2.0 + 0.1,
    )
    ax.set_xlim(1, 300)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(r"Common line-loss return, $R_{\mathrm{loss}}$")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", ncol=3, handlelength=2.8, columnspacing=1.4)
    ax.text(
        0.99,
        0.98,
        "Lines: three-seed mean; shading: seed range; 25-episode rolling mean",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_episode_data()
    summary = aggregate(data)
    summary.to_csv(SOURCE_DATA, index=False)
    fig = plot(summary)
    stem = OUTPUT_DIR / "final_training_reward_curves"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
