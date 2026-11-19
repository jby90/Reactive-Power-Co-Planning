"""Draw publication-ready single-line topologies for the VMOD benchmark feeders."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


NAVY = "#123A67"
BLUE = "#2D7DB8"
GREEN = "#149B68"
GREY = "#8D9AA8"
LIGHT = "#F5F8FA"


def tree_positions(edges: list[tuple[int, int]], root: int = 0) -> dict[int, tuple[float, float]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    parent = {root: -1}
    depth = {root: 0}
    queue = deque([root])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in sorted(adjacency[node]):
            if neighbour in parent:
                continue
            parent[neighbour] = node
            depth[neighbour] = depth[node] + 1
            queue.append(neighbour)
    children: dict[int, list[int]] = defaultdict(list)
    for node, ancestor in parent.items():
        if ancestor >= 0:
            children[ancestor].append(node)
    for node in children:
        children[node].sort()

    y: dict[int, float] = {}
    next_leaf = 0.0

    def assign(node: int) -> float:
        nonlocal next_leaf
        if not children[node]:
            y[node] = next_leaf
            next_leaf += 1.0
        else:
            y[node] = float(np.mean([assign(child) for child in children[node]]))
        return y[node]

    assign(root)
    centre = float(np.mean(list(y.values())))
    return {node: (float(depth[node]), y[node] - centre) for node in order}


def device_ids(env_id: int) -> tuple[list[int], list[int]]:
    if env_id == 33:
        return [17, 21, 24], [32]
    if env_id == 69:
        return [5, 23, 44, 57], [13]
    raise ValueError(f"Unsupported environment: {env_id}")


def feeder_data(env_id: int, case_path: Path) -> pd.DataFrame:
    case = loadmat(case_path, squeeze_me=True, struct_as_record=False)["mpc"]
    branch = np.asarray(case.branch, dtype=float)
    lines = pd.DataFrame(
        {
            "from_bus": branch[:, 0].astype(int) - 1,
            "to_bus": branch[:, 1].astype(int) - 1,
            "in_service": branch[:, 10].astype(int).astype(bool),
        }
    )
    lines["environment"] = env_id
    return lines


def draw_feeder(ax: plt.Axes, env_id: int, lines: pd.DataFrame) -> dict:
    closed = [
        (int(row.from_bus), int(row.to_bus))
        for row in lines.itertuples()
        if bool(row.in_service)
    ]
    opened = [
        (int(row.from_bus), int(row.to_bus))
        for row in lines.itertuples()
        if not bool(row.in_service)
    ]
    positions = tree_positions(closed)
    for left, right in closed:
        x = [positions[left][0], positions[right][0]]
        y = [positions[left][1], positions[right][1]]
        ax.plot(x, y, color=NAVY, linewidth=1.15, solid_capstyle="round", zorder=1)
    for left, right in opened:
        if left not in positions or right not in positions:
            continue
        x = [positions[left][0], positions[right][0]]
        y = [positions[left][1], positions[right][1]]
        ax.plot(x, y, color=GREY, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)

    pv_buses, svc_buses = device_ids(env_id)
    buses = sorted(positions)
    xs = [positions[node][0] for node in buses]
    ys = [positions[node][1] for node in buses]
    ax.scatter(xs, ys, s=12, facecolor="white", edgecolor=NAVY, linewidth=0.7, zorder=3)
    ax.scatter(
        [positions[node][0] for node in pv_buses],
        [positions[node][1] for node in pv_buses],
        s=34,
        marker="s",
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="PV inverter",
    )
    ax.scatter(
        [positions[node][0] for node in svc_buses],
        [positions[node][1] for node in svc_buses],
        s=42,
        marker="D",
        facecolor=GREEN,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="SVC",
    )
    ax.scatter(
        [positions[0][0]],
        [positions[0][1]],
        s=46,
        marker="o",
        facecolor=NAVY,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Grid connection",
    )
    for node in buses:
        ax.annotate(
            str(node + 1),
            positions[node],
            xytext=(0, 3.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=4.5 if env_id == 69 else 5.2,
            color=NAVY,
        )
    ax.set_title(
        f"{'a' if env_id == 33 else 'b'}  Modified IEEE {env_id}-bus feeder",
        loc="left",
        fontsize=8,
        fontweight="bold",
        color=NAVY,
        pad=3,
    )
    ax.set_facecolor(LIGHT)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {
        "environment": env_id,
        "buses": len(buses),
        "closed_branches": len(closed),
        "open_ties": len(opened),
        "pv_buses_one_based": [bus + 1 for bus in pv_buses],
        "svc_buses_one_based": [bus + 1 for bus in svc_buses],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case33", type=Path, default=Path("case33_bw.mat"))
    parser.add_argument("--case69", type=Path, default=Path("case69.mat"))
    parser.add_argument("--out_base", type=Path, required=True)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    lines33 = feeder_data(33, args.case33)
    lines69 = feeder_data(69, args.case69)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), constrained_layout=True)
    summaries = [draw_feeder(axes[0], 33, lines33), draw_feeder(axes[1], 69, lines69)]
    handles, labels = axes[0].get_legend_handles_labels()
    open_handle = mpl.lines.Line2D([], [], color=GREY, linestyle=(0, (2, 2)), linewidth=0.8)
    axes[0].legend(
        handles + [open_handle],
        labels + ["Normally open tie"],
        loc="upper right",
        ncol=4,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    args.out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(args.out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(args.out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = pd.concat([lines33, lines69], ignore_index=True)
    lines.to_csv(args.out_base.with_name(args.out_base.name + "_source_data.csv"), index=False)
    manifest = {
        "claim": "Benchmark radial topologies and controllable-device locations.",
        "panels": summaries,
        "bus_numbering": "One based in labels and manuscript; zero based in source columns.",
        "exports": ["pdf", "svg", "tiff_600dpi", "png_preview"],
    }
    args.out_base.with_name(args.out_base.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
