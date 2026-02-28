#!/usr/bin/env python3
"""Plot agent trajectories from benchmark results with trajectory data.

Usage:
    python scripts/plot_trajectories.py results/trajectory_capture.json \
        --world data/simulated/srtio3_clustered_easy.h5 \
        --agent raster_equipped --seed 0 -o trajectory.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(result_path: str, agent: str, seed: int = 0, world: str | None = None):
    """Load trajectory from a results JSON file."""
    with open(result_path) as f:
        data = json.load(f)

    for combo in data["results"]:
        if combo["agent"] != agent:
            continue
        if world and combo["world"] != world:
            continue
        for ep in combo["episodes"]:
            if ep["seed"] == seed:
                traj = ep.get("trajectory")
                if traj is None:
                    raise ValueError(f"No trajectory data for {agent} seed={seed}. Was save_trajectories enabled?")
                return traj, combo["world"]

    raise ValueError(f"No matching result for agent={agent}, seed={seed}, world={world}")


def plot_trajectory(traj: list[dict], world_path: str, output: str | None = None):
    """Plot trajectory on world overview image."""
    # Load overview from world HDF5
    with h5py.File(world_path, "r") as f:
        overview = f["overview"][:]
        fov_nm = (f["metadata"].attrs["fov_nm_h"], f["metadata"].attrs["fov_nm_w"])

    # Extract positions and dose info
    positions = []
    dose_fracs = []
    acquire_positions = []
    max_dose = max(e["dose_used"] for e in traj) or 1.0

    for entry in traj:
        pos = entry.get("position")
        if pos is None:
            continue
        positions.append(pos)
        dose_fracs.append(entry["dose_used"] / max_dose)
        if entry.get("action_type") == 1:  # ACQUIRE
            acquire_positions.append(pos)

    if not positions:
        print("No positions found in trajectory.")
        return

    positions = np.array(positions)
    dose_fracs = np.array(dose_fracs)
    acquire_positions = np.array(acquire_positions) if acquire_positions else np.empty((0, 2))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Show overview
    ax.imshow(overview, extent=[0, fov_nm[1], fov_nm[0], 0], cmap="gray", alpha=0.7)

    # Plot path color-coded by dose fraction
    for i in range(len(positions) - 1):
        ax.plot(
            positions[i:i+2, 0], positions[i:i+2, 1],
            color=plt.cm.viridis(dose_fracs[i]),
            linewidth=0.8, alpha=0.6,
        )

    # Mark acquire positions
    if len(acquire_positions) > 0:
        ax.scatter(
            acquire_positions[:, 0], acquire_positions[:, 1],
            c="red", s=15, marker="x", linewidths=0.8, label="ACQUIRE", zorder=5,
        )

    # Start/end markers
    ax.scatter(*positions[0], c="lime", s=60, marker="o", edgecolors="black", linewidths=1, zorder=10, label="Start")
    ax.scatter(*positions[-1], c="blue", s=60, marker="s", edgecolors="black", linewidths=1, zorder=10, label="End")

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"Agent trajectory ({len(acquire_positions)} acquires)")
    ax.legend(loc="upper right", fontsize=8)

    # Colorbar for dose
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, max_dose))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative dose (e-/A^2)")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot agent trajectories")
    parser.add_argument("result_json", help="Path to results JSON with trajectory data")
    parser.add_argument("--world", "-w", required=True, help="Path to world HDF5 file")
    parser.add_argument("--agent", "-a", required=True, help="Agent name")
    parser.add_argument("--seed", "-s", type=int, default=0, help="Seed")
    parser.add_argument("--output", "-o", default=None, help="Output file (PDF/PNG)")
    args = parser.parse_args()

    traj, world_name = load_trajectory(args.result_json, args.agent, args.seed)
    plot_trajectory(traj, args.world, args.output)


if __name__ == "__main__":
    main()
