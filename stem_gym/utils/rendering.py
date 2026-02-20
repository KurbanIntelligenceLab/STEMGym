"""Rendering utilities for STEMGym visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_episode_trajectory(
    overview: np.ndarray,
    positions: list[tuple[float, float]],
    pixel_size_nm: float,
    ax=None,
) -> Any:
    """Plot agent trajectory on top of overview image.

    Args:
        overview: (H, W) overview image.
        positions: List of (x_nm, y_nm) agent positions.
        pixel_size_nm: nm per pixel in the overview.
        ax: matplotlib Axes. If None, creates new figure.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(overview, cmap="gray", origin="lower")

    if positions:
        xs = [p[0] / pixel_size_nm for p in positions]
        ys = [p[1] / pixel_size_nm for p in positions]
        # Scale to overview resolution
        oh, ow = overview.shape
        # (overview is typically downsampled — caller handles scaling)
        ax.plot(xs, ys, "r-", alpha=0.5, linewidth=1)
        ax.plot(xs[0], ys[0], "go", markersize=8, label="start")
        ax.plot(xs[-1], ys[-1], "rs", markersize=8, label="end")
        ax.legend()

    ax.set_title("Agent Trajectory")
    return ax
