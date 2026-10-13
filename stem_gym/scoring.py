"""Ground-truth selection for explored-region scoring."""

from typing import Any
import numpy as np
from stem_gym.envs.world import World


def filter_gt_to_explored(
    gt: dict[str, Any],
    visited_mask: np.ndarray,
    world: World,
) -> dict[str, Any]:
    """Filter ground truth to only atoms within explored (visited) tiles.

    Converts atom positions (nm) to tile grid coordinates and keeps only
    atoms whose tile has been visited by the agent. This prevents penalizing
    agents for not detecting defects in regions they haven't explored.

    Args:
        gt: Full ground truth dict from world.get_ground_truth().
        visited_mask: Boolean array (grid_h, grid_w) of visited tiles.
        world: World object for coordinate conversion.

    Returns:
        Filtered ground truth dict (shallow copy with filtered arrays).
    """
    atom_positions = gt.get("atom_positions")
    if atom_positions is None or len(atom_positions) == 0:
        return gt

    # Convert atom positions (nm) to tile grid indices
    # atom_positions is (N, 2) with columns [x_nm, y_nm]
    px_x = atom_positions[:, 0] / world.pixel_size_nm
    px_y = atom_positions[:, 1] / world.pixel_size_nm

    rows = np.clip((px_y / world.stride_px).astype(int), 0, world.grid_shape[0] - 1)
    cols = np.clip((px_x / world.stride_px).astype(int), 0, world.grid_shape[1] - 1)

    # Boolean mask: True if atom's tile has been visited
    mask = visited_mask[rows, cols]

    # Filter all atom-level arrays
    filtered = dict(gt)
    filtered["atom_positions"] = atom_positions[mask]

    for key in (
        "atom_types",
        "defect_mask",
        "defect_types",
        "particle_diameters_nm",
        "particle_morphology_labels",
    ):
        if key in gt and gt[key] is not None and len(gt[key]) == len(atom_positions):
            filtered[key] = gt[key][mask]

    return filtered
