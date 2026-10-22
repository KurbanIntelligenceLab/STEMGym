from typing import Any
import numpy as np
from stem_gym.envs.world import World


def filter_gt_to_explored(
    gt: dict[str, Any],
    visited_mask: np.ndarray,
    world: World,
) -> dict[str, Any]:
    """Filter ground truth to only atoms within explored (visited) tiles."""
    atom_positions = gt.get("atom_positions")
    if atom_positions is None or len(atom_positions) == 0:
        return gt


    px_x = atom_positions[:, 0] / world.pixel_size_nm
    px_y = atom_positions[:, 1] / world.pixel_size_nm

    rows = np.clip((px_y / world.stride_px).astype(int), 0, world.grid_shape[0] - 1)
    cols = np.clip((px_x / world.stride_px).astype(int), 0, world.grid_shape[1] - 1)


    mask = visited_mask[rows, cols]


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
