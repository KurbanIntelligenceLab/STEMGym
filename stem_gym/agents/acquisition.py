"""Upper-confidence-bound acquisition used by the manuscript GP policies.
"""

from __future__ import annotations

import torch
from torch import Tensor


def evaluate_acquisition(
    model,
    candidates: Tensor,
    acq_func: str = "ucb",
    beta: float = 2.0,
    train_Y: Tensor | None = None,
) -> Tensor:
    """Evaluate acquisition function over candidate points.

    Args:
        model: Fitted BoTorch SingleTaskGP model (in eval mode).
        candidates: (N, D) tensor of candidate points.
        acq_func: The supported value is "ucb".
        beta: UCB exploration parameter (only used for "ucb").
        train_Y: Unused compatibility argument.

    Returns:
        (N,) tensor of acquisition values (higher = better).
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1)

    if acq_func == "ucb":
        return mean + beta * std

    raise ValueError(f"Unsupported acquisition function: {acq_func!r}; the manuscript uses 'ucb'.")
