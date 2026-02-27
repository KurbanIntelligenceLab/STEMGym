"""Shared acquisition function dispatch for GP-based agents.

Provides UCB, Expected Improvement (EI), and Thompson Sampling (TS)
acquisition functions. Used by both Navigator (STEMAgent) and GPUCBAgent.
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
        acq_func: One of "ucb", "ei", "ts".
        beta: UCB exploration parameter (only used for "ucb").
        train_Y: Normalized training targets (only used for "ei" to get best_f).

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

    elif acq_func == "ei":
        # Expected Improvement: EI(x) = E[max(f(x) - f*, 0)]
        if train_Y is not None:
            best_f = train_Y.max()
        else:
            best_f = mean.max()
        z = (mean - best_f) / std.clamp(min=1e-8)
        # Use normal CDF/PDF approximation
        normal = torch.distributions.Normal(0, 1)
        ei = std * (z * normal.cdf(z) + normal.log_prob(z).exp())
        return ei.clamp(min=0.0)

    elif acq_func == "ts":
        # Thompson Sampling: draw one sample from posterior
        with torch.no_grad():
            sample = posterior.rsample()  # (1, N, 1)
        return sample.squeeze(-1).squeeze(0)

    else:
        raise ValueError(f"Unknown acquisition function: {acq_func!r}. Use 'ucb', 'ei', or 'ts'.")
