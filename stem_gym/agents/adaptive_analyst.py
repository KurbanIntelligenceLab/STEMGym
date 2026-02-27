"""Adaptive Analyst: online self-supervised fine-tuning of AtomFinder.

Wraps an existing Analyst instance and periodically fine-tunes the
AtomFinder ensemble using consistency between ensemble members as
pseudo-supervision. Member 0's high-confidence predictions serve as
pseudo-labels for members 1 and 2.

This improves cross-family transfer (e.g. perovskite-trained models
on SiGe or Pt worlds) by adapting the atom detector to the local
intensity patterns of the new material.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from stem_gym.agents.analyst import Analyst
from stem_gym.agents.simple_analyst import AnalysisResult

logger = logging.getLogger(__name__)


class AdaptiveAnalyst:
    """Wraps Analyst with online self-supervised fine-tuning.

    Delegates all analysis calls to the base analyst. After every
    ``update_every_n_tiles`` acquisitions, runs a few gradient steps
    on buffered tiles using ensemble consistency loss.

    Args:
        base_analyst: A loaded Analyst instance with trained models.
        lr: Learning rate for fine-tuning.
        update_every_n_tiles: Run fine-tuning after this many new tiles.
        n_gradient_steps: Number of gradient steps per fine-tuning round.
        tile_buffer_size: Max tiles to keep in the replay buffer.
        confidence_threshold: Minimum probability for pseudo-labels.
    """

    def __init__(
        self,
        base_analyst: Analyst,
        lr: float = 1e-5,
        update_every_n_tiles: int = 5,
        n_gradient_steps: int = 3,
        tile_buffer_size: int = 20,
        confidence_threshold: float = 0.5,
    ):
        self._base = base_analyst
        self._lr = lr
        self._update_every = update_every_n_tiles
        self._n_steps = n_gradient_steps
        self._conf_thresh = confidence_threshold
        self._tile_buffer: deque[np.ndarray] = deque(maxlen=tile_buffer_size)
        self._tiles_since_update = 0
        self._optimizer: torch.optim.Optimizer | None = None
        self._setup_finetuning()

    def _setup_finetuning(self) -> None:
        """Set up optimizer for AtomFinder fine-tuning."""
        atom_finder = self._get_atom_finder()
        if atom_finder is None:
            self._can_finetune = False
            return

        # Freeze BatchNorm layers — only train conv weights
        for module in atom_finder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        # Collect trainable parameters (conv weights only)
        params = [p for p in atom_finder.parameters() if p.requires_grad]
        if not params:
            self._can_finetune = False
            return

        self._optimizer = torch.optim.Adam(params, lr=self._lr)
        self._can_finetune = True
        logger.info(
            "AdaptiveAnalyst: fine-tuning enabled, %d trainable params",
            sum(p.numel() for p in params),
        )

    def _get_atom_finder(self):
        """Access the underlying AtomFinderEnsemble (handles torch.compile)."""
        af = self._base.atom_finder
        if af is None:
            return None
        # torch.compile wraps the model; access the original via _orig_mod
        return getattr(af, "_orig_mod", af)

    # -- Delegation to base analyst --

    def reset(self) -> None:
        self._base.reset()
        self._tile_buffer.clear()
        self._tiles_since_update = 0

    def analyze_tile(
        self,
        tile: np.ndarray,
        tile_position_nm: tuple[float, float] | None = None,
        pixel_size_nm: float = 0.005,
        tile_key: tuple[int, int] | None = None,
    ) -> AnalysisResult:
        result = self._base.analyze_tile(
            tile, tile_position_nm=tile_position_nm,
            pixel_size_nm=pixel_size_nm, tile_key=tile_key,
        )

        # Buffer tile for fine-tuning
        if self._can_finetune and tile.max() > 0:
            self._tile_buffer.append(tile.copy())
            self._tiles_since_update += 1

            if self._tiles_since_update >= self._update_every:
                self._finetune_step()
                self._tiles_since_update = 0

        return result

    def get_defect_census(self) -> dict[str, int]:
        return self._base.get_defect_census()

    def get_corrected_tile_phases(self):
        return self._base.get_corrected_tile_phases()

    def get_total_atoms(self) -> int:
        return self._base.get_total_atoms()

    def get_confidence_summary(self) -> dict[str, float]:
        return self._base.get_confidence_summary()

    # Forward internal state for findings assembly
    @property
    def _all_atom_positions_nm(self):
        return self._base._all_atom_positions_nm

    @property
    def _all_defect_labels(self):
        return self._base._all_defect_labels

    @property
    def _tile_results(self):
        return self._base._tile_results

    def _finetune_step(self) -> None:
        """Run self-supervised fine-tuning on buffered tiles.

        Uses member 0's high-confidence predictions as pseudo-labels
        for members 1 and 2 (BCE loss).
        """
        atom_finder = self._get_atom_finder()
        if atom_finder is None or not self._tile_buffer:
            return

        device = self._base.device

        # Prepare batch from buffer
        tiles = list(self._tile_buffer)
        batch = []
        for tile in tiles:
            t = torch.from_numpy(tile.astype(np.float32))
            t_min, t_max = t.min(), t.max()
            if t_max > t_min:
                t = (t - t_min) / (t_max - t_min)
            batch.append(t.unsqueeze(0))

        x = torch.stack(batch, dim=0).to(device)  # (B, 1, H, W)

        members = atom_finder.members

        # Step 1: Get pseudo-labels from member 0 (no grad)
        members[0].eval()
        with torch.no_grad():
            pseudo_probs = members[0](x)  # (B, 1, H, W)

        # Mask: only use high-confidence pixels as pseudo-labels
        confident_mask = (pseudo_probs > self._conf_thresh) | (pseudo_probs < (1 - self._conf_thresh))

        if confident_mask.sum() == 0:
            return

        pseudo_labels = (pseudo_probs > self._conf_thresh).float()

        # Step 2: Fine-tune members 1+ against pseudo-labels
        for _ in range(self._n_steps):
            total_loss = torch.tensor(0.0, device=device)

            for m_idx in range(1, len(members)):
                members[m_idx].train()
                # Re-freeze BN layers
                for module in members[m_idx].modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        module.eval()

                pred = members[m_idx](x)
                # BCE loss only on confident pixels
                loss = nn.functional.binary_cross_entropy(
                    pred[confident_mask], pseudo_labels[confident_mask],
                )
                total_loss = total_loss + loss

            if total_loss.requires_grad:
                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in atom_finder.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self._optimizer.step()

        # Return all members to eval mode
        atom_finder.eval()
        for module in atom_finder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        logger.debug(
            "AdaptiveAnalyst: fine-tuned on %d tiles, loss=%.4f",
            len(tiles), total_loss.item() if total_loss.requires_grad else 0.0,
        )
