#!/usr/bin/env python3
"""Human Baseline UI: Gradio web app for interactive STEM microscopy sessions.

Presents a world as an interactive microscopy session where a human user
can navigate, acquire tiles, declare findings, and receive DEC-AUC scores
using the same pipeline as automated agents.

Usage:
    pip install gradio
    python stem_gym/scripts/human_baseline.py [--world data/simulated/test_world.h5] [--task defect_census] [--budget 5000]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stem_gym.envs.stem_env import STEMEnv, ActionType
from stem_gym.metrics import compute_dec
from stem_gym.runner import filter_gt_to_explored


# ---------------------------------------------------------------------------
# Session wrapper
# ---------------------------------------------------------------------------

class HumanSession:
    """Wraps STEMEnv for step-by-step human interaction."""

    def __init__(
        self,
        world_path: str,
        task_name: str = "defect_census",
        budget: float = 5000.0,
        max_steps: int = 500,
        score_every: int = 5,
    ):
        self.world_path = world_path
        self.task_name = task_name
        self.budget = budget
        self.max_steps = max_steps
        self.score_every = score_every

        self.env: STEMEnv | None = None
        self.episode_log: list[dict] = []
        self.findings: dict[str, Any] = {
            "defect_census": {},
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }
        self._acquires_since_score = 0
        self._step = 0
        self._last_score = 0.0
        self._started = False
        self._finished = False

    def start(self, seed: int = 42) -> np.ndarray:
        """Initialize environment and return overview image."""
        from experiments.run_benchmark import make_task
        task = make_task(self.task_name)
        self.env = STEMEnv(
            self.world_path, task=task,
            total_budget=self.budget, max_steps=self.max_steps,
        )
        self.obs, self.info = self.env.reset(seed=seed)
        self.episode_log = []
        self.findings = {
            "defect_census": {},
            "defect_positions": [],
            "phase_map": None,
            "atom_positions": None,
        }
        self._acquires_since_score = 0
        self._step = 0
        self._last_score = 0.0
        self._started = True
        self._finished = False

        # Return full overview
        overview = self.env.world.get_full_overview()
        return overview

    def navigate(self, x_nm: float, y_nm: float) -> str:
        """Navigate to a position (free, no dose cost)."""
        if not self._started or self._finished:
            return "Session not active."

        action = {
            "action_type": ActionType.NAVIGATE,
            "position": np.array([x_nm, y_nm], dtype=np.float32),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self._step += 1

        if term or trunc:
            self._finished = True
            return "Episode ended."

        pos = self.obs["position"]
        return f"Navigated to ({pos[0]:.2f}, {pos[1]:.2f}) nm"

    def acquire(self) -> tuple[np.ndarray | None, str]:
        """Acquire tile at current position (costs dose)."""
        if not self._started or self._finished:
            return None, "Session not active."

        action = {
            "action_type": ActionType.ACQUIRE,
            "position": self.obs["position"].copy(),
            "zoom": np.array([1.0], dtype=np.float32),
            "dwell": np.array([1.0], dtype=np.float32),
            "finding_type": 0,
        }
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self._step += 1
        self._acquires_since_score += 1

        # Log entry
        log_entry = {
            "step": self._step,
            "action_type": int(ActionType.ACQUIRE),
            "dose_used": self.env.state.dose_used,
            "tiles_acquired": self.env.state.tiles_acquired,
            "position": list(self.env.state.position),
            "task_score": None,
        }

        # Periodic scoring
        if self._acquires_since_score >= self.score_every:
            score = self._score_current()
            log_entry["task_score"] = score
            self._last_score = score
            self._acquires_since_score = 0

        if term or trunc:
            score = self._score_current()
            log_entry["task_score"] = score
            self._last_score = score
            self._finished = True

        self.episode_log.append(log_entry)

        tile = self.obs["tile"]
        budget_pct = float(self.obs["budget_remaining"][0]) * 100
        status = (
            f"Tile acquired at ({self.env.state.position[0]:.1f}, {self.env.state.position[1]:.1f}) nm | "
            f"Tiles: {self.env.state.tiles_acquired} | "
            f"Budget: {budget_pct:.1f}% | "
            f"Score: {self._last_score:.3f}"
        )
        if self._finished:
            status += " | EPISODE ENDED"

        return tile, status

    def declare_finding(self, finding_type: str, x_nm: float, y_nm: float) -> str:
        """Record a finding declaration."""
        if not self._started:
            return "Session not active."

        # Update findings dict
        census = self.findings["defect_census"]
        census[finding_type] = census.get(finding_type, 0) + 1

        self.findings["defect_positions"].append({
            "position": (x_nm, y_nm),
            "type": finding_type,
        })

        return f"Declared: {finding_type} at ({x_nm:.1f}, {y_nm:.1f}) nm | Census: {census}"

    def finish(self) -> dict[str, Any]:
        """End episode and compute final results."""
        if not self._started:
            return {"error": "Session not active."}

        # Final scoring
        score = self._score_current()
        self._last_score = score
        self._finished = True

        # Compute DEC
        dec = compute_dec(self.episode_log, self.budget)

        result = {
            "task": self.task_name,
            "world": Path(self.world_path).stem,
            "final_score": score,
            "dec_auc": dec.auc,
            "tiles_acquired": self.env.state.tiles_acquired,
            "dose_used": self.env.state.dose_used,
            "budget": self.budget,
            "findings": self.findings,
            "dec_dose_fractions": dec.dose_fractions.tolist(),
            "dec_scores": dec.scores.tolist(),
            "timestamp": datetime.now().isoformat(),
        }

        # Save to results dir
        out_dir = Path("results/human_baseline")
        out_dir.mkdir(parents=True, exist_ok=True)
        world_stem = Path(self.world_path).stem
        filename = f"human_{world_stem}_{self.task_name}_{int(time.time())}.json"
        out_path = out_dir / filename
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        result["saved_to"] = str(out_path)
        return result

    def _score_current(self) -> float:
        """Score current findings against ground truth."""
        gt = self.env.world.get_ground_truth()
        if self.env.task.name != "phase_mapping":
            gt = filter_gt_to_explored(gt, self.env.state.visited_mask, self.env.world)
        return self.env.task.score(self.findings, gt)

    def get_visited_overlay(self, overview: np.ndarray) -> np.ndarray:
        """Create overview with visited tiles highlighted."""
        if self.env is None:
            return overview

        # Normalize overview to [0, 1]
        img = overview.astype(np.float64)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        # Create RGB image
        h, w = img.shape
        rgb = np.stack([img, img, img], axis=-1)

        # Overlay visited mask (green tint)
        mask = self.env.state.visited_mask
        if mask.shape != (h, w):
            from scipy.ndimage import zoom
            scale_h = h / mask.shape[0]
            scale_w = w / mask.shape[1]
            mask = zoom(mask.astype(float), (scale_h, scale_w), order=0) > 0.5

        rgb[mask, 1] = np.clip(rgb[mask, 1] + 0.3, 0, 1)

        # Mark current position (red dot)
        if self.env.state is not None:
            pos = self.env.state.position
            fov = self.env.state.fov_nm
            px_x = int(pos[0] / fov[1] * w)
            px_y = int(pos[1] / fov[0] * h)
            r = max(2, min(h, w) // 50)
            y_lo = max(0, px_y - r)
            y_hi = min(h, px_y + r)
            x_lo = max(0, px_x - r)
            x_hi = min(w, px_x + r)
            rgb[y_lo:y_hi, x_lo:x_hi, 0] = 1.0
            rgb[y_lo:y_hi, x_lo:x_hi, 1] = 0.0
            rgb[y_lo:y_hi, x_lo:x_hi, 2] = 0.0

        return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(session: HumanSession):
    """Build the Gradio Blocks interface."""
    import gradio as gr

    overview_img = [None]  # mutable container for overview

    def start_session(seed):
        seed = int(seed) if seed else 42
        overview = session.start(seed=seed)
        overview_img[0] = overview
        overlay = session.get_visited_overlay(overview)
        return (
            overlay,
            None,
            f"Session started (seed={seed}). Click overview to navigate, then Acquire.",
            f"Budget: 100.0% | Tiles: 0 | Score: 0.000",
        )

    def on_overview_click(evt: gr.SelectData):
        if not session._started or session._finished:
            return session.get_visited_overlay(overview_img[0]) if overview_img[0] is not None else None, None, "Session not active."

        # Convert pixel coordinates to nm
        fov = session.env.state.fov_nm
        overview_h, overview_w = overview_img[0].shape
        x_nm = evt.index[0] / overview_w * fov[1]
        y_nm = evt.index[1] / overview_h * fov[0]

        msg = session.navigate(x_nm, y_nm)
        overlay = session.get_visited_overlay(overview_img[0])
        return overlay, None, msg

    def on_acquire():
        if not session._started or session._finished:
            return (
                session.get_visited_overlay(overview_img[0]) if overview_img[0] is not None else None,
                None, "Session not active.", "",
            )

        tile, status = session.acquire()

        # Normalize tile for display
        tile_display = None
        if tile is not None:
            t = tile.astype(np.float64)
            if t.max() > t.min():
                t = (t - t.min()) / (t.max() - t.min())
            tile_display = (t * 255).astype(np.uint8)

        overlay = session.get_visited_overlay(overview_img[0])

        budget_pct = float(session.obs["budget_remaining"][0]) * 100
        stats = (
            f"Budget: {budget_pct:.1f}% | "
            f"Tiles: {session.env.state.tiles_acquired} | "
            f"Score: {session._last_score:.3f}"
        )

        return overlay, tile_display, status, stats

    def on_declare(finding_type):
        if not session._started:
            return "Session not active."
        pos = session.env.state.position
        return session.declare_finding(finding_type, pos[0], pos[1])

    def on_finish():
        if not session._started:
            return "Session not active."
        result = session.finish()
        lines = [
            f"Final Score: {result.get('final_score', 0):.4f}",
            f"DEC-AUC: {result.get('dec_auc', 0):.4f}",
            f"Tiles: {result.get('tiles_acquired', 0)}",
            f"Dose Used: {result.get('dose_used', 0):.0f}",
        ]
        if "saved_to" in result:
            lines.append(f"Saved: {result['saved_to']}")
        return "\n".join(lines)

    with gr.Blocks(title="STEMGym Human Baseline") as demo:
        gr.Markdown("# STEMGym Human Baseline")
        gr.Markdown(
            f"**World:** `{Path(session.world_path).stem}` | "
            f"**Task:** `{session.task_name}` | "
            f"**Budget:** {session.budget:.0f} e⁻/Å²"
        )

        with gr.Row():
            seed_input = gr.Number(value=42, label="Seed", precision=0)
            start_btn = gr.Button("Start Session", variant="primary")

        status_bar = gr.Textbox(label="Status", interactive=False)
        stats_bar = gr.Textbox(label="Stats", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                overview_display = gr.Image(
                    label="Overview (click to navigate)",
                    interactive=True,
                    type="numpy",
                )
            with gr.Column(scale=1):
                tile_display = gr.Image(
                    label="Last Acquired Tile (128x128)",
                    interactive=False,
                    type="numpy",
                )

        with gr.Row():
            acquire_btn = gr.Button("Acquire Tile", variant="primary")
            done_btn = gr.Button("Done (End Episode)", variant="stop")

        with gr.Row():
            finding_dropdown = gr.Dropdown(
                choices=["vacancy", "substitution", "interstitial"],
                value="vacancy",
                label="Finding Type",
            )
            declare_btn = gr.Button("Declare Finding at Current Position")

        result_output = gr.Textbox(label="Results", lines=5, interactive=False)

        # Wire events
        start_btn.click(
            start_session, inputs=[seed_input],
            outputs=[overview_display, tile_display, status_bar, stats_bar],
        )
        overview_display.select(
            on_overview_click,
            outputs=[overview_display, tile_display, status_bar],
        )
        acquire_btn.click(
            on_acquire,
            outputs=[overview_display, tile_display, status_bar, stats_bar],
        )
        declare_btn.click(
            on_declare, inputs=[finding_dropdown],
            outputs=[status_bar],
        )
        done_btn.click(on_finish, outputs=[result_output])

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STEMGym Human Baseline UI")
    parser.add_argument(
        "--world", "-w", type=str,
        default="data/simulated/test_world.h5",
        help="Path to world HDF5 file",
    )
    parser.add_argument("--task", "-t", type=str, default="defect_census")
    parser.add_argument("--budget", "-b", type=float, default=5000.0)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    world_path = args.world
    # Try resolving from registry if not a direct path
    if not Path(world_path).exists():
        from experiments.run_benchmark import resolve_world
        world_path = resolve_world(world_path)

    session = HumanSession(
        world_path=world_path,
        task_name=args.task,
        budget=args.budget,
    )

    demo = build_ui(session)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
