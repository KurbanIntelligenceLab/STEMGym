"""VLMAnalyst: Vision-language model analyst for STEM tile analysis.

Uses VLMs via OpenRouter to analyze HAADF-STEM tiles, replacing the
neural network Analyst with a general-purpose VLM. Matches the Analyst
interface consumed by _assemble_findings() in equipped_baselines.py.

Supported models (all via OpenRouter):
- anthropic/claude-haiku-4.5
- openai/gpt-5-mini
- meta-llama/llama-4-scout

Requires OPENROUTER_API_KEY environment variable.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any

import numpy as np

from stem_gym.agents.simple_analyst import AnalysisResult

# H6: retry budget for transient JSON parse failures.
_VLM_RETRY_ATTEMPTS = 3
_VLM_RETRY_SLEEP_S = 1.0

logger = logging.getLogger(__name__)

DEFECT_CLASSES = ["pristine", "vacancy", "substitution", "interstitial"]

# Provider → OpenRouter model ID
PROVIDER_MODELS = {
    "claude": "anthropic/claude-haiku-4.5",
    "gpt5": "openai/gpt-5-mini",
    "llama": "meta-llama/llama-4-scout",
    "gemini_flash": "google/gemini-2.0-flash-001",
    "claude_opus": "anthropic/claude-opus-4.7",
}

# System prompt for VLM tile analysis
_SYSTEM_PROMPT = """\
You are an expert materials scientist analyzing HAADF-STEM (High-Angle Annular \
Dark-Field Scanning Transmission Electron Microscopy) images of crystalline materials.

In HAADF-STEM images:
- Bright spots are atomic columns (intensity ~ Z^1.7, heavier atoms appear brighter)
- Vacancies appear as MISSING or abnormally DIM spots in the lattice
- Substitutional defects appear as spots that are significantly BRIGHTER or DIMMER \
than their neighbors (different atomic number)
- The image is a 128x128 pixel tile from a larger specimen

Analyze the tile and respond with ONLY a JSON object (no markdown, no explanation)."""

_USER_PROMPT = """\
Analyze this HAADF-STEM tile image.

First, reason step by step about what you see: identify the atomic lattice, \
estimate the column spacing, and note any visible defects (missing atoms, \
unusually bright/dim atoms, regions of different crystal structure). Keep \
this reasoning concise (2-4 sentences).

Then emit a single JSON object with the findings:
{{
  "n_atoms": <estimated number of atomic columns visible>,
  "defects": [
    {{"row": <pixel row 0-127>, "col": <pixel col 0-127>, "type": "vacancy"|"substitution"}}
  ],
  "phase": <integer phase label: 0 for primary phase, 1+ for different crystal structure>,
  "mean_intensity": <estimated mean brightness 0.0-1.0>,
  "confidence": <your confidence in the analysis 0.0-1.0>
}}

If no defects are visible, return an empty defects array.
Focus on clear, unambiguous defects only — do not over-report.
Wrap the JSON in a single ```json``` code block at the end of your response \
so it can be parsed reliably."""


def _tile_to_png_base64(tile: np.ndarray) -> str:
    """Convert a float32 tile to base64-encoded PNG."""
    from PIL import Image

    # Normalize to 0-255
    t = tile.copy()
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t = (t - t_min) / (t_max - t_min)
    img = Image.fromarray((t * 255).astype(np.uint8), mode="L")

    # Upscale 2x for better VLM visibility
    img = img.resize((256, 256), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json(text: str) -> dict:
    """Extract a JSON object from the VLM response.

    Handles three cases (in order of preference):
    1. A fenced ```json...``` block (preferred for CoT responses).
    2. A bare ```...``` fenced block.
    3. The last balanced {...} substring in the text.

    Raises json.JSONDecodeError if no valid JSON object can be extracted.
    """
    text = text.strip()

    # 1. Fenced json block, last occurrence
    json_fence_marker = "```json"
    if json_fence_marker in text:
        after = text.rsplit(json_fence_marker, 1)[1]
        body = after.split("```", 1)[0].strip()
        return json.loads(body)

    # 2. Generic fenced block
    if "```" in text:
        parts = text.split("```")
        # Take the last fenced segment
        for candidate in reversed(parts[1:-1] + (parts[-1:] if len(parts) > 1 else [])):
            stripped = candidate.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                return json.loads(stripped)

    # 3. Brace-matched outermost {...}: scan for the largest balanced JSON object.
    # Required because reasoning models (e.g. gpt-5-mini) sometimes emit valid
    # nested JSON without wrapping in a fenced block; rfind('{') would land
    # inside an inner array element and produce an invalid slice.
    obj = _scan_outermost_object(text)
    if obj is not None:
        return json.loads(obj)

    # No JSON found — raise the same exception type the retry loop watches for
    raise json.JSONDecodeError("No JSON object found in VLM response", text, 0)


def _scan_outermost_object(text: str) -> str | None:
    """Return the substring of the first balanced top-level {...} object, or None.

    Tracks brace depth, ignoring braces inside string literals. Picks the
    largest top-level object so a response that opens with a stray `{` then
    later emits the real JSON still parses.
    """
    best: str | None = None
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        start = i
        while i < n:
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        if best is None or len(candidate) > len(best):
                            best = candidate
                        i += 1
                        break
            i += 1
        else:
            # ran off end of string with depth > 0; abandon
            break
    return best


def _call_openrouter(image_b64: str, model: str) -> dict:
    """Call a VLM via OpenRouter API and parse the response.

    Raises ``json.JSONDecodeError`` if the response cannot be parsed; the
    caller (``VLMAnalyst._call_vlm``) handles retries up to
    ``_VLM_RETRY_ATTEMPTS`` before falling back to an empty findings dict.
    """
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ],
    )

    text = response.choices[0].message.content
    if not text:
        # Reasoning models (notably gpt-5-mini under a low max_tokens cap) sometimes
        # return None or empty content when internal reasoning consumes the budget
        # before the structured output is emitted. Raise JSONDecodeError so the
        # retry loop in _call_vlm catches it instead of bombing on text.strip().
        raise json.JSONDecodeError("VLM returned empty content", "", 0)
    return _extract_json(text)


class VLMAnalyst:
    """Vision-language model analyst for STEM tile analysis.

    Drop-in replacement for Analyst that uses VLMs via OpenRouter
    instead of trained neural networks. Matches the interface consumed
    by _assemble_findings() in equipped_baselines.py.

    Args:
        provider: "claude", "gpt5", or "llama" — selects the VLM model.
    """

    def __init__(self, provider: str = "claude"):
        if provider not in PROVIDER_MODELS:
            raise ValueError(
                f"Unknown VLM provider: {provider}. "
                f"Available: {list(PROVIDER_MODELS.keys())}"
            )
        self.provider = provider
        self.model = PROVIDER_MODELS[provider]

        # Match Analyst's accumulated state attributes
        self._tile_results: list[AnalysisResult] = []
        self._all_atom_positions_nm: list[np.ndarray] = []
        self._all_defect_labels: list[str] = []

        # Phase tracking (same as Analyst heuristic)
        self._phase_intensities: list[float] = []
        self._phase_tile_keys: list[tuple[int, int]] = []
        self._phase_labels: list[int] = []

        self._call_count = 0
        self._error_count = 0
        # H6: per-call diagnostics for parse-failure reporting (V23).
        self._diagnostics: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._tile_results.clear()
        self._all_atom_positions_nm.clear()
        self._all_defect_labels.clear()
        self._phase_intensities.clear()
        self._phase_tile_keys.clear()
        self._phase_labels.clear()
        self._call_count = 0
        self._error_count = 0
        self._diagnostics.clear()

    def _call_vlm(self, tile: np.ndarray) -> dict:
        """Call the VLM API with up to ``_VLM_RETRY_ATTEMPTS`` retries.

        Retries are triggered only by ``json.JSONDecodeError`` (the
        I/O+parse boundary). Other exceptions short-circuit to the empty
        findings fallback. Per-attempt outcomes are appended to
        ``self._diagnostics`` for downstream parse-failure reporting.
        """
        image_b64 = _tile_to_png_base64(tile)

        last_excerpt = ""
        for attempt in range(1, _VLM_RETRY_ATTEMPTS + 1):
            try:
                result = _call_openrouter(image_b64, self.model)
                self._diagnostics.append(
                    {
                        "model": self.model,
                        "attempt": attempt,
                        "success": True,
                        "parse_failure_reason": None,
                        "response_excerpt": "",
                    }
                )
                return result
            except json.JSONDecodeError as e:
                last_excerpt = (e.doc or "")[:200]
                self._diagnostics.append(
                    {
                        "model": self.model,
                        "attempt": attempt,
                        "success": False,
                        "parse_failure_reason": f"JSONDecodeError: {e.msg}",
                        "response_excerpt": last_excerpt,
                    }
                )
                if attempt < _VLM_RETRY_ATTEMPTS:
                    time.sleep(_VLM_RETRY_SLEEP_S)
                continue
            except Exception as e:
                self._error_count += 1
                self._diagnostics.append(
                    {
                        "model": self.model,
                        "attempt": attempt,
                        "success": False,
                        "parse_failure_reason": f"{type(e).__name__}: {e}",
                        "response_excerpt": "",
                    }
                )
                logger.warning(
                    f"VLM call failed ({self.provider}/{self.model}): {e}"
                )
                return {
                    "n_atoms": 0,
                    "defects": [],
                    "phase": 0,
                    "confidence": 0.0,
                }

        # All retry attempts exhausted via JSONDecodeError
        self._error_count += 1
        logger.warning(
            f"VLM JSON parse failed after {_VLM_RETRY_ATTEMPTS} attempts "
            f"({self.provider}/{self.model}); last excerpt: {last_excerpt!r}"
        )
        return {"n_atoms": 0, "defects": [], "phase": 0, "confidence": 0.0}

    def analyze_tile(
        self,
        tile: np.ndarray,
        tile_position_nm: tuple[float, float] | None = None,
        pixel_size_nm: float = 0.005,
        tile_key: tuple[int, int] | None = None,
    ) -> AnalysisResult:
        """Full analysis pipeline for a single tile via VLM.

        Same interface as Analyst.analyze_tile().
        """
        if tile.max() <= 0:
            return AnalysisResult(
                atom_positions=np.empty((0, 2)),
                atom_intensities=np.empty(0),
            )

        self._call_count += 1
        response = self._call_vlm(tile)

        n_atoms = int(response.get("n_atoms", 0))
        defects = response.get("defects", [])
        phase_label = int(response.get("phase", 0))
        mean_intensity = float(response.get("mean_intensity", tile.mean()))

        # VLM can't reliably enumerate all atoms, so we use peak detection
        # for atom positions and VLM only for defect classification
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max

        smoothed = gaussian_filter(tile.astype(np.float64), sigma=1.5)
        thresh = smoothed.max() * 0.2
        positions = peak_local_max(smoothed, min_distance=6, threshold_abs=thresh)

        if len(positions) == 0:
            positions = np.empty((0, 2), dtype=np.float32)
            intensities = np.empty(0, dtype=np.float32)
        else:
            positions = positions.astype(np.float32)
            intensities = np.array(
                [tile[int(r), int(c)] for r, c in positions], dtype=np.float32
            )

        # Match VLM-reported defects to detected atom positions
        defect_labels = np.zeros(len(positions), dtype=np.int32)
        for defect in defects:
            try:
                d_row = int(defect["row"])
                d_col = int(defect["col"])
                d_type = str(defect.get("type", "vacancy"))

                if len(positions) > 0:
                    dists = np.sqrt(
                        (positions[:, 0] - d_row) ** 2 +
                        (positions[:, 1] - d_col) ** 2
                    )
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < 10:  # within 10 pixels
                        if d_type == "vacancy":
                            defect_labels[nearest] = 1
                        elif d_type == "substitution":
                            defect_labels[nearest] = 2
            except (KeyError, ValueError):
                continue

        n_vac = int(np.sum(defect_labels == 1))
        n_sub = int(np.sum(defect_labels == 2))

        # Track phase
        self._phase_intensities.append(mean_intensity)
        self._phase_labels.append(phase_label)
        if tile_key is not None:
            self._phase_tile_keys.append(tile_key)

        # Convert positions to world nm if tile position given
        if tile_position_nm is not None and len(positions) > 0:
            tile_h, tile_w = tile.shape
            world_positions = np.column_stack([
                tile_position_nm[0] + (positions[:, 1] - tile_w / 2) * pixel_size_nm,
                tile_position_nm[1] + (positions[:, 0] - tile_h / 2) * pixel_size_nm,
            ])
            self._all_atom_positions_nm.append(world_positions)

        # Record defect labels
        for lbl in defect_labels:
            self._all_defect_labels.append(DEFECT_CLASSES[lbl])

        result = AnalysisResult(
            atom_positions=positions,
            atom_intensities=intensities,
            n_atoms=len(positions),
            n_vacancies=n_vac,
            n_substitutions=n_sub,
            mean_intensity=float(intensities.mean()) if len(intensities) > 0 else 0.0,
            intensity_std=float(intensities.std()) if len(intensities) > 1 else 0.0,
            phase_label=phase_label,
        )
        self._tile_results.append(result)
        return result

    def get_defect_census(self) -> dict[str, int]:
        total_vac = sum(r.n_vacancies for r in self._tile_results)
        total_sub = sum(r.n_substitutions for r in self._tile_results)
        census = {}
        if total_vac > 0:
            census["vacancy"] = total_vac
        if total_sub > 0:
            census["substitution"] = total_sub
        return census

    def get_corrected_tile_phases(self) -> dict[tuple[int, int], int] | None:
        """Return tile-level phase labels from VLM predictions.

        Uses VLM's phase predictions directly (unlike heuristic's
        median-split + bimodality test). Returns None if VLM always
        predicts the same phase (no evidence of multi-phase).
        """
        if len(self._phase_tile_keys) < 4:
            return None

        unique = set(self._phase_labels[:len(self._phase_tile_keys)])
        if len(unique) < 2:
            return None

        return {
            key: label
            for key, label in zip(self._phase_tile_keys, self._phase_labels)
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Per-VLM parse-failure diagnostics aggregated over the episode.

        Used by H6/V23 to populate the VLM diagnostics table in Appendix O.
        ``successful_calls`` counts unique calls that ultimately succeeded
        (one per ``analyze_tile`` invocation that produced a valid JSON);
        ``parse_failures`` counts attempts that raised ``JSONDecodeError``;
        ``api_errors`` counts non-parse exceptions.
        """
        successful_calls = sum(1 for d in self._diagnostics if d["success"])
        parse_failures = sum(
            1
            for d in self._diagnostics
            if not d["success"]
            and d["parse_failure_reason"]
            and d["parse_failure_reason"].startswith("JSONDecodeError")
        )
        api_errors = sum(
            1
            for d in self._diagnostics
            if not d["success"]
            and d["parse_failure_reason"]
            and not d["parse_failure_reason"].startswith("JSONDecodeError")
        )
        retry_distribution: dict[int, int] = {}
        for d in self._diagnostics:
            if d["success"]:
                retry_distribution[d["attempt"]] = (
                    retry_distribution.get(d["attempt"], 0) + 1
                )
        return {
            "model": self.model,
            "total_attempts": len(self._diagnostics),
            "successful_calls": successful_calls,
            "parse_failures": parse_failures,
            "api_errors": api_errors,
            "retry_distribution": retry_distribution,
            "raw_log": list(self._diagnostics),
        }
