from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from stem_gym.agents.analyst import Analyst
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.baselines import Agent
from stem_gym.baselines.gp_ucb import GPUCBAgent
from stem_gym.envs.stem_env import STEMEnv, ActionType
from stem_gym.runner import run_episode
from stem_gym.tasks.defect_census import DefectCensusTask

WORLDS = [
    "srtio3_clustered_easy",
    "batio3_interface_easy",
    "sige_gradient_clustered_easy",
    "gan_easy",
]
POLICIES = [
    "raster_heuristic",
    "raster_trained",
    "random_trained",
    "gp_intensity_trained",
    "gp_defect_trained",
]


@dataclass(frozen=True)
class EvaluationConfig:
    acquisitions: int = 50
    dose_per_tile: float = 100.0
    score_every: int = 5

    def __post_init__(self):
        if (
            not isinstance(self.acquisitions, int)
            or isinstance(self.acquisitions, bool)
            or not isinstance(self.score_every, int)
            or isinstance(self.score_every, bool)
            or self.acquisitions < 1
            or self.score_every < 1
            or not np.isfinite(self.dose_per_tile)
            or self.dose_per_tile <= 0
        ):
            raise ValueError("Evaluation counts and dose must be positive")


class ReferenceAgent(Agent):


    def __init__(self, policy: str, analyst, config: EvaluationConfig):
        if policy not in POLICIES:
            raise ValueError(f"Unknown reference policy: {policy}")
        self.policy, self.analyst, self.config = policy, analyst, config
        self.navigator = GPUCBAgent(initial_random=6)

    def reset(self, obs, info, *, seed=None):
        self.analyst.reset()
        self.navigator.reset(obs, info, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.geometry = info["geometry"]
        self.tiles = []
        self._observed_acquisitions = 0
        self.navigate = True

    def act(self, obs, info):
        if len(self.tiles) >= self.config.acquisitions:
            return self._make_done_action()
        if not self.navigate:
            self.navigate = True
            return {"action_type": ActionType.ACQUIRE}
        candidates = np.argwhere(obs["visited_mask"] < 0.5)
        if len(candidates) == 0:
            return self._make_done_action()
        if self.policy.startswith("raster"):
            tile = tuple(map(int, candidates[0]))
        elif self.policy.startswith("random"):
            tile = tuple(map(int, candidates[self.rng.integers(len(candidates))]))
        else:
            tile = self.navigator.select_tile(obs)
        self.navigate = False
        return {
            "action_type": ActionType.NAVIGATE,
            "position": np.asarray(self.geometry.tile_to_nm(*tile)),
        }

    def observe(self, obs, info):
        count = info["acquisitions"]
        if count <= self._observed_acquisitions:
            return
        tile = info["tile_position"]
        kwargs = dict(
            tile_position_nm=self.geometry.tile_to_nm(*tile),
            pixel_size_nm=self.geometry.pixel_size_nm,
        )
        if isinstance(self.analyst, Analyst):
            kwargs["tile_key"] = tile
        result = self.analyst.analyze_tile(obs["tile"], **kwargs)
        self.tiles.append(list(tile))
        self._observed_acquisitions = count
        if self.policy.startswith("gp"):
            signal = (
                result.n_vacancies + result.n_substitutions
                if "defect" in self.policy
                else float(np.std(obs["tile"]))
            )
            self.navigator.update(
                tile, signal, fit=len(self.tiles) < self.config.acquisitions
            )

    def get_findings(self):
        if isinstance(self.analyst, Analyst):
            return self.analyst.get_findings(self.geometry.grid_shape)
        return {
            "defect_census": self.analyst.get_defect_census(),
            "defect_positions": [],
        }


def evaluate_episode(
    job: dict, root: Path, config: EvaluationConfig = EvaluationConfig()
) -> dict:
    """Run one reference configuration, always closing the world on failure."""
    import time

    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    trained = job["policy"] != "raster_heuristic"
    analyst = (
        Analyst.from_checkpoint(root / "checkpoints", device="cpu")
        if trained
        else SimpleAnalyst()
    )
    if trained and any(
        getattr(analyst, name) is None
        for name in ("atom_finder", "defect_classifier", "phase_identifier")
    ):
        raise ValueError("All three reference analyst checkpoints are required")
    env = STEMEnv(
        root / "data/simulated" / (job["world"] + ".h5"),
        task=DefectCensusTask(),
        total_budget=config.acquisitions * config.dose_per_tile,
        dose_per_acquire=config.dose_per_tile,
        max_steps=max(200, config.acquisitions * 2),
    )
    agent = ReferenceAgent(job["policy"], analyst, config)
    start = time.perf_counter()
    try:
        result = run_episode(
            env, agent, seed=job["seed"], score_every_n_acquires=config.score_every
        )
        if (
            len(agent.tiles) != config.acquisitions
            or len({tuple(t) for t in agent.tiles}) != config.acquisitions
        ):
            raise RuntimeError(
                "Reference evaluation did not acquire the expected distinct tiles"
            )
        return {
            **job,
            "dec_auc": result.dec.auc,
            "final_score": float(result.final_score),
            "tiles": agent.tiles,
            "analysed_tiles": analyst.analysed_tiles,
            "dose_used": result.total_dose_used,
            "score_log": [
                {"dose_used": e["dose_used"], "task_score": e["task_score"]}
                for e in result.episode_log
                if e["task_score"] is not None
            ],
            "dec_dose_fractions": result.dec.dose_fractions.tolist(),
            "dec_scores": result.dec.scores.tolist(),
            "models_active_at_end": {
                k: getattr(analyst, k, None) is not None
                for k in ("atom_finder", "defect_classifier", "phase_identifier")
            },
            "elapsed_seconds": time.perf_counter() - start,
        }
    finally:
        env.close()
