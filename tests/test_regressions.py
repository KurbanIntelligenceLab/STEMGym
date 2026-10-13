"""Regression checks for acquisition order, resets, and preservation boundaries."""

from pathlib import Path
import numpy as np
import pytest
import torch
from stem_gym.envs.stem_env import STEMEnv
from stem_gym.baselines.raster_scan import RasterAgent
from stem_gym.baselines.random_agent import RandomAgent
from stem_gym.baselines.gp_ucb import GPUCBAgent
from stem_gym.agents.analyst import Analyst
from stem_gym.agents.simple_analyst import SimpleAnalyst
from stem_gym.runner import run_episode
from stem_gym.tasks import Task
from stem_gym.artifacts import verify_preservation, writable_output
from stem_gym.evaluation import ReferenceAgent, EvaluationConfig

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data/simulated/test_world.h5"


class AcquisitionScore(Task):
    name = "test"

    def __init__(self, agent):
        self.agent = agent

    def score(self, findings, ground_truth):
        return self.agent.analyst.analysed_tiles / 5


@pytest.mark.parametrize("factory", [RasterAgent, RandomAgent, GPUCBAgent])
def test_final_acquisition_is_analysed_before_score(factory):
    agent = factory()
    with_env = STEMEnv(
        WORLD, total_budget=500, max_steps=20, task=AcquisitionScore(agent)
    )
    try:
        result = run_episode(with_env, agent, seed=13)
        assert agent.analyst.analysed_tiles == with_env.state.acquisitions == 5
        assert result.final_score == 1
    finally:
        with_env.close()


@pytest.mark.parametrize("factory", [RandomAgent, GPUCBAgent])
def test_reused_agent_repeats_seed(factory):
    env = STEMEnv(WORLD, total_budget=700, max_steps=20)
    agent = factory()
    try:
        one = run_episode(env, agent, seed=7)
        two = run_episode(env, agent, seed=7)
        assert [e["position"] for e in one.episode_log] == [
            e["position"] for e in two.episode_log
        ]
        np.testing.assert_array_equal(one.dec.scores, two.dec.scores)
    finally:
        env.close()


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": 1, "dwell": [-1.0]},
        {"action_type": 1, "dwell": [0.0]},
        {"action_type": 1, "dwell": [np.nan]},
        {"action_type": 1, "zoom": [np.inf]},
        {"action_type": 1, "zoom": [5.0]},
        {"action_type": 1, "zoom": 2.0},
        {"action_type": 0, "position": [np.nan, 0]},
        {"action_type": 0, "position": [0]},
        {"action_type": -1},
        {"action_type": 1.5},
        {"action_type": 2, "finding_type": -1},
    ],
)
def test_invalid_action_is_atomic(action):
    env = STEMEnv(WORLD, total_budget=500)
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step(action)
        assert (
            env.state.dose_used == env.state.step_count == env.state.acquisitions == 0
        )
        assert env.observation_space.contains(env._get_obs())
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_budget": 0},
        {"total_budget": np.nan},
        {"dose_per_acquire": -1},
        {"max_steps": 0},
        {"max_steps": 1.5},
        {"overview_size": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        STEMEnv(WORLD, **kwargs)


def test_revisits_trigger_scoring():
    class Revisit(RasterAgent):
        def act(self, obs, info):
            return {"action_type": 1}

    agent = Revisit()
    env = STEMEnv(WORLD, total_budget=500, task=AcquisitionScore(agent))
    try:
        result = run_episode(env, agent, score_every_n_acquires=1)
        assert env.state.acquisitions == 5 and env.state.tiles_acquired == 1
        assert [e["task_score"] for e in result.episode_log] == [
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
        with pytest.raises(RuntimeError):
            env.step({"action_type": 1})
    finally:
        env.close()


def test_analyst_reset_restores_fallback_heads():
    class ConstantPhase(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[10.0, 0.0]]).repeat(x.shape[0], 1)

    model = ConstantPhase()
    analyst = Analyst(phase_identifier=model, device="cpu")
    for _ in range(5):
        analyst.identify_phase(np.ones((128, 128), dtype=np.float32))
    assert analyst.phase_identifier is None
    analyst.reset()
    assert analyst.phase_identifier is model
    assert analyst.analysed_tiles == 0


def test_empty_tiles_are_recorded():
    analyst = SimpleAnalyst()
    analyst.analyze_tile(np.zeros((128, 128)))
    assert analyst.analysed_tiles == 1


def test_manuscript_evidence_preserved():
    assert verify_preservation(ROOT / "results/manuscript")["evidence_files"] >= 10
    for path in [
        "results/manuscript/reference_results.json",
        "development/protected.pdf",
    ]:
        with pytest.raises(ValueError):
            writable_output(ROOT / path, ROOT)


@pytest.mark.parametrize(
    "policy",
    ["raster_heuristic", "random_trained", "gp_intensity_trained", "gp_defect_trained"],
)
def test_reference_policy_uses_shared_synchronous_loop(policy):
    agent = ReferenceAgent(policy, SimpleAnalyst(), EvaluationConfig(acquisitions=5))
    env = STEMEnv(WORLD, total_budget=500, task=AcquisitionScore(agent))
    try:
        result = run_episode(env, agent, seed=5)
        assert agent.analyst.analysed_tiles == 5
        assert len({tuple(t) for t in agent.tiles}) == 5
        assert result.final_score == 1
    finally:
        env.close()


def test_runner_rejects_nonterminating_declarations():
    class DeclaringAgent(RasterAgent):
        def act(self, obs, info):
            return {"action_type": 2}

    env = STEMEnv(WORLD, max_steps=2)
    try:
        with pytest.raises(RuntimeError, match="interaction limit"):
            run_episode(env, DeclaringAgent())
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"acquisitions": 1.5},
        {"acquisitions": True},
        {"score_every": 0},
        {"dose_per_tile": float("nan")},
    ],
)
def test_evaluation_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


def test_download_verifies_before_publishing(tmp_path, monkeypatch):
    import io
    import hashlib
    import stem_gym.data as data

    expected = b"expected payload"
    manifest = {
        "id": "fixture",
        "sha": "pinned",
        "siblings": [
            {
                "rfilename": "worlds/test.h5",
                "lfs": {"sha256": hashlib.sha256(expected).hexdigest()},
            }
        ],
    }
    destination = tmp_path / "test.h5"
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="Download does not match"):
        data.download_verified(manifest, "worlds/test.h5", destination)
    assert not destination.exists() and not list(tmp_path.iterdir())
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: io.BytesIO(expected))
    data.download_verified(manifest, "worlds/test.h5", destination)
    assert destination.read_bytes() == expected
    destination.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="Existing file"):
        data.download_verified(manifest, "worlds/test.h5", destination)
    assert destination.read_bytes() == b"corrupt cache"


def test_preserved_study_verifies_without_manuscript():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / 'reproducibility/verify_reference.py')],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '220 episodes' in result.stdout


@pytest.mark.parametrize('delta,accepted', [(1e-15, True), (1e-6, False)])
def test_reference_curve_precision(monkeypatch, delta, accepted):
    """Accept roundoff while still rejecting changed scientific values."""
    import runpy

    monkeypatch.syspath_prepend(str(ROOT / 'reproducibility'))
    verifier = runpy.run_path(str(ROOT / 'reproducibility/verify_reference.py'))
    interpolate = np.interp

    def perturbed_interpolation(*args, **kwargs):
        scores = interpolate(*args, **kwargs)
        scores[10] += delta
        return scores

    monkeypatch.setattr(np, 'interp', perturbed_interpolation)
    if accepted:
        assert verifier['verify'](ROOT / 'results/manuscript/reference_results.json')['episodes'] == 220
    else:
        with pytest.raises(ValueError, match='differ from stored curves'):
            verifier['verify'](ROOT / 'results/manuscript/reference_results.json')
