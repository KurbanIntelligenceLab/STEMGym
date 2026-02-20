"""Integration tests for scenario generators.

These tests verify that the abTEM simulation pipeline works end-to-end
with small supercells. They do NOT test physical accuracy (that requires
visual inspection of larger simulations).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "scripts"))

from sim_utils import (
    SimConfig,
    GroundTruth,
    build_supercell,
    plant_vacancies,
    plant_substitutions,
    record_ground_truth,
    run_stem_simulation,
    write_world_hdf5,
)
from stem_gym.envs.world import World
from stem_gym.envs.stem_env import STEMEnv, ActionType


# Mark all tests in this module as slow (require abTEM)
pytestmark = pytest.mark.slow


def _mini_sim_config(**overrides) -> SimConfig:
    """Config for fast mini simulations."""
    defaults = dict(
        scenario="test",
        sampling=0.05,
        num_phonon_configs=2,
        phonon_sigmas={"Si": 0.078},
        device="cpu",
        interpolation=2,
        tile_size_px=32,
        overlap_px=2,
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


class TestSimUtils:
    def test_plant_vacancies(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (3, 3, 3)
        rng = np.random.default_rng(42)
        modified, vac_idx = plant_vacancies(si, "Si", rate=0.1, rng=rng)
        assert len(modified) < len(si)
        assert len(vac_idx) > 0

    def test_plant_substitutions(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (3, 3, 3)
        rng = np.random.default_rng(42)
        modified, sub_idx = plant_substitutions(si, "Si", "Ge", rate=0.2, rng=rng)
        assert len(modified) == len(si)
        assert len(sub_idx) > 0
        symbols = modified.get_chemical_symbols()
        assert "Ge" in symbols

    def test_record_ground_truth(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (3, 3, 3)
        gt = record_ground_truth(si)
        assert gt.atom_positions_nm.shape == (len(si), 2)
        assert len(gt.atom_types) == len(si)
        assert all(t == "pristine" for t in gt.defect_types)


class TestSrTiO3Generator:
    def test_unit_cell(self):
        from generate_srtio3 import build_srtio3_unit_cell
        uc = build_srtio3_unit_cell()
        assert len(uc) == 5  # Sr + Ti + 3 O
        assert set(uc.get_chemical_symbols()) == {"Sr", "Ti", "O"}

    def test_pipeline(self, tmp_path):
        import abtem
        from generate_srtio3 import build_srtio3_unit_cell

        uc = build_srtio3_unit_cell()
        sc = build_supercell(uc, (20.0, 20.0, 40.0))
        rng = np.random.default_rng(42)
        defected, vac_idx = plant_vacancies(sc, "O", rate=0.05, rng=rng)
        gt = record_ground_truth(sc, vacancy_indices=vac_idx)

        config = _mini_sim_config(
            scenario="srtio3",
            phonon_sigmas={"Sr": 0.084, "Ti": 0.060, "O": 0.085},
        )
        image = run_stem_simulation(defected, config)
        assert image.ndim == 2
        assert image.max() > 0

        out = tmp_path / "srtio3_test.h5"
        write_world_hdf5(out, image, gt, config)
        assert out.exists()

        with World(out) as w:
            assert w.grid_shape[0] > 0
            tile = w.get_tile(0, 0)
            assert tile.shape == (32, 32)


class TestBaTiO3Generator:
    def test_unit_cells(self):
        from generate_batio3 import build_batio3_cell
        cubic = build_batio3_cell("cubic")
        tetra = build_batio3_cell("tetragonal")
        assert len(cubic) == 5
        assert len(tetra) == 5
        # Tetragonal c > cubic a
        assert tetra.cell[2, 2] > cubic.cell[2, 2]

    def test_phase_map(self):
        from generate_batio3 import make_phase_boundary_mask
        rng = np.random.default_rng(42)
        pm = make_phase_boundary_mask((100, 100), n_boundaries=1, waviness=0.0, rng=rng)
        assert pm.shape == (100, 100)
        assert set(np.unique(pm)) == {0, 1}
        # Both phases present
        assert np.sum(pm == 0) > 0
        assert np.sum(pm == 1) > 0

    def test_pipeline(self, tmp_path):
        from generate_batio3 import build_batio3_cell, make_phase_boundary_mask, build_composite_structure

        phase_map = make_phase_boundary_mask((100, 100), n_boundaries=1)
        composite, atom_phases = build_composite_structure(20.0, 40.0, phase_map, 0.1)
        gt = record_ground_truth(composite)
        gt.phase_map = phase_map

        config = _mini_sim_config(
            scenario="batio3",
            phonon_sigmas={"Ba": 0.080, "Ti": 0.055, "O": 0.090},
        )
        image = run_stem_simulation(composite, config)
        assert image.ndim == 2

        from scipy.ndimage import zoom as sz
        gt.phase_map = sz(phase_map, (image.shape[0]/100, image.shape[1]/100), order=0).astype(np.int32)

        out = tmp_path / "batio3_test.h5"
        write_world_hdf5(out, image, gt, config)

        with World(out) as w:
            gt_loaded = w.get_ground_truth()
            assert "phase_map" in gt_loaded


class TestSiGeGenerator:
    def test_si_110_supercell(self):
        from generate_sige import build_si_supercell_110
        sc = build_si_supercell_110(20.0, 40.0)
        assert len(sc) > 0
        assert all(s == "Si" for s in sc.get_chemical_symbols())

    def test_ge_gradient(self):
        from generate_sige import build_si_supercell_110, apply_ge_gradient
        sc = build_si_supercell_110(20.0, 40.0)
        rng = np.random.default_rng(42)
        modified, sub_idx, ge_profile = apply_ge_gradient(
            sc, interface_position=0.5, gradient_width_frac=0.3,
            ge_max_fraction=0.5, rng=rng,
        )
        assert "Ge" in modified.get_chemical_symbols()
        assert len(sub_idx) > 0
        assert len(ge_profile) == 100

    def test_pipeline(self, tmp_path):
        from generate_sige import build_si_supercell_110, apply_ge_gradient, make_interface_phase_map

        sc = build_si_supercell_110(20.0, 40.0)
        rng = np.random.default_rng(42)
        modified, sub_idx, ge_profile = apply_ge_gradient(
            sc, 0.5, 0.3, 0.5, rng,
        )
        gt = record_ground_truth(sc, substitution_indices=sub_idx)
        gt.extra["ge_fraction_profile"] = ge_profile
        gt.phase_map = make_interface_phase_map((100, 100), 0.5, 0.3)

        config = _mini_sim_config(
            scenario="sige",
            phonon_sigmas={"Si": 0.078, "Ge": 0.065},
        )
        image = run_stem_simulation(modified, config)

        from scipy.ndimage import zoom as sz
        gt.phase_map = sz(gt.phase_map, (image.shape[0]/100, image.shape[1]/100), order=0).astype(np.int32)

        out = tmp_path / "sige_test.h5"
        write_world_hdf5(out, image, gt, config)

        with World(out) as w:
            gt_loaded = w.get_ground_truth()
            assert "ge_fraction_profile" in gt_loaded


class TestEnvIntegration:
    """Test that generated worlds work with STEMEnv."""

    @pytest.fixture(params=["srtio3_test_mini", "batio3_test_mini", "sige_test_mini"])
    def world_path(self, request):
        p = Path("data/simulated") / f"{request.param}.h5"
        if not p.exists():
            pytest.skip(f"{p} not generated")
        return p

    def test_env_runs(self, world_path):
        env = STEMEnv(world_path, total_budget=1e4, max_steps=10)
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)

        for _ in range(5):
            action = {
                "action_type": ActionType.ACQUIRE,
                "position": np.array([0.1, 0.1], dtype=np.float32),
                "zoom": np.array([1.0], dtype=np.float32),
                "dwell": np.array([1.0], dtype=np.float32),
                "finding_type": 0,
            }
            obs, r, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs)
            if term or trunc:
                break
        env.close()
