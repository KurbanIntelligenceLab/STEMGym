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
    plant_vacancies_weighted,
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


class TestClusteredUtilities:
    """Tests for plant_vacancies_weighted (no abTEM required)."""

    pytestmark = []  # Override module-level slow marker

    def test_correct_vacancy_count(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (5, 5, 5)
        rng = np.random.default_rng(42)
        n_si = sum(1 for s in si.get_chemical_symbols() if s == "Si")
        weights = rng.random(n_si)
        modified, vac_idx = plant_vacancies_weighted(
            si, "Si", rate=0.1, weights=weights, rng=rng,
        )
        expected = max(1, int(n_si * 0.1))
        assert len(vac_idx) == expected
        assert len(modified) == len(si) - expected

    def test_weighted_bias(self):
        """Higher-weighted atoms should be selected more often than uniform."""
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (8, 8, 8)
        n_si = sum(1 for s in si.get_chemical_symbols() if s == "Si")

        # Create weights: first half gets weight 10, second half gets weight 0.01
        weights = np.ones(n_si)
        half = n_si // 2
        weights[:half] = 10.0
        weights[half:] = 0.01

        symbols = np.array(si.get_chemical_symbols())
        si_indices = np.where(symbols == "Si")[0]

        rng = np.random.default_rng(123)
        _, vac_idx = plant_vacancies_weighted(
            si, "Si", rate=0.05, weights=weights, rng=rng, cluster_fraction=0.8,
        )

        # Map vacancy indices back to local Si indices
        si_idx_set = set(si_indices.tolist())
        si_local = {idx: local for local, idx in enumerate(si_indices)}
        n_first_half = sum(1 for idx in vac_idx if si_local[idx] < half)

        # With 80% cluster fraction and 10:0.01 weight ratio,
        # vast majority of vacancies should be in the first half
        assert n_first_half > len(vac_idx) * 0.5, (
            f"Expected >50% of vacancies in high-weight half, got "
            f"{n_first_half}/{len(vac_idx)}"
        )

    def test_reproducibility(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (4, 4, 4)
        n_si = sum(1 for s in si.get_chemical_symbols() if s == "Si")
        weights = np.random.default_rng(0).random(n_si)

        _, vac1 = plant_vacancies_weighted(
            si, "Si", rate=0.1, weights=weights,
            rng=np.random.default_rng(42),
        )
        _, vac2 = plant_vacancies_weighted(
            si, "Si", rate=0.1, weights=weights,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(sorted(vac1), sorted(vac2))

    def test_weights_length_mismatch_raises(self):
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.431) * (3, 3, 3)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="weights length"):
            plant_vacancies_weighted(
                si, "Si", rate=0.1, weights=np.ones(5), rng=rng,
            )


class TestSrTiO3Clustered:
    """Tests for the SrTiO3 clustered-defect generator."""

    pytestmark = []  # Override module-level slow marker for non-pipeline tests

    def test_grain_boundary_weights(self):
        from generate_srtio3 import build_srtio3_unit_cell
        from generate_srtio3_clustered import compute_grain_boundary_weights

        uc = build_srtio3_unit_cell()
        sc = build_supercell(uc, (40.0, 40.0, 20.0))
        weights = compute_grain_boundary_weights(sc, line_width_frac=0.05)
        n_o = sum(1 for s in sc.get_chemical_symbols() if s == "O")
        assert len(weights) == n_o
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_clustering_near_boundary(self):
        """~70% of vacancies should be within 1 sigma of the grain boundary."""
        from generate_srtio3 import build_srtio3_unit_cell
        from generate_srtio3_clustered import (
            compute_grain_boundary_weights,
            GB_START, GB_END,
        )

        uc = build_srtio3_unit_cell()
        sc = build_supercell(uc, (60.0, 60.0, 20.0))
        line_width_frac = 0.05

        weights = compute_grain_boundary_weights(sc, line_width_frac)
        rng = np.random.default_rng(42)
        _, vac_idx = plant_vacancies_weighted(
            sc, "O", rate=0.03, weights=weights, rng=rng, cluster_fraction=0.70,
        )

        # Compute distance of each vacancy to the grain boundary
        positions = sc.get_positions()
        cell_x = sc.cell[0, 0]
        cell_y = sc.cell[1, 1]
        p0 = GB_START * np.array([cell_x, cell_y])
        p1 = GB_END * np.array([cell_x, cell_y])
        line_vec = p1 - p0
        line_dir = line_vec / np.linalg.norm(line_vec)

        diag_len = np.sqrt(cell_x**2 + cell_y**2)
        sigma = line_width_frac * diag_len

        vac_positions = positions[vac_idx, :2]
        dp = vac_positions - p0
        distances = np.abs(dp[:, 0] * line_dir[1] - dp[:, 1] * line_dir[0])

        frac_within_sigma = np.mean(distances < sigma)
        # With 70% cluster_fraction and Gaussian weights, we expect a clear
        # bias towards the boundary. Use a lenient threshold for statistical test.
        assert frac_within_sigma > 0.25, (
            f"Expected >25% of vacancies within 1σ of boundary, got "
            f"{frac_within_sigma:.1%}"
        )

    @pytest.mark.slow
    def test_pipeline(self, tmp_path):
        from generate_srtio3 import build_srtio3_unit_cell
        from generate_srtio3_clustered import compute_grain_boundary_weights

        uc = build_srtio3_unit_cell()
        sc = build_supercell(uc, (20.0, 20.0, 40.0))
        weights = compute_grain_boundary_weights(sc, line_width_frac=0.05)
        rng = np.random.default_rng(42)
        defected, vac_idx = plant_vacancies_weighted(
            sc, "O", rate=0.05, weights=weights, rng=rng,
        )
        gt = record_ground_truth(sc, vacancy_indices=vac_idx)

        config = _mini_sim_config(
            scenario="srtio3_clustered",
            phonon_sigmas={"Sr": 0.084, "Ti": 0.060, "O": 0.085},
        )
        image = run_stem_simulation(defected, config)
        assert image.ndim == 2

        out = tmp_path / "srtio3_clustered_test.h5"
        write_world_hdf5(out, image, gt, config)
        assert out.exists()


class TestBaTiO3Interface:
    """Tests for the BaTiO3 interface generator."""

    pytestmark = []

    def test_boundary_distance_weights(self):
        from generate_batio3 import build_batio3_cell, make_phase_boundary_mask, build_composite_structure
        from generate_batio3_interface import compute_boundary_distance_weights

        phase_map = make_phase_boundary_mask((200, 200), n_boundaries=1, waviness=0.0)
        composite, _ = build_composite_structure(20.0, 20.0, phase_map, 0.1)
        weights = compute_boundary_distance_weights(
            composite, phase_map, boundary_sigma_px=50,
        )
        n_o = sum(1 for s in composite.get_chemical_symbols() if s == "O")
        assert len(weights) == n_o
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_defects_near_boundary(self):
        """Defects should be closer to boundary than random expectation."""
        from generate_batio3 import build_batio3_cell, make_phase_boundary_mask, build_composite_structure
        from generate_batio3_interface import compute_boundary_distance_weights
        from scipy.ndimage import distance_transform_edt

        phase_map = make_phase_boundary_mask((200, 200), n_boundaries=1, waviness=0.0)
        composite, _ = build_composite_structure(30.0, 20.0, phase_map, 0.1)

        weights = compute_boundary_distance_weights(
            composite, phase_map, boundary_sigma_px=50,
        )
        rng = np.random.default_rng(42)
        _, vac_idx = plant_vacancies_weighted(
            composite, "O", rate=0.04, weights=weights,
            rng=rng, cluster_fraction=0.70,
        )

        # Compute mean distance-to-boundary for vacancies vs all O atoms
        boundary_mask = np.zeros_like(phase_map, dtype=bool)
        boundary_mask[:-1, :] |= phase_map[:-1, :] != phase_map[1:, :]
        boundary_mask[:, :-1] |= phase_map[:, :-1] != phase_map[:, 1:]
        dist_field = distance_transform_edt((~boundary_mask).astype(float))

        positions = composite.get_positions()
        symbols = np.array(composite.get_chemical_symbols())
        o_mask = symbols == "O"
        o_pos = positions[o_mask, :2]
        cell_x, cell_y = composite.cell[0, 0], composite.cell[1, 1]
        map_h, map_w = phase_map.shape

        def mean_boundary_dist(pos_xy):
            px = (pos_xy[:, 0] / cell_x * map_w).astype(int) % map_w
            py = (pos_xy[:, 1] / cell_y * map_h).astype(int) % map_h
            return dist_field[py, px].mean()

        vac_dist = mean_boundary_dist(positions[vac_idx, :2])
        all_o_dist = mean_boundary_dist(o_pos)

        assert vac_dist < all_o_dist, (
            f"Vacancy mean dist ({vac_dist:.1f}px) should be less than "
            f"all-O mean dist ({all_o_dist:.1f}px)"
        )

    @pytest.mark.slow
    def test_pipeline(self, tmp_path):
        from generate_batio3 import make_phase_boundary_mask, build_composite_structure
        from generate_batio3_interface import compute_boundary_distance_weights

        phase_map = make_phase_boundary_mask((100, 100), n_boundaries=1)
        composite, _ = build_composite_structure(20.0, 40.0, phase_map, 0.1)
        weights = compute_boundary_distance_weights(composite, phase_map, 50)
        rng = np.random.default_rng(42)
        defected, vac_idx = plant_vacancies_weighted(
            composite, "O", rate=0.04, weights=weights, rng=rng,
        )
        gt = record_ground_truth(composite, vacancy_indices=vac_idx)
        gt.phase_map = phase_map

        config = _mini_sim_config(
            scenario="batio3_interface",
            phonon_sigmas={"Ba": 0.080, "Ti": 0.055, "O": 0.090},
        )
        image = run_stem_simulation(defected, config)
        assert image.ndim == 2

        out = tmp_path / "batio3_interface_test.h5"
        write_world_hdf5(out, image, gt, config)
        assert out.exists()


class TestSiGeGradientClustered:
    """Tests for the SiGe gradient-clustered generator."""

    pytestmark = []

    def test_ge_in_target_quadrant(self):
        """~70% of Ge should be in the target quadrant."""
        from generate_sige import build_si_supercell_110
        from generate_sige_gradient_clustered import (
            apply_ge_gradient_clustered,
            QUADRANT_CENTER,
        )

        sc = build_si_supercell_110(30.0, 30.0)
        rng = np.random.default_rng(42)
        modified, sub_idx, _ = apply_ge_gradient_clustered(
            sc, ge_max=0.5, cluster_fraction=0.70,
            quadrant_sigma_frac=0.15, rng=rng,
        )

        # Check what fraction of Ge is in the target quadrant
        positions = modified.get_positions()
        symbols = modified.get_chemical_symbols()
        cell_x = modified.cell[0, 0]
        cell_y = modified.cell[1, 1]

        n_ge_total = 0
        n_ge_quadrant = 0
        for pos, sym in zip(positions, symbols):
            if sym == "Ge":
                n_ge_total += 1
                fx = pos[0] / cell_x
                fy = pos[1] / cell_y
                # "Target quadrant" = within 0.25 of center in both dims
                if (abs(fx - QUADRANT_CENTER[0]) < 0.25 and
                        abs(fy - QUADRANT_CENTER[1]) < 0.25):
                    n_ge_quadrant += 1

        if n_ge_total > 0:
            frac_in_quadrant = n_ge_quadrant / n_ge_total
            # With cluster_fraction=0.70 and Gaussian, expect clear concentration
            assert frac_in_quadrant > 0.30, (
                f"Expected >30% of Ge in target quadrant, got "
                f"{frac_in_quadrant:.1%} ({n_ge_quadrant}/{n_ge_total})"
            )

    def test_clustered_phase_map(self):
        from generate_sige_gradient_clustered import make_clustered_phase_map
        pm = make_clustered_phase_map((200, 200), quadrant_sigma_frac=0.15)
        assert pm.shape == (200, 200)
        assert set(np.unique(pm)).issubset({0, 1, 2})
        # All three phases should be present
        assert np.sum(pm == 0) > 0
        assert np.sum(pm == 2) > 0  # high-Ge core

    def test_gradient_output_shape(self):
        from generate_sige import build_si_supercell_110
        from generate_sige_gradient_clustered import apply_ge_gradient_clustered

        sc = build_si_supercell_110(20.0, 30.0)
        rng = np.random.default_rng(42)
        modified, sub_idx, ge_frac_2d = apply_ge_gradient_clustered(
            sc, ge_max=0.5, cluster_fraction=0.70,
            quadrant_sigma_frac=0.15, rng=rng,
        )
        assert len(modified) == len(sc)  # substitutions don't remove atoms
        assert ge_frac_2d.shape == (50, 50)
        assert ge_frac_2d.min() >= 0.0
        assert ge_frac_2d.max() <= 1.0

    @pytest.mark.slow
    def test_pipeline(self, tmp_path):
        from generate_sige import build_si_supercell_110
        from generate_sige_gradient_clustered import (
            apply_ge_gradient_clustered, make_clustered_phase_map,
        )

        sc = build_si_supercell_110(20.0, 40.0)
        rng = np.random.default_rng(42)
        modified, sub_idx, ge_frac_2d = apply_ge_gradient_clustered(
            sc, ge_max=0.5, cluster_fraction=0.70,
            quadrant_sigma_frac=0.15, rng=rng,
        )
        gt = record_ground_truth(sc, substitution_indices=sub_idx)
        gt.extra["ge_fraction_2d"] = ge_frac_2d
        gt.phase_map = make_clustered_phase_map((100, 100), 0.15)

        config = _mini_sim_config(
            scenario="sige_gradient_clustered",
            phonon_sigmas={"Si": 0.078, "Ge": 0.065},
        )
        image = run_stem_simulation(modified, config)

        from scipy.ndimage import zoom as sz
        gt.phase_map = sz(gt.phase_map, (image.shape[0]/100, image.shape[1]/100), order=0).astype(np.int32)

        out = tmp_path / "sige_gradient_clustered_test.h5"
        write_world_hdf5(out, image, gt, config)
        assert out.exists()


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
