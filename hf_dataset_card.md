---
license: cc-by-4.0
task_categories:
  - reinforcement-learning
tags:
  - stem-microscopy
  - electron-microscopy
  - materials-science
  - gymnasium
  - benchmark
  - dose-efficiency
  - autonomous-microscopy
pretty_name: STEMGym
size_categories:
  - 1K<n<10K
---

# STEMGym: A Gymnasium Environment for Benchmarking Dose-Efficient Autonomous Scanning Transmission Electron Microscopy

This dataset accompanies the STEMGym benchmark — a Gymnasium-based environment for evaluating autonomous dose-efficient scanning transmission electron microscopy (STEM) agents.

## Dataset Description

STEMGym provides simulated STEM specimens as HDF5 world files. Each world contains:

- **Overview image**: Low-magnification survey of the full specimen
- **Tile grid**: High-resolution STEM-HAADF images (128x128 px tiles, 4px overlap, stride=124) arranged in an 8x8 grid
- **Ground truth annotations**: Atom positions, defect types, and phase maps for scoring
- **Metadata**: Pixel size, accelerating voltage, detector geometry, material parameters

### Materials

| Material | Zone Axis | FOV | Defect Types | Task |
|----------|-----------|-----|-------------|------|
| SrTiO3 | [001] | ~100 nm | O vacancies clustered along grain boundary | Defect Census |
| BaTiO3 | [001] | ~100 nm | Cubic/tetragonal phase boundaries + O vacancies | Phase Mapping + Defect Census |
| SiGe | [110] | ~50 nm | Ge substitutions concentrated in one quadrant | Targeted Characterization |
| GaN | [11-20] | ~80 nm | InGaN quantum-well substitutions | Defect Census |
| Pt nanoparticles | — | ~60 nm | Pt nanoparticles on amorphous carbon | Particle Census |

### Difficulty Levels

Each material is provided at three difficulty levels:

| Level | Vacancy Rate | Phonon Configs | Dose (e-/A^2) |
|-------|-------------|----------------|---------------|
| Easy | 5% | 4 | 1e4 |
| Medium | 3% | 8 | 5e3 |
| Hard | 1% | 16 | 1e3 |

## Files

### Simulated Worlds (`worlds/`)

| File | Material | Difficulty | Size |
|------|----------|-----------|------|
| `test_world.h5` | Synthetic (Gaussian blobs) | — | 9.7 MB |
| `replay_world.h5` | Synthetic (replay validation) | — | 2.5 MB |
| `srtio3_clustered_easy.h5` | SrTiO3 | Easy | 884 MB |
| `srtio3_clustered_medium.h5` | SrTiO3 | Medium | 884 MB |
| `srtio3_clustered_hard.h5` | SrTiO3 | Hard | 885 MB |
| `batio3_interface_easy.h5` | BaTiO3 | Easy | 911 MB |
| `batio3_interface_medium.h5` | BaTiO3 | Medium | 911 MB |
| `batio3_interface_hard.h5` | BaTiO3 | Hard | 911 MB |
| `sige_gradient_clustered_easy.h5` | SiGe | Easy | 150 MB |
| `sige_gradient_clustered_medium.h5` | SiGe | Medium | 148 MB |
| `sige_gradient_clustered_hard.h5` | SiGe | Hard | 147 MB |
| `gan_easy.h5` | GaN | Easy | 451 MB |
| `gan_medium.h5` | GaN | Medium | 449 MB |
| `gan_hard.h5` | GaN | Hard | 448 MB |
| `pt_nanoparticles_easy.h5` | Pt | Easy | 129 MB |
| `pt_nanoparticles_medium.h5` | Pt | Medium | 129 MB |
| `pt_nanoparticles_hard.h5` | Pt | Hard | 129 MB |

### Model Checkpoints (`checkpoints/`)

| File | Model | Description | Size |
|------|-------|-------------|------|
| `atom_finder.pt` | AtomFinderUNet | Atomic column detection ensemble (3 members) | 88 MB |
| `defect_classifier.pt` | DefectClassifierCNN | Defect type classification | 1.4 MB |
| `phase_identifier.pt` | PhaseIdentifierResNet | Material phase identification | 7.4 MB |
| `dqn_agent.zip` | DQN (SB3) | RL navigation baseline | 0.8 MB |
| `ppo_agent.zip` | PPO (SB3) | RL navigation baseline | 1.0 MB |
| `sac_agent.zip` | SAC (SB3) | RL navigation baseline | 8.4 MB |

### Transfer Checkpoints (`checkpoints/transfer/`)

Material-specific analyst models trained on individual materials for transfer experiments:

| Directory | Contents |
|-----------|----------|
| `transfer/srtio3/` | atom_finder.pt, defect_classifier.pt, phase_identifier.pt, training_metadata.json |
| `transfer/batio3/` | atom_finder.pt, defect_classifier.pt, phase_identifier.pt, training_metadata.json |
| `transfer/sige/` | atom_finder.pt, defect_classifier.pt, phase_identifier.pt, training_metadata.json |
| `transfer/gan/` | atom_finder.pt, defect_classifier.pt, phase_identifier.pt, training_metadata.json |

## HDF5 World Format

Each world file follows this layout:

```
/metadata/
    pixel_size_nm          # Physical pixel size
    tile_size_px           # Tile dimensions (128)
    grid_shape             # Grid dimensions (rows, cols)
    fov_nm                 # Field of view in nm
    scenario               # Material/scenario name
    difficulty             # easy / medium / hard
/overview                  # (H, W) low-res overview image
/tiles/{row}_{col}         # (128, 128) float32 normalized HAADF tiles
/ground_truth/
    atom_positions         # (N, 2) in nm
    atom_types             # (N,) int [0=pristine, 1=vacancy, 2=substitution]
    defect_mask            # (N,) bool
    defect_types           # (N,) str ["pristine"/"vacancy"/"substitution"]
    phase_map              # (H, W) int32, optional
/valid_region              # (H, W) bool
```

## Usage

```bash
# Install STEMGym
pip install -e .

# Download this data
python stem_gym/scripts/download_data.py

# Run a benchmark
stemgym run --agent raster_equipped --task defect_census --world srtio3_clustered_easy --seeds 3
```

## License

This dataset is released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Citation

```bibtex
@inproceedings{anonymous2026stemgym,
  title={STEMGym: A Gymnasium Environment for Benchmarking Dose-Efficient Autonomous Scanning Transmission Electron Microscopy},
  author={Anonymous},
  year={2026}
}
```
