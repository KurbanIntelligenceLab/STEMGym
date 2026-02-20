# Clustered-Defect World Generation — Status

## Completed

### 1. `plant_vacancies_weighted` in `data/scripts/sim_utils.py`
- Added after `plant_vacancies()` (~55 lines)
- Splits vacancies into clustered (weighted sampling) + background (uniform)
- Same return interface as `plant_vacancies`

### 2. `data/scripts/generate_srtio3_clustered.py`
- O vacancies along diagonal grain boundary `(0.15,0.35) → (0.85,0.65)`
- Gaussian distance-to-line weights
- 3 difficulty levels (easy/medium/hard)
- Default output: `data/simulated/srtio3_clustered.h5`

### 3. `data/scripts/generate_batio3_interface.py`
- Phase boundaries + O vacancies clustered near boundaries
- Uses `scipy.ndimage.distance_transform_edt` for boundary distance
- 3 difficulty levels with varying boundary count, waviness, sigma
- Default output: `data/simulated/batio3_interface.h5`

### 4. `data/scripts/generate_sige_gradient_clustered.py`
- Ge concentrated in quadrant `(0.75, 0.75)` via 2D Gaussian
- Replaces 1D sigmoid with clustered blob
- 3-class phase map (pure Si / low-Ge transition / high-Ge core)
- Default output: `data/simulated/sige_gradient_clustered.h5`

### 5. Tests in `tests/test_generators.py`
- 4 new test classes, 11 tests total
- All 11 non-slow tests passing
- 4 pipeline tests marked `@pytest.mark.slow` (require abTEM)

---

## Upcoming (GPU machine)

### 6. Generate medium-difficulty worlds

```bash
cd /path/to/STEMGym

python data/scripts/generate_srtio3_clustered.py --device gpu
python data/scripts/generate_batio3_interface.py --device gpu
python data/scripts/generate_sige_gradient_clustered.py --device gpu
```

This produces:
- `data/simulated/srtio3_clustered.h5`
- `data/simulated/batio3_interface.h5`
- `data/simulated/sige_gradient_clustered.h5`

### 7. Smoke test with existing agents

```bash
stemgym run --agent random raster --task defect_census \
            --world srtio3_clustered batio3_interface sige_gradient_clustered \
            --seeds 1 --budget 3000
```

Verify the HDF5 files load correctly and agents can run against them.

### 8. Commit + push

```bash
git add data/scripts/sim_utils.py \
        data/scripts/generate_srtio3_clustered.py \
        data/scripts/generate_batio3_interface.py \
        data/scripts/generate_sige_gradient_clustered.py \
        tests/test_generators.py

git commit -m "Add clustered-defect world generators (SrTiO3 grain boundary, BaTiO3 interface, SiGe quadrant)"
git push
```

### 9. Upload worlds to HuggingFace

```bash
huggingface-cli upload johnpolat/STEMGym \
    data/simulated/srtio3_clustered.h5 worlds/srtio3_clustered.h5 \
    --repo-type=dataset

huggingface-cli upload johnpolat/STEMGym \
    data/simulated/batio3_interface.h5 worlds/batio3_interface.h5 \
    --repo-type=dataset

huggingface-cli upload johnpolat/STEMGym \
    data/simulated/sige_gradient_clustered.h5 worlds/sige_gradient_clustered.h5 \
    --repo-type=dataset
```

---

## No changes needed to
- `stem_gym/` (env, agents, tasks, metrics — all format-agnostic)
- `experiments/run_benchmark.py` (auto-discovers worlds)
- `experiments/configs/benchmark/` (already references these world names)
- Existing generation scripts (untouched)
