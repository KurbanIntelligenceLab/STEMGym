"""Bounded training checks; convergence tests remain in test_analyst.py."""
from pathlib import Path

import numpy as np
import pytest
import torch

from stem_gym.agents.models import AtomFinderEnsemble, DefectClassifierCNN
from stem_gym.scripts.train_analyst import (
    TileDataset, train_atom_finder, train_defect_classifier,
)


@pytest.mark.training
@pytest.mark.training_smoke
@pytest.mark.parametrize('kind', ['atom_finder', 'defect_classifier'])
def test_training_checkpoint_roundtrip(kind, tmp_path):
    world = Path(__file__).resolve().parents[1] / 'data/simulated/test_world.h5'
    assert world.exists(), 'Run reproducibility/prepare_test_data.py first'
    dataset = TileDataset([world], augment=False, max_tiles_per_world=8)
    assert len(dataset) == 8
    torch.manual_seed(42)
    constructor = (lambda: AtomFinderEnsemble(n_members=2)) if kind == 'atom_finder' else DefectClassifierCNN
    initial = constructor()
    before = {name: value.detach().clone() for name, value in initial.named_parameters()}
    torch.manual_seed(42)
    if kind == 'atom_finder':
        model, metadata = train_atom_finder(dataset, n_members=2, epochs=1, batch_size=4)
        curves = metadata['loss_curves']['member_0']
        assert len(metadata['loss_curves']) == 2
        for member in metadata['loss_curves'].values():
            assert len(member['train']) == len(member['val']) == 1
            assert np.isfinite(member['train'] + member['val']).all()
        sample = dataset[0]['tile'].unsqueeze(0)
    else:
        model, metadata = train_defect_classifier(dataset, epochs=1, batch_size=8, max_patches=32)
        assert 5 <= metadata['n_patches'] <= 32
        curves = metadata['loss_curves']
        sample = torch.zeros(1, 1, 32, 32)
    for split in ['train', 'val']:
        assert len(curves[split]) == 1
        assert np.isfinite(curves[split]).all()
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    assert all(torch.isfinite(value).all() for value in model.parameters())
    checkpoint = tmp_path / f'{kind}.pt'
    torch.save(model.state_dict(), checkpoint)
    restored = constructor()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected, actual = model(sample), restored(sample)
    if isinstance(expected, tuple):
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
