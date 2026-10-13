"""Download and verify benchmark worlds and analyst checkpoints."""
import argparse
from pathlib import Path
from stem_gym.data import download_verified, release_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds', action='store_true')
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    manifest = release_manifest()
    all_files = not (args.worlds or args.checkpoints or args.test_only)
    for record in manifest['siblings']:
        remote = record['rfilename']
        name = Path(remote).name
        if remote.startswith('worlds/') and (all_files or args.worlds or args.test_only and name in ('test_world.h5', 'replay_world.h5')):
            destination = args.data_dir / ('replay' if name == 'replay_world.h5' else 'simulated') / name
        elif remote in ('checkpoints/atom_finder.pt', 'checkpoints/defect_classifier.pt', 'checkpoints/phase_identifier.pt') and (all_files or args.checkpoints):
            destination = args.checkpoint_dir / name
        else:
            continue
        download_verified(manifest, remote, destination)
        print(f'{destination}: matches pinned release')

if __name__ == '__main__':
    main()
