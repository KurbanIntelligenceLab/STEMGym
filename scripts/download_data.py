#!/usr/bin/env python3
"""Download STEMGym data and model checkpoints from Hugging Face.

Usage:
    python scripts/download_data.py              # download everything
    python scripts/download_data.py --worlds     # only HDF5 worlds
    python scripts/download_data.py --checkpoints # only model checkpoints
    python scripts/download_data.py --test-only  # only test_world.h5 (minimal)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF_DATASET_REPO = "johnpolat/STEMGym"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "simulated"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Files to download: (remote_path, local_dir, filename)
WORLD_FILES = [
    "test_world.h5",
    "srtio3_clustered_easy.h5",
    "srtio3_clustered_medium.h5",
    "srtio3_clustered_hard.h5",
    "batio3_interface_easy.h5",
    "batio3_interface_medium.h5",
    "batio3_interface_hard.h5",
    "sige_gradient_clustered_easy.h5",
    "sige_gradient_clustered_medium.h5",
    "sige_gradient_clustered_hard.h5",
]

CHECKPOINT_FILES = [
    "atom_finder.pt",
    "defect_classifier.pt",
    "phase_identifier.pt",
]


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress display."""
    if dest.exists():
        print(f"  [skip] {dest.name} (already exists)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        print(f"  [download] {dest.name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f" {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_worlds(test_only: bool = False) -> int:
    """Download HDF5 world files. Returns number of failures."""
    files = ["test_world.h5"] if test_only else WORLD_FILES
    print(f"\nDownloading world files to {DATA_DIR}/")
    failures = 0
    for fname in files:
        url = f"{HF_BASE_URL}/worlds/{fname}"
        if not download_file(url, DATA_DIR / fname):
            failures += 1
    return failures


def download_checkpoints() -> int:
    """Download model checkpoint files. Returns number of failures."""
    print(f"\nDownloading checkpoints to {CHECKPOINT_DIR}/")
    failures = 0
    for fname in CHECKPOINT_FILES:
        url = f"{HF_BASE_URL}/checkpoints/{fname}"
        if not download_file(url, CHECKPOINT_DIR / fname):
            failures += 1
    return failures


def main():
    parser = argparse.ArgumentParser(description="Download STEMGym data from Hugging Face")
    parser.add_argument("--worlds", action="store_true", help="Download only world files")
    parser.add_argument("--checkpoints", action="store_true", help="Download only checkpoints")
    parser.add_argument("--test-only", action="store_true", help="Download only test_world.h5")
    args = parser.parse_args()

    # Default: download everything
    download_all = not (args.worlds or args.checkpoints or args.test_only)

    print(f"STEMGym Data Downloader")
    print(f"Source: huggingface.co/datasets/{HF_DATASET_REPO}")

    failures = 0

    if download_all or args.worlds or args.test_only:
        failures += download_worlds(test_only=args.test_only)

    if download_all or args.checkpoints:
        failures += download_checkpoints()

    if failures:
        print(f"\nWARNING: {failures} download(s) failed (some worlds may not be available yet).")
        print("Continuing with available worlds...")
    else:
        print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
