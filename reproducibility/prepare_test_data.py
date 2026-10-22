from pathlib import Path
from stem_gym.data import download_verified, release_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = release_manifest()
    for name, directory in [
        ("test_world.h5", "simulated"),
        ("replay_world.h5", "replay"),
    ]:
        download_verified(manifest, "worlds/" + name, root / "data" / directory / name)
        print(f"{directory}/{name}: matches pinned release")


if __name__ == "__main__":
    main()
