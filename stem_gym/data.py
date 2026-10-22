import hashlib
import json
from pathlib import Path
from importlib.resources import files
from urllib.request import urlopen


def release_manifest() -> dict:
    local = (
        Path(__file__).resolve().parents[1] / "results/manuscript/release_manifest.json"
    )
    if local.exists():
        return json.loads(local.read_text())
    return json.loads(
        files("stem_gym").joinpath("data/release_manifest.json").read_text()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(manifest: dict, remote: str, destination: Path) -> Path:
    import tempfile

    record = next(r for r in manifest["siblings"] if r["rfilename"] == remote)
    expected = record["lfs"]["sha256"]
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(
                f"Existing file does not match the pinned release: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{manifest['id']}/resolve/{manifest['sha']}/{remote}"
    partial = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name,
            suffix=".download",
            delete=False,
        ) as out:
            partial = Path(out.name)
            with urlopen(url, timeout=90) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    out.write(block)
        if sha256(partial) != expected:
            raise ValueError(f"Download does not match the pinned release: {remote}")
        partial.replace(destination)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return destination
