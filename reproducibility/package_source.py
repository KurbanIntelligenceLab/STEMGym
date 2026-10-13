"""Build an allowlisted review archive, including frozen evidence and uv.lock."""

import hashlib
import json
from pathlib import Path
import zipfile
from stem_gym.artifacts import verify_preservation


def main():
    root = Path(__file__).resolve().parents[1]
    verify_preservation(root / "results/manuscript")
    output = root / "development/releases/stemgym_mlst_source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        root / n
        for n in [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            ".gitignore",
            "croissant.json",
        ]
    ]
    for directory, suffixes in [
        ("stem_gym", {".py"}),
        ("tests", {".py"}),
        ("reproducibility", {".py", ".md"}),
        ("results/manuscript", {".py", ".json", ".zip", ".md"}),
        (".github", {".yml", ".yaml"}),
    ]:
        paths.extend(
            (
                p
                for p in (root / directory).rglob("*")
                if p.is_file()
                and p.suffix in suffixes
                and ("__pycache__" not in p.parts)
            )
        )
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(root / name, name)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or any(
            (
                hashlib.sha256(archive.read(n)).hexdigest() != h
                for (n, h) in manifest.items()
            )
        ):
            raise ValueError("Archive failed integrity verification")
    print(f"{len(manifest)} files packaged and verified: {output}")


if __name__ == "__main__":
    main()
