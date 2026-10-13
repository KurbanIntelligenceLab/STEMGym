"""Protect reference files and validate source identities."""

from pathlib import Path
import hashlib
import json
import zipfile


def writable_output(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    for directory in ("results/manuscript", "development"):
        protected = (root / directory).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                f"Refusing to overwrite protected files: {resolved}"
            )
    return resolved


def verify_preservation(evidence: Path) -> dict:
    manifest = json.loads((evidence / "preservation.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Preserved evidence changed: {name}")
    with zipfile.ZipFile(evidence / "original_source.zip") as archive:
        sources = json.loads(archive.read("SOURCE_MANIFEST.json"))
        for name, digest in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Preserved source changed: {name}")
    return {"evidence_files": len(manifest["files"]), "source_files": len(sources)}
