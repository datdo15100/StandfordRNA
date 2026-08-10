#!/usr/bin/env python
"""Package the small code/prior overlay needed by the hybrid Kaggle kernel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "interim" / "kaggle_hybrid_runtime_upload"
STAGING = REPO_ROOT / "data" / "interim" / "kaggle_hybrid_runtime_staging"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (STAGING / "src").mkdir(parents=True)
    (STAGING / "configs").mkdir(parents=True)
    (STAGING / "kaggle").mkdir(parents=True)
    (STAGING / "wheels").mkdir(parents=True)
    OUTPUT.mkdir(parents=True)

    shutil.copytree(REPO_ROOT / "src" / "rna3d", STAGING / "src" / "rna3d")
    shutil.copy2(REPO_ROOT / "configs" / "paths.yaml", STAGING / "configs" / "paths.yaml")
    shutil.copy2(REPO_ROOT / "kaggle" / "__init__.py", STAGING / "kaggle" / "__init__.py")
    shutil.copy2(
        REPO_ROOT / "kaggle" / "hybrid_inference.py",
        STAGING / "kaggle" / "hybrid_inference.py",
    )
    shutil.copy2(
        REPO_ROOT / "data" / "processed" / "geofuse_geometry_v2_priors.json",
        STAGING / "geofuse_geometry_v2_priors.json",
    )
    wheel_dir = REPO_ROOT / "data" / "interim" / "kaggle_hybrid_wheels"
    wheels = sorted(wheel_dir.glob("biopython-*-cp310-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"expected one Biopython cp310 wheel in {wheel_dir}, found {wheels}")
    shutil.copy2(wheels[0], STAGING / "wheels" / wheels[0].name)

    archive = OUTPUT / "rna3d_hybrid_runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                handle.add(path, arcname=path.relative_to(STAGING))
    manifest = {
        "archive": archive.name,
        "sha256": sha256(archive),
        "files": sum(1 for path in STAGING.rglob("*") if path.is_file()),
        "purpose": "native-blind TBM + DRfold2 + Geometry v2 Kaggle inference",
    }
    (OUTPUT / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "RNA3D Thesis Hybrid Runtime v1",
                "id": "datdo151000/rna3d-thesis-hybrid-runtime-v1",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(OUTPUT)


if __name__ == "__main__":
    main()
