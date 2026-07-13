"""Build the compact, read-only artifact bundle used by the Kaggle kernel.

The bundle contains derived search/coordinate artifacts, the exact inference code,
and a relocatable MMseqs executable. It intentionally excludes raw competition CSV,
MSA and PDB files because Kaggle mounts those through the competition dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import Bio


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "data" / "interim" / "kaggle_artifact_bundle"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundled_libraries(executable: Path) -> list[Path]:
    output = subprocess.run(
        ["ldd", str(executable)], check=True, capture_output=True, text=True
    ).stdout
    libraries = []
    env_root = Path(sys.executable).resolve().parents[1]
    for line in output.splitlines():
        if "=>" not in line:
            continue
        candidate = line.split("=>", 1)[1].strip().split(" ", 1)[0]
        path = Path(candidate)
        if path.is_file() and env_root in path.parents:
            libraries.append(path)
    return libraries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise SystemExit(f"bundle already exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    artifacts = {
        REPO / "data/processed/geometry_priors.json": output / "geometry_priors.json",
        REPO / "data/processed/template_meta.parquet": output / "template_meta.parquet",
        REPO / "data/processed/top1_template_meta.parquet": output / "top1_template_meta.parquet",
        REPO / "data/cache/template_coords.pkl": output / "template_coords.pkl",
        REPO / "data/cache/top1_template_coords.pkl": output / "top1_template_coords.pkl",
        REPO / "data/cache/template_db.fasta": output / "template_db.fasta",
        REPO / "configs/paths.yaml": output / "configs/paths.yaml",
        REPO / "kaggle/__init__.py": output / "kaggle/__init__.py",
        REPO / "kaggle/inference_pipeline.py": output / "kaggle/inference_pipeline.py",
    }
    for source, destination in artifacts.items():
        copy_file(source, destination)

    shutil.copytree(
        REPO / "src/rna3d",
        output / "src/rna3d",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(REPO / "data/cache/mmseqs", output / "mmseqs")

    mmseqs = Path(sys.executable).resolve().parent / "mmseqs"
    copy_file(mmseqs, output / "bin/mmseqs")
    for library in bundled_libraries(mmseqs):
        copy_file(library, output / "lib" / library.name)

    # Kaggle's base Python image does not guarantee Biopython. Bundle the exact
    # wheel so the notebook can install it offline into /kaggle/working.
    wheels = output / "wheels"
    wheels.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(wheels),
            f"biopython=={Bio.__version__}",
        ],
        check=True,
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output)
    print(f"files={len(files)} bytes={manifest['total_bytes']}")


if __name__ == "__main__":
    main()
