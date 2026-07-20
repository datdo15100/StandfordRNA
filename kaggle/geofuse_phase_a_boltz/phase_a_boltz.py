"""Kaggle GPU fallback: reproduce top-1 Boltz-1 inference for R1138.

DRfold2 exhausts a P100's 16 GB on this 720-nt validation target.  The winning
hybrid notebook routes sequences longer than 600 nt through Boltz instead.  This
private kernel follows that native-blind branch and exports the mmCIF plus the
model-produced confidence sidecars for the GeoFuse Phase-A candidate cache.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


TARGET_ID = "R1138"
EXPECTED_LENGTH = 720
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
PACKAGE_DIR = WORKING / "boltz"
MODEL_INPUT_DIR = WORKING / "geofuse_boltz_input"
RAW_OUTPUT_DIR = WORKING / "geofuse_boltz_raw"
OUTPUT_DIR = WORKING / "geofuse_boltz"


def find_unique(paths: list[Path], description: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one {description}, found {[str(path) for path in unique]}"
        )
    return unique[0]


def install_offline_dependencies() -> None:
    """Install exactly the wheel bundles attached by the top-1 notebook."""
    groups = (
        "boltz-dependencies/*.whl",
        "fairscale-0413/*.whl",
        "biopython/*.whl",
    )
    for suffix in groups:
        wheels = sorted(glob.glob(str(INPUT / "**" / suffix), recursive=True))
        if not wheels:
            raise FileNotFoundError(f"no offline wheels found for {suffix}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheels],
            check=True,
        )


def prepare_boltz_package() -> Path:
    source = find_unique(
        [path.parent for path in INPUT.rglob("src/boltz/main.py")],
        "Boltz source package",
    )
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    shutil.copytree(source, PACKAGE_DIR)
    return find_unique(
        [
            path.parent
            for path in INPUT.rglob("boltz1_conf.ckpt")
            if (path.parent / "ccd.pkl").is_file()
        ],
        "Boltz checkpoint cache",
    )


def load_target_sequence() -> str:
    import pandas as pd

    sequence_file = find_unique(
        [
            path
            for path in INPUT.rglob("validation_sequences.csv")
            if "stanford-rna-3d-folding" in str(path)
        ],
        "competition validation_sequences.csv",
    )
    rows = pd.read_csv(sequence_file)
    matches = rows.loc[rows["target_id"] == TARGET_ID, "sequence"]
    if len(matches) != 1:
        raise KeyError(f"expected one validation sequence for {TARGET_ID}, found {len(matches)}")
    sequence = str(matches.iloc[0])
    if len(sequence) != EXPECTED_LENGTH:
        raise ValueError(
            f"{TARGET_ID} length changed: expected {EXPECTED_LENGTH}, found {len(sequence)}"
        )
    return sequence


def write_yaml(sequence: str) -> Path:
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_INPUT_DIR / f"{TARGET_ID}.yaml"
    path.write_text(
        "constraints: []\n"
        "sequences:\n"
        "- rna:\n"
        "    id:\n"
        "    - A1\n"
        f"    sequence: {sequence}\n"
    )
    return path


def run_boltz(cache_dir: Path) -> tuple[subprocess.CompletedProcess, float, Path]:
    """Use the exact inference settings from the top-1 hybrid notebook."""
    log_path = OUTPUT_DIR / TARGET_ID / "boltz_top1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "boltz.main",
        "predict",
        str(MODEL_INPUT_DIR),
        "--out_dir",
        str(RAW_OUTPUT_DIR),
        "--cache",
        str(cache_dir),
        "--diffusion_samples",
        "1",
        "--recycling_steps",
        "10",
        "--accelerator",
        "gpu",
        "--sampling_steps",
        "500",
        "--seed",
        "42",
        "--override",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKING)
    started = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=WORKING,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed, time.time() - started, log_path


def validate_and_export(seconds: float, log_path: Path) -> dict:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    import numpy as np

    prediction_dir = (
        RAW_OUTPUT_DIR
        / f"boltz_results_{MODEL_INPUT_DIR.stem}"
        / "predictions"
        / TARGET_ID
    )
    structure = prediction_dir / f"{TARGET_ID}_model_0.cif"
    confidence = prediction_dir / f"confidence_{TARGET_ID}_model_0.json"
    plddt = prediction_dir / f"plddt_{TARGET_ID}_model_0.npz"
    for path in (structure, confidence, plddt):
        if not path.is_file():
            raise FileNotFoundError(f"Boltz did not produce {path}")

    document = MMCIF2Dict(str(structure))
    atom_names = document.get("_atom_site.label_atom_id", [])
    # MMCIF2Dict already removes CIF quoting. Preserve the apostrophe that is
    # part of the RNA atom name (stripping single quotes would turn C1' into C1).
    c1_count = sum(str(name).strip('"') in {"C1'", "C1*"} for name in atom_names)
    if c1_count != EXPECTED_LENGTH:
        raise ValueError(f"expected {EXPECTED_LENGTH} C1' atoms, found {c1_count}")
    with np.load(plddt, allow_pickle=False) as payload:
        local_confidence = np.asarray(payload["plddt"]).squeeze()
    if local_confidence.shape != (EXPECTED_LENGTH,):
        raise ValueError(
            f"expected residue confidence shape {(EXPECTED_LENGTH,)}, "
            f"found {local_confidence.shape}"
        )

    candidate_dir = OUTPUT_DIR / TARGET_ID / "boltz"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(structure, candidate_dir / "model_1.cif")
    shutil.copy2(confidence, candidate_dir / "confidence_model_1.json")
    shutil.copy2(plddt, candidate_dir / "plddt_model_1.npz")
    confidence_document = json.loads(confidence.read_text())
    status = {
        "status": "complete",
        "target_id": TARGET_ID,
        "length": EXPECTED_LENGTH,
        "c1_atoms": c1_count,
        "plddt_values": int(local_confidence.size),
        "confidence_score": confidence_document.get("confidence_score"),
        "ptm": confidence_document.get("ptm"),
        "seconds": round(seconds, 1),
        "settings": {
            "diffusion_samples": 1,
            "recycling_steps": 10,
            "sampling_steps": 500,
            "seed": 42,
        },
        "log": str(log_path),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    install_offline_dependencies()
    cache_dir = prepare_boltz_package()
    sequence = load_target_sequence()
    write_yaml(sequence)

    import torch

    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        flush=True,
    )
    completed, seconds, log_path = run_boltz(cache_dir)
    if completed.returncode != 0:
        status = {
            "status": "failed",
            "returncode": completed.returncode,
            "seconds": round(seconds, 1),
            "log": str(log_path),
        }
        (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(log_path.read_text()[-8000:], flush=True)
        raise RuntimeError(f"Boltz exited with status {completed.returncode}")

    status = validate_and_export(seconds, log_path)
    archive = shutil.make_archive(
        str(WORKING / "geofuse_boltz_r1138"), "zip", root_dir=OUTPUT_DIR
    )
    print(json.dumps(status, indent=2), flush=True)
    print(f"output archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
