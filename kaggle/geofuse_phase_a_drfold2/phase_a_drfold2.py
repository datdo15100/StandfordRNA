"""Kaggle GPU fallback: raw DRfold2 cfg97 candidates for long validation RNA.

This kernel intentionally stops before PotentialFold CPU optimization.  It
exports the five highest-confidence direct checkpoint hypotheses, after Arena
fills C1' and other missing atoms, together with safe pLDDT/distogram sidecars.
Native validation labels are never read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


TARGET_IDS = {"R1138"}  # 720 nt: exceeds the local 8 GB GPU gate.
N_CANDIDATES = 5
CFG = "cfg_97"

INPUT = Path("/kaggle/input")
TEMP = Path("/kaggle/temp/geofuse_drfold2")
WORKING = Path("/kaggle/working")
OUTPUT = WORKING / "geofuse_drfold2_e2e"


def find_unique(paths: list[Path], description: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one {description}, found: {[str(p) for p in unique]}")
    return unique[0]


def prepare_drfold2() -> Path:
    source = find_unique(
        [path.parent for path in INPUT.rglob("DRfold_infer.py") if "drfold" in str(path).lower()],
        "DRfold2 source directory",
    )
    weight_source = find_unique(
        [
            path
            for path in INPUT.rglob("model_hub")
            if (path / CFG).is_dir() and (path / "RCLM").is_dir()
        ],
        "DRfold2 model_hub",
    )
    repo = TEMP / "DRfold2"
    if TEMP.exists():
        shutil.rmtree(TEMP)
    shutil.copytree(source, repo)
    # cfg97 also loads the shared RCLM checkpoint via a path relative to the repo.
    for name in ("RCLM", CFG):
        shutil.copytree(weight_source / name, repo / "model_hub" / name)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("no C++ compiler is available for Arena")
    subprocess.run(
        [compiler, "-O3", str(repo / "Arena" / "Arena.cpp"), "-o", str(repo / "Arena" / "Arena")],
        check=True,
    )
    return repo


def residue_confidence(pair_or_residue: np.ndarray) -> np.ndarray:
    confidence = np.asarray(pair_or_residue, dtype=np.float32)
    if confidence.ndim == 2:
        confidence = 0.5 * (confidence.mean(0) + confidence.mean(1))
    elif confidence.ndim != 1:
        raise ValueError(f"unexpected pLDDT shape: {confidence.shape}")
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def run_target(repo: Path, target_id: str, sequence: str) -> dict:
    started = time.time()
    scratch = TEMP / "predictions" / target_id
    ret_dir = scratch / "rets_dir"
    ret_dir.mkdir(parents=True, exist_ok=True)
    fasta = scratch / f"{target_id}.fasta"
    fasta.write_text(f">{target_id}\n{sequence}\n")
    log_path = OUTPUT / target_id / "drfold2_e2e.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo / CFG / "test_modeldir.py"),
        "cuda" if torch.cuda.is_available() else "cpu",
        str(fasta),
        str(ret_dir / f"{CFG}_"),
        str(repo / "model_hub" / CFG),
    ]
    print(f"[{target_id}] start L={len(sequence)} on {command[2]}", flush=True)
    with open(log_path, "w") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ret_paths = sorted(ret_dir.glob("*.ret"))
    if completed.returncode != 0 or not ret_paths:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "ret_files": len(ret_paths),
            "seconds": round(time.time() - started, 1),
            "log": str(log_path),
        }

    ranked = []
    for ret_path in ret_paths:
        # Trusted pickle: generated in this process by the attached official repo.
        with open(ret_path, "rb") as handle:
            payload = pickle.load(handle)
        confidence = residue_confidence(payload["plddt"])
        ranked.append((float(confidence.mean()), ret_path))
        del payload
    ranked.sort(key=lambda item: (-item[0], item[1].name))

    candidate_dir = OUTPUT / target_id / "e2e_relax"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, (score, ret_path) in enumerate(ranked[:N_CANDIDATES], start=1):
        with open(ret_path, "rb") as handle:
            payload = pickle.load(handle)
        confidence = residue_confidence(payload["plddt"])
        raw_pdb = ret_path.with_suffix(".pdb")
        output_pdb = candidate_dir / f"model_{rank}.pdb"
        subprocess.run(
            [str(repo / "Arena" / "Arena"), str(raw_pdb), str(output_pdb), "7"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        np.savez_compressed(candidate_dir / f"plddt_model_{rank}.npz", plddt=confidence)
        priors = {
            key: np.asarray(payload[key], dtype=np.float16)
            for key in ("dist_p", "dist_c", "dist_n")
            if key in payload
        }
        np.savez_compressed(candidate_dir / f"priors_model_{rank}.npz", **priors)
        manifest.append(
            {"rank": rank, "checkpoint_ret": ret_path.name, "global_confidence": score}
        )
        del payload
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "complete_e2e",
        "length": len(sequence),
        "ret_files": len(ret_paths),
        "e2e_models": len(manifest),
        "best_confidence": manifest[0]["global_confidence"] if manifest else None,
        "seconds": round(time.time() - started, 1),
        "log": str(log_path),
    }


def main() -> None:
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        flush=True,
    )
    repo = prepare_drfold2()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sequence_file = find_unique(
        [
            path
            for path in INPUT.rglob("validation_sequences.csv")
            if "stanford-rna-3d-folding" in str(path)
        ],
        "competition validation_sequences.csv",
    )
    sequences = pd.read_csv(sequence_file)
    sequences = sequences[sequences["target_id"].isin(TARGET_IDS)].copy()
    missing = TARGET_IDS - set(sequences["target_id"])
    if missing:
        raise KeyError(f"competition data is missing targets: {sorted(missing)}")

    status = {}
    for row in sequences.sort_values("target_id").itertuples(index=False):
        try:
            status[row.target_id] = run_target(repo, row.target_id, row.sequence)
        except Exception as exc:
            status[row.target_id] = {"status": "exception", "error": repr(exc)}
        (OUTPUT / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(f"[{row.target_id}] {status[row.target_id]}", flush=True)
        torch.cuda.empty_cache()

    archive = shutil.make_archive(
        str(WORKING / "geofuse_drfold2_e2e_r1138"), "zip", root_dir=OUTPUT
    )
    print(f"output archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
