"""Kaggle GPU runner for the leakage-audited GeoFuse real-OOF pilot."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


TARGET_IDS = {
    "8QEQ_R", "9G6K_l0", "9G6K_lN", "8Z1P_B", "8RR3_T",
    "8UPT_A", "8H8E_G", "8XZE_A", "9CEV_W", "9C3I_A",
    "8K0Y_A", "8Y9M_B", "9DE6_B", "9B0Q_AP", "9DCF_C",
}
N_CANDIDATES = 2
CFG = "cfg_97"
INPUT = Path("/kaggle/input")
TEMP = Path("/kaggle/temp/geofuse_real_oof")
OUTPUT = Path("/kaggle/working/geofuse_drfold2_real_oof")


def find_unique(paths: list[Path], description: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one {description}, found {[str(path) for path in unique]}")
    return unique[0]


def prepare_drfold2() -> Path:
    source = find_unique(
        [path.parent for path in INPUT.rglob("DRfold_infer.py") if "drfold" in str(path).lower()],
        "DRfold2 source directory",
    )
    weights = find_unique(
        [path for path in INPUT.rglob("model_hub") if (path / CFG).is_dir() and (path / "RCLM").is_dir()],
        "DRfold2 model_hub",
    )
    repo = TEMP / "DRfold2"
    if TEMP.exists():
        shutil.rmtree(TEMP)
    shutil.copytree(source, repo)
    for name in ("RCLM", CFG):
        shutil.copytree(weights / name, repo / "model_hub" / name)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Arena requires a C++ compiler")
    subprocess.run(
        [compiler, "-O3", str(repo / "Arena" / "Arena.cpp"), "-o", str(repo / "Arena" / "Arena")],
        check=True,
    )
    return repo


def residue_confidence(values: np.ndarray) -> np.ndarray:
    confidence = np.asarray(values, dtype=np.float32)
    if confidence.ndim == 2:
        confidence = 0.5 * (confidence.mean(0) + confidence.mean(1))
    if confidence.ndim != 1:
        raise ValueError(f"unexpected pLDDT shape {confidence.shape}")
    return np.clip(confidence, 0.0, 1.0)


def run_target(repo: Path, target_id: str, sequence: str) -> dict:
    started = time.time()
    scratch = TEMP / "predictions" / target_id
    ret_dir = scratch / "rets_dir"
    ret_dir.mkdir(parents=True, exist_ok=True)
    fasta = scratch / f"{target_id}.fasta"
    fasta.write_text(f">{target_id}\n{sequence}\n")
    target_output = OUTPUT / target_id
    target_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(repo / CFG / "test_modeldir.py"), "cuda", str(fasta),
        str(ret_dir / f"{CFG}_"), str(repo / "model_hub" / CFG),
    ]
    with (target_output / "drfold2_e2e.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    ret_paths = sorted(ret_dir.glob("*.ret"))
    if completed.returncode or not ret_paths:
        return {"status": "failed", "returncode": completed.returncode, "ret_files": len(ret_paths)}
    ranked = []
    for path in ret_paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # trusted output created by this kernel
        ranked.append((float(residue_confidence(payload["plddt"]).mean()), path))
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    candidate_dir = target_output / "e2e_relax"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, (score, ret_path) in enumerate(ranked[:N_CANDIDATES], start=1):
        with ret_path.open("rb") as handle:
            payload = pickle.load(handle)
        model_path = candidate_dir / f"model_{rank}.pdb"
        subprocess.run(
            [str(repo / "Arena" / "Arena"), str(ret_path.with_suffix(".pdb")), str(model_path), "7"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        np.savez_compressed(
            candidate_dir / f"plddt_model_{rank}.npz",
            plddt=residue_confidence(payload["plddt"]),
        )
        priors = {
            key: np.asarray(payload[key], dtype=np.float16)
            for key in ("dist_p", "dist_c", "dist_n") if key in payload
        }
        np.savez_compressed(candidate_dir / f"priors_model_{rank}.npz", **priors)
        manifest.append({"rank": rank, "ret": ret_path.name, "confidence": score})
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "complete", "length": len(sequence), "ret_files": len(ret_paths),
        "models": len(manifest), "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this kernel requires a Kaggle GPU")
    repo = prepare_drfold2()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sequence_file = find_unique(
        [path for path in INPUT.rglob("train_sequences.v2.csv") if "stanford-rna-3d-folding" in str(path)],
        "competition train_sequences.v2.csv",
    )
    sequences = pd.read_csv(sequence_file, dtype=str)
    sequences = sequences[sequences["target_id"].isin(TARGET_IDS)].copy()
    missing = TARGET_IDS - set(sequences["target_id"])
    if missing:
        raise KeyError(f"missing pilot targets: {sorted(missing)}")
    status = {}
    for row in sequences.sort_values(["temporal_cutoff", "target_id"]).itertuples(index=False):
        print(f"[{row.target_id}] start L={len(row.sequence)}", flush=True)
        try:
            status[row.target_id] = run_target(repo, row.target_id, row.sequence)
        except Exception as exc:
            status[row.target_id] = {"status": "exception", "error": repr(exc)}
        (OUTPUT / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(f"[{row.target_id}] {status[row.target_id]}", flush=True)
        torch.cuda.empty_cache()
    shutil.make_archive(
        "/kaggle/working/geofuse_drfold2_real_oof_pilot", "zip", root_dir=OUTPUT
    )


if __name__ == "__main__":
    main()
