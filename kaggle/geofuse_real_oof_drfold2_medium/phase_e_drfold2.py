"""Kaggle GPU runner for the frozen 60/20/20 GeoFuse real-OOF experiment."""
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
    # Frozen train (60)
    "8T2A_R", "8T2O_R", "8UPT_A", "8OUF_B", "8PM4_B", "8Y0X_E",
    "8IPA_cl", "8IEW_B", "8PEG_X", "8SCZ_B", "8QCQ_w", "8QOA_Z",
    "8ITS_A", "8U5Z_A", "8H8E_G", "8T7S_H", "9BH8_Y", "8JY0_D",
    "8KAG_A", "8QBM_L", "8CBL_T", "8P7B_R", "8P7B_T", "8HZE_B",
    "8HZL_A", "8URU_D", "8V1H_A", "9ENE_C", "8K29_P", "8OPS_C",
    "8VMB_R", "8W35_C", "8XZE_A", "9BDP_TIRN", "8XAK_D", "8YDB_C",
    "8RO0_2", "8VU0_D", "8KDA_O", "8UAU_R", "8ZYD_C", "8Q7Z_V",
    "9BH5_DA", "9CEV_W", "9CF1_W", "8QWE_C", "8VPV_A", "8Z0K_L",
    "8WND_T", "9DTT_B", "8RFJ_H", "8YUR_X", "8P6P_7", "8P7X_8",
    "8P8V_6", "9GCH_T", "9BLM_A", "9BZ1_B", "8XPP_B", "9C3I_A",
    # Frozen calibration (20)
    "8QEQ_R", "8T2Y_tA", "8VB6_F", "8VBH_F", "8ZOL_A", "9FI8_hB",
    "9FI8_hL", "9FIA_bJ", "9G6K_l1", "9G6K_l3", "9G6K_l9", "9G6K_lA",
    "9G6K_lB", "9G6K_lK", "8UP5_T", "9D85_T", "8Y03_G", "8Y08_B",
    "8Y0B_B", "8RR3_T",
    # Frozen newest validation (20)
    "8K0Y_A", "8K1E_A", "8K30_B", "8K7W_A", "8VZ6_S", "8Y9N_B",
    "9DRS_C", "8KEB_A", "8YIH_C", "9DE6_A", "9DE7_A", "9DPB_C",
    "8WFA_B", "8Z8Q_B", "8Z9K_B", "9B0S_Et", "9N2C_Pt", "9B2K_B",
    "9DCF_C", "9E2Z_F",
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
        raise KeyError(f"missing frozen medium targets: {sorted(missing)}")
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
        "/kaggle/working/geofuse_drfold2_real_oof_medium", "zip", root_dir=OUTPUT
    )


if __name__ == "__main__":
    main()
