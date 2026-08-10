"""Offline Kaggle inference for TBM + DRfold2 + Geometry v2 (no GeoFuse).

The module is designed for a code-competition rerun: it reads the runtime test
sequences, builds candidates without native labels, and always returns a valid
five-structure submission even when DRfold2 times out or cannot process a long
target.
"""
from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd


CFG = "cfg_97"


def _find_unique(paths: list[Path], description: str) -> Path:
    unique = sorted({path.resolve() for path in paths})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one {description}, found {[str(path) for path in unique]}"
        )
    return unique[0]


def prepare_drfold2(input_root: Path, temp_root: Path) -> Path:
    """Copy the attached DRfold2 source/weights to writable disk and compile Arena."""
    source = _find_unique(
        [
            path.parent
            for path in input_root.rglob("DRfold_infer.py")
            if "drfold" in str(path).lower()
        ],
        "DRfold2 source directory",
    )
    weights = _find_unique(
        [
            path
            for path in input_root.rglob("model_hub")
            if (path / CFG).is_dir() and (path / "RCLM").is_dir()
        ],
        "DRfold2 model_hub",
    )
    repo = temp_root / "DRfold2"
    if repo.exists():
        shutil.rmtree(repo)
    repo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, repo)
    for name in ("RCLM", CFG):
        destination = repo / "model_hub" / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(weights / name, destination)
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("DRfold2 Arena requires a C++ compiler")
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
        raise ValueError(f"unexpected DRfold2 pLDDT shape {confidence.shape}")
    return np.clip(confidence, 0.0, 1.0)


def read_c1_pdb(path: Path, expected_length: int) -> np.ndarray:
    """Read one C1' coordinate per residue from an Arena-produced PDB."""
    records: list[tuple[tuple[str, str, str], list[float]]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() not in {"C1'", "C1*"}:
                continue
            residue = (line[21:22], line[22:26], line[26:27])
            if residue in seen:
                continue
            seen.add(residue)
            records.append(
                (residue, [float(line[30:38]), float(line[38:46]), float(line[46:54])])
            )
    coords = np.asarray([record[1] for record in records], dtype=np.float32)
    if coords.shape != (expected_length, 3) or not np.isfinite(coords).all():
        raise ValueError(
            f"{path}: expected {(expected_length, 3)} finite C1' coordinates, got {coords.shape}"
        )
    return coords


def run_drfold2_candidates(
    repo: Path,
    temp_root: Path,
    target_id: str,
    sequence: str,
    *,
    candidates: int = 2,
) -> tuple[list[dict], dict]:
    """Return the highest model-confidence direct DRfold2 hypotheses."""
    started = time.time()
    scratch = temp_root / "predictions" / target_id
    ret_dir = scratch / "rets_dir"
    ret_dir.mkdir(parents=True, exist_ok=True)
    fasta = scratch / f"{target_id}.fasta"
    fasta.write_text(f">{target_id}\n{sequence}\n")
    log_path = scratch / "drfold2.log"
    command = [
        sys.executable,
        str(repo / CFG / "test_modeldir.py"),
        "cuda",
        str(fasta),
        str(ret_dir / f"{CFG}_"),
        str(repo / "model_hub" / CFG),
    ]
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ret_paths = sorted(ret_dir.glob("*.ret"))
    if completed.returncode != 0 or not ret_paths:
        return [], {
            "status": "model_failed",
            "returncode": completed.returncode,
            "ret_files": len(ret_paths),
            "seconds": round(time.time() - started, 1),
            "log_tail": log_path.read_text(errors="replace")[-2000:],
        }

    ranked: list[tuple[float, Path, np.ndarray]] = []
    for ret_path in ret_paths:
        with ret_path.open("rb") as handle:
            payload = pickle.load(handle)  # trusted output created in this process
        confidence = residue_confidence(payload["plddt"])
        if confidence.shape == (len(sequence),):
            ranked.append((float(confidence.mean()), ret_path, confidence))
    ranked.sort(key=lambda item: (-item[0], item[1].name))

    outputs: list[dict] = []
    arena_failures = 0
    for score, ret_path, confidence in ranked:
        output_pdb = scratch / f"arena_{len(outputs) + 1}.pdb"
        arena = subprocess.run(
            [str(repo / "Arena" / "Arena"), str(ret_path.with_suffix(".pdb")), str(output_pdb), "7"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if arena.returncode:
            arena_failures += 1
            continue
        try:
            coords = read_c1_pdb(output_pdb, len(sequence))
        except (OSError, ValueError):
            arena_failures += 1
            continue
        outputs.append(
            {
                "coords": coords,
                "confidence": confidence,
                "global_confidence": score,
                "candidate_id": ret_path.stem,
            }
        )
        if len(outputs) == candidates:
            break
    return outputs, {
        "status": "complete" if len(outputs) == candidates else "partial",
        "ret_files": len(ret_paths),
        "candidates": len(outputs),
        "arena_failures": arena_failures,
        "seconds": round(time.time() - started, 1),
    }


def _tbm_candidate_bank(
    test_sequences: pd.DataFrame,
    artifacts: Path,
    work_dir: Path,
) -> tuple[dict[str, list[dict]], dict]:
    """Run the production temporal-safe composite TBM search without refinement."""
    # The frozen coordinate pickle was produced with NumPy 2, whose internal
    # module moved from ``numpy.core`` to ``numpy._core``.  The pinned DRfold2
    # GPU image uses NumPy 1; alias the one referenced module before unpickling.
    try:
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        numpy_core = importlib.import_module("numpy.core")
        numpy_numeric = importlib.import_module("numpy.core.numeric")
        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

    from rna3d.geometry.denovo import de_novo_ensemble
    from rna3d.pipeline.tbm import build_tbm_candidates
    from rna3d.template import db, mmseqs_search

    priors_v1 = json.loads((artifacts / "geometry_priors.json").read_text())
    meta = db.load_meta()
    query_path = work_dir / "query.fasta"
    with query_path.open("w") as handle:
        for row in test_sequences.itertuples(index=False):
            handle.write(f">{row.target_id}\n{row.sequence}\n")
    hits = mmseqs_search.search(query_path, work_dir / "hits.m8")

    banks: dict[str, list[dict]] = {}
    status: dict[str, dict] = {}
    for row in test_sequences.itertuples(index=False):
        target_id, sequence = str(row.target_id), str(row.sequence)
        cutoff = getattr(row, "temporal_cutoff", "9999-12-31") or "9999-12-31"
        candidates = build_tbm_candidates(
            target_id,
            sequence,
            cutoff,
            hits[hits["query"] == target_id],
            meta,
            rng=np.random.default_rng(0),
            adj_dist=float(priors_v1["adjacent_c1"]["mean"]),
            max_candidates=5,
        )
        bank = [
            {
                "coords": candidate.coords,
                "confidence": candidate.conf_residue,
                "global_confidence": candidate.confidence,
                "candidate_id": f"tbm__{candidate.chain_key}",
            }
            for candidate in candidates
        ]
        if len(bank) < 5:
            for index, coords in enumerate(
                de_novo_ensemble(sequence, n=5 - len(bank), base_seed=0), start=1
            ):
                bank.append(
                    {
                        "coords": coords,
                        "confidence": np.full(len(sequence), 0.1, dtype=np.float32),
                        "global_confidence": 0.1,
                        "candidate_id": f"denovo__{index}",
                    }
                )
        banks[target_id] = bank[:5]
        status[target_id] = {
            "templates": len(candidates),
            "candidate_ids": [candidate["candidate_id"] for candidate in bank[:5]],
        }
    return banks, status


def run_hybrid_inference(
    test_sequences: pd.DataFrame,
    artifacts: Path,
    runtime: Path,
    input_root: Path,
    *,
    work_dir: Path,
    sample_submission: pd.DataFrame | None = None,
    geometry_steps: int = 300,
    drfold_max_len: int = 600,
    drfold_deadline_seconds: float = 6.5 * 60 * 60,
) -> tuple[pd.DataFrame, dict]:
    """Run native-blind 3-TBM + 2-DRfold2 selection and Geometry v2."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts.resolve()
    runtime = runtime.resolve()
    os.environ["RNA3D_PROCESSED"] = str(artifacts)
    os.environ["RNA3D_CACHE"] = str(artifacts)
    sys.path.insert(0, str(runtime / "src"))

    from rna3d.data import io
    from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
    from rna3d.template import mmseqs_search

    priors_v1 = json.loads((artifacts / "geometry_priors.json").read_text())
    priors_v2 = json.loads((runtime / "geofuse_geometry_v2_priors.json").read_text())
    runtime_mmseqs = work_dir / "mmseqs"
    shutil.copy2(artifacts / "bin" / "mmseqs", runtime_mmseqs)
    runtime_mmseqs.chmod(0o755)
    mmseqs_search.mmseqs_bin = lambda: str(runtime_mmseqs)
    tbm_banks, tbm_status = _tbm_candidate_bank(test_sequences, artifacts, work_dir)

    started = time.time()
    drfold_status: dict[str, dict] = {}
    pretrained: dict[str, list[dict]] = {}
    temp_root = work_dir / "drfold2"
    try:
        repo = prepare_drfold2(input_root, temp_root)
    except Exception as exc:
        repo = None
        drfold_status["setup"] = {"status": "failed", "error": repr(exc)}

    ordered = test_sequences.assign(
        sequence_length=test_sequences["sequence"].str.len()
    ).sort_values(["sequence_length", "target_id"])
    for row in ordered.itertuples(index=False):
        target_id, sequence = str(row.target_id), str(row.sequence)
        elapsed = time.time() - started
        if repo is None:
            pretrained[target_id] = []
            continue
        if len(sequence) > drfold_max_len:
            pretrained[target_id] = []
            drfold_status[target_id] = {
                "status": "skipped_length",
                "length": len(sequence),
                "limit": drfold_max_len,
            }
            continue
        if elapsed >= drfold_deadline_seconds:
            pretrained[target_id] = []
            drfold_status[target_id] = {
                "status": "skipped_deadline",
                "elapsed_seconds": round(elapsed, 1),
            }
            continue
        try:
            values, status = run_drfold2_candidates(
                repo, temp_root, target_id, sequence, candidates=2
            )
        except Exception as exc:
            values, status = [], {"status": "exception", "error": repr(exc)}
        pretrained[target_id] = values
        drfold_status[target_id] = status
        print(f"[DRfold2:{target_id}] {status}", flush=True)

    final_predictions: dict[str, np.ndarray] = {}
    refinement_status: dict[str, list[dict]] = {}
    config = GeometryV2Config(steps=geometry_steps)
    for row in test_sequences.itertuples(index=False):
        target_id, sequence = str(row.target_id), str(row.sequence)
        dl = pretrained.get(target_id, [])[:2]
        n_tbm = 5 - len(dl)
        selected = tbm_banks[target_id][:n_tbm] + dl
        refined: list[np.ndarray] = []
        target_refinement: list[dict] = []
        for candidate in selected:
            try:
                coords, _ = refine_structure_v2(
                    candidate["coords"],
                    sequence,
                    priors_v1,
                    priors_v2,
                    source_confidence=candidate["confidence"],
                    global_confidence=candidate["global_confidence"],
                    cfg=config,
                    device="cuda",
                )
                status = "complete"
            except Exception as exc:
                coords = np.asarray(candidate["coords"], dtype=np.float32)
                status = f"raw_fallback:{type(exc).__name__}"
            refined.append(np.asarray(coords, dtype=np.float32))
            target_refinement.append(
                {"candidate_id": candidate["candidate_id"], "status": status}
            )
        final_predictions[target_id] = np.stack(refined, axis=0)
        refinement_status[target_id] = target_refinement

    submission = io.build_submission(final_predictions, test_sequences)
    io.validate_submission(submission, test_sequences)
    if sample_submission is not None:
        submission = io.order_submission_like(submission, sample_submission)
    manifest = {
        "pipeline": "3 TBM + 2 DRfold2 when feasible, then Geometry v2; no GeoFuse",
        "native_labels_used": False,
        "geometry_v2_steps": geometry_steps,
        "drfold_max_len": drfold_max_len,
        "drfold_deadline_seconds": drfold_deadline_seconds,
        "tbm": tbm_status,
        "drfold2": drfold_status,
        "refinement": refinement_status,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    return submission, manifest
