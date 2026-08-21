"""Offline Kaggle deployment adapter for the frozen V4 thesis pipeline.

The scientific method is imported from the pre-opening V4 freeze.  This module
only adapts that method to Kaggle's native-blind runtime: retained exhaustive
TBM, a fixed five-candidate 3T+2D bank, and Selected Geometry.  Missing DRfold2
slots use the next frozen template candidates, exactly as specified by the
freeze.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from kaggle.hybrid_inference import prepare_drfold2, run_drfold2_candidates


FROZEN_COMMIT = "4ea1044cca5e2ebcb5e1b76ab02d955e84d21758"
GEOMETRY_CONFIG = {
    "steps": 300,
    "lr": 0.04,
    "w_source": 3.0,
    "w_backbone": 1.0,
    "w_clash": 0.3,
    "w_rg": 0.0,
    "w_angle": 0.3,
    "w_torsion": 0.15,
    "w_kink": 20.0,
    "adaptive_strength": True,
    "fixed_strength": 1.0,
    "context_mode": "global",
    "kink_floor_deg": 70.0,
    "kink_margin_deg": 5.0,
    "backbone_huber_delta": 2.0,
    "rg_huber_delta": 5.0,
    "sep_clash": 2,
    "huber_delta": 2.0,
    "grad_clip": 10.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy_pickle_compatibility() -> None:
    """Allow a NumPy-2 coordinate pickle to load in Kaggle's NumPy-1 image."""
    try:
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        numpy_core = importlib.import_module("numpy.core")
        numpy_numeric = importlib.import_module("numpy.core.numeric")
        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def _cutoff(row: object) -> str:
    value = getattr(row, "temporal_cutoff", "9999-12-31")
    if value is None or pd.isna(value) or not str(value).strip():
        return "9999-12-31"
    return str(value)


def _build_tbm_banks(
    test_sequences: pd.DataFrame,
    artifacts: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict]]:
    """Materialize the exact retained V4 TBM method without native labels."""
    _numpy_pickle_compatibility()
    from rna3d.pipeline.v4_frozen import build_thesis_tbm_bank
    from rna3d.template import db

    meta = db.load_meta()
    coordinates = db.load_coords()
    adjacent = float(json.loads((artifacts / "geometry_priors.json").read_text())["adjacent_c1"]["mean"])
    banks: dict[str, dict[str, np.ndarray]] = {}
    status: dict[str, dict] = {}
    for row in test_sequences.itertuples(index=False):
        target_id = str(row.target_id)
        sequence = str(row.sequence)
        started = time.time()
        bank = build_thesis_tbm_bank(
            target_id=target_id,
            sequence=sequence,
            cutoff=_cutoff(row),
            excluded_pdb_ids=(),
            meta=meta,
            coordinates=coordinates,
            adjacent_distance=adjacent,
            n=5,
        )
        banks[target_id] = {
            "coords": bank.coords,
            "confidence": bank.confidence,
            "global_confidence": bank.global_confidence,
        }
        status[target_id] = {
            "candidate_ids": list(bank.candidate_ids),
            "pdb_ids": list(bank.pdb_ids),
            "fallback_slots": int(bank.fallback_slots),
            "seconds": round(time.time() - started, 2),
        }
        print(f"[V4 TBM:{target_id}] fallback={bank.fallback_slots}", flush=True)
    return banks, status


def _assemble_3t2d(
    tbm: dict[str, np.ndarray],
    deep: list[dict],
) -> tuple[list[dict], dict]:
    """Assemble the frozen 3T+2D order, including its recorded T fallback."""
    selected = [
        {
            "coords": tbm["coords"][index],
            "confidence": tbm["confidence"][index],
            "global_confidence": float(tbm["global_confidence"][index]),
            "candidate_id": f"tbm_{index + 1}",
            "source": "template",
        }
        for index in range(3)
    ]
    selected.extend({**item, "source": "drfold2_e2e"} for item in deep[:2])
    fallback_indices = []
    deep_available = min(len(deep), 2)
    while len(selected) < 5:
        # This mirrors scripts/run_v4_final_native_blind.py: if one D candidate
        # is available, the fifth T candidate fills the remaining D slot.
        template_index = 3 + (len(selected) - 3)
        fallback_indices.append(template_index)
        selected.append(
            {
                "coords": tbm["coords"][template_index],
                "confidence": tbm["confidence"][template_index],
                "global_confidence": float(tbm["global_confidence"][template_index]),
                "candidate_id": f"tbm_{template_index + 1}_d_fallback",
                "source": "template_fallback",
            }
        )
    return selected, {
        "drfold2_available": deep_available,
        "fallback_template_indices_zero_based": fallback_indices,
        "candidate_sources": [item["source"] for item in selected],
        "candidate_ids": [item["candidate_id"] for item in selected],
    }


def run_v4_frozen_inference(
    test_sequences: pd.DataFrame,
    artifacts: Path,
    runtime: Path,
    input_root: Path,
    *,
    work_dir: Path,
    sample_submission: pd.DataFrame | None = None,
    drfold_max_len: int = 600,
    drfold_deadline_seconds: float = 6.5 * 60 * 60,
) -> tuple[pd.DataFrame, dict]:
    """Run the frozen V4 complete pipeline and return a valid five-model CSV."""
    started_all = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts.resolve()
    runtime = runtime.resolve()
    os.environ["RNA3D_PROCESSED"] = str(artifacts)
    os.environ["RNA3D_CACHE"] = str(artifacts)
    sys.path.insert(0, str(runtime / "src"))

    from rna3d.data import io
    from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
    from rna3d.pipeline.v4_frozen import stable_seed

    priors_v1 = json.loads((artifacts / "geometry_priors.json").read_text())
    priors_v2 = json.loads((runtime / "geofuse_geometry_v2_priors.json").read_text())
    tbm_banks, tbm_status = _build_tbm_banks(test_sequences, artifacts)

    drfold_started = time.time()
    temp_root = work_dir / "drfold2"
    pretrained: dict[str, list[dict]] = {}
    drfold_status: dict[str, dict] = {}
    try:
        repo = prepare_drfold2(input_root, temp_root)
    except Exception as error:
        repo = None
        drfold_status["setup"] = {"status": "failed", "error": repr(error)}

    ordered = test_sequences.assign(sequence_length=test_sequences["sequence"].str.len()).sort_values(
        ["sequence_length", "target_id"]
    )
    for row in ordered.itertuples(index=False):
        target_id, sequence = str(row.target_id), str(row.sequence)
        elapsed = time.time() - drfold_started
        if repo is None:
            pretrained[target_id] = []
            continue
        if len(sequence) > drfold_max_len:
            pretrained[target_id] = []
            drfold_status[target_id] = {
                "status": "skipped_length_resource_fallback",
                "length": len(sequence),
                "limit": drfold_max_len,
            }
            continue
        if elapsed >= drfold_deadline_seconds:
            pretrained[target_id] = []
            drfold_status[target_id] = {
                "status": "skipped_deadline_resource_fallback",
                "elapsed_seconds": round(elapsed, 1),
            }
            continue
        try:
            values, status = run_drfold2_candidates(
                repo, temp_root, target_id, sequence, candidates=2
            )
        except Exception as error:
            values, status = [], {"status": "exception", "error": repr(error)}
        pretrained[target_id] = values
        drfold_status[target_id] = status
        print(f"[V4 DRfold2:{target_id}] {status}", flush=True)

    config = GeometryV2Config(**GEOMETRY_CONFIG)
    predictions: dict[str, np.ndarray] = {}
    assembly_status: dict[str, dict] = {}
    refinement_status: dict[str, list[dict]] = {}
    for row in test_sequences.itertuples(index=False):
        target_id, sequence = str(row.target_id), str(row.sequence)
        selected, assembly = _assemble_3t2d(
            tbm_banks[target_id], pretrained.get(target_id, [])
        )
        refined = []
        target_refinement = []
        for index, candidate in enumerate(selected):
            seed = stable_seed("v4-final-refine", target_id, "Thesis", index, "Geometry")
            try:
                coords, _ = refine_structure_v2(
                    candidate["coords"],
                    sequence,
                    priors_v1,
                    priors_v2,
                    source_confidence=candidate["confidence"],
                    global_confidence=float(candidate["global_confidence"]),
                    cfg=config,
                    device="cuda",
                    seed=seed,
                )
                coords = np.asarray(coords, dtype=np.float32)
                if coords.shape != (len(sequence), 3) or not np.isfinite(coords).all():
                    raise FloatingPointError("invalid Geometry output")
                refinement_state = "complete"
            except Exception as error:
                coords = np.asarray(candidate["coords"], dtype=np.float32)
                refinement_state = f"raw_fallback:{type(error).__name__}"
            refined.append(coords)
            target_refinement.append(
                {
                    "candidate_index": index,
                    "candidate_id": candidate["candidate_id"],
                    "source": candidate["source"],
                    "seed": seed,
                    "status": refinement_state,
                }
            )
        predictions[target_id] = np.stack(refined, axis=0)
        assembly_status[target_id] = assembly
        refinement_status[target_id] = target_refinement

    submission = io.build_submission(predictions, test_sequences)
    io.validate_submission(submission, test_sequences)
    if sample_submission is not None:
        submission = io.order_submission_like(submission, sample_submission)
    manifest = {
        "pipeline": "frozen V4: retained TBM + direct DRfold2 3T+2D + Selected Geometry",
        "frozen_commit": FROZEN_COMMIT,
        "native_labels_used": False,
        "scientific_method_changed_after_confirmatory_opening": False,
        "deployment_adapter_sha256": sha256(Path(__file__)),
        "artifact_hashes": {
            name: sha256(artifacts / name)
            for name in (
                "template_meta.parquet",
                "template_coords.pkl",
                "geometry_priors.json",
            )
        },
        "geometry_prior_sha256": sha256(runtime / "geofuse_geometry_v2_priors.json"),
        "geometry_config": asdict(config),
        "drfold2": {
            "source": "cfg97 20-checkpoint direct e2e, top two by mean pLDDT",
            "max_length_resource_limit": drfold_max_len,
            "deadline_seconds": drfold_deadline_seconds,
            "targets": drfold_status,
        },
        "tbm": tbm_status,
        "candidate_assembly": assembly_status,
        "refinement": refinement_status,
        "target_n": int(len(test_sequences)),
        "row_n": int(len(submission)),
        "elapsed_seconds": round(time.time() - started_all, 1),
    }
    return submission, manifest
