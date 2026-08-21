#!/usr/bin/env python
"""Build V5 design manifests without running or scoring a prediction method."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import re
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rna3d.data import io


OUT = REPO / "reports" / "thesis_v5"
DATA = REPO / "data" / "stanford-rna-3d-folding"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_pdb_ids(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return sorted({item.upper() for item in re.findall(r">([0-9][A-Za-z0-9]{3})_", value)})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sequences_path = DATA / "validation_sequences.csv"
    labels_path = DATA / "validation_labels.csv"
    sequences = pd.read_csv(sequences_path)
    labels = pd.read_csv(labels_path)
    rows = []
    for row in sequences.itertuples(index=False):
        references = io.get_reference_coords(labels, row.target_id)
        rows.append(
            {
                "target_id": row.target_id,
                "sequence": row.sequence,
                "sequence_sha256": hashlib.sha256(row.sequence.encode()).hexdigest(),
                "sequence_length": len(row.sequence),
                "temporal_cutoff": row.temporal_cutoff,
                "excluded_native_pdb_ids": ",".join(native_pdb_ids(row.all_sequences)),
                "native_reference_conformations": len(references),
                "v5_role": "CASP15_LOCAL_VALIDATION_AND_MODEL_DEVELOPMENT",
                "labels_visible_during_development": True,
            }
        )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 12 or manifest["target_id"].duplicated().any():
        raise RuntimeError("CASP15 manifest must contain 12 unique targets")
    manifest_path = OUT / "casp15_target_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    artifacts = {
        "casp15_sequences": sequences_path,
        "casp15_labels": labels_path,
        "local_test_sequences_alias": DATA / "test_sequences.csv",
        "sample_submission": DATA / "sample_submission.csv",
        "template_meta": REPO / "data" / "processed" / "template_meta.parquet",
        "template_coordinates": REPO / "data" / "cache" / "template_coords.pkl",
        "composite_meta": REPO / "data" / "processed" / "top1_template_meta.parquet",
        "composite_coordinates": REPO / "data" / "cache" / "top1_template_coords.pkl",
        "p0_distance_rg": REPO / "data" / "processed" / "geometry_priors.json",
        "p0_angle_torsion": REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json",
        "usalign": REPO / "external" / "binaries" / "USalign",
        "historical_tbm_runner": REPO / "data" / "interim" / "kaggle_artifact_bundle" / "kaggle" / "inference_pipeline.py",
        "historical_tbm_module": REPO / "data" / "interim" / "kaggle_artifact_bundle" / "src" / "rna3d" / "pipeline" / "tbm.py",
        "historical_hybrid_runner": REPO / "kaggle" / "hybrid_inference.py",
        "drfold_checkpoint_manifest": REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "drfold2_cfg97_checkpoint_manifest.csv",
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing V5 design artifacts: {missing}")
    full_meta = pd.read_parquet(artifacts["template_meta"])
    composite_meta = pd.read_parquet(artifacts["composite_meta"])
    with artifacts["template_coordinates"].open("rb") as handle:
        full_coordinates = pickle.load(handle)
    with artifacts["composite_coordinates"].open("rb") as handle:
        composite_coordinates = pickle.load(handle)
    receipt = {
        "status": "V5_DESIGN_INPUTS_AUDITED_BEFORE_FULL_CASP15_MATRIX",
        "prediction_methods_run": False,
        "v4_97_target_evidence_used": False,
        "casp15_manifest": {
            "path": str(manifest_path.relative_to(REPO)),
            "sha256": sha256(manifest_path),
            "target_n": len(manifest),
            "length_min": int(manifest["sequence_length"].min()),
            "length_median": float(manifest["sequence_length"].median()),
            "length_max": int(manifest["sequence_length"].max()),
        },
        "template_universe": {
            "full_chain_rows": int(len(full_meta)),
            "full_pdb_entries": int(full_meta["pdb_id"].astype(str).str.upper().nunique()),
            "full_coordinate_keys": len(full_coordinates),
            "composite_sequence_rows": int(len(composite_meta)),
            "composite_pdb_entries": int(composite_meta["pdb_id"].astype(str).str.upper().nunique()),
            "composite_coordinate_keys": len(composite_coordinates),
            "composite_is_full_key_subset": set(composite_meta["target_id"]).issubset(set(full_meta["chain_key"])),
        },
        "artifacts": {
            name: {"path": str(path.relative_to(REPO)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in artifacts.items()
        },
        "code_sha256": sha256(Path(__file__)),
    }
    (OUT / "v5_design_input_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
