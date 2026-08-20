#!/usr/bin/env python
"""Freeze the exposed V4 development cohort and protocol before scoring it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.geofuse.candidate import CandidateCache


OUT = REPO / "reports" / "thesis_v4" / "development"
MEDIUM = REPO / "data" / "processed" / "geofuse_real_oof_v2" / "medium_manifest.csv"
MASTER = REPO / "reports" / "thesis_v4" / "preregistration" / "v4_master_rna_ledger.csv"
EXPOSURE = REPO / "reports" / "thesis_v4" / "preregistration" / "development_exposure_ledger.csv"
DB_FREEZE = REPO / "reports" / "thesis_v4" / "phase1_controlled_db" / "db_controlled_freeze.json"
P0 = REPO / "reports" / "thesis_v4" / "phase1_p0" / "p0_reproduction_audit.json"
DRFOLD = REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "pretrained_provenance_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def main() -> None:
    if (OUT / "results" / "primary_inference.json").exists():
        raise RuntimeError("development performance already exists; refusing to refreeze protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    medium = pd.read_csv(MEDIUM, dtype=str)
    selected = medium[medium["split"] == "calibration"].copy()
    if len(selected) != 20:
        raise RuntimeError("historical calibration cohort no longer contains exactly 20 targets")
    master = pd.read_csv(MASTER, dtype=str).fillna("")
    selected = selected.merge(
        master[
            [
                "target_id",
                "mmseqs_sequence_similarity_cluster",
                "development_exposed",
                "known_exposure_reason",
            ]
        ],
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    if not (selected["development_exposed"] == "True").all():
        raise RuntimeError("development cohort contains a target not marked exposed")

    cache = CandidateCache(REPO / "data" / "cache" / "geofuse_candidates", "train_v2")
    inventory_rows = []
    for row in selected.itertuples(index=False):
        deep = [
            candidate
            for candidate in cache.load_target(row.target_id, row.sequence)
            if candidate.kind == "pretrained" and candidate.source.startswith("drfold2")
        ]
        for candidate in deep:
            path = cache.candidate_path(candidate)
            inventory_rows.append(
                {
                    "target_id": row.target_id,
                    "candidate_id": candidate.candidate_id,
                    "path": rel(path),
                    "sha256": sha256(path),
                    "model": candidate.model,
                    "global_confidence": candidate.global_confidence,
                }
            )
        if len(deep) < 2:
            raise RuntimeError(f"{row.target_id}: fixed-N development requires two cached DRfold2 candidates")
    inventory = pd.DataFrame(inventory_rows).sort_values(["target_id", "candidate_id"])
    inventory_path = OUT / "drfold2_candidate_manifest.csv"
    inventory.to_csv(inventory_path, index=False)

    columns = [
        "target_id",
        "sequence",
        "seq_len",
        "date",
        "excluded_pdb_ids",
        "mmseqs_sequence_similarity_cluster",
        "development_exposed",
        "known_exposure_reason",
    ]
    manifest = selected[columns].copy()
    manifest["v4_role"] = "DEVELOPMENT_COMPONENT_SELECTION_ONLY"
    manifest["final_eligible"] = False
    manifest_path = OUT / "development_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    protocol = {
        "status": "FROZEN_BEFORE_V4_DEVELOPMENT_SCORING",
        "performance_accessed_by_this_script": False,
        "cohort": {
            "selection_rule": "all 20 targets in the historical medium-manifest calibration split",
            "reason": "all were already native/per-target exposed and are eligible for development but never confirmatory evidence",
            "target_n": len(manifest),
            "cluster_n": int(manifest["mmseqs_sequence_similarity_cluster"].nunique()),
            "manifest_path": rel(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "common_control": json.loads(DB_FREEZE.read_text())["database"],
        "raw_boundary": "after coordinate transfer and frozen gap completion; before every refiner",
        "candidate_budget": 5,
        "h1": "J-controlled-TBM-raw versus Thesis-TBM-raw",
        "h2": "5T versus 3T+2D using the exact same Thesis TBM ranked list",
        "h3": "Geometry versus Simple using identical 3T+2D raw candidates and raw-selected native conformation",
        "refiners": {
            "Raw": "identity",
            "John-original": "captured rules with candidate raw confidence",
            "John-fixed": "captured rules with confidence=0.5",
            "Simple": "300 Adam steps, lr=0.04, source=3.0, backbone=1.0, fixed strength; other terms off",
            "Geometry": json.loads(P0.read_text())["production"]["geometry_config"],
        },
        "fallback": {
            "missing_tbm": "deterministic method-specific de-novo candidates",
            "missing_drfold2_slot": "next Thesis TBM fallback while preserving all-target H2",
            "refiner_failure": "raw candidate plus failure flag",
            "whole_bank_failure": "bank TM=0",
        },
        "statistics": {
            "target_unit": "RNA",
            "block_unit": "regenerated MMseqs sequence-similarity cluster",
            "bootstrap_replicates": 10000,
            "permutation_replicates": 100000,
            "seed": 20260819,
            "development_interpretation": "selection evidence only; not confirmatory H1-H3 claims",
        },
        "pretrained": {
            "drfold2_candidate_manifest_path": rel(inventory_path),
            "drfold2_candidate_manifest_sha256": sha256(inventory_path),
            "rclm_membership": "UNAVAILABLE",
            "complete_pretrained_time_safe_claim": False,
            "audit_sha256": sha256(DRFOLD),
        },
        "source_files": {
            rel(MEDIUM): sha256(MEDIUM),
            rel(MASTER): sha256(MASTER),
            rel(EXPOSURE): sha256(EXPOSURE),
            rel(DB_FREEZE): sha256(DB_FREEZE),
            rel(P0): sha256(P0),
        },
        "base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip(),
    }
    protocol_path = OUT / "development_protocol_freeze.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
