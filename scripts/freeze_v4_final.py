#!/usr/bin/env python
"""Freeze the V4 final method and confirmatory manifest without reading performance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "thesis_v4" / "final_freeze"
MASTER = REPO / "reports" / "thesis_v4" / "preregistration" / "v4_master_rna_ledger.csv"
TRAIN = REPO / "data" / "stanford-rna-3d-folding" / "train_sequences.v2.csv"
OVERLAP = REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "drfold2_overlap_by_target.csv"
EXTERNAL = REPO / "reports" / "thesis_v4" / "phase1_completion" / "external_exposure_audit.csv"
DEV_SUMMARY = REPO / "reports" / "thesis_v4" / "development" / "results" / "development_summary.json"
SELECTED = REPO / "reports" / "thesis_v4" / "development" / "results" / "rq3_refinement" / "selected_inference_and_decision.json"
SELECTED_FREEZE = REPO / "reports" / "thesis_v4" / "development" / "selected_refiner_freeze.json"
RUNNER = REPO / "scripts" / "run_v4_final_native_blind.py"
PIPELINE_MODULE = REPO / "src" / "rna3d" / "pipeline" / "v4_frozen.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def artifact(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": rel(path), "sha256": sha256(path)}


def normalized_sequence(value: str) -> str:
    return str(value).upper().replace("T", "U")


def referenced_pdbs(target_id: str, all_sequences: str) -> str:
    values = {target_id.split("_", 1)[0].upper()}
    values.update(value.upper() for value in re.findall(r">([0-9][A-Za-z0-9]{3})_", all_sequences))
    return ",".join(sorted(value for value in values if re.fullmatch(r"[0-9A-Z]{4}", value)))


def main() -> None:
    existing = OUT / "final_method_freeze.json"
    if existing.exists():
        raise FileExistsError("final method is already frozen; do not replace it after checkpoint creation")
    selected = json.loads(SELECTED.read_text())
    if not selected["development_gate_pass"] or selected["selected_final_refiner"] != "Geometry-selected":
        raise RuntimeError("development decision does not retain selected Geometry")

    master = pd.read_csv(MASTER, dtype=str).fillna("")
    overlap = pd.read_csv(OVERLAP, dtype=str).fillna("")
    external = pd.read_csv(EXTERNAL, dtype=str).fillna("")
    sequences = pd.read_csv(TRAIN, dtype=str).fillna("")
    candidates = master[master["provisional_after_repository_audit"].str.lower() == "true"].copy()
    candidates = candidates.merge(overlap, on="target_id", how="left", validate="one_to_one")
    candidates = candidates.merge(external, on="target_id", how="left", validate="one_to_one")
    candidates = candidates.merge(
        sequences[["target_id", "sequence", "all_sequences"]],
        on="target_id",
        how="left",
        validate="one_to_one",
    )
    eligible = candidates[
        (candidates["v4_status"] == "STRUCTURAL_OVERLAP_PASS")
        & (candidates["audit_decision"] == "NO_ADDITIONAL_EVIDENCE_FOUND")
        & (candidates["development_exposed"].str.lower() == "false")
        & (candidates["exposed_cluster"].str.lower() == "false")
        & (candidates["exact_duplicate_excluded"].str.lower() == "false")
    ].copy()
    eligible["sequence"] = eligible["sequence"].map(normalized_sequence)
    observed_hash = eligible["sequence"].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    if not (observed_hash == eligible["normalized_sequence_sha256"]).all():
        bad = eligible.loc[observed_hash != eligible["normalized_sequence_sha256"], "target_id"].tolist()
        raise RuntimeError(f"sequence hash mismatch: {bad}")
    if not (eligible["sequence"].str.len() == eligible["sequence_length"].astype(int)).all():
        raise RuntimeError("sequence length mismatch")
    eligible["excluded_pdb_ids"] = [
        referenced_pdbs(target_id, all_sequences)
        for target_id, all_sequences in zip(eligible["target_id"], eligible["all_sequences"])
    ]
    eligible["v4_role"] = "FINAL_CONFIRMATORY_LOCKED_NATIVE_BLIND"
    eligible["native_performance_opened"] = False
    manifest = eligible[
        [
            "target_id",
            "sequence",
            "sequence_length",
            "temporal_cutoff",
            "excluded_pdb_ids",
            "mmseqs_sequence_similarity_cluster",
            "v4_status",
            "audit_decision",
            "drfold_language_model_provenance_status",
            "v4_role",
            "native_performance_opened",
        ]
    ].rename(columns={"sequence_length": "seq_len", "v4_status": "drfold_structural_overlap_status"})
    manifest = manifest.sort_values("target_id").reset_index(drop=True)
    if len(manifest) != 97 or manifest["mmseqs_sequence_similarity_cluster"].nunique() != 86:
        raise RuntimeError(
            f"unexpected final pool: N={len(manifest)}, K={manifest['mmseqs_sequence_similarity_cluster'].nunique()}"
        )

    OUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUT / "final_target_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    hashes = {
        "master_ledger": artifact(MASTER),
        "train_sequences_v2": artifact(TRAIN),
        "structural_overlap_audit": artifact(OVERLAP),
        "external_exposure_audit": artifact(EXTERNAL),
        "development_summary": artifact(DEV_SUMMARY),
        "selected_refiner_pre_score_freeze": artifact(SELECTED_FREEZE),
        "selected_refiner_result": artifact(SELECTED),
        "retained_tbm_freeze": artifact(REPO / "reports" / "thesis_v4" / "development" / "retained_tbm_freeze.json"),
        "controlled_db_freeze": artifact(REPO / "reports" / "thesis_v4" / "phase1_controlled_db" / "db_controlled_freeze.json"),
        "template_meta": artifact(REPO / "data" / "processed" / "template_meta.parquet"),
        "template_coordinates": artifact(REPO / "data" / "cache" / "template_coords.pkl"),
        "p0_distance_rg": artifact(REPO / "data" / "processed" / "geometry_priors.json"),
        "p0_angle_torsion": artifact(REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json"),
        "drfold2_checkpoint_manifest": artifact(REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "drfold2_cfg97_checkpoint_manifest.csv"),
        "drfold2_provenance": artifact(REPO / "reports" / "thesis_v4" / "phase1_pretrained" / "pretrained_provenance_audit.json"),
        "statistics_implementation": artifact(REPO / "src" / "rna3d" / "eval" / "v4_statistics.py"),
        "drfold2_generation_runner": artifact(REPO / "scripts" / "run_drfold2_candidates.py"),
        "pretrained_structure_import": artifact(REPO / "src" / "rna3d" / "geofuse" / "structure_io.py"),
        "geometry_refiner": artifact(REPO / "src" / "rna3d" / "geofuse" / "refine_v2.py"),
        "john_rule_refiner": artifact(REPO / "src" / "rna3d" / "refine" / "rule_based.py"),
        "john_tbm_baseline": artifact(REPO / "src" / "rna3d" / "baselines" / "top1.py"),
        "gap_completion": artifact(REPO / "src" / "rna3d" / "template" / "gap_fill.py"),
        "coordinate_alignment": artifact(REPO / "src" / "rna3d" / "template" / "align.py"),
        "de_novo_fallback": artifact(REPO / "src" / "rna3d" / "geometry" / "denovo.py"),
        "environment": artifact(REPO / "environment.yml"),
    }
    decisions = [
        ("P0 production prior", "KEEP", "exactly reproduced; no P1 was selected"),
        ("MMseqs retrieval branch", "DROP", "no TM or availability gain on development"),
        ("full composite score", "DROP", "global-only retrieval was better on development"),
        ("global-only exhaustive retrieval", "KEEP", "development-supported simplification"),
        ("identity by coverage rank", "KEEP_HEURISTIC", "positive but inconclusive; not a demonstrated contribution"),
        ("template completeness rank", "DROP", "zero incremental effect beyond coverage"),
        ("distinct-PDB selection", "KEEP_SAFEGUARD", "redundancy safeguard; no accuracy claim"),
        ("linear gap completion", "KEEP", "simplest method not beaten by curved completion"),
        ("3T+2D allocation", "KEEP_CONFIRMATORY", "positive but inconclusive H2 development evidence"),
        ("Boltz", "DROP", "required artifacts and provenance unavailable"),
        ("adaptive source strength", "KEEP", "fixed control narrowly failed the TM preservation gate"),
        ("candidate-derived angle/torsion context", "DROP", "did not beat unconditional global prior"),
        ("Rg objective", "DROP", "no clear independent benefit"),
        ("selected Geometry", "KEEP", "passed development H3 and TM preservation gate"),
    ]
    decision_frame = pd.DataFrame(decisions, columns=["component", "decision", "development_basis"])
    decision_path = OUT / "component_decisions.csv"
    decision_frame.to_csv(decision_path, index=False)

    freeze = {
        "status": "FROZEN_BEFORE_FINAL_CONFIRMATORY_NATIVE_PERFORMANCE_OPENING",
        "final_test_native_performance_accessed": False,
        "freeze_date": "2026-08-20",
        "target_manifest": {
            "path": rel(manifest_path),
            "sha256": sha256(manifest_path),
            "target_n": len(manifest),
            "cluster_n": int(manifest["mmseqs_sequence_similarity_cluster"].nunique()),
            "selection": "all 97 provisional targets passing DRfold2 structural exact/homolog screen and accessible exposure audit",
            "sampling": "none",
        },
        "final_pipeline": {
            "template_method": "global-only exhaustive retrieval; identity*coverage rank; distinct-PDB safeguard; linear gap completion",
            "candidate_bank": "3T+2D, fixed N=5; missing D slots use the next frozen T candidates",
            "pretrained_source": "direct DRfold2 cfg97 20-checkpoint e2e candidates, top two by mean pLDDT",
            "refiner": "Geometry-selected",
            "geometry_config": selected["selected_config"],
            "prior": "P0-production",
        },
        "confirmatory_comparisons": {
            "H1": "retained Thesis 5T raw minus J-controlled 5T raw",
            "H2": "Thesis 3T+2D raw minus Thesis 5T raw",
            "H3": "Simple minus selected Geometry on SW-RMSD9, plus Geometry minus Simple bank TM safeguard",
            "factorial": "Raw, John-original, John-fixed, Simple, selected Geometry on identical raw banks",
        },
        "statistics": {
            "target_unit": "RNA",
            "dependence_block": "MMseqs sequence-similarity cluster",
            "bootstrap_replicates": 10000,
            "permutation_replicates": 100000,
            "seed": 20260819,
            "holm_family": ["H1", "H2", "H3"],
            "h3_tm_noninferiority_margin": -0.005,
        },
        "limitations": {
            "rclm_membership": "UNAVAILABLE; no complete pretrained-model time-safety claim",
            "external_memory": "researcher memory could not be independently audited; later positive evidence must trigger exclusion before opening",
            "john_hybrid": "partial public artifact reproduction only; not exact winning submission",
            "boltz": "excluded because model artifacts and provenance were unavailable",
        },
        "artifact_hashes": hashes,
        "generation_code": {
            "runner_path": rel(RUNNER),
            "runner_sha256": sha256(RUNNER),
            "pipeline_module_path": rel(PIPELINE_MODULE),
            "pipeline_module_sha256": sha256(PIPELINE_MODULE),
            "development_tbm_replay": "20/20 exact coordinate, residue-confidence and global-confidence arrays",
        },
        "base_commit_before_final_freeze": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "opening_rule": "STOP. Do not load final native coordinates or compute TM, SW-RMSD, lDDT, or diagnostics until user reviews this freeze.",
    }
    freeze_path = OUT / "final_method_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")

    execution = {
        "status": "READY_NATIVE_BLIND_ONLY",
        "final_performance_accessed": False,
        "validate": "python scripts/run_v4_final_native_blind.py validate",
        "tbm": "python scripts/run_v4_final_native_blind.py build-tbm",
        "drfold2": "python scripts/run_drfold2_candidates.py --repo /home/datdo/.cache/rna3d/external/DRfold2 --output-root data/cache/v4_final_drfold2 --manifest reports/thesis_v4/final_freeze/final_target_manifest.csv --mode cfg97 --e2e-only --e2e-candidates 2",
        "assemble": "python scripts/run_v4_final_native_blind.py assemble-raw --drfold-root data/cache/v4_final_drfold2",
        "refine": "python scripts/run_v4_final_native_blind.py refine --device cuda --workers 8",
        "status": "python scripts/run_v4_final_native_blind.py status",
        "evaluation": "BLOCKED_NOT_IMPLEMENTED_BEFORE_FINAL_OPENING_RECEIPT",
    }
    (OUT / "native_blind_execution_plan.json").write_text(json.dumps(execution, indent=2) + "\n")
    opening = {
        "checkpoint": "MANDATORY_USER_REVIEW_BEFORE_FINAL_LABEL_OPENING",
        "final_performance_accessed": False,
        "method_frozen": True,
        "manifest_frozen": True,
        "generation_setup_ready": True,
        "evaluation_authorized": False,
        "required_next_receipt": "user authorization plus final freeze commit and tag",
    }
    (OUT / "final_opening_checkpoint.json").write_text(json.dumps(opening, indent=2) + "\n")
    print(json.dumps({"target_n": len(manifest), "cluster_n": manifest["mmseqs_sequence_similarity_cluster"].nunique(), "freeze": rel(freeze_path), "native_performance_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
