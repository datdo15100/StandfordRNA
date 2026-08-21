#!/usr/bin/env python
"""Freeze V4 native-blind outputs and run the one-time confirmatory evaluation.

The ``prepare-opening`` command never loads native coordinates.  It validates and
hashes every scientific output required by the frozen final protocol, then writes
the explicit receipt that authorizes one native-label opening.  The ``evaluate``
command refuses to run without that receipt and refuses to overwrite an existing
opening event or confirmatory result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.data import io
from rna3d.eval.local_metrics import local_accuracy_metrics
from rna3d.eval.usalign import score_target
from rna3d.eval.v4_statistics import holm_step_down, primary_inference
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.structure_io import import_structure
from rna3d.paths import comp_file, usalign_bin


FINAL = REPO / "reports" / "thesis_v4" / "final_freeze"
RESULTS = REPO / "reports" / "thesis_v4" / "confirmatory"
MANIFEST = FINAL / "final_target_manifest.csv"
METHOD_FREEZE = FINAL / "final_method_freeze.json"
NATIVE_BLIND_ROOT = REPO / "data" / "cache" / "v4_final_native_blind"
DRFOLD_ROOT = REPO / "data" / "cache" / "v4_final_drfold2"
OUTPUT_MANIFEST = FINAL / "native_blind_output_manifest.csv"
OUTPUT_FREEZE = FINAL / "native_blind_output_freeze.json"
OPENING_RECEIPT = FINAL / "final_opening_receipt.json"
OPENING_EVENT = RESULTS / "native_label_opening_event.json"
FINAL_INFERENCE = RESULTS / "primary_inference.json"
P0_DISTANCE = REPO / "data" / "processed" / "geometry_priors.json"
P0_GEOMETRY = REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json"
EXECUTION_AUDIT = FINAL / "native_blind_execution_audit.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def frozen_inputs() -> tuple[dict, pd.DataFrame]:
    freeze = json.loads(METHOD_FREEZE.read_text())
    if sha256(MANIFEST) != freeze["target_manifest"]["sha256"]:
        raise RuntimeError("final target manifest differs from the frozen hash")
    for name, record in freeze["artifact_hashes"].items():
        path = REPO / record["path"]
        if not path.exists() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input mismatch: {name} ({path})")
    generation = freeze["generation_code"]
    for name, path_key, hash_key in (
        ("native-blind runner", "runner_path", "runner_sha256"),
        ("frozen pipeline module", "pipeline_module_path", "pipeline_module_sha256"),
    ):
        path = REPO / generation[path_key]
        if not path.exists() or sha256(path) != generation[hash_key]:
            raise RuntimeError(f"frozen generation code mismatch: {name} ({path})")
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    if len(manifest) != 97 or manifest["mmseqs_sequence_similarity_cluster"].nunique() != 86:
        raise RuntimeError("final manifest no longer contains the frozen 97 targets / 86 clusters")
    if (manifest["native_performance_opened"].str.lower() != "false").any():
        raise RuntimeError("manifest native-performance flag is no longer false")
    return freeze, manifest


def _expected_refined_shapes(length: int) -> dict[str, tuple[int, ...]]:
    return {
        f"{bank}__{setting}": (5, length, 3)
        for bank in ("J", "Thesis")
        for setting in ("Raw", "John_original", "John_fixed", "Simple", "Geometry")
    }


def validate_scientific_outputs(manifest: pd.DataFrame) -> dict:
    errors: list[str] = []
    counts = {
        "target_n": len(manifest),
        "tbm_banks": 0,
        "raw_banks": 0,
        "refined_banks": 0,
        "drfold_ret": 0,
        "drfold_selected_pdb": 0,
        "drfold_confidence": 0,
        "drfold_prior": 0,
        "drfold_fallback_targets": 0,
    }
    for row in manifest.itertuples(index=False):
        target_id = row.target_id
        length = len(row.sequence)
        target = NATIVE_BLIND_ROOT / target_id
        tbm_path = target / "tbm_banks.npz"
        raw_path = target / "raw_banks.npz"
        refined_path = target / "refined_banks.npz"
        for name, path in (("TBM", tbm_path), ("Raw", raw_path), ("Refined", refined_path)):
            if not path.exists():
                errors.append(f"{target_id}: missing {name} bank")
        if errors and (not tbm_path.exists() or not raw_path.exists() or not refined_path.exists()):
            continue
        counts["tbm_banks"] += 1
        counts["raw_banks"] += 1
        counts["refined_banks"] += 1
        with np.load(tbm_path, allow_pickle=False) as payload:
            expected = {
                "j_coords": (5, length, 3), "j_conf": (5, length), "j_global_conf": (5,),
                "t_coords": (5, length, 3), "t_conf": (5, length), "t_global_conf": (5,),
            }
            for key, shape in expected.items():
                if key not in payload.files or payload[key].shape != shape or not np.isfinite(payload[key]).all():
                    errors.append(f"{target_id}: invalid TBM array {key}")
        with np.load(raw_path, allow_pickle=False) as payload:
            expected = {
                "j_coords": (5, length, 3), "t_coords": (5, length, 3),
                "d_coords": (2, length, 3), "thesis_coords": (5, length, 3),
                "j_conf": (5, length), "t_conf": (5, length),
                "d_conf": (2, length), "thesis_conf": (5, length),
                "j_global_conf": (5,), "t_global_conf": (5,),
                "d_global_conf": (2,), "thesis_global_conf": (5,),
            }
            for key, shape in expected.items():
                if key not in payload.files or payload[key].shape != shape or not np.isfinite(payload[key]).all():
                    errors.append(f"{target_id}: invalid Raw array {key}")
            available = int(payload["drfold2_available"].item())
            raw_deep_coords = np.asarray(payload["d_coords"], dtype=np.float32)
            if available != 2:
                counts["drfold_fallback_targets"] += 1
            if not np.array_equal(payload["thesis_coords"][:3], payload["t_coords"][:3]):
                errors.append(f"{target_id}: 3T allocation mismatch")
            if available == 2 and not np.array_equal(payload["thesis_coords"][3:], payload["d_coords"]):
                errors.append(f"{target_id}: 2D allocation mismatch")
        with np.load(refined_path, allow_pickle=False) as payload:
            expected = _expected_refined_shapes(length)
            if set(payload.files) != set(expected):
                errors.append(f"{target_id}: refined bank keys mismatch")
            for key, shape in expected.items():
                if key not in payload.files or payload[key].shape != shape or not np.isfinite(payload[key]).all():
                    errors.append(f"{target_id}: invalid refined array {key}")
        drfold = DRFOLD_ROOT / target_id
        rets = sorted((drfold / "rets_dir").glob("*.ret"))
        selected = sorted((drfold / "e2e_relax").glob("model_*.pdb"))
        confidence = sorted((drfold / "e2e_relax").glob("plddt_model_*.npz"))
        priors = sorted((drfold / "e2e_relax").glob("priors_model_*.npz"))
        counts["drfold_ret"] += len(rets)
        counts["drfold_selected_pdb"] += len(selected)
        counts["drfold_confidence"] += len(confidence)
        counts["drfold_prior"] += len(priors)
        if (len(rets), len(selected), len(confidence), len(priors)) != (20, 2, 2, 2):
            errors.append(f"{target_id}: incomplete DRfold2 artifacts")
        if any(path.stat().st_size == 0 for path in rets):
            errors.append(f"{target_id}: empty DRfold2 checkpoint output")
        imported = []
        for index, path in enumerate(selected, start=1):
            try:
                candidate = import_structure(
                    path,
                    target_id=target_id,
                    sequence=row.sequence,
                    candidate_id=f"freeze_check_{index}",
                    source="drfold2_e2e",
                    model="cfg97_20ckpt_e2e",
                    default_confidence=0.5,
                )
                if candidate.coords.shape != (length, 3) or not np.isfinite(candidate.coords).all():
                    errors.append(f"{target_id}: invalid selected DRfold2 coordinates")
                imported.append(candidate)
            except Exception as error:
                errors.append(f"{target_id}: DRfold2 import error {type(error).__name__}:{error}")
        imported = sorted(imported, key=lambda item: (-item.global_confidence, item.candidate_id))
        if len(imported) == 2 and not np.array_equal(
            raw_deep_coords, np.asarray([item.coords for item in imported], dtype=np.float32)
        ):
            errors.append(f"{target_id}: assembled 2D coordinates differ from selected PDB files")
        for path in confidence:
            try:
                with np.load(path, allow_pickle=False) as payload:
                    values = payload["plddt"]
                    if values.shape != (length,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                        errors.append(f"{target_id}: invalid {path.name}")
            except Exception as error:
                errors.append(f"{target_id}: confidence error {type(error).__name__}:{error}")
        for path in priors:
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if not payload.files or any(not np.isfinite(payload[key]).all() for key in payload.files):
                        errors.append(f"{target_id}: invalid {path.name}")
            except Exception as error:
                errors.append(f"{target_id}: prior error {type(error).__name__}:{error}")
        try:
            records = json.loads((drfold / "e2e_relax" / "manifest.json").read_text())
            scores = np.asarray([item["global_confidence"] for item in records], dtype=float)
            if [item["rank"] for item in records] != [1, 2] or not np.isfinite(scores).all():
                errors.append(f"{target_id}: invalid DRfold2 ranking manifest")
        except Exception as error:
            errors.append(f"{target_id}: DRfold2 manifest error {type(error).__name__}:{error}")
    failure_path = NATIVE_BLIND_ROOT / "refinement_failures.csv"
    if not failure_path.exists():
        errors.append("missing refinement failure ledger")
        refinement_failures = None
    else:
        failures = pd.read_csv(failure_path)
        expected_rows = len(manifest) * 2 * 2 * 5
        if len(failures) != expected_rows:
            errors.append(f"refinement failure ledger has {len(failures)} rows, expected {expected_rows}")
        refinement_failures = int(failures["failed"].astype(bool).sum())
    if errors:
        raise RuntimeError("native-blind completeness gate failed:\n" + "\n".join(errors[:50]))
    counts["refinement_failure_count"] = refinement_failures
    return counts


def scientific_files(manifest: pd.DataFrame) -> Iterable[tuple[str, str, Path]]:
    for row in manifest.itertuples(index=False):
        target_id = row.target_id
        native = NATIVE_BLIND_ROOT / target_id
        for name in ("tbm_banks.npz", "tbm_banks.json", "raw_banks.npz", "refined_banks.npz"):
            yield "final_bank", target_id, native / name
        drfold = DRFOLD_ROOT / target_id
        for path in sorted((drfold / "rets_dir").glob("*.ret")):
            yield "drfold_checkpoint_output", target_id, path
        for pattern in ("model_*.pdb", "plddt_model_*.npz", "priors_model_*.npz", "manifest.json"):
            for path in sorted((drfold / "e2e_relax").glob(pattern)):
                yield "drfold_selected_output", target_id, path
    for path in (
        NATIVE_BLIND_ROOT / "tbm_generation_manifest.csv",
        NATIVE_BLIND_ROOT / "raw_generation_manifest.csv",
        NATIVE_BLIND_ROOT / "refinement_generation_manifest.csv",
        NATIVE_BLIND_ROOT / "refinement_failures.csv",
    ):
        yield "generation_ledger", "ALL", path


def cmd_prepare_opening(_: argparse.Namespace) -> None:
    if FINAL_INFERENCE.exists() or OPENING_EVENT.exists():
        raise RuntimeError("native labels have already been opened; preparation cannot be repeated")
    freeze, manifest = frozen_inputs()
    counts = validate_scientific_outputs(manifest)
    rows = []
    for group, target_id, path in scientific_files(manifest):
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "artifact_group": group,
                "target_id": target_id,
                "path": str(path.relative_to(REPO)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["artifact_group", "target_id", "path"])
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_MANIFEST, index=False)
    output_freeze = {
        "status": "NATIVE_BLIND_FINAL_OUTPUTS_FROZEN_BEFORE_LABEL_OPENING",
        "created_at_utc": now_utc(),
        "final_performance_accessed": False,
        "frozen_commit": git_output("rev-parse", "HEAD"),
        "frozen_tag": "v4-pre-final-opening-2026-08-20",
        "target_manifest_sha256": sha256(MANIFEST),
        "method_freeze_sha256": sha256(METHOD_FREEZE),
        "output_manifest_path": str(OUTPUT_MANIFEST.relative_to(REPO)),
        "output_manifest_sha256": sha256(OUTPUT_MANIFEST),
        "artifact_file_count": len(frame),
        "artifact_total_bytes": int(frame["bytes"].sum()),
        "completeness": counts,
        "method_change_after_freeze": False,
    }
    if not EXECUTION_AUDIT.exists():
        raise FileNotFoundError("native-blind execution audit must be finalized before opening")
    write_json(OUTPUT_FREEZE, output_freeze)
    evaluator_paths = {
        "native_blind_execution_audit": EXECUTION_AUDIT,
        "confirmatory_orchestrator": Path(__file__),
        "statistics": REPO / "src/rna3d/eval/v4_statistics.py",
        "tm_wrapper": REPO / "src/rna3d/eval/usalign.py",
        "usalign_binary": usalign_bin(),
        "local_metrics": REPO / "src/rna3d/eval/local_metrics.py",
        "geometry_diagnostics": REPO / "src/rna3d/geofuse/geometry_v2.py",
        "native_io": REPO / "src/rna3d/data/io.py",
    }
    receipt = {
        "receipt": "EXPLICIT_FINAL_NATIVE_LABEL_OPENING_AUTHORIZATION",
        "created_at_utc": now_utc(),
        "operator": f"{os.environ.get('USER', 'UNKNOWN')} with Codex",
        "user_authorization": "GO for the confirmatory final evaluation",
        "native_labels_opened_at_receipt_creation": False,
        "allowed_openings": 1,
        "base_commit_before_final_freeze": freeze.get("base_commit_before_final_freeze"),
        "final_freeze_commit": git_output("rev-parse", "HEAD"),
        "final_freeze_tag": "v4-pre-final-opening-2026-08-20",
        "target_n": int(len(manifest)),
        "cluster_n": int(manifest["mmseqs_sequence_similarity_cluster"].nunique()),
        "target_manifest_sha256": sha256(MANIFEST),
        "method_freeze_sha256": sha256(METHOD_FREEZE),
        "native_blind_output_freeze_sha256": sha256(OUTPUT_FREEZE),
        "native_blind_output_manifest_sha256": sha256(OUTPUT_MANIFEST),
        "evaluator_hashes": {
            name: {"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
            for name, path in evaluator_paths.items()
        },
        "statistics": freeze["statistics"],
        "scientific_method_change_after_output_freeze": False,
        "next_action": "Run evaluate once; do not retune or overwrite results",
    }
    write_json(OPENING_RECEIPT, receipt)
    print(json.dumps({"output_freeze": output_freeze, "opening_receipt": receipt}, indent=2))


def validate_opening_receipt() -> tuple[dict, dict, pd.DataFrame]:
    freeze, manifest = frozen_inputs()
    if not OUTPUT_FREEZE.exists() or not OUTPUT_MANIFEST.exists() or not OPENING_RECEIPT.exists():
        raise RuntimeError("prepare-opening must complete before native-label evaluation")
    receipt = json.loads(OPENING_RECEIPT.read_text())
    if receipt["allowed_openings"] != 1 or receipt["native_labels_opened_at_receipt_creation"]:
        raise RuntimeError("invalid final-opening receipt")
    if receipt["target_manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("opening receipt target hash mismatch")
    if receipt["method_freeze_sha256"] != sha256(METHOD_FREEZE):
        raise RuntimeError("opening receipt method hash mismatch")
    if receipt["native_blind_output_freeze_sha256"] != sha256(OUTPUT_FREEZE):
        raise RuntimeError("opening receipt output-freeze hash mismatch")
    if receipt["native_blind_output_manifest_sha256"] != sha256(OUTPUT_MANIFEST):
        raise RuntimeError("opening receipt output-manifest hash mismatch")
    for record in pd.read_csv(OUTPUT_MANIFEST).itertuples(index=False):
        path = REPO / record.path
        if not path.exists() or sha256(path) != record.sha256:
            raise RuntimeError(f"native-blind output changed after freeze: {record.path}")
    for name, record in receipt["evaluator_hashes"].items():
        path = REPO / record["path"]
        if not path.exists() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"evaluator changed after opening receipt: {name}")
    return freeze, receipt, manifest


def references_for(labels: pd.DataFrame, target_id: str) -> list[np.ndarray]:
    return io.get_reference_coords(labels, target_id)


def _array_digest(coords: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(coords, np.float32).tobytes()).hexdigest()


class Scorer:
    def __init__(self) -> None:
        self.cache: dict[tuple, float] = {}
        self.failures: list[dict] = []

    def bank(self, target_id: str, name: str, coords: np.ndarray, refs: list[np.ndarray], sequence: str) -> float:
        key = (target_id, "bank", _array_digest(coords))
        if key in self.cache:
            return self.cache[key]
        try:
            value = float(score_target(list(coords), refs, list(sequence)))
        except Exception as error:
            value = 0.0
            self.failures.append(
                {"target_id": target_id, "scope": "bank", "setting": name, "reason": f"{type(error).__name__}:{error}"}
            )
        self.cache[key] = value
        return value

    def candidate(self, target_id: str, name: str, coords: np.ndarray, reference: np.ndarray, sequence: str) -> float:
        key = (target_id, "candidate", _array_digest(coords), _array_digest(reference))
        if key in self.cache:
            return self.cache[key]
        try:
            value = float(score_target([coords], [reference], list(sequence)))
        except Exception as error:
            value = 0.0
            self.failures.append(
                {"target_id": target_id, "scope": "candidate", "setting": name, "reason": f"{type(error).__name__}:{error}"}
            )
        self.cache[key] = value
        return value


def locked_reference(
    scorer: Scorer, target_id: str, raw: np.ndarray, refs: list[np.ndarray], sequence: str
) -> tuple[int, float]:
    scores = [scorer.candidate(target_id, f"reference_lock_{index}", raw, ref, sequence) for index, ref in enumerate(refs)]
    index = int(np.argmax(scores))
    return index, float(scores[index])


def inference(name: str, frame: pd.DataFrame, delta: str) -> dict:
    result = primary_inference(name, frame[delta], frame["cluster_id"]).to_dict()
    result["evidence_role"] = "confirmatory final evidence"
    return result


def self_tm(scorer: Scorer, target_id: str, bank_name: str, bank: np.ndarray, sequence: str) -> tuple[float, float]:
    values = []
    for left in range(len(bank)):
        for right in range(left + 1, len(bank)):
            forward = scorer.candidate(
                target_id, f"self_tm_{bank_name}_{left}_{right}_forward", bank[left], bank[right], sequence
            )
            reverse = scorer.candidate(
                target_id, f"self_tm_{bank_name}_{left}_{right}_reverse", bank[right], bank[left], sequence
            )
            values.append((forward + reverse) / 2.0)
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(np.mean(array >= 0.95))


def cmd_evaluate(_: argparse.Namespace) -> None:
    if FINAL_INFERENCE.exists() or OPENING_EVENT.exists():
        raise RuntimeError("confirmatory native labels were already opened; refusing a second run")
    freeze, receipt, manifest = validate_opening_receipt()
    RESULTS.mkdir(parents=True, exist_ok=True)
    event = {
        "event": "FINAL_NATIVE_LABEL_OPENING",
        "opened_at_utc": now_utc(),
        "opening_number": 1,
        "allowed_openings": 1,
        "opening_receipt_sha256": sha256(OPENING_RECEIPT),
        "target_manifest_sha256": sha256(MANIFEST),
        "method_change_after_opening": False,
        "evaluation_completed": False,
    }
    write_json(OPENING_EVENT, event)

    # This is the single deliberate introduction of final native coordinates.
    labels_path = comp_file("train_labels_v2")
    labels_all = io.load_labels("train_v2")
    target_ids = labels_all["ID"].map(io.target_id_of)
    labels = labels_all[target_ids.isin(set(manifest["target_id"]))].copy()
    observed = set(labels["ID"].map(io.target_id_of))
    if observed != set(manifest["target_id"]):
        raise RuntimeError("opened labels do not cover exactly the frozen target manifest")
    manifest_sequences = manifest.set_index("target_id")["sequence"].to_dict()
    sequence_mismatches = [
        target_id
        for target_id in sorted(observed)
        if io.get_sequence_from_labels(labels, target_id) != manifest_sequences[target_id]
    ]
    if sequence_mismatches:
        raise RuntimeError(f"native-label sequence mismatch: {sequence_mismatches[:10]}")
    event.update(
        {
            "native_label_path": str(labels_path.relative_to(REPO)),
            "native_label_sha256": sha256(labels_path),
            "opened_target_n": len(observed),
            "label_rows_for_frozen_targets": len(labels),
        }
    )
    write_json(OPENING_EVENT, event)

    priors_v1 = json.loads(P0_DISTANCE.read_text())
    priors_v2 = json.loads(P0_GEOMETRY.read_text())
    scorer = Scorer()
    primary_rows: list[dict] = []
    bank_rows: list[dict] = []
    candidate_rows: list[dict] = []
    fixed_n_rows: list[dict] = []
    settings = ("Raw", "John_original", "John_fixed", "Simple", "Geometry")
    for number, target in enumerate(manifest.itertuples(index=False), start=1):
        refs = references_for(labels, target.target_id)
        if not refs:
            raise RuntimeError(f"{target.target_id}: no valid native conformation")
        raw_path = NATIVE_BLIND_ROOT / target.target_id / "raw_banks.npz"
        refined_path = NATIVE_BLIND_ROOT / target.target_id / "refined_banks.npz"
        with np.load(raw_path, allow_pickle=False) as raw, np.load(refined_path, allow_pickle=False) as refined:
            j5 = np.asarray(raw["j_coords"], dtype=np.float32)
            t5 = np.asarray(raw["t_coords"], dtype=np.float32)
            d2 = np.asarray(raw["d_coords"], dtype=np.float32)
            mixed = np.asarray(raw["thesis_coords"], dtype=np.float32)
            j_score = scorer.bank(target.target_id, "J-controlled-5T-Raw", j5, refs, target.sequence)
            t_score = scorer.bank(target.target_id, "Thesis-5T-Raw", t5, refs, target.sequence)
            mixed_score = scorer.bank(target.target_id, "Thesis-3T+2D-Raw", mixed, refs, target.sequence)
            primary_rows.append(
                {
                    "target_id": target.target_id,
                    "cluster_id": target.mmseqs_sequence_similarity_cluster,
                    "seq_len": int(target.seq_len),
                    "j_controlled_5t_raw_tm": j_score,
                    "thesis_5t_raw_tm": t_score,
                    "thesis_3t2d_raw_tm": mixed_score,
                    "h1_delta": t_score - j_score,
                    "h2_delta": mixed_score - t_score,
                    "drfold2_available": int(raw["drfold2_available"].item()),
                }
            )
            mechanism_banks = {
                "5T": t5,
                "3T+2D": mixed,
                "2T": t5[:2],
                "1T+1D": np.concatenate([t5[:1], d2[:1]]),
                "2D": d2,
            }
            for name, coords in mechanism_banks.items():
                required_deep = {"3T+2D": 2, "1T+1D": 1, "2D": 2}.get(name, 0)
                pair_tm, duplicate_fraction = self_tm(scorer, target.target_id, name, coords, target.sequence)
                fixed_n_rows.append(
                    {
                        "target_id": target.target_id,
                        "cluster_id": target.mmseqs_sequence_similarity_cluster,
                        "seq_len": int(target.seq_len),
                        "bank": name,
                        "n_candidates": len(coords),
                        "best_tm": scorer.bank(target.target_id, f"mechanism_{name}", coords, refs, target.sequence),
                        "mean_pairwise_self_tm": pair_tm,
                        "near_duplicate_pair_fraction": duplicate_fraction,
                        "source_available": int(raw["drfold2_available"].item()) >= required_deep,
                    }
                )
            for bank, prefix in (("J-controlled", "J"), ("Thesis-3T+2D", "Thesis")):
                raw_coords = np.asarray(refined[f"{prefix}__Raw"], dtype=np.float32)
                locked = [locked_reference(scorer, target.target_id, coords, refs, target.sequence) for coords in raw_coords]
                for setting in settings:
                    coords_bank = np.asarray(refined[f"{prefix}__{setting}"], dtype=np.float32)
                    bank_rows.append(
                        {
                            "target_id": target.target_id,
                            "cluster_id": target.mmseqs_sequence_similarity_cluster,
                            "seq_len": int(target.seq_len),
                            "bank": bank,
                            "setting": setting,
                            "best5_tm": scorer.bank(target.target_id, f"{bank}_{setting}", coords_bank, refs, target.sequence),
                        }
                    )
                    for index, coords in enumerate(coords_bank):
                        ref_index, raw_tm = locked[index]
                        reference = refs[ref_index]
                        candidate_rows.append(
                            {
                                "target_id": target.target_id,
                                "cluster_id": target.mmseqs_sequence_similarity_cluster,
                                "seq_len": int(target.seq_len),
                                "bank": bank,
                                "candidate_index": index,
                                "setting": setting,
                                "raw_reference_index": ref_index,
                                "raw_candidate_tm": raw_tm,
                                "candidate_tm_same_reference": scorer.candidate(
                                    target.target_id, f"{bank}_{setting}_{index}", coords, reference, target.sequence
                                ),
                                **local_accuracy_metrics(coords, reference, windows=(9, 15)),
                                **geometry_v2_metrics(coords, target.sequence, priors_v1, priors_v2),
                            }
                        )
        print(f"[{number:03d}/{len(manifest)} {target.target_id}] confirmatory scored", flush=True)

    primary = pd.DataFrame(primary_rows)
    banks = pd.DataFrame(bank_rows)
    candidates = pd.DataFrame(candidate_rows)
    fixed_n = pd.DataFrame(fixed_n_rows)
    primary.to_csv(RESULTS / "primary_target_scores.csv", index=False)
    banks.to_csv(RESULTS / "factorial_bank_metrics.csv", index=False)
    candidates.to_csv(RESULTS / "factorial_candidate_metrics.csv", index=False)
    fixed_n.to_csv(RESULTS / "fixed_n_candidate_source_metrics.csv", index=False)
    pd.DataFrame(
        scorer.failures, columns=["target_id", "scope", "setting", "reason"]
    ).to_csv(RESULTS / "evaluation_failures.csv", index=False)

    thesis_candidates = candidates[candidates["bank"] == "Thesis-3T+2D"]
    h3_target = thesis_candidates.pivot_table(
        index=["target_id", "cluster_id"], columns="setting", values="sw_rmsd_9", aggfunc="mean"
    ).reset_index()
    h3_target["h3_delta"] = h3_target["Simple"] - h3_target["Geometry"]
    h3_target.to_csv(RESULTS / "h3_target_deltas.csv", index=False)
    thesis_banks = banks[banks["bank"] == "Thesis-3T+2D"]
    tm_guard_target = thesis_banks.pivot(
        index=["target_id", "cluster_id"], columns="setting", values="best5_tm"
    ).reset_index()
    tm_guard_target["geometry_vs_simple_tm_delta"] = tm_guard_target["Geometry"] - tm_guard_target["Simple"]
    tm_guard_target.to_csv(RESULTS / "h3_tm_safeguard_target_deltas.csv", index=False)

    h1 = inference("H1-confirmatory", primary, "h1_delta")
    h2 = inference("H2-confirmatory", primary, "h2_delta")
    h3 = inference("H3-confirmatory", h3_target, "h3_delta")
    tm_guard = inference("H3-TM-safeguard-confirmatory", tm_guard_target, "geometry_vs_simple_tm_delta")
    holm = holm_step_down({"H1": h1["raw_one_sided_p"], "H2": h2["raw_one_sided_p"], "H3": h3["raw_one_sided_p"]})
    decisions = {
        "H1": bool(holm["H1"]["reject"] and h1["ci_lower"] > 0.0),
        "H2": bool(holm["H2"]["reject"] and h2["ci_lower"] > 0.0),
        "H3_local_superiority": bool(holm["H3"]["reject"] and h3["ci_lower"] > 0.0),
        "H3_tm_noninferiority": bool(tm_guard["ci_lower"] > freeze["statistics"]["h3_tm_noninferiority_margin"]),
    }
    decisions["H3"] = decisions["H3_local_superiority"] and decisions["H3_tm_noninferiority"]

    thesis_candidate_means = thesis_candidates.groupby("setting")["sw_rmsd_9"].mean()
    thesis_bank_means = thesis_banks.groupby("setting")["best5_tm"].mean()
    descriptive_scores = {
        "j_controlled_5t_raw_mean_tm": float(primary["j_controlled_5t_raw_tm"].mean()),
        "thesis_5t_raw_mean_tm": float(primary["thesis_5t_raw_tm"].mean()),
        "thesis_3t2d_raw_mean_tm": float(primary["thesis_3t2d_raw_tm"].mean()),
        "simple_mean_sw_rmsd9": float(thesis_candidate_means["Simple"]),
        "geometry_mean_sw_rmsd9": float(thesis_candidate_means["Geometry"]),
        "simple_mean_bank_tm": float(thesis_bank_means["Simple"]),
        "geometry_mean_bank_tm": float(thesis_bank_means["Geometry"]),
        "h1_relative_tm_change_percent": float(
            100.0 * h1["mean_delta"] / primary["j_controlled_5t_raw_tm"].mean()
        ),
        "h2_relative_tm_change_percent": float(
            100.0 * h2["mean_delta"] / primary["thesis_5t_raw_tm"].mean()
        ),
        "h3_relative_sw_rmsd9_reduction_percent": float(
            100.0 * h3["mean_delta"] / thesis_candidate_means["Simple"]
        ),
        "h3_relative_bank_tm_change_percent": float(
            100.0 * tm_guard["mean_delta"] / thesis_bank_means["Simple"]
        ),
    }

    factorial_target = candidates.groupby(
        ["target_id", "cluster_id", "seq_len", "bank", "setting"], as_index=False
    ).agg(
        candidate_tm_same_reference=("candidate_tm_same_reference", "mean"),
        c1_rmsd=("c1_rmsd", "mean"),
        c1_lddt=("c1_lddt", "mean"),
        sw_rmsd_9=("sw_rmsd_9", "mean"),
        sw_rmsd_15=("sw_rmsd_15", "mean"),
        bb_dev=("bb_dev", "mean"),
        clash_per_res=("clash_per_res", "mean"),
        sharp_kinks=("sharp_kinks", "mean"),
        rg_err=("rg_err", "mean"),
        angle_nll=("angle_nll", "mean"),
        torsion_nll=("torsion_nll", "mean"),
        pair_like_fraction=("pair_like_fraction", "mean"),
    )
    factorial_target.to_csv(RESULTS / "factorial_target_metrics.csv", index=False)
    factorial_summary = factorial_target.groupby(["bank", "setting"], as_index=False).mean(numeric_only=True)
    bank_summary = banks.groupby(["bank", "setting"], as_index=False)["best5_tm"].mean()
    factorial_summary.merge(bank_summary, on=["bank", "setting"]).to_csv(
        RESULTS / "factorial_summary.csv", index=False
    )
    fixed_n.groupby("bank", as_index=False).agg(
        target_n=("target_id", "count"),
        mean_best_tm=("best_tm", "mean"),
        mean_pairwise_self_tm=("mean_pairwise_self_tm", "mean"),
        near_duplicate_pair_fraction=("near_duplicate_pair_fraction", "mean"),
        source_availability=("source_available", "mean"),
    ).to_csv(RESULTS / "fixed_n_candidate_source_summary.csv", index=False)

    supporting: dict[str, dict] = {}
    thesis_factorial = factorial_target[factorial_target["bank"] == "Thesis-3T+2D"]
    for metric, higher_is_better in (
        ("sw_rmsd_15", False),
        ("c1_rmsd", False),
        ("c1_lddt", True),
        ("candidate_tm_same_reference", True),
    ):
        pivot = thesis_factorial.pivot(
            index=["target_id", "cluster_id"], columns="setting", values=metric
        ).reset_index()
        delta = f"geometry_vs_simple_{metric}_delta"
        pivot[delta] = (
            pivot["Geometry"] - pivot["Simple"]
            if higher_is_better
            else pivot["Simple"] - pivot["Geometry"]
        )
        supporting[f"Geometry_vs_Simple_{metric}"] = inference(
            f"supporting-Geometry-vs-Simple-{metric}", pivot, delta
        )
    for bank in ("J-controlled", "Thesis-3T+2D"):
        bank_group = banks[banks["bank"] == bank].pivot(
            index=["target_id", "cluster_id"], columns="setting", values="best5_tm"
        ).reset_index()
        for setting in ("John_fixed", "Simple", "Geometry"):
            delta = f"{setting}_vs_raw_tm_delta"
            bank_group[delta] = bank_group[setting] - bank_group["Raw"]
            supporting[f"{bank}_{setting}_vs_Raw_bank_TM"] = inference(
                f"supporting-{bank}-{setting}-vs-Raw-bank-TM", bank_group, delta
            )
        bank_group["john_original_vs_fixed_tm_delta"] = (
            bank_group["John_original"] - bank_group["John_fixed"]
        )
        supporting[f"{bank}_John_original_vs_fixed_bank_TM"] = inference(
            f"sensitivity-{bank}-John-original-vs-fixed-bank-TM",
            bank_group,
            "john_original_vs_fixed_tm_delta",
        )
        candidate_group = factorial_target[factorial_target["bank"] == bank].pivot(
            index=["target_id", "cluster_id"], columns="setting", values="sw_rmsd_9"
        ).reset_index()
        candidate_group["john_fixed_vs_original_sw9_delta"] = (
            candidate_group["John_original"] - candidate_group["John_fixed"]
        )
        supporting[f"{bank}_John_original_confidence_SW9_sensitivity"] = inference(
            f"sensitivity-{bank}-John-fixed-vs-original-SW9",
            candidate_group,
            "john_fixed_vs_original_sw9_delta",
        )
    write_json(
        RESULTS / "supporting_inference.json",
        {
            "evidence_role": "preregistered supporting and sensitivity evidence; not in the H1-H3 Holm family",
            "effects": supporting,
        },
    )

    length_bins = pd.cut(primary["seq_len"], bins=[29, 79, 149, 249, 400], labels=["30-79", "80-149", "150-249", "250-400"])
    sensitivities = []
    for length_bin, group in primary.assign(length_bin=length_bins).groupby("length_bin", observed=True):
        for hypothesis, delta in (("H1", "h1_delta"), ("H2", "h2_delta")):
            sensitivities.append({"analysis": "length_bin", "stratum": str(length_bin), **inference(f"{hypothesis}-{length_bin}", group, delta)})
        ids = set(group["target_id"])
        h3_group = h3_target[h3_target["target_id"].isin(ids)]
        sensitivities.append({"analysis": "length_bin", "stratum": str(length_bin), **inference(f"H3-{length_bin}", h3_group, "h3_delta")})
    pd.DataFrame(sensitivities).to_csv(RESULTS / "preregistered_sensitivity_inference.csv", index=False)

    result = {
        "status": "CONFIRMATORY_FINAL_EVALUATION_COMPLETE_NO_RETUNING",
        "completed_at_utc": now_utc(),
        "evidence_role": "97-target confirmatory final evidence; development results remain separate",
        "target_n": len(primary),
        "cluster_n": primary["cluster_id"].nunique(),
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "H3_tm_safeguard": tm_guard,
        "holm_family": holm,
        "preregistered_decisions": decisions,
        "descriptive_scores": descriptive_scores,
        "supporting_inference_path": "reports/thesis_v4/confirmatory/supporting_inference.json",
        "evaluation_failure_count": len(scorer.failures),
        "generation_refinement_failure_count": int(json.loads(OUTPUT_FREEZE.read_text())["completeness"]["refinement_failure_count"]),
        "source_available_targets": int((primary["drfold2_available"] >= 2).sum()),
        "complete_case_target_n": len(primary),
        "overlap_sensitivity": "All 97 frozen targets have STRUCTURAL_OVERLAP_PASS; no confirmatory overlap strata exist",
        "post_opening_method_change": False,
        "external_kaggle_attribution_allowed": False,
    }
    write_json(FINAL_INFERENCE, result)
    event.update(
        {
            "evaluation_completed": True,
            "evaluation_completed_at_utc": now_utc(),
            "primary_inference_path": str(FINAL_INFERENCE.relative_to(REPO)),
            "primary_inference_sha256": sha256(FINAL_INFERENCE),
        }
    )
    write_json(OPENING_EVENT, event)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-opening").set_defaults(func=cmd_prepare_opening)
    commands.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
