#!/usr/bin/env python
"""Generate the frozen V4 confirmatory candidates without native structures.

The runner intentionally has no evaluation command.  It stops after producing
hashed Raw, John, Simple, and selected-Geometry coordinates.  Native labels may
only be introduced by a separate post-checkpoint evaluation receipt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.geofuse.geometry_v2 import geometry_v2_metrics  # noqa: F401; diagnostic implementation freeze
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.geofuse.structure_io import import_structure
from rna3d.pipeline.v4_frozen import build_j_controlled_bank, build_thesis_tbm_bank, stable_seed
from rna3d.refine.rule_based import refine_rule_based
from rna3d.template import db


FINAL = REPO / "reports" / "thesis_v4" / "final_freeze"
MANIFEST = FINAL / "final_target_manifest.csv"
FREEZE = FINAL / "final_method_freeze.json"
CACHE = REPO / "data" / "cache" / "v4_final_native_blind"
P0_DISTANCE = REPO / "data" / "processed" / "geometry_priors.json"
P0_GEOMETRY = REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen_inputs() -> tuple[dict, pd.DataFrame]:
    freeze = json.loads(FREEZE.read_text())
    if sha256(MANIFEST) != freeze["target_manifest"]["sha256"]:
        raise RuntimeError("final target manifest changed after freeze")
    for key, record in freeze["artifact_hashes"].items():
        path = REPO / record["path"]
        if not path.exists() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen artifact mismatch: {key} ({path})")
    source_hashes = freeze["generation_code"]
    if sha256(Path(__file__)) != source_hashes["runner_sha256"]:
        raise RuntimeError("native-blind runner changed after method freeze")
    module_path = REPO / "src" / "rna3d" / "pipeline" / "v4_frozen.py"
    if sha256(module_path) != source_hashes["pipeline_module_sha256"]:
        raise RuntimeError("frozen V4 pipeline module changed after method freeze")
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    if len(manifest) != freeze["target_manifest"]["target_n"]:
        raise RuntimeError("final target count differs from freeze")
    if manifest["mmseqs_sequence_similarity_cluster"].nunique() != freeze["target_manifest"]["cluster_n"]:
        raise RuntimeError("final cluster count differs from freeze")
    return freeze, manifest


def target_dir(target_id: str) -> Path:
    return CACHE / target_id


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def cmd_validate(_: argparse.Namespace) -> None:
    freeze, manifest = read_frozen_inputs()
    source = Path(__file__).read_text()
    forbidden = ("load_" + "labels", "score_" + "target", "get_reference_" + "coords")
    present = [token for token in forbidden if token in source]
    if present:
        raise RuntimeError(f"native-blind source contains forbidden evaluator hooks: {present}")
    print(
        json.dumps(
            {
                "status": "NATIVE_BLIND_SETUP_VALID",
                "target_n": len(manifest),
                "cluster_n": manifest["mmseqs_sequence_similarity_cluster"].nunique(),
                "final_performance_accessed": False,
                "selected_bank": freeze["final_pipeline"]["candidate_bank"],
                "selected_refiner": freeze["final_pipeline"]["refiner"],
            },
            indent=2,
        )
    )


def cmd_build_tbm(args: argparse.Namespace) -> None:
    _, manifest = read_frozen_inputs()
    meta = db.load_meta()
    coordinates = db.load_coords()
    adjacent = float(json.loads(P0_DISTANCE.read_text())["adjacent_c1"]["mean"])
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = target_dir(row.target_id) / "tbm_banks.npz"
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists() and not args.replace:
            rows.append({"target_id": row.target_id, "path": str(path.relative_to(REPO)), "sha256": sha256(path), "status": "cached"})
            continue
        exclusions = [item for item in row.excluded_pdb_ids.split(",") if item]
        started = time.time()
        john = build_j_controlled_bank(
            target_id=row.target_id,
            sequence=row.sequence,
            cutoff=row.temporal_cutoff,
            excluded_pdb_ids=exclusions,
            meta=meta,
            coordinates=coordinates,
        )
        thesis = build_thesis_tbm_bank(
            target_id=row.target_id,
            sequence=row.sequence,
            cutoff=row.temporal_cutoff,
            excluded_pdb_ids=exclusions,
            meta=meta,
            coordinates=coordinates,
            adjacent_distance=adjacent,
        )
        save_npz(
            path,
            j_coords=john.coords,
            j_conf=john.confidence,
            j_global_conf=john.global_confidence,
            t_coords=thesis.coords,
            t_conf=thesis.confidence,
            t_global_conf=thesis.global_confidence,
        )
        metadata = {
            "target_id": row.target_id,
            "sequence_sha256": hashlib.sha256(row.sequence.encode()).hexdigest(),
            "j_candidate_ids": john.candidate_ids,
            "j_fallback_slots": john.fallback_slots,
            "t_candidate_ids": thesis.candidate_ids,
            "t_pdb_ids": thesis.pdb_ids,
            "t_fallback_slots": thesis.fallback_slots,
            "method": "J-controlled raw and retained V4 TBM raw",
        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({"target_id": row.target_id, "path": str(path.relative_to(REPO)), "sha256": sha256(path), "status": "generated"})
        print(f"[{number:03d}/{len(manifest)} {row.target_id}] TBM sec={time.time()-started:.1f}", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CACHE / "tbm_generation_manifest.csv", index=False)


def _deep_candidates(root: Path, target_id: str, sequence: str) -> list:
    files = sorted((root / target_id / "e2e_relax").glob("model_*.pdb"))
    candidates = []
    for index, path in enumerate(files):
        candidates.append(
            import_structure(
                path,
                target_id=target_id,
                sequence=sequence,
                candidate_id=f"drfold2_e2e__cfg97_20ckpt_e2e__{index+1:02d}",
                source="drfold2_e2e",
                model="cfg97_20ckpt_e2e",
                default_confidence=0.5,
                provenance={
                    "structural_training_cutoff": "2023-12-31",
                    "language_model_membership": "UNAVAILABLE",
                },
            )
        )
    return sorted(candidates, key=lambda item: (-item.global_confidence, item.candidate_id))[:2]


def cmd_assemble_raw(args: argparse.Namespace) -> None:
    _, manifest = read_frozen_inputs()
    root = args.drfold_root.resolve()
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = target_dir(row.target_id) / "raw_banks.npz"
        if path.exists() and not args.replace:
            rows.append({"target_id": row.target_id, "path": str(path.relative_to(REPO)), "sha256": sha256(path), "drfold2_available": "cached"})
            continue
        tbm_path = target_dir(row.target_id) / "tbm_banks.npz"
        if not tbm_path.exists():
            raise FileNotFoundError(f"build TBM before raw assembly: {tbm_path}")
        with np.load(tbm_path, allow_pickle=False) as tbm:
            j_coords, j_conf, j_global = tbm["j_coords"], tbm["j_conf"], tbm["j_global_conf"]
            t_coords, t_conf, t_global = tbm["t_coords"], tbm["t_conf"], tbm["t_global_conf"]
        deep = _deep_candidates(root, row.target_id, row.sequence)
        deep_coords = [item.coords for item in deep]
        deep_conf = [item.confidence for item in deep]
        deep_global = [item.global_confidence for item in deep]
        available = len(deep)
        while len(deep_coords) < 2:
            template_index = 3 + len(deep_coords)
            deep_coords.append(t_coords[template_index])
            deep_conf.append(t_conf[template_index])
            deep_global.append(t_global[template_index])
        thesis_coords = np.concatenate([t_coords[:3], np.asarray(deep_coords[:2])])
        thesis_conf = np.concatenate([t_conf[:3], np.asarray(deep_conf[:2])])
        thesis_global = np.concatenate([t_global[:3], np.asarray(deep_global[:2])])
        save_npz(
            path,
            j_coords=j_coords,
            j_conf=j_conf,
            j_global_conf=j_global,
            t_coords=t_coords,
            t_conf=t_conf,
            t_global_conf=t_global,
            d_coords=np.asarray([item.coords for item in deep], dtype=np.float32).reshape((available, len(row.sequence), 3)),
            d_conf=np.asarray([item.confidence for item in deep], dtype=np.float32).reshape((available, len(row.sequence))),
            d_global_conf=np.asarray([item.global_confidence for item in deep], dtype=np.float32),
            thesis_coords=np.asarray(thesis_coords, dtype=np.float32),
            thesis_conf=np.asarray(thesis_conf, dtype=np.float32),
            thesis_global_conf=np.asarray(thesis_global, dtype=np.float32),
            drfold2_available=np.asarray(available, dtype=np.int16),
        )
        rows.append({"target_id": row.target_id, "path": str(path.relative_to(REPO)), "sha256": sha256(path), "drfold2_available": available})
        print(f"[{number:03d}/{len(manifest)} {row.target_id}] D={available}/2", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "raw_generation_manifest.csv", index=False)


def _geometry_call(
    coords: np.ndarray,
    sequence: str,
    confidence: np.ndarray,
    global_confidence: float,
    cfg: GeometryV2Config,
    priors_v1: dict,
    priors_v2: dict,
    device: str,
    seed: int,
) -> tuple[np.ndarray, bool, str]:
    try:
        output, _ = refine_structure_v2(
            coords,
            sequence,
            priors_v1,
            priors_v2,
            source_confidence=confidence,
            global_confidence=global_confidence,
            cfg=cfg,
            device=device,
            seed=seed,
        )
        if not np.isfinite(output).all():
            raise FloatingPointError("nonfinite output")
        return np.asarray(output, dtype=np.float32), False, ""
    except Exception as error:
        return np.asarray(coords, dtype=np.float32), True, f"{type(error).__name__}:{error}"


def cmd_refine(args: argparse.Namespace) -> None:
    freeze, manifest = read_frozen_inputs()
    selected = GeometryV2Config(**freeze["final_pipeline"]["geometry_config"])
    simple = replace(
        selected,
        adaptive_strength=False,
        fixed_strength=1.0,
        context_mode="global",
        w_clash=0.0,
        w_rg=0.0,
        w_angle=0.0,
        w_torsion=0.0,
        w_kink=0.0,
    )
    priors_v1 = json.loads(P0_DISTANCE.read_text())
    priors_v2 = json.loads(P0_GEOMETRY.read_text())
    rows, failures = [], []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        output_path = target_dir(row.target_id) / "refined_banks.npz"
        if output_path.exists() and not args.replace:
            rows.append({"target_id": row.target_id, "path": str(output_path.relative_to(REPO)), "sha256": sha256(output_path), "status": "cached"})
            continue
        raw_path = target_dir(row.target_id) / "raw_banks.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"assemble raw banks before refinement: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as raw:
            banks = {
                "J": (raw["j_coords"], raw["j_conf"], raw["j_global_conf"]),
                "Thesis": (raw["thesis_coords"], raw["thesis_conf"], raw["thesis_global_conf"]),
            }
        arrays = {}
        tasks = {}
        started = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for bank, (coords, confidence, global_confidence) in banks.items():
                arrays[f"{bank}__Raw"] = np.asarray(coords, dtype=np.float32)
                arrays[f"{bank}__John_original"] = np.asarray(
                    [refine_rule_based(value, row.sequence, confidence=float(global_confidence[index])) for index, value in enumerate(coords)],
                    dtype=np.float32,
                )
                arrays[f"{bank}__John_fixed"] = np.asarray(
                    [refine_rule_based(value, row.sequence, confidence=0.5) for value in coords],
                    dtype=np.float32,
                )
                for index, value in enumerate(coords):
                    for name, cfg in (("Simple", simple), ("Geometry", selected)):
                        future = executor.submit(
                            _geometry_call,
                            value,
                            row.sequence,
                            confidence[index],
                            float(global_confidence[index]),
                            cfg,
                            priors_v1,
                            priors_v2,
                            args.device,
                            stable_seed("v4-final-refine", row.target_id, bank, index, name),
                        )
                        tasks[future] = (bank, name, index)
            outputs = {tasks[future]: future.result() for future in as_completed(tasks)}
        for bank in banks:
            for name in ("Simple", "Geometry"):
                values = []
                for index in range(5):
                    value, failed, reason = outputs[(bank, name, index)]
                    values.append(value)
                    failures.append({"target_id": row.target_id, "bank": bank, "setting": name, "candidate_index": index, "failed": failed, "reason": reason})
                arrays[f"{bank}__{name}"] = np.asarray(values, dtype=np.float32)
        save_npz(output_path, **arrays)
        rows.append({"target_id": row.target_id, "path": str(output_path.relative_to(REPO)), "sha256": sha256(output_path), "status": "generated"})
        print(f"[{number:03d}/{len(manifest)} {row.target_id}] refine sec={time.time()-started:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "refinement_generation_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(CACHE / "refinement_failures.csv", index=False)


def cmd_status(_: argparse.Namespace) -> None:
    _, manifest = read_frozen_inputs()
    rows = []
    for row in manifest.itertuples(index=False):
        directory = target_dir(row.target_id)
        rows.append(
            {
                "target_id": row.target_id,
                "tbm": (directory / "tbm_banks.npz").exists(),
                "raw": (directory / "raw_banks.npz").exists(),
                "refined": (directory / "refined_banks.npz").exists(),
            }
        )
    frame = pd.DataFrame(rows)
    print(frame[["tbm", "raw", "refined"]].sum().to_string())
    print("final_performance_accessed    0")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").set_defaults(func=cmd_validate)
    tbm = commands.add_parser("build-tbm")
    tbm.add_argument("--replace", action="store_true")
    tbm.set_defaults(func=cmd_build_tbm)
    raw = commands.add_parser("assemble-raw")
    raw.add_argument("--drfold-root", type=Path, required=True)
    raw.add_argument("--replace", action="store_true")
    raw.set_defaults(func=cmd_assemble_raw)
    refine = commands.add_parser("refine")
    refine.add_argument("--device", default="cuda")
    refine.add_argument("--workers", type=int, default=8)
    refine.add_argument("--replace", action="store_true")
    refine.set_defaults(func=cmd_refine)
    commands.add_parser("status").set_defaults(func=cmd_status)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
