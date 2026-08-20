#!/usr/bin/env python
"""Freeze the shared template database and smoke-test J-controlled without scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import re
import sys
import time

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.baselines.top1 import build_raw_candidates
from rna3d.data import io
from rna3d.paths import cache, processed


DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_v4" / "phase1_controlled_db"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(sequence: str, coordinates: np.ndarray) -> str:
    digest = hashlib.sha256(sequence.encode())
    digest.update(np.ascontiguousarray(coordinates, dtype=np.float32).tobytes())
    return digest.hexdigest()


def native_pdb_ids(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {item.upper() for item in re.findall(r">([0-9][A-Za-z0-9]{3})_", value)}


def main(args: argparse.Namespace) -> None:
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    meta_path = processed() / "template_meta.parquet"
    coords_path = cache() / "template_coords.pkl"
    meta = pd.read_parquet(meta_path).sort_values("chain_key").reset_index(drop=True)
    with coords_path.open("rb") as handle:
        coordinates = pickle.load(handle)
    if len(meta) != 23869 or len(coordinates) != 23869:
        raise RuntimeError("controlled template artifacts no longer contain 23,869 chains")

    rows = []
    for row in meta.itertuples(index=False):
        key = str(row.chain_key)
        if key not in coordinates:
            raise RuntimeError(f"coordinate store is missing {key}")
        record = coordinates[key]
        sequence = str(record["seq"])
        xyz = np.asarray(record["coords"], dtype=float)
        if xyz.shape != (len(sequence), 3) or not np.isfinite(xyz).any():
            raise RuntimeError(f"invalid controlled template {key}: {xyz.shape}")
        rows.append(
            {
                "chain_key": key,
                "pdb_id": str(row.pdb_id),
                "chain_id": str(row.chain_id),
                "release_date": str(row.release_date),
                "length": len(sequence),
                "resolved_c1_count": int(np.isfinite(xyz).all(axis=1).sum()),
                "normalized_sequence_sha256": hashlib.sha256(
                    sequence.upper().replace("T", "U").encode()
                ).hexdigest(),
                "sequence_coordinate_sha256": value_hash(sequence, xyz),
                "db_controlled_included": True,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = report / "db_controlled_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    validation = io.load_sequences("validation")
    target = validation.loc[validation["target_id"].eq(args.smoke_target)].iloc[0]
    excluded = native_pdb_ids(target.get("all_sequences"))
    allowed = meta[
        (meta["release_date"].astype(str) < str(target["temporal_cutoff"]))
        & ~meta["pdb_id"].astype(str).str.upper().isin(excluded)
    ]
    templates = [
        (
            str(row.chain_key),
            str(coordinates[str(row.chain_key)]["seq"]),
            np.asarray(coordinates[str(row.chain_key)]["coords"], dtype=float),
        )
        for row in allowed.itertuples(index=False)
    ]
    started = time.time()
    candidates = build_raw_candidates(
        str(target["sequence"]),
        args.smoke_target,
        templates,
        n=5,
        base_seed=args.base_seed,
    )
    seconds = time.time() - started
    if len(candidates) != 5:
        raise RuntimeError("J-controlled smoke did not return five raw candidates")
    candidate_hashes = [
        hashlib.sha256(
            np.ascontiguousarray(candidate.coords, dtype=np.float64).tobytes()
        ).hexdigest()
        for candidate in candidates
    ]
    audit = {
        "phase": "V4 Phase 1 DB-controlled freeze",
        "performance_accessed": False,
        "audit_driver": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": sha256(Path(__file__))},
        "database": {
            "definition": "all 23,869 valid parsed RNA chains; target-specific temporal and self filters applied identically at query time",
            "meta": {"path": str(meta_path.relative_to(REPO_ROOT)), "sha256": sha256(meta_path)},
            "coordinates": {"path": str(coords_path.relative_to(REPO_ROOT)), "sha256": sha256(coords_path)},
            "manifest": {"path": str(manifest_path.relative_to(REPO_ROOT)), "sha256": sha256(manifest_path)},
            "chains": len(manifest),
            "pdb_entries": int(manifest["pdb_id"].nunique()),
            "all_coordinate_records_valid": True,
        },
        "source_snapshot": {
            "pdb_release_dates_sha256": sha256(REPO_ROOT / "data" / "stanford-rna-3d-folding" / "PDB_RNA" / "pdb_release_dates_NA.csv"),
            "pdb_seqres_sha256": sha256(REPO_ROOT / "data" / "stanford-rna-3d-folding" / "PDB_RNA" / "pdb_seqres_NA.fasta"),
        },
        "j_controlled_native_blind_smoke": {
            "target_id": args.smoke_target,
            "sequence_length": len(str(target["sequence"])),
            "target_cutoff": str(target["temporal_cutoff"]),
            "self_pdb_ids_excluded": sorted(excluded),
            "templates_after_common_filter": len(templates),
            "candidate_sources": [candidate.source for candidate in candidates],
            "template_ids": [candidate.template_id for candidate in candidates],
            "candidate_coordinate_sha256": candidate_hashes,
            "base_seed": args.base_seed,
            "seconds": seconds,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    audit_path = report / "db_controlled_freeze.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    (report / "README.md").write_text(
        f"""# V4 Phase 1: DB-controlled freeze

- Native performance accessed: **No**.
- Shared database: {len(manifest):,} valid parsed chains from {manifest['pdb_id'].nunique():,} PDB entries.
- Metadata hash: `{sha256(meta_path)}`.
- Coordinate-store hash: `{sha256(coords_path)}`.
- Per-chain manifest hash: `{sha256(manifest_path)}`.
- J-controlled smoke: `{args.smoke_target}`, {len(templates):,} templates after the
  common temporal/self filter, five raw candidates generated.

Both John and thesis must query this same frozen database. Method-specific retrieval and
ranking may differ, but no comparator may substitute a private or larger template pool.
"""
    )
    print(json.dumps(audit["j_controlled_native_blind_smoke"], indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--smoke-target", default="R1117v2")
    result.add_argument("--base-seed", type=int, default=42)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
