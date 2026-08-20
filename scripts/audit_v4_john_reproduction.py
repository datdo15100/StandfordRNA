#!/usr/bin/env python
"""Audit the public John TBM capture and run a native-blind reproduction smoke test."""
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

import Bio
import numpy as np
import pandas as pd
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.baselines.top1 import build_raw_candidates
from rna3d.data import io
from rna3d.paths import cache, processed


DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_v4" / "phase1_john"
DEFAULT_SMOKE = processed() / "v4_john_smoke"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def native_pdb_ids(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {item.upper() for item in re.findall(r">([0-9][A-Za-z0-9]{3})_", value)}


def main(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.resolve()
    smoke_dir = args.smoke_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    public_tbm = REPO_ROOT / "utilities" / "top1_tbm.py"
    public_hybrid = REPO_ROOT / "utilities" / "top1_4_4_hybrid_final_take.py"
    port = REPO_ROOT / "src" / "rna3d" / "baselines" / "top1.py"
    refiner = REPO_ROOT / "src" / "rna3d" / "refine" / "rule_based.py"
    denovo = REPO_ROOT / "src" / "rna3d" / "geometry" / "denovo.py"
    meta_path = processed() / "top1_template_meta.parquet"
    coords_path = cache() / "top1_template_coords.pkl"

    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    with coords_path.open("rb") as handle:
        coords = pickle.load(handle)
    sequences = io.load_sequences("validation")
    selected = sequences.loc[sequences["target_id"].eq(args.target_id)]
    if len(selected) != 1:
        raise RuntimeError(f"smoke target {args.target_id!r} is not unique")
    target = selected.iloc[0]
    query = str(target["sequence"])
    cutoff = str(target["temporal_cutoff"])
    excluded_pdbs = native_pdb_ids(target.get("all_sequences"))

    allowed = meta[
        (meta["release_date"].astype(str) < cutoff)
        & ~meta["pdb_id"].astype(str).str.upper().isin(excluded_pdbs)
    ]
    templates = []
    for row in allowed.itertuples(index=False):
        record = coords[str(row.target_id)]
        templates.append(
            (
                str(row.target_id),
                str(record["seq"]),
                np.asarray(record["coords"], dtype=float),
            )
        )

    started = time.time()
    first = build_raw_candidates(
        query,
        args.target_id,
        templates,
        n=5,
        base_seed=args.base_seed,
    )
    elapsed = time.time() - started
    second = build_raw_candidates(
        query,
        args.target_id,
        templates,
        n=5,
        base_seed=args.base_seed,
    )
    repeatable = all(
        left.source == right.source
        and left.template_id == right.template_id
        and left.confidence == right.confidence
        and np.array_equal(left.coords, right.coords)
        for left, right in zip(first, second)
    )
    if len(first) != 5 or not repeatable:
        raise RuntimeError("John raw-candidate smoke test is not repeatable")

    stacked = np.stack([candidate.coords for candidate in first])
    smoke_path = smoke_dir / f"{args.target_id}_raw_candidates.npy"
    np.save(smoke_path, stacked, allow_pickle=False)
    source_text = public_tbm.read_text(errors="replace")
    audit = {
        "phase": "V4 Phase 1 public John reproduction audit",
        "baseline_name": "reproduced publicly released John pipeline",
        "status": "PARTIAL_DATASET_REPRODUCTION_NATIVE_BLIND_SMOKE_PASS",
        "performance_accessed": False,
        "public_capture": {
            "tbm_path": relative(public_tbm),
            "tbm_sha256": sha256(public_tbm),
            "hybrid_path": relative(public_hybrid),
            "hybrid_sha256": sha256(public_hybrid),
            "reported_coordinate_groups_in_captured_output": 18815 if "18815/18815" in source_text else "UNKNOWN",
            "exact_notebook_version": "UNKNOWN",
            "exact_attached_dataset_versions": "UNKNOWN",
            "exact_winning_submission_equivalence": "UNKNOWN",
        },
        "local_port": {
            "audit_driver": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
            "files": [
                {"path": relative(path), "sha256": sha256(path)}
                for path in [port, refiner, denovo]
            ],
            "raw_boundary": "after coordinate transfer and gap completion; before John rule refiner and Gaussian jitter",
            "base_seed": args.base_seed,
            "seed_contract": "SHA-256-derived 32-bit seeds replace process-randomized Python hash and global NumPy state",
        },
        "local_john_style_database": {
            "meta_path": relative(meta_path),
            "meta_sha256": sha256(meta_path),
            "coords_path": relative(coords_path),
            "coords_sha256": sha256(coords_path),
            "unique_sequences": int(len(meta)),
            "unique_pdb_ids": int(meta["pdb_id"].nunique()),
            "minimum_length": int(meta["length"].min()),
            "median_length": float(meta["length"].median()),
            "maximum_length": int(meta["length"].max()),
        },
        "native_blind_smoke": {
            "target_id": args.target_id,
            "sequence_length": len(query),
            "cutoff": cutoff,
            "direct_self_pdb_ids_excluded": sorted(excluded_pdbs),
            "allowed_template_count": len(templates),
            "candidate_count": len(first),
            "candidate_sources": [candidate.source for candidate in first],
            "template_ids": [candidate.template_id for candidate in first],
            "confidences": [candidate.confidence for candidate in first],
            "candidate_coordinate_sha256": [bytes_sha256(candidate.coords) for candidate in first],
            "repeatable_within_process": repeatable,
            "artifact_path": relative(smoke_path),
            "artifact_sha256": sha256(smoke_path),
            "first_pass_seconds": elapsed,
        },
        "known_deviations": [
            {
                "id": "DATASET-UNIVERSE",
                "status": "CONFIRMED",
                "detail": f"Local John-style database has {len(meta):,} unique sequences, not the captured notebook's 18,815 coordinate groups.",
                "consequence": "J-original-local is a partial data reproduction; it is not an exact notebook or winning-submission reproduction.",
            },
            {
                "id": "DATASET-NORMALIZATION",
                "status": "CONFIRMED",
                "detail": "Local reconstruction canonicalizes modified residues and keeps the earliest release representative per exact sequence.",
                "consequence": "The local input table differs from the public attached dataset.",
            },
            {
                "id": "RANDOMNESS",
                "status": "CONTROLLED_DEVIATION",
                "detail": "Public code uses process-randomized hash(target_id) and global NumPy/Python random state.",
                "consequence": "V4 freezes SHA-256-derived seeds and records this execution-only deviation.",
            },
            {
                "id": "RAW-BOUNDARY",
                "status": "CONTROLLED_ADDITION",
                "detail": "V4 exposes candidates before John rule refinement and Gaussian jitter.",
                "consequence": "H1 can compare candidate generation without conflating refinement.",
            },
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "biopython": Bio.__version__,
            "platform": platform.platform(),
        },
    }
    audit_path = report_dir / "john_reproduction_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    readme = f"""# V4 Phase 1: public John reproduction audit

Status: **{audit['status']}**

- Required name: **reproduced publicly released John pipeline**.
- Native performance accessed in this audit: **No**.
- Public TBM capture hash: `{audit['public_capture']['tbm_sha256']}`.
- Local John-style database: {len(meta):,} unique sequences versus 18,815 coordinate groups shown in captured notebook output.
- Native-blind smoke target: `{args.target_id}`, length {len(query)}, five raw candidates, repeatable with frozen seed: **{repeatable}**.
- Raw boundary: after transfer/gap completion and before John rule refiner/jitter.

This audit does not establish an exact winning-pipeline reproduction. It establishes a
hashable local port, makes data mismatch explicit, and removes process-dependent random
execution from the controlled baseline. `J-controlled` still requires the shared
DB-controlled snapshot and is tracked separately.
"""
    (report_dir / "README.md").write_text(readme)
    print(json.dumps(audit["native_blind_smoke"], indent=2))
    print(audit["status"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--target-id", default="R1117v2")
    result.add_argument("--base-seed", type=int, default=42)
    result.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
