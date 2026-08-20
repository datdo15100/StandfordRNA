#!/usr/bin/env python
"""Verify that the final TBM module exactly replays all frozen development banks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.pipeline.v4_frozen import build_thesis_tbm_bank
from rna3d.template import db


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = REPO / "reports" / "thesis_v4" / "development" / "development_manifest.csv"
    raw_root = REPO / "data" / "cache" / "v4_development_raw"
    module_path = REPO / "src" / "rna3d" / "pipeline" / "v4_frozen.py"
    output_path = REPO / "reports" / "thesis_v4" / "final_freeze" / "tbm_replay_audit.json"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    meta = db.load_meta()
    coordinates = db.load_coords()
    adjacent = float(
        json.loads((REPO / "data" / "processed" / "geometry_priors.json").read_text())["adjacent_c1"]["mean"]
    )
    rows = []
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        observed = build_thesis_tbm_bank(
            target_id=row.target_id,
            sequence=row.sequence,
            cutoff=row.date,
            excluded_pdb_ids=row.excluded_pdb_ids.split(","),
            meta=meta,
            coordinates=coordinates,
            adjacent_distance=adjacent,
            fallback_version="v4-dev-raw-1",
        )
        expected_path = raw_root / row.target_id / "retained_tbm.npz"
        with np.load(expected_path, allow_pickle=False) as expected:
            checks = {
                "coords": np.array_equal(observed.coords, expected["coords"]),
                "confidence": np.array_equal(observed.confidence, expected["confidence"]),
                "global_confidence": np.array_equal(observed.global_confidence, expected["global_confidence"]),
            }
        if not all(checks.values()):
            raise RuntimeError(f"{row.target_id}: replay mismatch {checks}")
        rows.append({"target_id": row.target_id, **checks, "expected_sha256": sha256(expected_path)})
        print(f"[{number:02d}/{len(manifest)}] {row.target_id} exact replay", flush=True)
    document = {
        "status": "PASS_ALL_20_EXACT",
        "native_performance_accessed": False,
        "target_n": len(rows),
        "arrays_checked": ["coords", "confidence", "global_confidence"],
        "manifest_sha256": sha256(manifest_path),
        "pipeline_module_sha256": sha256(module_path),
        "targets": rows,
    }
    output_path.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: document[key] for key in ("status", "target_n", "native_performance_accessed")}, indent=2))


if __name__ == "__main__":
    main()
