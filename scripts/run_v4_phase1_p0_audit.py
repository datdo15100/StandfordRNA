#!/usr/bin/env python
"""Rebuild and audit the frozen P0 production geometry priors.

The rebuild is written to a separate directory. Production artifacts are read-only,
and this script never calculates target prediction performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.geometry_v2 import estimate_geometry_v2_priors
from rna3d.geofuse.refine_v2 import GeometryV2Config
from rna3d.geometry.priors import compute_priors
from rna3d.paths import casp15_safe_cutoff, processed


DEFAULT_REBUILD = processed() / "v4_p0_rebuild"
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_v4" / "phase1_p0"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def compare_values(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
    ignore_paths: set[str] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Recursively compare JSON-like values and collect exact/numeric mismatches."""
    ignore_paths = ignore_paths or set()
    if path in ignore_paths:
        return [], 0.0
    if isinstance(expected, dict) and isinstance(observed, dict):
        mismatches: list[dict[str, Any]] = []
        maximum = 0.0
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}"
            if key not in expected or key not in observed:
                mismatches.append(
                    {
                        "path": child,
                        "expected": expected.get(key, "<MISSING>"),
                        "observed": observed.get(key, "<MISSING>"),
                        "kind": "missing_key",
                    }
                )
                continue
            found, error = compare_values(
                expected[key], observed[key], path=child, ignore_paths=ignore_paths
            )
            mismatches.extend(found)
            maximum = max(maximum, error)
        return mismatches, maximum
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            return [
                {
                    "path": path,
                    "expected": len(expected),
                    "observed": len(observed),
                    "kind": "list_length",
                }
            ], 0.0
        mismatches: list[dict[str, Any]] = []
        maximum = 0.0
        for index, (left, right) in enumerate(zip(expected, observed)):
            found, error = compare_values(
                left,
                right,
                path=f"{path}[{index}]",
                ignore_paths=ignore_paths,
            )
            mismatches.extend(found)
            maximum = max(maximum, error)
        return mismatches, maximum
    numeric = (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
    )
    if numeric:
        error = abs(float(expected) - float(observed))
        if error:
            return [
                {
                    "path": path,
                    "expected": expected,
                    "observed": observed,
                    "absolute_error": error,
                    "kind": "numeric",
                }
            ], error
        return [], 0.0
    if expected != observed:
        return [
            {
                "path": path,
                "expected": expected,
                "observed": observed,
                "kind": "value",
            }
        ], 0.0
    return [], 0.0


def main(args: argparse.Namespace) -> None:
    report_dir = args.report_dir.resolve()
    rebuild_dir = args.rebuild_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    rebuild_dir.mkdir(parents=True, exist_ok=True)

    sequences = io.load_sequences("train_v2")
    labels = io.load_labels("train_v2")
    cutoff = str(casp15_safe_cutoff())
    safe = sequences.loc[
        sequences["temporal_cutoff"] < cutoff,
        ["target_id", "temporal_cutoff", "sequence", "seq_len"],
    ].copy()
    safe["normalized_sequence_sha256"] = safe["sequence"].str.upper().str.replace(
        "T", "U"
    ).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    manifest = safe.rename(columns={"seq_len": "sequence_length"})[
        [
            "target_id",
            "temporal_cutoff",
            "sequence_length",
            "normalized_sequence_sha256",
        ]
    ].sort_values("target_id")
    if len(manifest) != 3397 or manifest["target_id"].nunique() != 3397:
        raise RuntimeError("P0 manifest is not the preregistered 3,397 unique targets")
    manifest_path = report_dir / "p0_production_target_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    safe_ids = set(manifest["target_id"])
    label_targets = labels["ID"].map(io.target_id_of)
    safe_labels = labels[label_targets.isin(safe_ids)].copy()
    present = set(safe_labels["ID"].map(io.target_id_of))
    if present != safe_ids:
        raise RuntimeError(f"P0 labels mismatch: missing={len(safe_ids - present)}")

    started_v1 = time.time()
    rebuilt_v1 = compute_priors(safe_labels)
    rebuilt_v1.pop("_raw")
    rebuilt_v1 = {
        "_meta": {
            "source": "train_v2",
            "cutoff": cutoff,
            "n_safe_chains": len(safe_ids),
        },
        **rebuilt_v1,
    }
    elapsed_v1 = time.time() - started_v1
    rebuilt_v1_path = rebuild_dir / "geometry_priors.json"
    rebuilt_v1_path.write_text(json.dumps(rebuilt_v1, indent=2))

    started_v2 = time.time()
    rebuilt_v2 = estimate_geometry_v2_priors(safe_labels, bins=72)
    elapsed_v2 = time.time() - started_v2
    rebuilt_v2["_meta"] = {
        "source": "train_v2",
        "cutoff_rule": f"temporal_cutoff < {cutoff}",
        "n_safe_target_ids": len(safe_ids),
        "seconds": round(elapsed_v2, 1),
    }
    rebuilt_v2_path = rebuild_dir / "geofuse_geometry_v2_priors.json"
    rebuilt_v2_path.write_text(json.dumps(rebuilt_v2, indent=2) + "\n")

    production_v1_path = processed() / "geometry_priors.json"
    production_v2_path = processed() / "geofuse_geometry_v2_priors.json"
    production_v1 = json.loads(production_v1_path.read_text())
    production_v2 = json.loads(production_v2_path.read_text())
    mismatches_v1, max_error_v1 = compare_values(production_v1, rebuilt_v1)
    mismatches_v2, max_error_v2 = compare_values(
        production_v2,
        rebuilt_v2,
        ignore_paths={"$._meta.seconds"},
    )
    tolerance = float(args.tolerance)
    structural_v1 = [m for m in mismatches_v1 if m.get("kind") != "numeric"]
    structural_v2 = [m for m in mismatches_v2 if m.get("kind") != "numeric"]
    match_v1 = not structural_v1 and max_error_v1 <= tolerance
    match_v2 = not structural_v2 and max_error_v2 <= tolerance

    code_paths = [
        Path(__file__),
        REPO_ROOT / "scripts" / "run_phase2_priors.py",
        REPO_ROOT / "scripts" / "build_geofuse_geometry_v2_priors.py",
        REPO_ROOT / "src" / "rna3d" / "geometry" / "priors.py",
        REPO_ROOT / "src" / "rna3d" / "geofuse" / "geometry_v2.py",
        REPO_ROOT / "src" / "rna3d" / "geofuse" / "refine_v2.py",
    ]
    input_paths = [
        REPO_ROOT / "data" / "stanford-rna-3d-folding" / "train_sequences.v2.csv",
        REPO_ROOT / "data" / "stanford-rna-3d-folding" / "train_labels.v2.csv",
    ]
    audit = {
        "phase": "V4 Phase 1 P0-production reproduction",
        "performance_accessed": False,
        "target_count": len(manifest),
        "cutoff_rule": f"temporal_cutoff < {cutoff}",
        "manifest": {"path": relative(manifest_path), "sha256": sha256(manifest_path)},
        "production": {
            "distance_rg": {"path": relative(production_v1_path), "sha256": sha256(production_v1_path)},
            "angle_torsion": {"path": relative(production_v2_path), "sha256": sha256(production_v2_path)},
            "geometry_config": asdict(GeometryV2Config()),
        },
        "rebuild": {
            "distance_rg": {"path": relative(rebuilt_v1_path), "sha256": sha256(rebuilt_v1_path), "seconds": elapsed_v1},
            "angle_torsion": {"path": relative(rebuilt_v2_path), "sha256": sha256(rebuilt_v2_path), "seconds": elapsed_v2},
        },
        "comparison": {
            "absolute_tolerance": tolerance,
            "distance_rg_match": match_v1,
            "distance_rg_max_absolute_error": max_error_v1,
            "distance_rg_mismatch_count": len(mismatches_v1),
            "angle_torsion_match_excluding_runtime": match_v2,
            "angle_torsion_max_absolute_error": max_error_v2,
            "angle_torsion_mismatch_count_excluding_runtime": len(mismatches_v2),
            "first_distance_rg_mismatches": mismatches_v1[:10],
            "first_angle_torsion_mismatches": mismatches_v2[:10],
        },
        "inputs": [{"path": relative(path), "sha256": sha256(path)} for path in input_paths],
        "code": [{"path": relative(path), "sha256": sha256(path)} for path in code_paths],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    audit_path = report_dir / "p0_reproduction_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    status = "PASS" if match_v1 and match_v2 else "FAIL"
    report = f"""# V4 Phase 1: P0-production reproduction

Status: **{status}**

- Target manifest: {len(manifest):,} train-V2 RNAs with `temporal_cutoff < {cutoff}`.
- Prediction/native performance accessed: **No**.
- Distance/Rg prior match at tolerance `{tolerance:g}`: **{match_v1}**; maximum absolute error `{max_error_v1:.3g}`.
- Angle/torsion prior match at tolerance `{tolerance:g}`, excluding runtime metadata: **{match_v2}**; maximum absolute error `{max_error_v2:.3g}`.
- Production artifacts were read-only. Rebuilt artifacts are stored under `{relative(rebuild_dir)}`.
- Geometry production configuration is serialized in `p0_reproduction_audit.json`, including `steps=300` and `w_source=3.0`.

This audit establishes whether `P0-production` can be reconstructed from its declared
3,397-RNA input and current frozen code. It does not test whether P0 improves an RNA
prediction and does not authorize any change to the prior.
"""
    (report_dir / "README.md").write_text(report)
    print(json.dumps(audit["comparison"], indent=2))
    print(f"P0 reproduction: {status}")
    if args.require_match and status != "PASS":
        raise SystemExit(1)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--rebuild-dir", type=Path, default=DEFAULT_REBUILD)
    result.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--tolerance", type=float, default=1e-10)
    result.add_argument("--require-match", action="store_true")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
