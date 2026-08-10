#!/usr/bin/env python
"""Export the native-blind TBM + pretrained + Geometry-v2 Kaggle submission.

This script deliberately never loads validation labels.  The public CASP15 test
sequences are matched to the already-normalized candidate cache, five candidates
are selected by the frozen source-balanced confidence rule, and Geometry v2 is
applied with its frozen configuration before the competition CSV is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.data import io
from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.phase_a import select_source_balanced
from rna3d.paths import cache, comp_file
from run_geofuse_phase_b import _cached_refinement, load_priors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_submission(
    output_dir: Path,
    *,
    candidate_split: str = "validation",
    candidates_per_target: int = 5,
    steps: int = 300,
    device: str = "cuda",
) -> tuple[Path, Path]:
    sequences = io.load_sequences("test")
    sample = pd.read_csv(comp_file("sample_submission"))
    store = CandidateCache(cache() / "geofuse_candidates", candidate_split)
    priors_v1, priors_v2 = load_priors()

    predictions: dict[str, np.ndarray] = {}
    targets: list[dict] = []
    for row in sequences.itertuples(index=False):
        target_id = str(row.target_id)
        sequence = str(row.sequence)
        bank = store.load_target(target_id, sequence)
        selected = select_source_balanced(bank, candidates_per_target)
        if len(selected) != candidates_per_target:
            raise RuntimeError(
                f"{target_id}: expected {candidates_per_target} selected candidates, "
                f"found {len(selected)}"
            )

        refined: list[np.ndarray] = []
        records: list[dict] = []
        for rank, candidate in enumerate(selected, start=1):
            coords, seconds = _cached_refinement(
                candidate,
                sequence,
                "v2",
                priors_v1,
                priors_v2,
                steps,
                device,
                False,
            )
            coords = np.asarray(coords, dtype=np.float32)
            if coords.shape != (len(sequence), 3) or not np.isfinite(coords).all():
                raise ValueError(f"{candidate.candidate_id}: invalid refined coordinates")
            refined.append(coords)
            records.append(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "source": candidate.source,
                    "model": candidate.model,
                    "global_confidence": candidate.global_confidence,
                    "support_fraction": float(candidate.support_mask.mean()),
                    "geometry_v2_seconds_cached_or_measured": seconds,
                }
            )

        predictions[target_id] = np.stack(refined, axis=0)
        targets.append(
            {
                "target_id": target_id,
                "sequence_length": len(sequence),
                "candidate_bank_size": len(bank),
                "selected": records,
            }
        )
        summary = ", ".join(f"{x['source']}:{x['candidate_id']}" for x in records)
        print(f"[{target_id}] {summary}")

    submission = io.build_submission(predictions, sequences)
    io.validate_submission(submission, sequences)
    submission = io.order_submission_like(submission, sample)
    coordinate_values = submission[io.SUBMISSION_COORD_COLS].to_numpy(dtype=float)
    if not np.isfinite(coordinate_values).all():
        raise ValueError("submission contains non-finite coordinates")

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = io.write_submission(submission, output_dir / "submission.csv")
    manifest = {
        "pipeline": "source-balanced TBM + pretrained candidates, then Geometry v2",
        "native_labels_used": False,
        "selection_rule": (
            "deterministic source-balanced round robin; within each source sort by "
            "descending model-side global confidence, then candidate_id"
        ),
        "candidate_cache_split": candidate_split,
        "candidates_per_target": candidates_per_target,
        "geometry_v2_steps": steps,
        "geometry_v2_device": device,
        "submission_rows": int(len(submission)),
        "submission_columns": list(submission.columns),
        "targets": targets,
    }
    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["submission_sha256"] = sha256(submission_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[submission] {submission_path}")
    print(f"[sha256] {manifest['submission_sha256']}")
    print(f"[manifest] {manifest_path}")
    return submission_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "kaggle_hybrid_geometry_v2",
    )
    parser.add_argument("--candidate-split", default="validation")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_submission(
        args.output_dir,
        candidate_split=args.candidate_split,
        candidates_per_target=args.candidates,
        steps=args.steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
