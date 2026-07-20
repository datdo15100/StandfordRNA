#!/usr/bin/env python
"""Audit exact sequence overlap between an evaluation split and model FASTAs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.overlap import audit_exact_sequence_overlap


def model_fasta(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected MODEL=/path/to/train.fasta")
    model, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser()
    if not model.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid model FASTA: {value}")
    return model.strip(), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="validation", choices=["train", "train_v2", "validation"]
    )
    parser.add_argument(
        "--model-fasta",
        action="append",
        required=True,
        type=model_fasta,
        help="MODEL=/path/to/training.fasta; repeat for multiple models",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT
        / "reports"
        / "tables"
        / "geofuse_phase_a"
        / "pretrained_exact_overlap.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_pretrained_overlap.md",
    )
    args = parser.parse_args()

    result = audit_exact_sequence_overlap(io.load_sequences(args.split), args.model_fasta)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    overlaps = result[result["exact_overlap"]].copy()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GeoFuse pretrained exact-sequence overlap audit",
        "",
        f"Evaluation split: `{args.split}`.",
        "",
        "This is a conservative exact normalized-sequence audit (DNA `T` and RNA `U` "
        "are treated as equivalent). It does not test structural or remote-homology "
        "overlap, and it only covers the supplied training manifests.",
        "",
        f"- Target/model pairs checked: {len(result)}",
        f"- Exact target/model overlaps: {len(overlaps)}",
        "",
        "## Exact matches",
        "",
    ]
    if overlaps.empty:
        lines.append("No exact matches found.")
    else:
        lines.append(
            overlaps[["target_id", "seq_len", "model", "matching_training_ids"]]
            .to_markdown(index=False)
        )
    lines.extend(
        [
            "",
            "## Evaluation rule",
            "",
            "Report the full pretrained result as competition-style evidence. For the "
            "thesis's retrospective validation claim, also report a sensitivity analysis "
            "that excludes every exact-overlap target for the relevant model.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines))
    print(overlaps[["target_id", "model", "matching_training_ids"]].to_string(index=False))
    print(f"[csv] {args.output_csv}")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
