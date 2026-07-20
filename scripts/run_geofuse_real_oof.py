#!/usr/bin/env python
"""Prepare and audit real out-of-fold data for the GeoFuse confidence gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.real_oof import (
    audit_pretrained_oof,
    audit_template_oof,
    grouped_temporal_split,
    sequence_group,
)
from rna3d.paths import cache, comp_file, processed


def _guess_pdb_ids(value: str | float) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {item.upper() for item in re.findall(r">([0-9][A-Za-z0-9]{3})_", value)}


def _complete_target_ids(requested: set[str], min_resolved_fraction: float) -> set[str]:
    counts: dict[str, int] = {target_id: 0 for target_id in requested}
    resolved: dict[str, int] = {target_id: 0 for target_id in requested}
    for chunk in pd.read_csv(
        comp_file("train_labels_v2"), usecols=["ID", "x_1"], chunksize=500_000
    ):
        target = chunk["ID"].str.rsplit("_", n=1).str[0]
        mask = target.isin(requested)
        if not mask.any():
            continue
        target = target[mask]
        is_resolved = chunk.loc[mask, "x_1"].to_numpy(float) > io.RESOLVED_THRESHOLD
        for target_id, group in pd.DataFrame(
            {"target_id": target.to_numpy(), "resolved": is_resolved}
        ).groupby("target_id"):
            counts[target_id] += len(group)
            resolved[target_id] += int(group["resolved"].sum())
    return {
        target_id
        for target_id in requested
        if counts[target_id] >= 3
        and resolved[target_id] / counts[target_id] >= min_resolved_fraction
    }


def _write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.target_id}\n{row.sequence}\n")


def _family_clusters(frame: pd.DataFrame, mmseqs: str | None, work_dir: Path) -> tuple[dict, str]:
    exact = dict(zip(frame["target_id"], frame["sequence_group"]))
    if not mmseqs:
        return exact, "exact_sequence_sha256"
    executable = shutil.which(mmseqs) or (str(Path(mmseqs)) if Path(mmseqs).exists() else None)
    if not executable:
        raise FileNotFoundError(f"MMseqs executable not found: {mmseqs}")
    fasta = work_dir / "eligible.fasta"
    _write_fasta(frame, fasta)
    with tempfile.TemporaryDirectory(prefix="mmseqs_family_", dir=work_dir) as temporary:
        temporary = Path(temporary)
        output = temporary / "family80"
        tmp = temporary / "tmp"
        subprocess.run(
            [
                executable,
                "easy-cluster",
                str(fasta),
                str(output),
                str(tmp),
                "--min-seq-id", "0.8",
                "-c", "0.8",
                "--cov-mode", "0",
                "--threads", "4",
            ],
            check=True,
        )
        clusters = pd.read_csv(
            f"{output}_cluster.tsv", sep="\t", names=["representative", "member"]
        )
    mapping = dict(zip(clusters["member"], clusters["representative"]))
    missing = set(frame["target_id"]) - set(mapping)
    if missing:
        raise RuntimeError(f"MMseqs family output omitted {len(missing)} targets")
    return mapping, "mmseqs_80pct_identity_80pct_coverage"


def cmd_prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = io.load_sequences("train_v2").copy()
    sequences["date"] = pd.to_datetime(sequences["temporal_cutoff"], errors="coerce")
    model_cutoff = pd.Timestamp(args.model_training_cutoff)
    last_date = pd.Timestamp(args.last_target_date) if args.last_target_date else sequences["date"].max()
    eligible = sequences[
        sequences["date"].notna()
        & (sequences["date"] > model_cutoff)
        & (sequences["date"] <= last_date)
        & sequences["seq_len"].between(args.min_len, args.max_len)
    ].copy()
    complete = _complete_target_ids(set(eligible["target_id"]), args.min_resolved_fraction)
    eligible = eligible[eligible["target_id"].isin(complete)].copy()
    eligible["sequence"] = eligible["sequence"].str.upper().str.replace("T", "U")
    eligible["sequence_group"] = eligible["sequence"].map(sequence_group)
    family_map, family_method = _family_clusters(eligible, args.mmseqs, output_dir)
    eligible["family_group"] = eligible["target_id"].map(family_map)
    eligible = grouped_temporal_split(
        eligible,
        args.calibration_fraction,
        args.validation_fraction,
        group_column="family_group",
    )
    eligible["excluded_pdb_ids"] = eligible["all_sequences"].map(
        lambda value: ";".join(sorted(_guess_pdb_ids(value)))
    )
    eligible["model_training_cutoff"] = str(model_cutoff.date())
    eligible["model_training_data"] = args.model_training_data
    columns = [
        "target_id", "sequence", "seq_len", "date", "split", "sequence_group",
        "family_group", "excluded_pdb_ids", "model_training_cutoff", "model_training_data",
    ]
    manifest = eligible[columns]
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    for split in ("train", "calibration", "validation"):
        subset = manifest[manifest["split"] == split]
        _write_fasta(subset, output_dir / f"{split}.fasta")
        (output_dir / f"{split}_targets.txt").write_text(
            "\n".join(subset["target_id"]) + ("\n" if len(subset) else "")
        )
    if args.pilot_per_split:
        pilot_parts = []
        for split in ("train", "calibration", "validation"):
            subset = manifest[
                (manifest["split"] == split) & (manifest["seq_len"] <= args.pilot_max_len)
            ].sort_values(["date", "seq_len", "target_id"])
            if len(subset) < args.pilot_per_split:
                raise ValueError(
                    f"{split}: only {len(subset)} targets satisfy pilot max length "
                    f"{args.pilot_max_len}"
                )
            indices = np.linspace(0, len(subset) - 1, args.pilot_per_split).round().astype(int)
            pilot_parts.append(subset.iloc[indices])
        pilot = pd.concat(pilot_parts).sort_values(["split", "date", "target_id"])
        pilot.to_csv(output_dir / "pilot_manifest.csv", index=False)
        (output_dir / "pilot_targets.txt").write_text("\n".join(pilot["target_id"]) + "\n")
        _write_fasta(pilot, output_dir / "pilot.fasta")
    summary = {
        "model_training_cutoff": str(model_cutoff.date()),
        "last_target_date": str(last_date.date()),
        "family_method": family_method,
        "targets": int(len(manifest)),
        "by_split": manifest["split"].value_counts().to_dict(),
        "length_min": int(manifest["seq_len"].min()),
        "length_max": int(manifest["seq_len"].max()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[manifest] {manifest_path}")


def cmd_audit(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, dtype={"target_id": str})
    store = CandidateCache(Path(args.cache_root), "train_v2")
    rows = []
    for row in manifest.itertuples(index=False):
        candidates = store.load_target(row.target_id, row.sequence)
        templates = [candidate for candidate in candidates if candidate.kind == "template"]
        pretrained = [candidate for candidate in candidates if candidate.kind == "pretrained"]
        excluded = set(str(row.excluded_pdb_ids).split(";")) - {"", "nan"}
        template_ok = 0
        pretrained_ok = 0
        errors = []
        for candidate in templates:
            try:
                audit_template_oof(candidate, row.date, excluded)
                template_ok += 1
            except ValueError as exc:
                errors.append(str(exc))
        for candidate in pretrained:
            try:
                audit_pretrained_oof(candidate, row.date)
                pretrained_ok += 1
            except ValueError as exc:
                errors.append(str(exc))
        rows.append(
            {
                "target_id": row.target_id,
                "split": row.split,
                "template_ok": template_ok,
                "pretrained_ok": pretrained_ok,
                "ready": template_ok > 0 and pretrained_ok > 0,
                "errors": " | ".join(errors[:3]),
            }
        )
    audit = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(
        audit.groupby("split").agg(
            targets=("target_id", "size"), ready=("ready", "sum"),
            templates=("template_ok", "sum"), pretrained=("pretrained_ok", "sum"),
        ).to_string()
    )
    print(f"[audit] {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="make a post-checkpoint temporal/family split")
    prepare.add_argument("--model-training-cutoff", required=True)
    prepare.add_argument("--model-training-data", required=True)
    prepare.add_argument("--last-target-date")
    prepare.add_argument("--min-len", type=int, default=30)
    prepare.add_argument("--max-len", type=int, default=400)
    prepare.add_argument("--min-resolved-fraction", type=float, default=0.8)
    prepare.add_argument("--calibration-fraction", type=float, default=0.15)
    prepare.add_argument("--validation-fraction", type=float, default=0.20)
    prepare.add_argument("--mmseqs", help="MMseqs executable; omit for exact-duplicate grouping")
    prepare.add_argument("--pilot-per-split", type=int, default=5)
    prepare.add_argument("--pilot-max-len", type=int, default=100)
    prepare.add_argument("--output-dir", default=str(processed() / "geofuse_real_oof"))
    prepare.set_defaults(func=cmd_prepare)

    audit = sub.add_parser("audit", help="verify cached TBM/pretrained pairs and provenance")
    audit.add_argument("--manifest", default=str(processed() / "geofuse_real_oof" / "manifest.csv"))
    audit.add_argument("--cache-root", default=str(cache() / "geofuse_candidates"))
    audit.add_argument("--output", default=str(cache() / "geofuse_real_oof_audit.csv"))
    audit.set_defaults(func=cmd_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
