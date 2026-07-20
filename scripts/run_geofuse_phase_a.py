#!/usr/bin/env python
"""Build/import GeoFuse candidates and run the Phase-A oracle-pool gate."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.candidate import CandidateCache, from_tbm_candidate, safe_name
from rna3d.geofuse.phase_a import evaluate_candidate_pool, write_phase_a_report
from rna3d.geofuse.structure_io import discover_structure_files, import_structure
from rna3d.paths import cache, casp15_safe_cutoff, processed
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.template import db, mmseqs_search


DEFAULT_PATTERNS = {
    "drfold2": ["**/{target_id}/relax/model_*.pdb"],
    "boltz": [
        "**/predictions/{target_id}/{target_id}_model_*.cif",
        "**/predictions/{target_id}/{target_id}_model_*.mmcif",
    ],
}


def guess_pdb_ids(all_sequences: str | float) -> tuple[str, ...]:
    if not isinstance(all_sequences, str):
        return ()
    ids = re.findall(r">([0-9][A-Za-z0-9]{3})_", all_sequences)
    return tuple(sorted({value.upper() for value in ids}))


def selected_sequences(args: argparse.Namespace) -> pd.DataFrame:
    sequences = io.load_sequences(args.split)
    if getattr(args, "target_ids", None):
        requested = {item.strip() for item in args.target_ids.split(",") if item.strip()}
        missing = requested - set(sequences["target_id"])
        if missing:
            raise KeyError(f"unknown target IDs for {args.split}: {sorted(missing)}")
        sequences = sequences[sequences["target_id"].isin(requested)]
    if getattr(args, "limit", None):
        sequences = sequences.head(args.limit)
    return sequences.reset_index(drop=True)


def get_cache(args: argparse.Namespace) -> CandidateCache:
    root = Path(args.cache_root) if args.cache_root else cache() / "geofuse_candidates"
    return CandidateCache(root, args.split)


def load_or_search_hits(sequences: pd.DataFrame, split: str, refresh: bool) -> pd.DataFrame:
    legacy = cache() / f"{split}_hits.m8"
    phase_dir = cache() / "geofuse_phase_a"
    phase_dir.mkdir(parents=True, exist_ok=True)
    hit_path = phase_dir / f"{split}_hits.m8"
    if not refresh:
        for existing in (hit_path, legacy):
            if existing.exists():
                hits = mmseqs_search.read_m8(existing)
                if set(sequences["target_id"]).issubset(set(hits["query"])):
                    print(f"[search] reuse {existing}")
                    return hits

    query_path = phase_dir / f"{split}_queries.fasta"
    with open(query_path, "w") as handle:
        for _, row in sequences.iterrows():
            handle.write(f">{row['target_id']}\n{row['sequence']}\n")
    print(f"[search] MMseqs2 -> {hit_path}")
    return mmseqs_search.search(query_path, hit_path)


def cmd_build_tbm(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    store = get_cache(args)
    hits = load_or_search_hits(sequences, args.split, args.refresh_search)
    meta = db.load_meta()
    db.load_coords()
    prior_path = processed() / "geometry_priors.json"
    adj_dist = 6.0
    if prior_path.exists():
        import json

        adj_dist = float(json.loads(prior_path.read_text())["adjacent_c1"]["mean"])

    written = 0
    for _, row in sequences.iterrows():
        target_id = str(row["target_id"])
        sequence = str(row["sequence"])
        if len(sequence) > args.max_len:
            print(f"[{target_id}] skip L={len(sequence)} > max-len={args.max_len}")
            continue
        cutoff = row.get("temporal_cutoff") or casp15_safe_cutoff()
        target_hits = hits[hits["query"] == target_id]
        candidates = build_tbm_candidates(
            target_id,
            sequence,
            cutoff,
            target_hits,
            meta,
            adj_dist=adj_dist,
            max_candidates=args.max_candidates,
            exclude_pdb_ids=guess_pdb_ids(row.get("all_sequences")),
            rng=np.random.default_rng(args.seed),
        )
        for raw in candidates:
            store.save(from_tbm_candidate(raw, sequence), overwrite=args.overwrite)
            written += 1
        print(f"[{target_id}] cached {len(candidates)} temporal-safe TBM candidates")
    print(f"[done] {written} TBM candidates in {store.split_dir}")


def cmd_import(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    store = get_cache(args)
    patterns = args.glob or DEFAULT_PATTERNS.get(
        args.source, ["**/{target_id}*.pdb", "**/{target_id}*.cif", "**/{target_id}*.mmcif"]
    )
    total = 0
    failures = []
    for _, row in sequences.iterrows():
        target_id = str(row["target_id"])
        sequence = str(row["sequence"])
        files = discover_structure_files(args.root, patterns, target_id)[: args.max_candidates]
        imported = 0
        for index, path in enumerate(files):
            candidate_id = f"{safe_name(args.source)}__{safe_name(args.model)}__{index + 1:02d}"
            try:
                candidate = import_structure(
                    path,
                    target_id=target_id,
                    sequence=sequence,
                    candidate_id=candidate_id,
                    source=args.source,
                    model=args.model,
                    default_confidence=args.default_confidence,
                )
                store.save(candidate, overwrite=args.overwrite)
                imported += 1
                total += 1
            except Exception as exc:
                failures.append((target_id, str(path), str(exc)))
                print(f"[{target_id}] reject {path}: {exc}")
                if args.fail_fast:
                    raise
        print(f"[{target_id}] imported {imported}/{len(files)} {args.source} candidates")
    print(f"[done] imported {total}; rejected {len(failures)}; cache={store.split_dir}")
    if failures:
        failure_path = cache() / "geofuse_phase_a" / f"{args.split}_{safe_name(args.source)}_failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures, columns=["target_id", "path", "error"]).to_csv(
            failure_path, index=False
        )
        print(f"[failures] {failure_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    labels = io.load_labels(args.split)
    candidates, targets, summary = evaluate_candidate_pool(
        sequences, labels, get_cache(args), max_selected=args.max_selected
    )
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "reports" / "tables" / "geofuse_phase_a"
    report_path = Path(args.report) if args.report else REPO_ROOT / "reports" / "thesis_notes" / "geofuse_phase_a_gate.md"
    result = write_phase_a_report(
        candidates,
        targets,
        summary,
        output_dir,
        report_path,
        min_mean_gain=args.min_mean_gain,
    )
    print(pd.Series(result).to_string())
    print(f"[tables] {output_dir}")
    print(f"[report] {report_path}")


def cmd_status(args: argparse.Namespace) -> None:
    inventory = pd.DataFrame(get_cache(args).inventory())
    if inventory.empty:
        print("No normalized candidates cached.")
        return
    summary = (
        inventory.groupby(["kind", "source", "model"])
        .agg(targets=("target_id", "nunique"), candidates=("candidate_id", "size"))
        .reset_index()
    )
    print(summary.to_string(index=False))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="validation", choices=["train", "train_v2", "validation"])
    parser.add_argument("--target-ids", help="comma-separated target IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-root", help="default: data/cache/geofuse_candidates")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    tbm = sub.add_parser("build-tbm", help="cache temporal-safe TBM candidates")
    add_common(tbm)
    tbm.add_argument("--max-candidates", type=int, default=10)
    # TBM transfer/gap filling is CPU-side and handled the 720-nt validation
    # target safely. Keep the default above the competition range; GPU model
    # length routing belongs in the model-specific runners instead.
    tbm.add_argument("--max-len", type=int, default=2000)
    tbm.add_argument("--seed", type=int, default=0)
    tbm.add_argument("--refresh-search", action="store_true")
    tbm.add_argument("--overwrite", action="store_true")
    tbm.set_defaults(func=cmd_build_tbm)

    imported = sub.add_parser("import", help="normalize DRfold2/Boltz structure outputs")
    add_common(imported)
    imported.add_argument("--source", required=True, help="e.g. drfold2 or boltz")
    imported.add_argument("--model", required=True, help="checkpoint/config label")
    imported.add_argument("--root", required=True, type=Path)
    imported.add_argument(
        "--glob", action="append", help="target-aware glob; may contain {target_id}; repeatable"
    )
    imported.add_argument("--max-candidates", type=int, default=10)
    imported.add_argument("--default-confidence", type=float, default=0.5)
    imported.add_argument("--overwrite", action="store_true")
    imported.add_argument("--fail-fast", action="store_true")
    imported.set_defaults(func=cmd_import)

    evaluate = sub.add_parser("evaluate", help="measure selected and oracle candidate-pool TM")
    add_common(evaluate)
    evaluate.add_argument("--max-selected", type=int, default=5)
    evaluate.add_argument("--min-mean-gain", type=float, default=0.0)
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--report")
    evaluate.set_defaults(func=cmd_evaluate)

    status = sub.add_parser("status", help="summarize the normalized candidate cache")
    add_common(status)
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
