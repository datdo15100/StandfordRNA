#!/usr/bin/env python
"""Secondary CASP15 mechanism experiments for V5 TBM.

This script runs only after the core same-sandbox John/V3 banks are frozen.  It
tests the historically deployed composite retrieval weights as a small,
scientifically interpretable family and isolates distinct-PDB selection.  These
are mechanism analyses; the exact reconstructed V3 bank remains a complete
contender regardless of any one 12-target component delta.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rna3d.baselines.top1 import (
    _diversity_clustering,
    build_raw_candidates,
    composite_similarity_components,
    rna_features,
)
from rna3d.data import io
from rna3d.eval.usalign import score_target
from rna3d.template import composite_search, db, mmseqs_search

import run_v5_casp15 as core


CACHE = REPO / "data" / "cache" / "v5_casp15_secondary"
OUT = REPO / "reports" / "thesis_v5" / "experiments" / "secondary_tbm"
MANIFEST = CACHE / "secondary_bank_manifest.csv"
RECEIPT = OUT / "secondary_bank_freeze.json"

SCORE_FAMILY = {
    # Historical constants and three parameterizations that preserve the
    # relative weights within a signal family.
    "FULL_4321": (0.4, 0.3, 0.2, 0.1),
    "ALIGNMENT_43": (4.0 / 7.0, 3.0 / 7.0, 0.0, 0.0),
    "FEATURE_KMER_21": (0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0),
    "GLOBAL_ONLY": (1.0, 0.0, 0.0, 0.0),
    "LOCAL_ONLY": (0.0, 1.0, 0.0, 0.0),
    "FEATURE_ONLY": (0.0, 0.0, 1.0, 0.0),
    "KMER3_ONLY": (0.0, 0.0, 0.0, 1.0),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _length_eligible(query: str, template: str) -> bool:
    relative = abs(len(template) - len(query)) / max(len(template), len(query))
    if len(query) < 50 or len(template) < 50:
        return relative <= 0.6
    if len(query) > 1000 or len(template) > 1000:
        return relative <= 0.2
    return relative <= 0.4


def _select_diverse(scored: list[dict], top_n: int = 8) -> list[dict]:
    scored = sorted(scored, key=lambda item: -float(item["retrieval_score"]))
    if len(scored) > 10:
        threshold = np.percentile([item["retrieval_score"] for item in scored], 80)
        scored = [item for item in scored if item["retrieval_score"] >= threshold][:50]
    else:
        scored = scored[:50]
    if len(scored) <= top_n:
        return scored[:top_n]
    features = np.asarray([rna_features(item["seq"]) for item in scored])
    clusters = min(top_n, len(scored))
    if len(scored) >= 15:
        labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(features)
    else:
        labels = _diversity_clustering(features, clusters)
    output = []
    for cluster in range(clusters):
        members = [item for index, item in enumerate(scored) if labels[index] == cluster]
        if members:
            output.append(max(members, key=lambda item: item["retrieval_score"]))
    return sorted(output, key=lambda item: -float(item["retrieval_score"]))[:top_n]


def _score_library(record: dict) -> dict[str, list[dict]]:
    meta, coords = composite_search._library()
    cutoff = str(record["temporal_cutoff"])
    excluded = set(core.exclusions(record["excluded_pdb_ids"]))
    query = str(record["sequence"])
    qfeat = rna_features(query)
    rows = []
    for row in meta.itertuples(index=False):
        if str(row.release_date) >= cutoff or str(row.pdb_id).upper() in excluded:
            continue
        sequence = str(row.sequence)
        if not _length_eligible(query, sequence):
            continue
        entry = coords.get(row.target_id)
        if entry is None:
            continue
        components = composite_similarity_components(query, sequence, qfeat)
        rows.append(
            {
                "chain_key": str(row.target_id),
                "seq": sequence,
                "coords": np.asarray(entry["coords"], dtype=float),
                "pdb_id": str(row.pdb_id).upper(),
                "release_date": str(row.release_date),
                **components,
            }
        )
    selected = {}
    for name, weights in SCORE_FAMILY.items():
        values = []
        for row in rows:
            score = (
                weights[0] * row["global"]
                + weights[1] * row["local"]
                + weights[2] * row["features"]
                + weights[3] * row["kmer3"]
            )
            if score > 0:
                values.append({**row, "retrieval_score": float(score)})
        selected[name] = _select_diverse(values, top_n=8)
    return selected


def _mm_items(record: dict, hits_records: list[dict], adjacent: float) -> list[dict]:
    meta = db.load_meta().set_index("chain_key")
    coords = db.load_coords()
    cutoff = str(record["temporal_cutoff"])
    excluded = set(core.exclusions(record["excluded_pdb_ids"]))
    output = []
    seen = set()
    for hit in sorted(hits_records, key=lambda value: -float(value["bits"])):
        key = str(hit["target"])
        if key in seen or key not in meta.index:
            continue
        seen.add(key)
        row = meta.loc[key]
        if str(row.release_date) >= cutoff or str(row.pdb_id).upper() in excluded:
            continue
        entry = coords.get(key)
        if entry is None:
            continue
        value = core._materialize(
            record["sequence"],
            {
                "chain_key": key,
                "seq": str(row.seq),
                "coords": entry["coords"],
                "pdb_id": row.pdb_id,
                "release_date": row.release_date,
            },
            adjacent,
            "mmseqs",
        )
        if value is not None:
            output.append(value)
        if len(output) == 40:
            break
    return output


def _target_build(record: dict, hits_records: list[dict], replace: bool) -> dict:
    target_id = record["target_id"]
    output = CACHE / target_id / "secondary_banks.npz"
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists() and not replace:
        return {
            "target_id": target_id,
            "path": str(output.relative_to(REPO)),
            "sha256": sha256(output),
            "metadata_sha256": sha256(metadata_path),
            "status": "cached",
        }
    adjacent = float(json.loads(core.P0.read_text())["adjacent_c1"]["mean"])
    selected_hits = _score_library(record)
    mm_items = _mm_items(record, hits_records, adjacent)
    arrays = {}
    banks = {}
    for score_name, hits in selected_hits.items():
        comp_items = []
        for hit in hits:
            item = core._materialize(record["sequence"], hit, adjacent, "composite")
            if item is not None:
                comp_items.append(item)
        seen = {item["chain_key"] for item in mm_items}
        merged = mm_items + [item for item in comp_items if item["chain_key"] not in seen]
        for source_name, items in (("COMPOSITE", comp_items), ("MERGED", merged)):
            name = f"{source_name}_{score_name}"
            chosen = core._rank(items, "ICS", distinct_pdb=True)
            bank = core._pad_v3(chosen, record["sequence"], "linear")
            arrays[f"{name}__coords"] = bank[0]
            banks[name] = {
                "candidate_ids": bank[4],
                "pdb_ids": bank[5],
                "sources": bank[6],
                "fallback_slots": bank[7],
            }
        if score_name == "FULL_4321":
            name = "MERGED_FULL_4321_NO_DISTINCT"
            chosen = core._rank(merged, "ICS", distinct_pdb=False)
            bank = core._pad_v3(chosen, record["sequence"], "linear")
            arrays[f"{name}__coords"] = bank[0]
            banks[name] = {
                "candidate_ids": bank[4],
                "pdb_ids": bank[5],
                "sources": bank[6],
                "fallback_slots": bank[7],
            }

    # Database-view sensitivity: John algorithm unchanged, but scan only the
    # V3 7,155-sequence derived view instead of the main full shared snapshot.
    meta, coords = composite_search._library()
    cutoff = str(record["temporal_cutoff"])
    excluded = set(core.exclusions(record["excluded_pdb_ids"]))
    templates = [
        (str(row.target_id), str(row.sequence), np.asarray(coords[row.target_id]["coords"], float))
        for row in meta.itertuples(index=False)
        if str(row.release_date) < cutoff
        and str(row.pdb_id).upper() not in excluded
        and row.target_id in coords
    ]
    john = build_raw_candidates(
        record["sequence"], target_id, templates, n=5, base_seed=core.SEED
    )
    arrays["J_SS_DEDUP_VIEW__coords"] = np.asarray(
        [item.coords for item in john], dtype=np.float32
    )
    banks["J_SS_DEDUP_VIEW"] = {
        "candidate_ids": [item.template_id or f"de_novo_{i}" for i, item in enumerate(john)],
        "sources": [item.source for item in john],
        "fallback_slots": sum(item.source != "template" for item in john),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    write_json(
        metadata_path,
        {
            "schema": "v5-casp15-secondary-tbm-v1",
            "target_id": target_id,
            "sequence_sha256": hashlib.sha256(record["sequence"].encode()).hexdigest(),
            "score_family": {name: list(weights) for name, weights in SCORE_FAMILY.items()},
            "banks": banks,
            "array_hashes": {name: core.array_sha256(value) for name, value in arrays.items()},
        },
    )
    return {
        "target_id": target_id,
        "path": str(output.relative_to(REPO)),
        "sha256": sha256(output),
        "metadata_sha256": sha256(metadata_path),
        "status": "generated",
    }


def cmd_build(args: argparse.Namespace) -> None:
    core.verify_raw()
    targets = core.read_targets()
    hits = mmseqs_search.read_m8(core.MMSEQS_HITS)
    by_target = {key: value.to_dict("records") for key, value in hits.groupby("query")}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _target_build,
                record,
                by_target.get(record["target_id"], []),
                args.replace,
            ): record["target_id"]
            for record in targets.to_dict("records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            value = future.result()
            rows.append(value)
            print(f"[{index:02d}/12 {value['target_id']}] {value['status']}", flush=True)
    frame = pd.DataFrame(rows).sort_values("target_id")
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST, index=False)
    write_json(
        RECEIPT,
        {
            "status": "V5_CASP15_SECONDARY_TBM_BANKS_FROZEN",
            "created_utc": now_utc(),
            "scientific_role": "secondary mechanism analysis on CASP15 development/validation",
            "target_n": len(frame),
            "core_raw_receipt_sha256": sha256(core.RAW_RECEIPT),
            "manifest_sha256": sha256(MANIFEST),
            "code_sha256": sha256(Path(__file__)),
            "score_family": {name: list(weights) for name, weights in SCORE_FAMILY.items()},
        },
    )


def verify() -> pd.DataFrame:
    receipt = json.loads(RECEIPT.read_text())
    if sha256(MANIFEST) != receipt["manifest_sha256"]:
        raise RuntimeError("secondary manifest changed after freeze")
    frame = pd.read_csv(MANIFEST, dtype=str).fillna("")
    for row in frame.itertuples(index=False):
        path = REPO / row.path
        if sha256(path) != row.sha256 or sha256(path.with_suffix(".json")) != row.metadata_sha256:
            raise RuntimeError(f"secondary artifact changed: {row.target_id}")
    return frame


def cmd_evaluate(_: argparse.Namespace) -> None:
    verify()
    targets = core.read_targets()
    labels = io.load_labels("validation")
    rows = []
    allocation_rows = []
    for index, target in enumerate(targets.itertuples(index=False), start=1):
        references = io.get_reference_coords(labels, target.target_id)
        score_cache: dict[str, float] = {}
        with np.load(CACHE / target.target_id / "secondary_banks.npz") as payload:
            row = {
                "target_id": target.target_id,
                "sequence_cluster": target.sequence_cluster,
                "length": len(target.sequence),
            }
            for key in payload.files:
                if key.endswith("__coords"):
                    name = key.removesuffix("__coords")
                    records = [
                        {"coords": np.asarray(coords, dtype=float)}
                        for coords in payload[key]
                    ]
                    row[name] = core._score_bank(
                        records,
                        references,
                        target.sequence,
                        cache=score_cache,
                    )
            drfold = core._drfold_records(target.target_id, target.sequence)
            for tbm_base in (
                "COMPOSITE_ALIGNMENT_43",
                "COMPOSITE_GLOBAL_ONLY",
                "COMPOSITE_FULL_4321",
                "J_SS_DEDUP_VIEW",
            ):
                templates = [
                    {
                        "coords": np.asarray(coords, dtype=float),
                        "candidate_id": f"{tbm_base}_{candidate_index}",
                        "source": "T",
                    }
                    for candidate_index, coords in enumerate(
                        payload[f"{tbm_base}__coords"]
                    )
                ]
                for allocation, (template_n, drfold_n) in {
                    "5T": (5, 0),
                    "4T+1D": (4, 1),
                    "3T+2D": (3, 2),
                }.items():
                    bank, realized = core.allocate_bank(
                        templates, drfold, template_n, drfold_n
                    )
                    allocation_rows.append(
                        {
                            "target_id": target.target_id,
                            "sequence_cluster": target.sequence_cluster,
                            "length": len(target.sequence),
                            "tbm_base": tbm_base,
                            "allocation": allocation,
                            "best_tm": core._score_bank(
                                bank,
                                references,
                                target.sequence,
                                cache=score_cache,
                            ),
                            **realized,
                        }
                    )
        # Core comparator cells are copied from their frozen evaluation table,
        # never regenerated under a subtly different boundary.
        rows.append(row)
        print(f"[{index:02d}/12 {target.target_id}] secondary scored", flush=True)
    frame = pd.DataFrame(rows)
    core_scores = pd.read_csv(core.OUT / "raw_results" / "rq1_target_best5_tm.csv")
    frame = frame.merge(
        core_scores[["target_id", "J_SS_RAW", "V3_COMPOSITE_ICS_LINEAR", "V3_EXACT_RAW"]],
        on="target_id",
        how="left",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "target_best5_tm.csv", index=False)
    allocations = pd.DataFrame(allocation_rows)
    allocations.to_csv(OUT / "candidate_allocation_target_tm.csv", index=False)
    variants = [
        value
        for value in frame.columns
        if value not in {"target_id", "sequence_cluster", "length"}
    ]
    clusters = frame["sequence_cluster"].to_numpy()
    summary_rows = []
    effects = {}
    for variant in variants:
        delta = frame[variant].to_numpy() - frame["COMPOSITE_FULL_4321"].to_numpy()
        value = core._bootstrap_effect(delta, clusters)
        effects[variant] = value
        summary_rows.append(
            {
                "variant": variant,
                "mean_best5_tm": float(frame[variant].mean()),
                "comparator": "COMPOSITE_FULL_4321",
                **value,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_best5_tm", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    write_json(OUT / "effects_vs_historical_full_composite.json", effects)

    # Interaction table: does adding MMseqs help each composite score family?
    interaction_rows = []
    for score_name in SCORE_FAMILY:
        before = f"COMPOSITE_{score_name}"
        after = f"MERGED_{score_name}"
        delta = frame[after].to_numpy() - frame[before].to_numpy()
        interaction_rows.append(
            {
                "score_family": score_name,
                "before": before,
                "after": after,
                **core._bootstrap_effect(delta, clusters),
            }
        )
    pd.DataFrame(interaction_rows).to_csv(OUT / "mmseqs_interactions.csv", index=False)
    allocation_summary = []
    allocation_effects = {}
    for tbm_base, group in allocations.groupby("tbm_base"):
        wide = group.pivot(
            index=["target_id", "sequence_cluster"],
            columns="allocation",
            values="best_tm",
        ).reset_index()
        for allocation in ("5T", "4T+1D", "3T+2D"):
            delta = wide[allocation].to_numpy() - wide["5T"].to_numpy()
            value = core._bootstrap_effect(
                delta, wide["sequence_cluster"].to_numpy()
            )
            allocation_effects[f"{tbm_base}:{allocation}_vs_5T"] = value
            allocation_summary.append(
                {
                    "tbm_base": tbm_base,
                    "allocation": allocation,
                    "mean_best_tm": float(wide[allocation].mean()),
                    **value,
                }
            )
    pd.DataFrame(allocation_summary).to_csv(
        OUT / "candidate_allocation_summary.csv", index=False
    )
    write_json(OUT / "candidate_allocation_effects.json", allocation_effects)
    write_json(
        OUT / "secondary_evaluation_receipt.json",
        {
            "status": "V5_CASP15_SECONDARY_TBM_COMPLETE",
            "created_utc": now_utc(),
            "target_n": 12,
            "sequence_cluster_n": targets["sequence_cluster"].nunique(),
            "target_scores_sha256": sha256(OUT / "target_best5_tm.csv"),
            "summary_sha256": sha256(OUT / "summary.csv"),
            "candidate_allocation_sha256": sha256(
                OUT / "candidate_allocation_target_tm.csv"
            ),
            "code_sha256": sha256(Path(__file__)),
        },
    )
    print(summary.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--workers", type=int, default=3)
    build.add_argument("--replace", action="store_true")
    build.set_defaults(func=cmd_build)
    evaluate = commands.add_parser("evaluate")
    evaluate.set_defaults(func=cmd_evaluate)
    return result


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    arguments = parser().parse_args()
    arguments.func(arguments)
