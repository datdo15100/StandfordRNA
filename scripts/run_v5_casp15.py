#!/usr/bin/env python
"""Run the CASP15 component experiments for the V5 scientific design.

The command boundaries deliberately separate native-blind candidate generation
from native-aware evaluation.  CASP15 is a development/validation benchmark in
V5, not an untouched confirmatory cohort, but this separation still makes every
result traceable to an immutable candidate artifact.

Commands implemented in this first executable stage:

``build-raw``
    Build the same-sandbox John raw bank and the historical V3 TBM ablation
    banks on the frozen 12-target CASP15 sandbox.  This command never loads a
    native structure.

``evaluate-raw``
    Verify the frozen raw artifacts, then evaluate RQ1 and the fixed-budget raw
    RQ2 allocations (5T, 4T+1D, 3T+2D, 2T, 1T+1D, 2D).

``audit``
    Verify every frozen input and cached direct-DRfold2 candidate used by RQ2.

Refinement is intentionally a later command in this same runner so the raw
candidate boundary can be audited before any refiner is applied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.baselines.top1 import build_raw_candidates
from rna3d.data import io
from rna3d.eval.self_tm import mean_pairwise_self_tm
from rna3d.eval.local_metrics import local_accuracy_metrics
from rna3d.eval.usalign import score_target
from rna3d.eval.v4_statistics import cluster_bootstrap_means
from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.geometry.denovo import de_novo_ensemble
from rna3d.refine.optimizer import RefineConfig, refine_structure
from rna3d.refine.rule_based import refine_rule_based
from rna3d.template import composite_search, db, mmseqs_search
from rna3d.template.align import align_and_transfer
from rna3d.template.gap_fill import fill_gaps, fill_gaps_john, fill_gaps_linear


OUT = REPO / "reports" / "thesis_v5" / "experiments"
CACHE = REPO / "data" / "cache" / "v5_casp15"
RAW_MANIFEST = CACHE / "raw_bank_manifest.csv"
RAW_RECEIPT = OUT / "raw_bank_freeze.json"
REFINED_MANIFEST = CACHE / "refined_bank_manifest.csv"
REFINED_RECEIPT = OUT / "refined_bank_freeze.json"
TARGET_MANIFEST = REPO / "reports" / "thesis_v5" / "casp15_target_manifest.csv"
DESIGN_AUDIT = REPO / "reports" / "thesis_v5" / "v5_design_input_audit.json"
P0 = REPO / "data" / "processed" / "geometry_priors.json"
P1 = REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json"
DRFOLD_CACHE = REPO / "data" / "cache" / "geofuse_candidates"
MMSEQS_HITS = CACHE / "casp15_hits.m8"
QUERY_FASTA = CACHE / "casp15_queries.fasta"
USALIGN = REPO / "external" / "binaries" / "USalign"
SEED = 20260821
TIE = 1e-6

# The first five cells recover the historically deployed V3 candidate bank one
# Lego at a time.  The last cells are the pre-specified interaction/mechanism
# comparisons needed to avoid attributing a merged effect to the wrong Lego.
V3_VARIANTS = (
    "V3_COMPOSITE_IC_LINEAR",
    "V3_MMSEQS_IC_LINEAR",
    "V3_MERGED_IC_LINEAR",
    "V3_MERGED_ICS_LINEAR",
    "V3_EXACT_RAW",
    "V3_MERGED_I_LINEAR",
    "V3_COMPOSITE_ICS_LINEAR",
    "V3_MERGED_ICS_JOHN_GAP",
)

FACTORIAL_BANKS = ("J_5T", "V3_5T", "J_3T2D", "V3_3T2D")
FACTORIAL_SETTINGS = (
    "Raw",
    "John_adaptive",
    "John_fixed",
    "Simple",
    "Geometry_historical",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def exclusions(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.strip().upper()
                for item in str(value).replace(";", ",").split(",")
                if item.strip()
            }
        )
    )


def read_targets() -> pd.DataFrame:
    frame = pd.read_csv(TARGET_MANIFEST, dtype=str).fillna("")
    frame = frame.rename(columns={"excluded_native_pdb_ids": "excluded_pdb_ids"})
    # Identical CASP15 sequences are one dependence cluster.  At present this
    # groups R1189/R1190 and leaves the other ten targets singleton; deriving the
    # identifier from sequence prevents an undocumented hand-coded grouping.
    frame["sequence_cluster"] = frame["sequence"].map(
        lambda value: "seq_" + hashlib.sha256(value.encode()).hexdigest()[:16]
    )
    required = {
        "target_id",
        "sequence",
        "temporal_cutoff",
        "excluded_pdb_ids",
        "sequence_cluster",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"CASP15 manifest is missing columns: {sorted(missing)}")
    if len(frame) != 12 or frame["target_id"].nunique() != 12:
        raise RuntimeError("V5 CASP15 manifest must contain exactly 12 unique targets")
    return frame


def _safe_full_templates(record: dict) -> tuple[list[tuple], int, int]:
    """John's single-table semantics over the shared frozen full snapshot."""
    meta = db.load_meta()
    coords = db.load_coords()
    cutoff = str(record["temporal_cutoff"])
    excluded = set(exclusions(record["excluded_pdb_ids"]))
    # Missing or malformed dates fail closed: ISO YYYY-MM-DD has length 10.
    release = meta["release_date"].astype(str)
    valid_date = release.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    safe = meta[
        valid_date
        & (release < cutoff)
        & ~meta["pdb_id"].astype(str).str.upper().isin(excluded)
    ]
    templates = []
    for row in safe.itertuples(index=False):
        entry = coords.get(row.chain_key)
        if entry is None:
            continue
        value = np.asarray(entry["coords"], dtype=float)
        templates.append((str(row.chain_key), str(row.seq), value))
    return templates, int(len(safe)), int(safe["pdb_id"].astype(str).str.upper().nunique())


def _materialize(
    target_sequence: str,
    hit: dict,
    adjacent: float,
    source: str,
) -> dict | None:
    transfer = align_and_transfer(
        target_sequence,
        {"seq": hit["seq"], "coords": np.asarray(hit["coords"], dtype=float)},
        hit["chain_key"],
    )
    if transfer.coverage <= 0:
        return None
    completeness = transfer.template_resolved / max(transfer.template_len, 1)
    linear, linear_conf = fill_gaps_linear(
        transfer.coords, transfer.mask, adj_dist=adjacent
    )
    curved, curved_conf = fill_gaps(
        transfer.coords,
        transfer.mask,
        adj_dist=adjacent,
        rng=np.random.default_rng(0),
    )
    john_gap, john_gap_conf = fill_gaps_john(
        transfer.coords, transfer.mask, adj_dist=5.9
    )
    return {
        "chain_key": str(hit["chain_key"]),
        "pdb_id": str(hit["pdb_id"]).upper(),
        "release_date": str(hit.get("release_date", "")),
        "source": source,
        "identity": float(transfer.identity),
        "coverage": float(transfer.coverage),
        "completeness": float(completeness),
        "I": float(transfer.identity),
        "IC": float(transfer.identity * transfer.coverage),
        "ICS": float(transfer.identity * transfer.coverage * completeness),
        "linear": linear,
        "linear_conf": linear_conf,
        "curved": curved,
        "curved_conf": curved_conf,
        "john_gap": john_gap,
        "john_gap_conf": john_gap_conf,
        "support_mask": np.asarray(transfer.mask, dtype=bool),
    }


def _rank(
    items: Iterable[dict],
    field: str,
    *,
    n: int = 5,
    distinct_pdb: bool = True,
) -> list[dict]:
    # Historical implementation merges MMseqs first and keeps the first instance
    # of a chain. Python's stable sort therefore preserves the deployed tie order.
    unique: dict[str, dict] = {}
    for item in items:
        unique.setdefault(item["chain_key"], item)
    ranked = sorted(unique.values(), key=lambda item: -float(item[field]))
    if not distinct_pdb:
        return ranked[:n]
    selected: list[dict] = []
    used_pdb: set[str] = set()
    for item in ranked:
        if item["pdb_id"] in used_pdb:
            continue
        selected.append(item)
        used_pdb.add(item["pdb_id"])
        if len(selected) == n:
            break
    if len(selected) < n:
        selected.extend(item for item in ranked if item not in selected)
    return selected[:n]


def _pad_v3(
    selected: list[dict],
    sequence: str,
    geometry: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str], int]:
    coords: list[np.ndarray] = []
    residue_confidence: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []
    global_confidence: list[float] = []
    candidate_ids: list[str] = []
    pdb_ids: list[str] = []
    sources: list[str] = []
    for item in selected[:5]:
        coords.append(np.asarray(item[geometry], dtype=float))
        residue_confidence.append(np.asarray(item[f"{geometry}_conf"], dtype=float))
        support_masks.append(np.asarray(item["support_mask"], dtype=bool))
        global_confidence.append(float(item["ICS"]))
        candidate_ids.append(item["chain_key"])
        pdb_ids.append(item["pdb_id"])
        sources.append(item["source"])
    missing = 5 - len(coords)
    if missing:
        for index, value in enumerate(de_novo_ensemble(sequence, n=missing, base_seed=0)):
            coords.append(np.asarray(value, dtype=float))
            residue_confidence.append(np.zeros(len(sequence), dtype=float))
            support_masks.append(np.zeros(len(sequence), dtype=bool))
            global_confidence.append(0.1)
            candidate_ids.append(f"de_novo_{index}")
            pdb_ids.append("")
            sources.append("de_novo_fallback")
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(residue_confidence, dtype=np.float32),
        np.asarray(support_masks, dtype=bool),
        np.asarray(global_confidence, dtype=np.float32),
        candidate_ids,
        pdb_ids,
        sources,
        missing,
    )


def _john_arrays(candidates, sequence: str):
    coords = np.asarray([c.coords for c in candidates], dtype=np.float32)
    # John exposes one composite confidence, not a per-residue confidence.  The
    # constant vector is an adapter for later controlled refiners, not a claim
    # that John estimated residue-level uncertainty.
    residue_confidence = np.asarray(
        [np.full(len(sequence), c.confidence, dtype=np.float32) for c in candidates]
    )
    support = np.asarray(
        [
            np.ones(len(sequence), dtype=bool)
            if c.source == "template"
            else np.zeros(len(sequence), dtype=bool)
            for c in candidates
        ]
    )
    global_confidence = np.asarray([c.confidence for c in candidates], dtype=np.float32)
    ids = [c.template_id or f"de_novo_{index}" for index, c in enumerate(candidates)]
    sources = [c.source for c in candidates]
    return coords, residue_confidence, support, global_confidence, ids, sources


def _target_build(record: dict, hits_records: list[dict], replace: bool) -> dict:
    target_id = str(record["target_id"])
    output = CACHE / target_id / "raw_banks.npz"
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists() and not replace:
        return {
            "target_id": target_id,
            "path": str(output.relative_to(REPO)),
            "sha256": sha256(output),
            "metadata_sha256": sha256(metadata_path),
            "status": "cached",
        }

    sequence = str(record["sequence"])
    cutoff = str(record["temporal_cutoff"])
    excluded = set(exclusions(record["excluded_pdb_ids"]))
    adjacent = float(json.loads(P0.read_text())["adjacent_c1"]["mean"])
    full_meta = db.load_meta()
    full_coords = db.load_coords()
    meta_idx = full_meta.set_index("chain_key")

    # Same-sandbox John: unchanged exhaustive composite retrieval and candidate
    # selection, but over the exact V3 full snapshot after common safety filters.
    john_templates, eligible_chains, eligible_pdbs = _safe_full_templates(record)
    john = build_raw_candidates(
        sequence,
        target_id,
        john_templates,
        n=5,
        base_seed=SEED,
    )
    (
        john_coords,
        john_conf,
        john_support,
        john_global,
        john_ids,
        john_sources,
    ) = _john_arrays(john, sequence)

    # Historical V3 MMseqs retrieval branch over the same full snapshot.
    mm_items: list[dict] = []
    seen_mmseqs: set[str] = set()
    for hit in sorted(hits_records, key=lambda value: -float(value["bits"])):
        key = str(hit["target"])
        if key in seen_mmseqs:
            continue
        seen_mmseqs.add(key)
        if key not in meta_idx.index:
            continue
        row = meta_idx.loc[key]
        if str(row["release_date"]) >= cutoff:
            continue
        if str(row["pdb_id"]).upper() in excluded:
            continue
        entry = full_coords.get(key)
        if entry is None:
            continue
        item = _materialize(
            sequence,
            {
                "chain_key": key,
                "seq": str(row["seq"]),
                "coords": entry["coords"],
                "pdb_id": row["pdb_id"],
                "release_date": row["release_date"],
            },
            adjacent,
            "mmseqs",
        )
        if item is not None:
            mm_items.append(item)
        if len(mm_items) == 40:
            break

    # Historical V3 exhaustive branch over its deployed 7,155-sequence derived
    # view.  This is an algorithmic view of the shared snapshot, not a different
    # source universe.
    comp_hits = composite_search.search(
        sequence,
        cutoff,
        exclude_pdb_ids=tuple(excluded),
        top_n=8,
    )
    comp_items: list[dict] = []
    for hit in comp_hits:
        item = _materialize(sequence, hit, adjacent, "composite")
        if item is not None:
            comp_items.append(item)

    seen = {item["chain_key"] for item in mm_items}
    merged = mm_items + [item for item in comp_items if item["chain_key"] not in seen]
    variants: dict[str, tuple[list[dict], str]] = {
        "V3_COMPOSITE_IC_LINEAR": (_rank(comp_items, "IC"), "linear"),
        "V3_MMSEQS_IC_LINEAR": (_rank(mm_items, "IC"), "linear"),
        "V3_MERGED_IC_LINEAR": (_rank(merged, "IC"), "linear"),
        "V3_MERGED_ICS_LINEAR": (_rank(merged, "ICS"), "linear"),
        "V3_EXACT_RAW": (_rank(merged, "ICS"), "curved"),
        "V3_MERGED_I_LINEAR": (_rank(merged, "I"), "linear"),
        "V3_COMPOSITE_ICS_LINEAR": (_rank(comp_items, "ICS"), "linear"),
        "V3_MERGED_ICS_JOHN_GAP": (_rank(merged, "ICS"), "john_gap"),
    }

    arrays: dict[str, np.ndarray] = {
        "J_SS_RAW__coords": john_coords,
        "J_SS_RAW__confidence": john_conf,
        "J_SS_RAW__support_mask": john_support,
        "J_SS_RAW__global_confidence": john_global,
    }
    banks: dict[str, dict] = {
        "J_SS_RAW": {
            "candidate_ids": john_ids,
            "sources": john_sources,
            "pdb_ids": [
                value.split("_")[0].upper() if value and not value.startswith("de_novo") else ""
                for value in john_ids
            ],
            "fallback_slots": int(sum(value != "template" for value in john_sources)),
            "selection": "John composite score plus sequence-feature diversity clustering",
            "database_view": "full frozen 23,869-chain snapshot after common safety filters",
        }
    }
    for name, (items, geometry) in variants.items():
        (
            coords,
            confidence,
            support,
            global_confidence,
            ids,
            pdb_ids,
            sources,
            missing,
        ) = _pad_v3(items, sequence, geometry)
        arrays[f"{name}__coords"] = coords
        arrays[f"{name}__confidence"] = confidence
        arrays[f"{name}__support_mask"] = support
        arrays[f"{name}__global_confidence"] = global_confidence
        banks[name] = {
            "candidate_ids": ids,
            "pdb_ids": pdb_ids,
            "sources": sources,
            "fallback_slots": missing,
            "distinct_pdb_n": len({value for value in pdb_ids if value}),
            "mean_identity": float(np.mean([item["identity"] for item in items])) if items else 0.0,
            "mean_coverage": float(np.mean([item["coverage"] for item in items])) if items else 0.0,
            "mean_completeness": float(np.mean([item["completeness"] for item in items])) if items else 0.0,
            "gap_geometry": geometry,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    write_json(
        metadata_path,
        {
            "schema": "v5-casp15-raw-banks-v1",
            "target_id": target_id,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "temporal_cutoff": cutoff,
            "excluded_pdb_ids": sorted(excluded),
            "eligible_full_database_chains": eligible_chains,
            "eligible_full_database_pdbs": eligible_pdbs,
            "mmseqs_realign_pool_n": len(mm_items),
            "composite_diversity_pool_n": len(comp_items),
            "merged_unique_pool_n": len({item["chain_key"] for item in merged}),
            "banks": banks,
            "array_hashes": {key: array_sha256(value) for key, value in arrays.items()},
        },
    )
    return {
        "target_id": target_id,
        "path": str(output.relative_to(REPO)),
        "sha256": sha256(output),
        "metadata_sha256": sha256(metadata_path),
        "john_fallback_slots": banks["J_SS_RAW"]["fallback_slots"],
        "v3_fallback_slots": banks["V3_EXACT_RAW"]["fallback_slots"],
        "status": "generated",
    }


def _direct_drfold_candidates(target_id: str, sequence: str):
    cache = CandidateCache(DRFOLD_CACHE, "validation")
    candidates = [
        candidate
        for candidate in cache.load_target(target_id, sequence)
        if candidate.candidate_id.startswith("drfold2_e2e__cfg97_20ckpt_e2e__")
        and candidate.model == "cfg97_20ckpt_e2e"
    ]
    candidates.sort(key=lambda value: (-value.global_confidence, value.candidate_id))
    return candidates


def _audit_drfold() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for target in read_targets().itertuples(index=False):
        candidates = _direct_drfold_candidates(target.target_id, target.sequence)
        for rank, candidate in enumerate(candidates, start=1):
            path = CandidateCache(DRFOLD_CACHE, "validation").candidate_path(candidate)
            rows.append(
                {
                    "target_id": target.target_id,
                    "length": len(target.sequence),
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "global_confidence": candidate.global_confidence,
                    "sha256": sha256(path),
                    "finite": bool(np.isfinite(candidate.coords).all()),
                    "path": str(path.relative_to(REPO)),
                }
            )
    frame = pd.DataFrame(rows)
    counts = frame.groupby("target_id").size().reindex(read_targets()["target_id"], fill_value=0)
    summary = {
        "target_n": 12,
        "targets_with_at_least_two_direct_candidates": int((counts >= 2).sum()),
        "targets_without_direct_candidates": counts[counts == 0].index.tolist(),
        "candidate_n": int(len(frame)),
        "all_finite": bool(frame["finite"].all()) if len(frame) else False,
    }
    return rows, summary


def cmd_audit(_: argparse.Namespace) -> None:
    targets = read_targets()
    audit = json.loads(DESIGN_AUDIT.read_text())
    expected = {
        REPO / "data" / "processed" / "template_meta.parquet": "80463138acfce71663a25504fd641521a57f75670e33cef1923f7a69fd346402",
        REPO / "data" / "cache" / "template_coords.pkl": "d5ce6232d88e1d92050307f6313c8ad1c3f0792ffa9c57d01d2b71af2405980f",
        REPO / "data" / "processed" / "top1_template_meta.parquet": "8e96ffb4e759e9aca200e0df3f4382e5bb681f1ea5094283c559b8f7701b4a8f",
        REPO / "data" / "cache" / "top1_template_coords.pkl": "b419405d7ee1447a470ada812f53fffc373674be10c2715ccd619e975656210a",
    }
    mismatches = {
        str(path.relative_to(REPO)): {"expected": wanted, "observed": sha256(path)}
        for path, wanted in expected.items()
        if not path.exists() or sha256(path) != wanted
    }
    drfold_rows, drfold_summary = _audit_drfold()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(drfold_rows).to_csv(OUT / "drfold2_direct_candidate_inventory.csv", index=False)
    payload = {
        "status": "PASS" if not mismatches and drfold_summary["all_finite"] else "FAIL",
        "created_utc": now_utc(),
        "target_n": len(targets),
        "input_snapshot_mismatches": mismatches,
        "design_audit_sha256": sha256(DESIGN_AUDIT),
        "target_manifest_sha256": sha256(TARGET_MANIFEST),
        "p0_sha256": sha256(P0),
        "p1_sha256": sha256(P1),
        "usalign_sha256": sha256(USALIGN),
        "drfold2": drfold_summary,
        "design_audit_summary": audit.get("summary", {}),
    }
    write_json(OUT / "v5_execution_input_audit.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError("V5 execution input audit failed")


def cmd_build_raw(args: argparse.Namespace) -> None:
    targets = read_targets()
    CACHE.mkdir(parents=True, exist_ok=True)
    QUERY_FASTA.write_text(
        "".join(f">{row.target_id}\n{row.sequence}\n" for row in targets.itertuples(index=False))
    )
    if args.refresh_mmseqs or not MMSEQS_HITS.exists():
        hits = mmseqs_search.search(QUERY_FASTA, MMSEQS_HITS)
    else:
        hits = mmseqs_search.read_m8(MMSEQS_HITS)
    by_target = {
        target_id: group.to_dict("records")
        for target_id, group in hits.groupby("query")
    }
    records = targets.to_dict("records")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _target_build,
                record,
                by_target.get(record["target_id"], []),
                args.replace,
            ): record["target_id"]
            for record in records
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            print(
                f"[{index:02d}/{len(records)} {result['target_id']}] {result['status']}",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values("target_id")
    frame.to_csv(RAW_MANIFEST, index=False)
    receipt = {
        "status": "V5_CASP15_RAW_BANKS_FROZEN",
        "created_utc": now_utc(),
        "scientific_role": "CASP15 development/validation; not untouched confirmation",
        "target_n": len(frame),
        "target_manifest_sha256": sha256(TARGET_MANIFEST),
        "design_audit_sha256": sha256(DESIGN_AUDIT),
        "mmseqs_hits_sha256": sha256(MMSEQS_HITS),
        "raw_manifest_sha256": sha256(RAW_MANIFEST),
        "generation_code_sha256": sha256(Path(__file__)),
        "variants": ["J_SS_RAW", *V3_VARIANTS],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(RAW_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


def verify_raw() -> pd.DataFrame:
    receipt = json.loads(RAW_RECEIPT.read_text())
    if receipt["status"] != "V5_CASP15_RAW_BANKS_FROZEN":
        raise RuntimeError("unexpected raw freeze status")
    if sha256(TARGET_MANIFEST) != receipt["target_manifest_sha256"]:
        raise RuntimeError("target manifest changed after raw generation")
    if sha256(RAW_MANIFEST) != receipt["raw_manifest_sha256"]:
        raise RuntimeError("raw bank manifest changed after freeze")
    frame = pd.read_csv(RAW_MANIFEST, dtype=str).fillna("")
    if len(frame) != 12:
        raise RuntimeError("raw manifest is incomplete")
    for row in frame.itertuples(index=False):
        path = REPO / row.path
        if sha256(path) != row.sha256:
            raise RuntimeError(f"raw candidate artifact changed: {row.target_id}")
        metadata = path.with_suffix(".json")
        if sha256(metadata) != row.metadata_sha256:
            raise RuntimeError(f"raw metadata artifact changed: {row.target_id}")
    return frame


def _load_template_bank(target_id: str, variant: str) -> tuple[np.ndarray, list[dict]]:
    path = CACHE / target_id / "raw_banks.npz"
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload[f"{variant}__coords"], dtype=float)
    metadata = json.loads(path.with_suffix(".json").read_text())["banks"][variant]
    records = []
    for index, value in enumerate(coords):
        records.append(
            {
                "coords": value,
                "candidate_id": metadata["candidate_ids"][index],
                "source": "T",
                "global_confidence": float(
                    np.load(path, allow_pickle=False)[f"{variant}__global_confidence"][index]
                ),
            }
        )
    return coords, records


def _template_records(target_id: str, variant: str) -> list[dict]:
    path = CACHE / target_id / "raw_banks.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())["banks"][variant]
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload[f"{variant}__coords"], dtype=float)
        confidence = np.asarray(payload[f"{variant}__confidence"], dtype=float)
        support = np.asarray(payload[f"{variant}__support_mask"], dtype=bool)
        global_confidence = np.asarray(
            payload[f"{variant}__global_confidence"], dtype=float
        )
    return [
        {
            "coords": coords[index],
            "confidence": confidence[index],
            "support_mask": support[index],
            "global_confidence": float(global_confidence[index]),
            "candidate_id": metadata["candidate_ids"][index],
            "source": "T",
            "underlying_source": metadata["sources"][index],
        }
        for index in range(len(coords))
    ]


def _drfold_records(target_id: str, sequence: str) -> list[dict]:
    return [
        {
            "coords": np.asarray(candidate.coords, dtype=float),
            "confidence": np.asarray(candidate.confidence, dtype=float),
            "support_mask": np.asarray(candidate.support_mask, dtype=bool),
            "global_confidence": float(candidate.global_confidence),
            "candidate_id": candidate.candidate_id,
            "source": "D",
            "underlying_source": "direct_drfold2_cfg97_20ckpt",
        }
        for candidate in _direct_drfold_candidates(target_id, sequence)
    ]


def allocate_bank(
    templates: list[dict],
    drfold: list[dict],
    template_n: int,
    drfold_n: int,
) -> tuple[list[dict], dict]:
    """Allocate a fixed budget; missing D slots fall back to the next T."""
    selected = list(templates[:template_n]) + list(drfold[:drfold_n])
    realized_d = min(len(drfold), drfold_n)
    missing_d = drfold_n - realized_d
    if missing_d:
        selected.extend(templates[template_n : template_n + missing_d])
    expected = template_n + drfold_n
    if len(selected) < expected:
        # Both sources failed.  This is retained as an evaluation failure rather
        # than silently dropping the target; caller assigns TM=0.
        return selected, {
            "requested_T": template_n,
            "requested_D": drfold_n,
            "realized_T": sum(value["source"] == "T" for value in selected),
            "realized_D": sum(value["source"] == "D" for value in selected),
            "complete": False,
        }
    return selected[:expected], {
        "requested_T": template_n,
        "requested_D": drfold_n,
        "realized_T": sum(value["source"] == "T" for value in selected[:expected]),
        "realized_D": sum(value["source"] == "D" for value in selected[:expected]),
        "complete": True,
    }


def _score_bank(
    records: list[dict],
    references: list[np.ndarray],
    sequence: str,
    cache: dict[str, float] | None = None,
) -> float:
    if not records or not all(np.isfinite(value["coords"]).all() for value in records):
        return 0.0
    if cache is not None:
        keyed = [
            (array_sha256(np.asarray(record["coords"], dtype=np.float32)), record)
            for record in records
        ]
        missing = [(key, record) for key, record in keyed if key not in cache]

        def score_missing(value: tuple[str, dict]) -> tuple[str, float]:
            key, record = value
            try:
                score = float(
                    score_target([record["coords"]], references, list(sequence))
                )
            except Exception:
                score = 0.0
            return key, score

        if missing:
            with ThreadPoolExecutor(max_workers=min(5, len(missing))) as executor:
                for key, score in executor.map(score_missing, missing):
                    cache[key] = score
        values = [cache[key] for key, _ in keyed]
        return float(max(values, default=0.0))
    try:
        return float(
            score_target(
                [value["coords"] for value in records], references, list(sequence)
            )
        )
    except Exception:
        return 0.0


def _bootstrap_effect(delta: np.ndarray, clusters: np.ndarray) -> dict:
    values = cluster_bootstrap_means(
        np.asarray(delta, dtype=float),
        np.asarray(clusters),
        replicates=10_000,
        seed=SEED,
    )
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "wins": int(np.sum(delta > TIE)),
        "ties": int(np.sum(np.abs(delta) <= TIE)),
        "losses": int(np.sum(delta < -TIE)),
        "target_n": int(len(delta)),
        "cluster_n": int(len(np.unique(clusters))),
        "bootstrap_replicates": 10_000,
        "seed": SEED,
    }


def _exact_sign_flip(
    delta: np.ndarray, clusters: np.ndarray | None = None
) -> dict:
    values = np.asarray(delta, dtype=float)
    original_n = len(values)
    if clusters is not None:
        grouped = pd.DataFrame(
            {"delta": values, "cluster": np.asarray(clusters)}
        ).groupby("cluster", sort=False)["delta"].sum()
        sign_units = grouped.to_numpy(dtype=float)
    else:
        sign_units = values
    observed = float(values.mean())
    n = len(sign_units)
    # CASP15 has only 12 targets, so all 2^12 sign allocations are exact.
    signs = 1.0 - 2.0 * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    # One shared sign per dependence cluster while retaining the target-weighted
    # estimand used by the cluster bootstrap.
    null = (signs * sign_units[None, :]).sum(axis=1) / original_n
    two_sided = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    greater = float(np.mean(null >= observed - 1e-15))
    return {
        "observed_mean_delta": observed,
        "target_n": int(original_n),
        "independent_cluster_n": int(n),
        "permutations": int(1 << n),
        "two_sided_p": two_sided,
        "greater_p": greater,
    }


def cmd_evaluate_raw(args: argparse.Namespace) -> None:
    verify_raw()
    targets = read_targets()
    labels = io.load_labels("validation")
    labels = labels[labels["ID"].map(io.target_id_of).isin(set(targets["target_id"]))]
    raw_variants = ["J_SS_RAW", *V3_VARIANTS]
    allocations = {
        "5T": (5, 0),
        "4T+1D": (4, 1),
        "3T+2D": (3, 2),
        "2T": (2, 0),
        "1T+1D": (1, 1),
        "2D": (0, 2),
    }
    rows: list[dict] = []
    allocation_rows: list[dict] = []
    diversity_rows: list[dict] = []
    for index, target in enumerate(targets.itertuples(index=False), start=1):
        references = io.get_reference_coords(labels, target.target_id)
        score_cache: dict[str, float] = {}
        row = {
            "target_id": target.target_id,
            "sequence_cluster": target.sequence_cluster,
            "length": len(target.sequence),
            "native_conformations": len(references),
        }
        for variant in raw_variants:
            bank = _template_records(target.target_id, variant)
            row[variant] = _score_bank(
                bank, references, target.sequence, cache=score_cache
            )
            if args.diversity:
                diversity_rows.append(
                    {
                        "target_id": target.target_id,
                        "tbm_base": variant,
                        "allocation": "5T",
                        "mean_pairwise_self_tm": mean_pairwise_self_tm(
                            [value["coords"] for value in bank], list(target.sequence)
                        ),
                    }
                )
        drfold = _drfold_records(target.target_id, target.sequence)
        for tbm_base in ("J_SS_RAW", "V3_EXACT_RAW"):
            templates = _template_records(target.target_id, tbm_base)
            for name, (template_n, drfold_n) in allocations.items():
                bank, realized = allocate_bank(
                    templates, drfold, template_n, drfold_n
                )
                score = _score_bank(
                    bank, references, target.sequence, cache=score_cache
                )
                allocation_rows.append(
                    {
                        "target_id": target.target_id,
                        "sequence_cluster": target.sequence_cluster,
                        "length": len(target.sequence),
                        "tbm_base": tbm_base,
                        "allocation": name,
                        "best_tm": score,
                        **realized,
                        "candidate_ids": "|".join(value["candidate_id"] for value in bank),
                    }
                )
                if args.diversity and len(bank) >= 2:
                    diversity_rows.append(
                        {
                            "target_id": target.target_id,
                            "tbm_base": tbm_base,
                            "allocation": name,
                            "mean_pairwise_self_tm": mean_pairwise_self_tm(
                                [value["coords"] for value in bank], list(target.sequence)
                            ),
                        }
                    )
        rows.append(row)
        print(f"[{index:02d}/12 {target.target_id}] raw RQ1/RQ2 scored", flush=True)

    results = OUT / "raw_results"
    results.mkdir(parents=True, exist_ok=True)
    target_scores = pd.DataFrame(rows)
    target_scores.to_csv(results / "rq1_target_best5_tm.csv", index=False)
    allocations_frame = pd.DataFrame(allocation_rows)
    allocations_frame.to_csv(results / "rq2_allocation_target_tm.csv", index=False)
    if diversity_rows:
        pd.DataFrame(diversity_rows).to_csv(
            results / "candidate_bank_diversity.csv", index=False
        )

    clusters = target_scores["sequence_cluster"].to_numpy()
    rq1_rows: list[dict] = []
    effects: dict[str, dict] = {}
    for variant in raw_variants:
        delta = (
            target_scores[variant].to_numpy()
            - target_scores["J_SS_RAW"].to_numpy()
        )
        effects[variant] = _bootstrap_effect(delta, clusters)
        rq1_rows.append(
            {
                "variant": variant,
                "mean_best5_tm": float(target_scores[variant].mean()),
                **effects[variant],
            }
        )
    rq1_summary = pd.DataFrame(rq1_rows)
    rq1_summary.to_csv(results / "rq1_summary.csv", index=False)
    write_json(results / "rq1_effects.json", effects)

    # Staged exact-V3 reconstruction is reported separately from the alternate
    # mechanism/interaction cells.
    stages = [
        ("composite_to_merged_retrieval", "V3_COMPOSITE_IC_LINEAR", "V3_MERGED_IC_LINEAR"),
        ("add_template_completeness", "V3_MERGED_IC_LINEAR", "V3_MERGED_ICS_LINEAR"),
        ("linear_to_curved_gap", "V3_MERGED_ICS_LINEAR", "V3_EXACT_RAW"),
        ("identity_to_identity_coverage", "V3_MERGED_I_LINEAR", "V3_MERGED_IC_LINEAR"),
        ("john_gap_to_curved_gap", "V3_MERGED_ICS_JOHN_GAP", "V3_EXACT_RAW"),
    ]
    stage_rows: list[dict] = []
    for label, before, after in stages:
        delta = target_scores[after].to_numpy() - target_scores[before].to_numpy()
        stage_rows.append(
            {"stage": label, "before": before, "after": after, **_bootstrap_effect(delta, clusters)}
        )
    pd.DataFrame(stage_rows).to_csv(results / "rq1_staged_effects.csv", index=False)

    rq2_summary_rows: list[dict] = []
    rq2_effects: dict[str, dict] = {}
    for tbm_base in ("J_SS_RAW", "V3_EXACT_RAW"):
        group = allocations_frame[allocations_frame["tbm_base"] == tbm_base]
        wide = group.pivot(index="target_id", columns="allocation", values="best_tm").reindex(
            target_scores["target_id"]
        )
        for allocation in allocations:
            delta = wide[allocation].to_numpy() - wide["5T"].to_numpy()
            key = f"{tbm_base}:{allocation}_vs_5T"
            rq2_effects[key] = {
                **_bootstrap_effect(delta, clusters),
                "exact_sign_flip": _exact_sign_flip(delta, clusters),
            }
            realized = group[group["allocation"] == allocation]
            rq2_summary_rows.append(
                {
                    "tbm_base": tbm_base,
                    "allocation": allocation,
                    "mean_best_tm": float(wide[allocation].mean()),
                    "delta_vs_5T": float(delta.mean()),
                    "targets_complete": int(realized["complete"].sum()),
                    "mean_realized_T": float(realized["realized_T"].mean()),
                    "mean_realized_D": float(realized["realized_D"].mean()),
                    **_bootstrap_effect(delta, clusters),
                }
            )
    pd.DataFrame(rq2_summary_rows).to_csv(results / "rq2_summary.csv", index=False)
    write_json(results / "rq2_effects.json", rq2_effects)

    headline = {
        "rq1_v3_exact_raw_vs_j_ss_raw": {
            **_bootstrap_effect(
                target_scores["V3_EXACT_RAW"].to_numpy()
                - target_scores["J_SS_RAW"].to_numpy(),
                clusters,
            ),
            "exact_sign_flip": _exact_sign_flip(
                target_scores["V3_EXACT_RAW"].to_numpy()
                - target_scores["J_SS_RAW"].to_numpy(),
                clusters,
            ),
        }
    }
    for base in ("J_SS_RAW", "V3_EXACT_RAW"):
        group = allocations_frame[allocations_frame["tbm_base"] == base]
        wide = group.pivot(index="target_id", columns="allocation", values="best_tm").reindex(
            target_scores["target_id"]
        )
        delta = wide["3T+2D"].to_numpy() - wide["5T"].to_numpy()
        headline[f"rq2_{base}_3T2D_vs_5T"] = {
            **_bootstrap_effect(delta, clusters),
            "exact_sign_flip": _exact_sign_flip(delta, clusters),
        }
    write_json(results / "headline_effects.json", headline)
    write_json(
        results / "raw_evaluation_receipt.json",
        {
            "status": "V5_CASP15_RAW_RQ1_RQ2_COMPLETE",
            "created_utc": now_utc(),
            "scientific_role": "development/validation",
            "target_n": len(target_scores),
            "sequence_cluster_n": target_scores["sequence_cluster"].nunique(),
            "raw_bank_receipt_sha256": sha256(RAW_RECEIPT),
            "rq1_target_scores_sha256": sha256(results / "rq1_target_best5_tm.csv"),
            "rq2_target_scores_sha256": sha256(results / "rq2_allocation_target_tm.csv"),
            "evaluation_code_sha256": sha256(Path(__file__)),
        },
    )
    print("\nRQ1 summary")
    print(rq1_summary.to_string(index=False))
    print("\nRQ2 summary")
    print(pd.DataFrame(rq2_summary_rows).to_string(index=False))


def _factorial_raw_banks(target_id: str, sequence: str) -> tuple[dict[str, list[dict]], dict]:
    john = _template_records(target_id, "J_SS_RAW")
    v3 = _template_records(target_id, "V3_EXACT_RAW")
    drfold = _drfold_records(target_id, sequence)
    j_mixed, j_realized = allocate_bank(john, drfold, 3, 2)
    v3_mixed, v3_realized = allocate_bank(v3, drfold, 3, 2)
    banks = {
        "J_5T": john,
        "V3_5T": v3,
        "J_3T2D": j_mixed,
        "V3_3T2D": v3_mixed,
    }
    return banks, {
        "J_3T2D": j_realized,
        "V3_3T2D": v3_realized,
        "direct_drfold2_available": len(drfold),
    }


def _stable_uint32(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _john_complete(
    bank: list[dict], target_id: str, sequence: str
) -> np.ndarray:
    """Complete controlled John bank: rule refiner plus deterministic MT19937 jitter.

    The public notebook used the legacy global NumPy RNG, but its global state and
    Python hash salt are unrecoverable.  V5 preserves the MT19937 distribution
    while defining a target/candidate-local seed contract.  This is a controlled
    same-sandbox reproduction, not byte-equivalence to the public Kaggle run.
    """
    output = []
    for index, candidate in enumerate(bank):
        confidence = float(candidate["global_confidence"])
        value = refine_rule_based(candidate["coords"], sequence, confidence=confidence)
        if candidate["underlying_source"] != "de_novo_fallback":
            scale = max(0.05, 0.8 - confidence)
            rng = np.random.RandomState(
                _stable_uint32("v5-john-mt19937", SEED, target_id, index)
            )
            value = value + rng.normal(0.0, scale, value.shape)
        output.append(value)
    return np.asarray(output, dtype=np.float32)


def _refine_one(
    candidate: dict,
    sequence: str,
    setting: str,
    priors_v1: dict,
    priors_v2: dict,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    coords = np.asarray(candidate["coords"], dtype=float)
    global_confidence = float(candidate["global_confidence"])
    confidence = np.asarray(candidate["confidence"], dtype=float)
    if setting == "Raw":
        return coords.copy(), {"setting": setting}
    if setting == "John_adaptive":
        return (
            refine_rule_based(coords, sequence, confidence=global_confidence),
            {
                "setting": setting,
                "confidence_adapter": "candidate source global confidence",
            },
        )
    if setting == "John_fixed":
        return (
            refine_rule_based(coords, sequence, confidence=0.5),
            {"setting": setting, "fixed_confidence": 0.5},
        )
    if setting == "Simple":
        config = GeometryV2Config(
            steps=300,
            lr=0.04,
            w_source=3.0,
            w_backbone=1.0,
            w_clash=0.0,
            w_rg=0.0,
            w_angle=0.0,
            w_torsion=0.0,
            w_kink=0.0,
            adaptive_strength=True,
            fixed_strength=1.0,
            context_mode="global",
        )
        value, info = refine_structure_v2(
            coords,
            sequence,
            priors_v1,
            priors_v2,
            source_confidence=confidence,
            global_confidence=global_confidence,
            cfg=config,
            device=device,
            seed=seed,
        )
        return value, info
    if setting == "Geometry_historical":
        config = GeometryV2Config()
    else:
        raise ValueError(f"unknown factorial setting {setting}")
    value, info = refine_structure_v2(
        coords,
        sequence,
        priors_v1,
        priors_v2,
        source_confidence=confidence,
        global_confidence=global_confidence,
        cfg=config,
        device=device,
        seed=seed,
    )
    return value, info


def _build_refined_target(
    target_id: str,
    sequence: str,
    device: str,
    replace_output: bool,
) -> dict:
    output = CACHE / target_id / "refined_banks.npz"
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists() and not replace_output:
        metadata = json.loads(metadata_path.read_text())
        return {
            "target_id": target_id,
            "path": str(output.relative_to(REPO)),
            "sha256": sha256(output),
            "metadata_sha256": sha256(metadata_path),
            "failure_count": int(metadata.get("failure_count", 0)),
            "status": "cached",
        }
    priors_v1 = json.loads(P0.read_text())
    priors_v2 = json.loads(P1.read_text())
    banks, allocation = _factorial_raw_banks(target_id, sequence)
    arrays: dict[str, np.ndarray] = {}
    setting_info: dict[str, list[dict]] = {}
    failures: list[dict] = []
    memo: dict[tuple, tuple[np.ndarray, dict]] = {}

    for bank_name, bank in banks.items():
        if len(bank) != 5:
            failures.append(
                {
                    "bank": bank_name,
                    "setting": "ALL",
                    "candidate_index": -1,
                    "error": f"incomplete raw bank: {len(bank)}/5",
                    "fallback": "retain available candidates; bank evaluation will score zero",
                }
            )
        for setting in FACTORIAL_SETTINGS:
            values: list[np.ndarray] = []
            infos: list[dict] = []
            for index, candidate in enumerate(bank):
                # Memoization is safe because each setting is a deterministic
                # function of exactly these raw inputs and the frozen config.
                key = (
                    setting,
                    array_sha256(candidate["coords"]),
                    array_sha256(candidate["confidence"]),
                    float(candidate["global_confidence"]),
                )
                if key in memo:
                    value, info = memo[key]
                    info = {**info, "memoized": True}
                else:
                    try:
                        value, info = _refine_one(
                            candidate,
                            sequence,
                            setting,
                            priors_v1,
                            priors_v2,
                            device,
                            _stable_uint32(SEED, target_id, setting, index),
                        )
                        if value.shape != candidate["coords"].shape or not np.isfinite(value).all():
                            raise ValueError("refiner returned invalid coordinates")
                    except Exception as error:
                        value = np.asarray(candidate["coords"], dtype=float).copy()
                        info = {"setting": setting, "fallback_to_raw": True}
                        failures.append(
                            {
                                "bank": bank_name,
                                "setting": setting,
                                "candidate_index": index,
                                "candidate_id": candidate["candidate_id"],
                                "error": f"{type(error).__name__}:{error}",
                                "fallback": "raw candidate",
                            }
                        )
                    memo[key] = (np.asarray(value, dtype=np.float32), info)
                values.append(np.asarray(value, dtype=np.float32))
                infos.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "source": candidate["source"],
                        "global_confidence": candidate["global_confidence"],
                        **info,
                    }
                )
            arrays[f"{bank_name}__{setting}"] = np.asarray(values, dtype=np.float32)
            setting_info[f"{bank_name}__{setting}"] = infos

    # Extra complete-system cells.  They are not substituted into the same-
    # candidate RQ3 factorial.
    arrays["J_5T__John_complete"] = _john_complete(
        banks["J_5T"], target_id, sequence
    )
    setting_info["J_5T__John_complete"] = [
        {
            "candidate_id": candidate["candidate_id"],
            "method": "John rule refiner plus deterministic legacy-MT19937 Gaussian jitter",
        }
        for candidate in banks["J_5T"]
    ]

    # The V3 TBM-only Kaggle notebook applied this older gradient stage after its
    # exact raw bank.  It remains a complete historical contender, separate from
    # the later hybrid's Geometry-v2 stage.
    old_config = RefineConfig(
        steps=300,
        lr=0.05,
        w_tpl=1.0,
        w_bb=1.0,
        w_clash=0.5,
        w_rg=0.05,
        w_dist=0.0,
        sep_clash=2,
    )
    old_values: list[np.ndarray] = []
    old_info: list[dict] = []
    for index, candidate in enumerate(banks["V3_5T"]):
        try:
            is_fallback = candidate["underlying_source"] == "de_novo_fallback"
            value, info = refine_structure(
                candidate["coords"],
                priors_v1,
                template_coords=None if is_fallback else candidate["coords"],
                conf_residue=None if is_fallback else candidate["confidence"],
                template_confidence=float(candidate["global_confidence"]),
                cfg=old_config,
                device=device,
                seed=index,
            )
            if not np.isfinite(value).all():
                raise ValueError("historical V3 gradient returned invalid coordinates")
        except Exception as error:
            value = np.asarray(candidate["coords"], dtype=float).copy()
            info = {"fallback_to_raw": True}
            failures.append(
                {
                    "bank": "V3_5T",
                    "setting": "V3_deployed_gradient",
                    "candidate_index": index,
                    "candidate_id": candidate["candidate_id"],
                    "error": f"{type(error).__name__}:{error}",
                    "fallback": "raw candidate",
                }
            )
        old_values.append(np.asarray(value, dtype=np.float32))
        old_info.append({"candidate_id": candidate["candidate_id"], **info})
    arrays["V3_5T__V3_deployed_gradient"] = np.asarray(old_values, dtype=np.float32)
    setting_info["V3_5T__V3_deployed_gradient"] = old_info

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    write_json(
        metadata_path,
        {
            "schema": "v5-casp15-refinement-factorial-v1",
            "target_id": target_id,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "device_requested": device,
            "banks": list(banks),
            "settings": list(FACTORIAL_SETTINGS),
            "allocation": allocation,
            "configs": {
                "Simple": asdict(
                    GeometryV2Config(
                        steps=300,
                        lr=0.04,
                        w_source=3.0,
                        w_backbone=1.0,
                        w_clash=0.0,
                        w_rg=0.0,
                        w_angle=0.0,
                        w_torsion=0.0,
                        w_kink=0.0,
                        adaptive_strength=True,
                        context_mode="global",
                    )
                ),
                "Geometry_historical": asdict(GeometryV2Config()),
                "V3_deployed_gradient": asdict(old_config),
            },
            "failures": failures,
            "failure_count": len(failures),
            "settings_info": setting_info,
            "array_hashes": {key: array_sha256(value) for key, value in arrays.items()},
        },
    )
    return {
        "target_id": target_id,
        "path": str(output.relative_to(REPO)),
        "sha256": sha256(output),
        "metadata_sha256": sha256(metadata_path),
        "failure_count": len(failures),
        "status": "generated",
    }


def cmd_build_refined(args: argparse.Namespace) -> None:
    verify_raw()
    targets = read_targets()
    rows = []
    # A single process owns CUDA.  Multiple Python processes duplicating a torch
    # CUDA context is slower here and can exhaust the 8-GB device.
    for index, target in enumerate(targets.itertuples(index=False), start=1):
        result = _build_refined_target(
            target.target_id, target.sequence, args.device, args.replace
        )
        rows.append(result)
        print(
            f"[{index:02d}/12 {target.target_id}] refinement {result['status']} "
            f"failures={result.get('failure_count', 'cached')}",
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values("target_id")
    frame.to_csv(REFINED_MANIFEST, index=False)
    receipt = {
        "status": "V5_CASP15_REFINEMENT_FACTORIAL_FROZEN",
        "created_utc": now_utc(),
        "scientific_role": "CASP15 development/validation",
        "target_n": len(frame),
        "raw_bank_receipt_sha256": sha256(RAW_RECEIPT),
        "refined_manifest_sha256": sha256(REFINED_MANIFEST),
        "generation_code_sha256": sha256(Path(__file__)),
        "factorial_banks": list(FACTORIAL_BANKS),
        "factorial_settings": list(FACTORIAL_SETTINGS),
        "extra_complete_cells": [
            "J_5T__John_complete",
            "V3_5T__V3_deployed_gradient",
        ],
        "failure_count": int(frame["failure_count"].astype(int).sum()),
        "device": args.device,
    }
    write_json(REFINED_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))


def verify_refined() -> pd.DataFrame:
    verify_raw()
    receipt = json.loads(REFINED_RECEIPT.read_text())
    if receipt["status"] != "V5_CASP15_REFINEMENT_FACTORIAL_FROZEN":
        raise RuntimeError("unexpected refined freeze status")
    if sha256(REFINED_MANIFEST) != receipt["refined_manifest_sha256"]:
        raise RuntimeError("refined manifest changed after freeze")
    frame = pd.read_csv(REFINED_MANIFEST, dtype=str).fillna("")
    if len(frame) != 12:
        raise RuntimeError("refined manifest is incomplete")
    for row in frame.itertuples(index=False):
        path = REPO / row.path
        if sha256(path) != row.sha256:
            raise RuntimeError(f"refined artifact changed: {row.target_id}")
        if sha256(path.with_suffix(".json")) != row.metadata_sha256:
            raise RuntimeError(f"refined metadata changed: {row.target_id}")
    return frame


class _MetricScorer:
    def __init__(self) -> None:
        self.cache: dict[tuple, float] = {}
        self.failures: list[dict] = []

    def candidate(
        self,
        target_id: str,
        name: str,
        coords: np.ndarray,
        reference: np.ndarray,
        sequence: str,
    ) -> float:
        key = (
            target_id,
            "candidate",
            array_sha256(np.asarray(coords, np.float32)),
            array_sha256(np.asarray(reference, np.float32)),
        )
        if key not in self.cache:
            try:
                self.cache[key] = float(
                    score_target([coords], [reference], list(sequence))
                )
            except Exception as error:
                self.cache[key] = 0.0
                self.failures.append(
                    {
                        "target_id": target_id,
                        "scope": "candidate",
                        "setting": name,
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
        return self.cache[key]

    def bank(
        self,
        target_id: str,
        name: str,
        coords: np.ndarray,
        references: list[np.ndarray],
        sequence: str,
    ) -> float:
        key = (target_id, "bank", array_sha256(np.asarray(coords, np.float32)))
        if key not in self.cache:
            reference_key = tuple(
                array_sha256(np.asarray(reference, np.float32))
                for reference in references
            )

            def candidate_best(candidate: np.ndarray) -> float:
                candidate_key = (
                    target_id,
                    "candidate_best_any_reference",
                    array_sha256(np.asarray(candidate, np.float32)),
                    reference_key,
                )
                if candidate_key not in self.cache:
                    try:
                        self.cache[candidate_key] = float(
                            score_target([candidate], references, list(sequence))
                        )
                    except Exception as error:
                        self.cache[candidate_key] = 0.0
                        self.failures.append(
                            {
                                "target_id": target_id,
                                "scope": "bank_candidate",
                                "setting": name,
                                "reason": f"{type(error).__name__}:{error}",
                            }
                        )
                return self.cache[candidate_key]

            with ThreadPoolExecutor(max_workers=min(5, len(coords))) as executor:
                candidate_values = list(executor.map(candidate_best, list(coords)))
            self.cache[key] = float(max(candidate_values, default=0.0))
        return self.cache[key]


def _locked_reference(
    scorer: _MetricScorer,
    target_id: str,
    raw: np.ndarray,
    references: list[np.ndarray],
    sequence: str,
) -> tuple[int, float]:
    scores = [
        scorer.candidate(
            target_id, f"reference_lock_{index}", raw, reference, sequence
        )
        for index, reference in enumerate(references)
    ]
    index = int(np.argmax(scores))
    return index, float(scores[index])


def cmd_evaluate_refined(_: argparse.Namespace) -> None:
    verify_refined()
    targets = read_targets()
    labels = io.load_labels("validation")
    labels = labels[labels["ID"].map(io.target_id_of).isin(set(targets["target_id"]))]
    priors_v1 = json.loads(P0.read_text())
    priors_v2 = json.loads(P1.read_text())
    scorer = _MetricScorer()
    candidate_rows: list[dict] = []
    bank_rows: list[dict] = []
    complete_rows: list[dict] = []
    for number, target in enumerate(targets.itertuples(index=False), start=1):
        references = io.get_reference_coords(labels, target.target_id)
        path = CACHE / target.target_id / "refined_banks.npz"
        with np.load(path, allow_pickle=False) as payload:
            for bank_name in FACTORIAL_BANKS:
                raw = np.asarray(payload[f"{bank_name}__Raw"], dtype=np.float32)
                locks = [
                    _locked_reference(
                        scorer,
                        target.target_id,
                        value,
                        references,
                        target.sequence,
                    )
                    for value in raw
                ]
                for setting in FACTORIAL_SETTINGS:
                    coords_bank = np.asarray(
                        payload[f"{bank_name}__{setting}"], dtype=np.float32
                    )
                    bank_rows.append(
                        {
                            "target_id": target.target_id,
                            "sequence_cluster": target.sequence_cluster,
                            "length": len(target.sequence),
                            "bank": bank_name,
                            "setting": setting,
                            "best5_tm": scorer.bank(
                                target.target_id,
                                f"{bank_name}_{setting}",
                                coords_bank,
                                references,
                                target.sequence,
                            ),
                        }
                    )
                    for candidate_index, coords in enumerate(coords_bank):
                        ref_index, raw_tm = locks[candidate_index]
                        reference = references[ref_index]
                        candidate_rows.append(
                            {
                                "target_id": target.target_id,
                                "sequence_cluster": target.sequence_cluster,
                                "length": len(target.sequence),
                                "bank": bank_name,
                                "candidate_index": candidate_index,
                                "setting": setting,
                                "raw_reference_index": ref_index,
                                "raw_candidate_tm": raw_tm,
                                "candidate_tm_same_reference": scorer.candidate(
                                    target.target_id,
                                    f"{bank_name}_{setting}_{candidate_index}",
                                    coords,
                                    reference,
                                    target.sequence,
                                ),
                                **local_accuracy_metrics(
                                    coords, reference, windows=(9, 15)
                                ),
                                **geometry_v2_metrics(
                                    coords,
                                    target.sequence,
                                    priors_v1,
                                    priors_v2,
                                ),
                            }
                        )

            extras = {
                "J_same_sandbox_complete": np.asarray(
                    payload["J_5T__John_complete"], dtype=np.float32
                ),
                "V3_TBM_raw": np.asarray(payload["V3_5T__Raw"], dtype=np.float32),
                "V3_TBM_deployed_gradient": np.asarray(
                    payload["V3_5T__V3_deployed_gradient"], dtype=np.float32
                ),
                "V3_hybrid_raw": np.asarray(
                    payload["V3_3T2D__Raw"], dtype=np.float32
                ),
                "V3_hybrid_historical_geometry": np.asarray(
                    payload["V3_3T2D__Geometry_historical"], dtype=np.float32
                ),
                "V5_V3bank_hybrid_simple": np.asarray(
                    payload["V3_3T2D__Simple"], dtype=np.float32
                ),
                "V5_V3bank_hybrid_geometry": np.asarray(
                    payload["V3_3T2D__Geometry_historical"], dtype=np.float32
                ),
                "V5_Johnbank_hybrid_simple": np.asarray(
                    payload["J_3T2D__Simple"], dtype=np.float32
                ),
                "V5_Johnbank_hybrid_geometry": np.asarray(
                    payload["J_3T2D__Geometry_historical"], dtype=np.float32
                ),
            }
            for name, coords in extras.items():
                complete_rows.append(
                    {
                        "target_id": target.target_id,
                        "sequence_cluster": target.sequence_cluster,
                        "length": len(target.sequence),
                        "pipeline": name,
                        "best5_tm": scorer.bank(
                            target.target_id,
                            name,
                            coords,
                            references,
                            target.sequence,
                        ),
                    }
                )
        print(f"[{number:02d}/12 {target.target_id}] RQ3/full pipelines scored", flush=True)

    results = OUT / "refinement_results"
    results.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    banks = pd.DataFrame(bank_rows)
    complete = pd.DataFrame(complete_rows)
    candidates.to_csv(results / "factorial_candidate_metrics.csv", index=False)
    banks.to_csv(results / "factorial_bank_metrics.csv", index=False)
    complete.to_csv(results / "complete_pipeline_target_tm.csv", index=False)
    pd.DataFrame(
        scorer.failures,
        columns=["target_id", "scope", "setting", "reason"],
    ).to_csv(results / "evaluation_failures.csv", index=False)

    target_metrics = candidates.groupby(
        ["target_id", "sequence_cluster", "length", "bank", "setting"],
        as_index=False,
    ).mean(numeric_only=True)
    target_metrics.to_csv(results / "factorial_target_metrics.csv", index=False)
    summary = target_metrics.groupby(["bank", "setting"], as_index=False).mean(
        numeric_only=True
    )
    bank_summary = banks.groupby(["bank", "setting"], as_index=False)[
        "best5_tm"
    ].mean()
    summary = summary.merge(bank_summary, on=["bank", "setting"])
    summary.to_csv(results / "factorial_summary.csv", index=False)

    # RQ3 headline and supporting metric conflicts. Positive deltas favour the
    # method named first; no metric is hidden when directions disagree.
    effects: dict[str, dict] = {}
    effect_rows: list[dict] = []
    metric_directions = {
        "sw_rmsd_9": False,
        "sw_rmsd_15": False,
        "c1_rmsd": False,
        "c1_lddt": True,
        "candidate_tm_same_reference": True,
    }
    comparisons = [
        ("Geometry_historical", "Simple"),
        ("Simple", "Raw"),
        ("John_adaptive", "Raw"),
        ("John_fixed", "Raw"),
    ]
    for bank_name in FACTORIAL_BANKS:
        frame = target_metrics[target_metrics["bank"] == bank_name]
        for first, second in comparisons:
            for metric, higher_better in metric_directions.items():
                wide = frame.pivot(
                    index=["target_id", "sequence_cluster"],
                    columns="setting",
                    values=metric,
                ).reset_index()
                delta = (
                    wide[first].to_numpy() - wide[second].to_numpy()
                    if higher_better
                    else wide[second].to_numpy() - wide[first].to_numpy()
                )
                key = f"{bank_name}:{first}_vs_{second}:{metric}"
                value = {
                    **_bootstrap_effect(delta, wide["sequence_cluster"].to_numpy()),
                    "exact_sign_flip": _exact_sign_flip(
                        delta, wide["sequence_cluster"].to_numpy()
                    ),
                    "positive_favours": first,
                }
                effects[key] = value
                effect_rows.append(
                    {
                        "bank": bank_name,
                        "first": first,
                        "second": second,
                        "metric": metric,
                        **{k: v for k, v in value.items() if k != "exact_sign_flip"},
                        "sign_flip_two_sided_p": value["exact_sign_flip"]["two_sided_p"],
                    }
                )
        bank_wide = banks[banks["bank"] == bank_name].pivot(
            index=["target_id", "sequence_cluster"],
            columns="setting",
            values="best5_tm",
        ).reset_index()
        for first, second in comparisons:
            delta = bank_wide[first].to_numpy() - bank_wide[second].to_numpy()
            key = f"{bank_name}:{first}_vs_{second}:best5_tm"
            value = {
                **_bootstrap_effect(delta, bank_wide["sequence_cluster"].to_numpy()),
                "exact_sign_flip": _exact_sign_flip(
                    delta, bank_wide["sequence_cluster"].to_numpy()
                ),
                "positive_favours": first,
            }
            effects[key] = value
            effect_rows.append(
                {
                    "bank": bank_name,
                    "first": first,
                    "second": second,
                    "metric": "best5_tm",
                    **{k: v for k, v in value.items() if k != "exact_sign_flip"},
                    "sign_flip_two_sided_p": value["exact_sign_flip"]["two_sided_p"],
                }
            )
    pd.DataFrame(effect_rows).to_csv(results / "paired_effects.csv", index=False)
    write_json(results / "paired_effects.json", effects)

    complete_summary_rows: list[dict] = []
    complete_effects: dict[str, dict] = {}
    complete_wide = complete.pivot(
        index=["target_id", "sequence_cluster"],
        columns="pipeline",
        values="best5_tm",
    ).reset_index()
    baseline = "J_same_sandbox_complete"
    for pipeline in sorted(complete["pipeline"].unique()):
        delta = complete_wide[pipeline].to_numpy() - complete_wide[baseline].to_numpy()
        value = {
            **_bootstrap_effect(delta, complete_wide["sequence_cluster"].to_numpy()),
            "exact_sign_flip": _exact_sign_flip(
                delta, complete_wide["sequence_cluster"].to_numpy()
            ),
        }
        complete_effects[pipeline] = value
        complete_summary_rows.append(
            {
                "pipeline": pipeline,
                "mean_best5_tm": float(complete_wide[pipeline].mean()),
                "baseline": baseline,
                **{k: v for k, v in value.items() if k != "exact_sign_flip"},
                "sign_flip_two_sided_p": value["exact_sign_flip"]["two_sided_p"],
            }
        )
    complete_summary = pd.DataFrame(complete_summary_rows).sort_values(
        "mean_best5_tm", ascending=False
    )
    complete_summary.to_csv(results / "complete_pipeline_summary.csv", index=False)
    write_json(results / "complete_pipeline_effects.json", complete_effects)

    # Explicit metric-conflict table for the thesis-specific Geometry question.
    conflicts: list[dict] = []
    for bank_name in FACTORIAL_BANKS:
        frame = target_metrics[target_metrics["bank"] == bank_name]
        for geometry in ("Geometry_historical",):
            sw = frame.pivot(
                index="target_id", columns="setting", values="sw_rmsd_9"
            )
            ld = frame.pivot(index="target_id", columns="setting", values="c1_lddt")
            for target_id in sw.index:
                conflicts.append(
                    {
                        "target_id": target_id,
                        "bank": bank_name,
                        "geometry": geometry,
                        "sw9_improvement_simple_minus_geometry": float(
                            sw.loc[target_id, "Simple"] - sw.loc[target_id, geometry]
                        ),
                        "lddt_improvement_geometry_minus_simple": float(
                            ld.loc[target_id, geometry] - ld.loc[target_id, "Simple"]
                        ),
                    }
                )
    pd.DataFrame(conflicts).to_csv(results / "geometry_metric_conflicts.csv", index=False)
    write_json(
        results / "refinement_evaluation_receipt.json",
        {
            "status": "V5_CASP15_RQ3_AND_COMPLETE_PIPELINES_COMPLETE",
            "created_utc": now_utc(),
            "scientific_role": "development/validation",
            "target_n": 12,
            "sequence_cluster_n": targets["sequence_cluster"].nunique(),
            "refined_bank_receipt_sha256": sha256(REFINED_RECEIPT),
            "candidate_metrics_sha256": sha256(
                results / "factorial_candidate_metrics.csv"
            ),
            "complete_pipeline_summary_sha256": sha256(
                results / "complete_pipeline_summary.csv"
            ),
            "evaluation_failure_count": len(scorer.failures),
            "evaluation_code_sha256": sha256(Path(__file__)),
        },
    )
    print("\nRQ3 factorial summary")
    print(summary.to_string(index=False))
    print("\nComplete pipeline summary")
    print(complete_summary.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.set_defaults(func=cmd_audit)
    build = commands.add_parser("build-raw")
    build.add_argument("--workers", type=int, default=3)
    build.add_argument("--refresh-mmseqs", action="store_true")
    build.add_argument("--replace", action="store_true")
    build.set_defaults(func=cmd_build_raw)
    evaluate = commands.add_parser("evaluate-raw")
    evaluate.add_argument("--diversity", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate_raw)
    refine = commands.add_parser("build-refined")
    refine.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda"),
        help="torch device; a single process owns the selected device",
    )
    refine.add_argument("--replace", action="store_true")
    refine.set_defaults(func=cmd_build_refined)
    evaluate_refined = commands.add_parser("evaluate-refined")
    evaluate_refined.set_defaults(func=cmd_evaluate_refined)
    return result


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    arguments = parser().parse_args()
    arguments.func(arguments)
