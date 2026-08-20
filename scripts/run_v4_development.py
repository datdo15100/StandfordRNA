#!/usr/bin/env python
"""Run the frozen V4 development ladder on already exposed RNA targets.

Commands deliberately separate native-blind raw generation from scoring:

``build-raw`` -> hash/freeze raw banks -> ``score-tbm`` -> ``score-bank`` ->
``score-refinement`` -> ``summarize``.

Nothing in this script reads or writes the V4 final-test manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.baselines.top1 import (
    build_raw_candidates,
    composite_similarity_components,
    rna_features,
)
from rna3d.data import io
from rna3d.eval.local_metrics import local_accuracy_metrics, sliding_window_c1_rmsd
from rna3d.eval.usalign import score_target
from rna3d.eval.v4_statistics import holm_step_down, primary_inference
from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.geometry.denovo import de_novo_structure
from rna3d.refine.rule_based import refine_rule_based
from rna3d.template import db, mmseqs_search
from rna3d.template.align import align_and_transfer
from rna3d.template.confidence import template_confidence
from rna3d.template.gap_fill import fill_gaps, fill_gaps_john, fill_gaps_linear


DEV = REPO / "reports" / "thesis_v4" / "development"
RESULTS = DEV / "results"
RAW_ROOT = REPO / "data" / "cache" / "v4_development_raw"
REFINE_ROOT = REPO / "data" / "cache" / "v4_development_refined"
COMPONENT_ROOT = REPO / "data" / "cache" / "v4_development_components"
MANIFEST = DEV / "development_manifest.csv"
PROTOCOL = DEV / "development_protocol_freeze.json"
DB_FREEZE = REPO / "reports" / "thesis_v4" / "phase1_controlled_db" / "db_controlled_freeze.json"
P0_AUDIT = REPO / "reports" / "thesis_v4" / "phase1_p0" / "p0_reproduction_audit.json"
P0_DISTANCE = REPO / "data" / "processed" / "geometry_priors.json"
P0_GEOMETRY = REPO / "data" / "processed" / "geofuse_geometry_v2_priors.json"
DEEP_CACHE = REPO / "data" / "cache" / "geofuse_candidates"
RAW_VERSION = "v4-dev-raw-1"
GAP_BINS = [0, 2, 5, 8, 20, np.inf]
GAP_LABELS = ["1-2", "3-5", "6-8", "9-20", ">20"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    value = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def read_manifest() -> pd.DataFrame:
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(MANIFEST) != protocol["cohort"]["manifest_sha256"]:
        raise RuntimeError("development manifest changed after freeze")
    return pd.read_csv(MANIFEST, dtype=str).fillna("")


def excluded_pdbs(value: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.strip().upper()
                for item in str(value).replace(";", ",").split(",")
                if item.strip() and item.strip().lower() != "nan"
            }
        )
    )


def length_allowed(query_len: int, template_len: int) -> bool:
    ratio = abs(template_len - query_len) / max(template_len, query_len)
    if query_len < 50 or template_len < 50:
        return ratio <= 0.6
    if query_len > 1000 or template_len > 1000:
        return ratio <= 0.2
    return ratio <= 0.4


def component_frame(target, meta: pd.DataFrame) -> pd.DataFrame:
    freeze = json.loads(DB_FREEZE.read_text())
    digest = hashlib.sha256(
        f"{RAW_VERSION}|{target.target_id}|{target.sequence}|{freeze['database']['meta']['sha256']}".encode()
    ).hexdigest()[:16]
    path = COMPONENT_ROOT / f"{target.target_id}__{digest}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    qfeat = rna_features(target.sequence)
    excluded = set(excluded_pdbs(target.excluded_pdb_ids))
    rows = []
    for row in meta.itertuples(index=False):
        if str(row.pdb_id).upper() in excluded:
            continue
        if not length_allowed(len(target.sequence), int(row.length)):
            continue
        values = composite_similarity_components(target.sequence, row.seq, qfeat)
        rows.append(
            {
                "chain_key": row.chain_key,
                "pdb_id": str(row.pdb_id).upper(),
                "release_date": str(row.release_date),
                **values,
            }
        )
    frame = pd.DataFrame(rows)
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def composite_rank(frame: pd.DataFrame, weights: tuple[float, ...], n: int) -> list[str]:
    if frame.empty:
        return []
    score = sum(weight * frame[name] for weight, name in zip(weights, ["global", "local", "features", "kmer3"]))
    return (
        frame.assign(_score=score)
        .sort_values(["_score", "chain_key"], ascending=[False, True])["chain_key"]
        .drop_duplicates()
        .head(n)
        .tolist()
    )


def item_from_key(
    key: str,
    target_seq: str,
    meta_index: pd.DataFrame,
    coords: dict,
    adj_dist: float,
    source: str,
) -> dict:
    row = meta_index.loc[key]
    transfer = align_and_transfer(target_seq, coords[key], key)
    completeness = transfer.template_resolved / max(transfer.template_len, 1)
    curved, curved_conf = fill_gaps(transfer.coords, transfer.mask, adj_dist=adj_dist, rng=np.random.default_rng(0))
    linear, linear_conf = fill_gaps_linear(transfer.coords, transfer.mask, adj_dist=adj_dist)
    john, john_conf = fill_gaps_john(transfer.coords, transfer.mask, adj_dist=5.9)
    return {
        "chain_key": key,
        "pdb_id": str(row["pdb_id"]).upper(),
        "source": source,
        "identity": float(transfer.identity),
        "coverage": float(transfer.coverage),
        "completeness": float(completeness),
        "confidence": template_confidence(transfer.identity, transfer.coverage, completeness),
        "transfer": transfer.coords,
        "mask": transfer.mask,
        "curved": curved,
        "linear": linear,
        "john_gap": john,
        "curved_conf": curved_conf,
        "linear_conf": linear_conf,
        "john_conf": john_conf,
    }


def select_items(items: Iterable[dict], rank: str, n: int = 5, distinct: bool = False) -> list[dict]:
    unique: dict[str, dict] = {}
    for item in items:
        previous = unique.get(item["chain_key"])
        if previous is None or item["confidence"] > previous["confidence"]:
            unique[item["chain_key"]] = item
    ranked = sorted(unique.values(), key=lambda item: (-float(item[rank]), item["chain_key"]))
    chosen, used = [], set()
    for item in ranked:
        if distinct and item["pdb_id"] in used:
            continue
        chosen.append(item)
        used.add(item["pdb_id"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        chosen.extend(item for item in ranked if item not in chosen)
    return chosen[:n]


def pad_coords(
    coords: list[np.ndarray], sequence: str, method: str, target_id: str, n: int = 5
) -> tuple[np.ndarray, int]:
    available = len(coords)
    result = [np.asarray(value, dtype=float) for value in coords[:n]]
    while len(result) < n:
        index = len(result)
        result.append(de_novo_structure(sequence, stable_seed(RAW_VERSION, method, target_id, index)))
    return np.stack(result), max(0, n - available)


def save_target_raw(target, banks: dict[str, list[dict]], john_candidates, deep) -> tuple[Path, dict]:
    target_dir = RAW_ROOT / target.target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict = {"target_id": target.target_id, "sequence": target.sequence, "banks": {}}
    for name, items in banks.items():
        coords, fallbacks = pad_coords(
            [item["curved"] for item in items], target.sequence, name, target.target_id
        )
        arrays[f"bank__{name}"] = coords.astype(np.float32)
        metadata["banks"][name] = {
            "available_templates": len(items),
            "fallback_slots": fallbacks,
            "candidate_ids": [item["chain_key"] for item in items] + [
                f"de_novo_{index}" for index in range(fallbacks)
            ],
            "pdb_ids": [item["pdb_id"] for item in items],
        }

    j_coords = np.stack([candidate.coords for candidate in john_candidates])
    arrays["j_coords"] = j_coords.astype(np.float32)
    arrays["j_global_conf"] = np.asarray([candidate.confidence for candidate in john_candidates], dtype=np.float32)
    arrays["j_conf"] = np.stack(
        [np.full(len(target.sequence), candidate.confidence, dtype=np.float32) for candidate in john_candidates]
    )
    metadata["j_candidate_ids"] = [candidate.template_id or f"john_de_novo_{index}" for index, candidate in enumerate(john_candidates)]
    metadata["j_sources"] = [candidate.source for candidate in john_candidates]

    thesis = banks["thesis_final"]
    thesis_coords, thesis_fallbacks = pad_coords(
        [item["curved"] for item in thesis], target.sequence, "thesis_final", target.target_id
    )
    thesis_conf = [item["curved_conf"] for item in thesis]
    thesis_global = [item["confidence"] for item in thesis]
    while len(thesis_conf) < 5:
        thesis_conf.append(np.full(len(target.sequence), 0.1))
        thesis_global.append(0.1)
    arrays["thesis_coords"] = thesis_coords.astype(np.float32)
    arrays["thesis_conf"] = np.stack(thesis_conf[:5]).astype(np.float32)
    arrays["thesis_global_conf"] = np.asarray(thesis_global[:5], dtype=np.float32)

    deep = sorted(deep, key=lambda candidate: (-candidate.global_confidence, candidate.candidate_id))[:2]
    arrays["deep_coords"] = np.stack([candidate.coords for candidate in deep]).astype(np.float32)
    arrays["deep_conf"] = np.stack([candidate.confidence for candidate in deep]).astype(np.float32)
    arrays["deep_global_conf"] = np.asarray([candidate.global_confidence for candidate in deep], dtype=np.float32)
    metadata["deep_candidate_ids"] = [candidate.candidate_id for candidate in deep]

    # Freeze same-correspondence gap inputs for the final thesis templates only.
    if thesis:
        arrays["gap_transfer"] = np.stack([item["transfer"] for item in thesis]).astype(np.float32)
        arrays["gap_mask"] = np.stack([item["mask"] for item in thesis])
        arrays["gap_john"] = np.stack([item["john_gap"] for item in thesis]).astype(np.float32)
        arrays["gap_linear"] = np.stack([item["linear"] for item in thesis]).astype(np.float32)
        arrays["gap_curved"] = np.stack([item["curved"] for item in thesis]).astype(np.float32)
        metadata["gap_candidate_ids"] = [item["chain_key"] for item in thesis]

    path = target_dir / "raw_banks.npz"
    np.savez_compressed(path, **arrays)
    metadata_path = target_dir / "raw_banks.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path, metadata


def cmd_build_raw(args: argparse.Namespace) -> None:
    if (DEV / "raw_candidate_freeze.json").exists() and not args.replace:
        raise FileExistsError("raw candidate freeze exists; use --replace only before any V4 development scoring")
    if RESULTS.exists() and any(RESULTS.rglob("*")):
        raise RuntimeError("development scores exist; raw candidates are immutable")
    manifest = read_manifest()
    meta = db.load_meta()
    coordinates = db.load_coords()
    meta_index = meta.set_index("chain_key")
    adj_dist = float(json.loads(P0_DISTANCE.read_text())["adjacent_c1"]["mean"])

    query = RAW_ROOT / "development_queries.fasta"
    hits_path = RAW_ROOT / "development_hits.m8"
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    query.write_text("".join(f">{row.target_id}\n{row.sequence}\n" for row in manifest.itertuples(index=False)))
    hits = mmseqs_search.search(query, hits_path) if args.refresh_mmseqs or not hits_path.exists() else mmseqs_search.read_m8(hits_path)
    deep_store = CandidateCache(DEEP_CACHE, "train_v2")
    artifact_rows = []

    for number, target in enumerate(manifest.itertuples(index=False), start=1):
        started = time.time()
        excluded = set(excluded_pdbs(target.excluded_pdb_ids))
        safe_meta = meta[
            (meta["release_date"].astype(str) < str(target.date))
            & ~meta["pdb_id"].str.upper().isin(excluded)
        ]
        templates = [
            (row.chain_key, row.seq, coordinates[row.chain_key]["coords"])
            for row in safe_meta.itertuples(index=False)
        ]
        john = build_raw_candidates(target.sequence, target.target_id, templates, n=5, base_seed=42)

        components_all = component_frame(target, meta)
        components_safe = components_all[components_all["release_date"] < str(target.date)].copy()
        full_keys = composite_rank(components_safe, (0.4, 0.3, 0.2, 0.1), 50)
        global_keys = composite_rank(components_safe, (1.0, 0.0, 0.0, 0.0), 50)
        unsafe_keys = composite_rank(components_all, (0.4, 0.3, 0.2, 0.1), 50)

        materialized: dict[str, dict] = {}
        for key in set(full_keys + global_keys + unsafe_keys):
            materialized[key] = item_from_key(key, target.sequence, meta_index, coordinates, adj_dist, "composite")

        mm_keys = []
        target_hits = hits[hits["query"] == target.target_id].sort_values("bits", ascending=False)
        for key in target_hits["target"].drop_duplicates():
            if key not in meta_index.index:
                continue
            row = meta_index.loc[key]
            if str(row["release_date"]) >= str(target.date) or str(row["pdb_id"]).upper() in excluded:
                continue
            if key not in materialized:
                materialized[key] = item_from_key(key, target.sequence, meta_index, coordinates, adj_dist, "mmseqs")
            mm_keys.append(key)
            if len(mm_keys) == 40:
                break

        mm_items = [materialized[key] for key in mm_keys]
        full_items = [materialized[key] for key in full_keys]
        global_items = [materialized[key] for key in global_keys]
        unsafe_items = [materialized[key] for key in unsafe_keys]
        union = mm_items + full_items
        for item in union:
            item["identity_only"] = item["identity"]
            item["identity_coverage"] = item["identity"] * item["coverage"]
            item["identity_coverage_completeness"] = item["confidence"]

        banks = {
            "mmseqs_only": select_items(mm_items, "confidence", distinct=True),
            "composite_only": select_items(full_items, "confidence", distinct=True),
            "mmseqs_plus_composite": select_items(union, "confidence", distinct=True),
            "global_only": select_items(global_items, "confidence", distinct=True),
            "full_composite": select_items(full_items, "confidence", distinct=True),
            "unsafe_dates": select_items(unsafe_items, "confidence", distinct=True),
            "rank_identity": select_items(union, "identity_only", distinct=False),
            "rank_identity_coverage": select_items(union, "identity_coverage", distinct=False),
            "rank_with_completeness": select_items(union, "identity_coverage_completeness", distinct=False),
            "rank_distinct_pdb": select_items(union, "identity_coverage_completeness", distinct=True),
        }
        banks["thesis_final"] = banks["rank_distinct_pdb"]
        deep = [candidate for candidate in deep_store.load_target(target.target_id, target.sequence) if candidate.kind == "pretrained"]
        raw_path, metadata = save_target_raw(target, banks, john, deep)
        artifact_rows.append(
            {
                "target_id": target.target_id,
                "raw_path": str(raw_path.relative_to(REPO)),
                "raw_sha256": sha256(raw_path),
                "metadata_sha256": sha256(raw_path.with_suffix(".json")),
                "j_template_candidates": sum(candidate.source == "template" for candidate in john),
                "thesis_template_candidates": metadata["banks"]["thesis_final"]["available_templates"],
                "drfold2_candidates": len(deep),
                "mmseqs_candidates": len(mm_items),
                "safe_composite_pool": len(full_items),
            }
        )
        print(
            f"[{number:02d}/{len(manifest)} {target.target_id}] J={artifact_rows[-1]['j_template_candidates']} "
            f"T={artifact_rows[-1]['thesis_template_candidates']} D={len(deep)} sec={time.time()-started:.1f}",
            flush=True,
        )

    artifact = pd.DataFrame(artifact_rows)
    artifact_path = DEV / "raw_candidate_manifest.csv"
    artifact.to_csv(artifact_path, index=False)
    freeze = {
        "status": "FROZEN_BEFORE_V4_DEVELOPMENT_NATIVE_SCORING",
        "performance_accessed": False,
        "raw_version": RAW_VERSION,
        "development_manifest_sha256": sha256(MANIFEST),
        "raw_candidate_manifest": str(artifact_path.relative_to(REPO)),
        "raw_candidate_manifest_sha256": sha256(artifact_path),
        "target_n": len(artifact),
        "all_have_five_j_outputs": bool((artifact["j_template_candidates"] <= 5).all()),
        "all_have_five_thesis_outputs_after_fallback": True,
        "all_have_two_drfold2": bool((artifact["drfold2_candidates"] >= 2).all()),
        "db_meta_sha256": json.loads(DB_FREEZE.read_text())["database"]["meta"]["sha256"],
        "db_coords_sha256": json.loads(DB_FREEZE.read_text())["database"]["coordinates"]["sha256"],
        "mmseqs_hits_sha256": sha256(hits_path),
        "p0_distance_sha256": sha256(P0_DISTANCE),
        "generation_code_sha256": sha256(Path(__file__)),
    }
    (DEV / "raw_candidate_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


def labels_for(manifest: pd.DataFrame) -> pd.DataFrame:
    labels = io.load_labels("train_v2")
    target_ids = labels["ID"].map(io.target_id_of)
    return labels[target_ids.isin(set(manifest["target_id"]))].copy()


def references_for(labels: pd.DataFrame, target_id: str) -> list[np.ndarray]:
    return io.get_reference_coords(labels, target_id)


def score_bank(coords: np.ndarray, refs: list[np.ndarray], sequence: str) -> float:
    try:
        return float(score_target(list(coords), refs, list(sequence)))
    except Exception:
        return 0.0


def inference_document(name: str, frame: pd.DataFrame, delta: str) -> dict:
    result = primary_inference(name, frame[delta], frame["cluster_id"]).to_dict()
    result["evidence_role"] = "development component-selection evidence; not confirmatory final evidence"
    return result


def cmd_score_tbm(_: argparse.Namespace) -> None:
    freeze = json.loads((DEV / "raw_candidate_freeze.json").read_text())
    manifest = read_manifest()
    labels = labels_for(manifest)
    rows, gap_rows = [], []
    for target in manifest.itertuples(index=False):
        raw_path = RAW_ROOT / target.target_id / "raw_banks.npz"
        expected = pd.read_csv(DEV / "raw_candidate_manifest.csv").set_index("target_id").loc[target.target_id, "raw_sha256"]
        if sha256(raw_path) != expected:
            raise RuntimeError(f"raw candidate hash changed for {target.target_id}")
        refs = references_for(labels, target.target_id)
        with np.load(raw_path, allow_pickle=False) as payload:
            bank_names = sorted(key.removeprefix("bank__") for key in payload.files if key.startswith("bank__"))
            scores = {name: score_bank(payload[f"bank__{name}"], refs, target.sequence) for name in bank_names}
            j_score = score_bank(payload["j_coords"], refs, target.sequence)
            row = {
                "target_id": target.target_id,
                "cluster_id": target.mmseqs_sequence_similarity_cluster,
                "j_controlled_tbm": j_score,
                **scores,
            }
            row["h1_delta"] = row["thesis_final"] - row["j_controlled_tbm"]
            rows.append(row)

            if "gap_transfer" in payload.files:
                for candidate_index, transfer in enumerate(payload["gap_transfer"]):
                    mask = payload["gap_mask"][candidate_index].astype(bool)
                    # Reference locking uses only directly transferred coordinates.
                    ref_scores = [score_bank(np.asarray([transfer]), [reference], target.sequence) for reference in refs]
                    reference = refs[int(np.argmax(ref_scores))]
                    variants = {
                        "john_gap": payload["gap_john"][candidate_index],
                        "linear": payload["gap_linear"][candidate_index],
                        "curved": payload["gap_curved"][candidate_index],
                    }
                    local = {
                        name: sliding_window_c1_rmsd(coords, reference, window=9)["per_residue"]
                        for name, coords in variants.items()
                    }
                    missing = ~mask
                    start = None
                    for index, value in enumerate(np.r_[missing, False]):
                        if value and start is None:
                            start = index
                        elif not value and start is not None:
                            stop = index
                            length = stop - start
                            gap_bin = pd.cut([length], bins=GAP_BINS, labels=GAP_LABELS, include_lowest=True)[0]
                            finite = np.logical_and.reduce([np.isfinite(values[start:stop]) for values in local.values()])
                            if finite.any():
                                gap_rows.append(
                                    {
                                        "target_id": target.target_id,
                                        "cluster_id": target.mmseqs_sequence_similarity_cluster,
                                        "candidate_index": candidate_index,
                                        "gap_length": length,
                                        "gap_bin": str(gap_bin),
                                        **{name: float(values[start:stop][finite].mean()) for name, values in local.items()},
                                    }
                                )
                            start = None
    output = RESULTS / "rq1_tbm"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "target_scores.csv", index=False)
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(output / "gap_instances.csv", index=False)
    summary_rows = []
    for column in [value for value in frame.columns if value not in {"target_id", "cluster_id", "h1_delta"}]:
        summary_rows.append({"setting": column, "mean_best5_tm": frame[column].mean(), "availability": 1.0})
    pd.DataFrame(summary_rows).to_csv(output / "setting_summary.csv", index=False)
    effects = {
        "H1": inference_document("H1-development", frame, "h1_delta"),
        "add_mmseqs_to_composite": inference_document(
            "T-SEARCH-development",
            frame.assign(delta=frame["mmseqs_plus_composite"] - frame["composite_only"]),
            "delta",
        ),
        "full_composite_vs_global": inference_document(
            "T-COMP-development",
            frame.assign(delta=frame["full_composite"] - frame["global_only"]),
            "delta",
        ),
        "coverage_beyond_identity": inference_document(
            "T-RANK-COVERAGE-development",
            frame.assign(delta=frame["rank_identity_coverage"] - frame["rank_identity"]),
            "delta",
        ),
        "completeness_beyond_coverage": inference_document(
            "T-RANK-COMPLETENESS-development",
            frame.assign(delta=frame["rank_with_completeness"] - frame["rank_identity_coverage"]),
            "delta",
        ),
        "distinct_pdb": inference_document(
            "T-RANK-DISTINCT-development",
            frame.assign(delta=frame["rank_distinct_pdb"] - frame["rank_with_completeness"]),
            "delta",
        ),
        "unsafe_temporal_leakage": inference_document(
            "T-UNSAFE-development",
            frame.assign(delta=frame["unsafe_dates"] - frame["full_composite"]),
            "delta",
        ),
    }
    if not gap.empty:
        target_gap = gap.groupby(["target_id", "cluster_id", "gap_bin"], as_index=False)[["john_gap", "linear", "curved"]].mean()
        target_gap.to_csv(output / "gap_target_scores.csv", index=False)
        gap_effects = []
        for gap_bin, group in target_gap.groupby("gap_bin"):
            for baseline in ("john_gap", "linear"):
                # Lower SW-RMSD is better, so positive means Curved is better.
                result = inference_document(
                    f"T-GAP-{gap_bin}-curved-vs-{baseline}",
                    group.assign(delta=group[baseline] - group["curved"]),
                    "delta",
                )
                gap_effects.append({"gap_bin": gap_bin, "baseline": baseline, **result})
        pd.DataFrame(gap_effects).to_csv(output / "gap_effects.csv", index=False)
    (output / "inference.json").write_text(json.dumps(effects, indent=2) + "\n")
    print(frame.mean(numeric_only=True).to_string())
    print(json.dumps(effects["H1"], indent=2))


def cmd_build_selected_tbm(args: argparse.Namespace) -> None:
    """Apply frozen RQ1 gates, then freeze a simplified TBM before scoring it."""
    selection_freeze = DEV / "selected_tbm_freeze.json"
    if selection_freeze.exists() and not args.replace:
        raise FileExistsError("selected TBM freeze exists")
    if (RESULTS / "rq1_tbm" / "selected_h1_inference.json").exists():
        raise RuntimeError("selected TBM has already been scored and cannot be regenerated")
    rq1 = json.loads((RESULTS / "rq1_tbm" / "inference.json").read_text())
    decisions = {
        "retrieval_source": {
            "decision": "DROP_MMSEQS_USE_GLOBAL_ONLY_COMPOSITE_SCAN",
            "evidence": rq1["add_mmseqs_to_composite"],
        },
        "composite_terms": {
            "decision": "DROP_LOCAL_FEATURE_KMER_TERMS_USE_GLOBAL_ALIGNMENT_ONLY",
            "evidence": rq1["full_composite_vs_global"],
        },
        "coverage": {
            "decision": "DROP_AS_DEMONSTRATED_TERM",
            "evidence": rq1["coverage_beyond_identity"],
        },
        "completeness": {
            "decision": "DROP_AS_RANKING_TERM",
            "evidence": rq1["completeness_beyond_coverage"],
        },
        "distinct_pdb": {
            "decision": "KEEP_AS_DIVERSITY_SAFEGUARD_NOT_ACCURACY_CONTRIBUTION",
            "evidence": rq1["distinct_pdb"],
            "tolerance": -0.005,
        },
        "gap": {
            "decision": "USE_LINEAR_COMPLETION",
            "reason": "Curved completion did not show a stable improvement beyond Linear in any preregistered gap bin.",
        },
    }
    if rq1["distinct_pdb"]["mean_delta"] <= -0.005:
        raise RuntimeError("distinct-PDB exceeded the preregistered TM harm tolerance")

    manifest = read_manifest()
    meta = db.load_meta()
    coordinates = db.load_coords()
    meta_index = meta.set_index("chain_key")
    adj_dist = float(json.loads(P0_DISTANCE.read_text())["adjacent_c1"]["mean"])
    artifacts = []
    for target in manifest.itertuples(index=False):
        components = component_frame(target, meta)
        safe = components[components["release_date"] < str(target.date)]
        keys = composite_rank(safe, (1.0, 0.0, 0.0, 0.0), 50)
        items = [
            item_from_key(key, target.sequence, meta_index, coordinates, adj_dist, "global_only")
            for key in keys
        ]
        # Coverage/completeness are intentionally absent from this rank. Distinct
        # PDB remains only as the preregistered low-harm diversity safeguard.
        for item in items:
            item["identity_only"] = item["identity"]
        selected = select_items(items, "identity_only", distinct=True)
        coords, fallbacks = pad_coords(
            [item["linear"] for item in selected], target.sequence, "selected_tbm", target.target_id
        )
        confidence = [item["linear_conf"] for item in selected]
        global_confidence = [item["identity"] for item in selected]
        while len(confidence) < 5:
            confidence.append(np.full(len(target.sequence), 0.1))
            global_confidence.append(0.1)
        target_dir = RAW_ROOT / target.target_id
        path = target_dir / "selected_tbm.npz"
        np.savez_compressed(
            path,
            coords=np.asarray(coords, np.float32),
            confidence=np.asarray(confidence[:5], np.float32),
            global_confidence=np.asarray(global_confidence[:5], np.float32),
        )
        metadata = {
            "target_id": target.target_id,
            "candidate_ids": [item["chain_key"] for item in selected],
            "pdb_ids": [item["pdb_id"] for item in selected],
            "fallback_slots": fallbacks,
            "method": "global-only retrieval; identity rank; distinct-PDB safeguard; linear gap",
        }
        meta_path = target_dir / "selected_tbm.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        artifacts.append(
            {
                "target_id": target.target_id,
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "metadata_sha256": sha256(meta_path),
                "candidate_count": len(coords),
                "fallback_slots": fallbacks,
                "distinct_pdb_count": len(set(metadata["pdb_ids"])),
            }
        )
    artifact = pd.DataFrame(artifacts)
    artifact_path = DEV / "selected_tbm_candidate_manifest.csv"
    artifact.to_csv(artifact_path, index=False)
    document = {
        "status": "FROZEN_BEFORE_SELECTED_TBM_NATIVE_SCORING",
        "performance_used_for_selection": "RQ1 development scores only",
        "final_test_performance_accessed": False,
        "decisions": decisions,
        "method": "global-only retrieval; identity rank; distinct-PDB safeguard; linear gap completion",
        "target_n": len(artifact),
        "fallback_slots": int(artifact["fallback_slots"].sum()),
        "candidate_manifest": str(artifact_path.relative_to(REPO)),
        "candidate_manifest_sha256": sha256(artifact_path),
        "generation_code_sha256": sha256(Path(__file__)),
    }
    selection_freeze.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


def cmd_score_selected_tbm(_: argparse.Namespace) -> None:
    manifest = read_manifest()
    labels = labels_for(manifest)
    expected = pd.read_csv(DEV / "selected_tbm_candidate_manifest.csv").set_index("target_id")
    rows = []
    for target in manifest.itertuples(index=False):
        selected_path = RAW_ROOT / target.target_id / "selected_tbm.npz"
        if sha256(selected_path) != expected.loc[target.target_id, "sha256"]:
            raise RuntimeError(f"selected TBM changed for {target.target_id}")
        refs = references_for(labels, target.target_id)
        with np.load(selected_path, allow_pickle=False) as selected, np.load(
            RAW_ROOT / target.target_id / "raw_banks.npz", allow_pickle=False
        ) as original:
            selected_tm = score_bank(selected["coords"], refs, target.sequence)
            john_tm = score_bank(original["j_coords"], refs, target.sequence)
        rows.append(
            {
                "target_id": target.target_id,
                "cluster_id": target.mmseqs_sequence_similarity_cluster,
                "j_controlled_tbm": john_tm,
                "selected_thesis_tbm": selected_tm,
                "h1_delta": selected_tm - john_tm,
            }
        )
    frame = pd.DataFrame(rows)
    inference = inference_document("H1-selected-development", frame, "h1_delta")
    output = RESULTS / "rq1_tbm"
    frame.to_csv(output / "selected_h1_target_scores.csv", index=False)
    (output / "selected_h1_inference.json").write_text(json.dumps(inference, indent=2) + "\n")
    print(frame.mean(numeric_only=True).to_string())
    print(json.dumps(inference, indent=2))


def cmd_build_retained_tbm(args: argparse.Namespace) -> None:
    """Freeze the gate-consistent engineering TBM retained for RQ2/RQ3."""
    freeze_path = DEV / "retained_tbm_freeze.json"
    if freeze_path.exists() and not args.replace:
        raise FileExistsError("retained TBM freeze exists")
    if (RESULTS / "rq1_tbm" / "retained_h1_inference.json").exists():
        raise RuntimeError("retained TBM was already scored")
    selected_result = json.loads((RESULTS / "rq1_tbm" / "selected_h1_inference.json").read_text())
    manifest = read_manifest()
    meta = db.load_meta()
    coordinates = db.load_coords()
    meta_index = meta.set_index("chain_key")
    adj_dist = float(json.loads(P0_DISTANCE.read_text())["adjacent_c1"]["mean"])
    rows = []
    for target in manifest.itertuples(index=False):
        components = component_frame(target, meta)
        safe = components[components["release_date"] < str(target.date)]
        keys = composite_rank(safe, (1.0, 0.0, 0.0, 0.0), 50)
        items = [
            item_from_key(key, target.sequence, meta_index, coordinates, adj_dist, "global_only")
            for key in keys
        ]
        for item in items:
            item["identity_coverage"] = item["identity"] * item["coverage"]
        retained = select_items(items, "identity_coverage", distinct=True)
        coords, fallbacks = pad_coords(
            [item["linear"] for item in retained], target.sequence, "retained_tbm", target.target_id
        )
        confidence = [item["linear_conf"] for item in retained]
        global_confidence = [item["identity_coverage"] for item in retained]
        while len(confidence) < 5:
            confidence.append(np.full(len(target.sequence), 0.1))
            global_confidence.append(0.1)
        path = RAW_ROOT / target.target_id / "retained_tbm.npz"
        np.savez_compressed(
            path,
            coords=np.asarray(coords, np.float32),
            confidence=np.asarray(confidence[:5], np.float32),
            global_confidence=np.asarray(global_confidence[:5], np.float32),
        )
        metadata = {
            "target_id": target.target_id,
            "candidate_ids": [item["chain_key"] for item in retained],
            "pdb_ids": [item["pdb_id"] for item in retained],
            "fallback_slots": fallbacks,
            "method": "global-only retrieval; identity*coverage rank; distinct-PDB safeguard; linear gap",
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(
            {
                "target_id": target.target_id,
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
                "metadata_sha256": sha256(meta_path),
                "fallback_slots": fallbacks,
                "distinct_pdb_count": len(set(metadata["pdb_ids"])),
            }
        )
    artifacts = pd.DataFrame(rows)
    manifest_path = DEV / "retained_tbm_candidate_manifest.csv"
    artifacts.to_csv(manifest_path, index=False)
    freeze = {
        "status": "FROZEN_BEFORE_RETAINED_TBM_NATIVE_SCORING",
        "final_test_performance_accessed": False,
        "development_evidence_seen": [
            "initial RQ1 component ablations",
            "strict simplification H1-selected development result",
        ],
        "strict_simplification_result": selected_result,
        "method": "global-only retrieval; identity*coverage rank; distinct-PDB safeguard; linear gap",
        "claim_contract": {
            "global_only": "development-supported replacement for full composite",
            "coverage": "retained engineering heuristic; not a demonstrated contribution",
            "distinct_pdb": "diversity safeguard; not an accuracy contribution",
            "linear_gap": "simplest non-inferior development completion",
        },
        "candidate_manifest": str(manifest_path.relative_to(REPO)),
        "candidate_manifest_sha256": sha256(manifest_path),
        "target_n": len(artifacts),
        "fallback_slots": int(artifacts["fallback_slots"].sum()),
        "generation_code_sha256": sha256(Path(__file__)),
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


def cmd_score_retained_tbm(_: argparse.Namespace) -> None:
    manifest = read_manifest()
    labels = labels_for(manifest)
    expected = pd.read_csv(DEV / "retained_tbm_candidate_manifest.csv").set_index("target_id")
    rows = []
    for target in manifest.itertuples(index=False):
        retained_path = RAW_ROOT / target.target_id / "retained_tbm.npz"
        if sha256(retained_path) != expected.loc[target.target_id, "sha256"]:
            raise RuntimeError(f"retained TBM changed for {target.target_id}")
        refs = references_for(labels, target.target_id)
        with np.load(retained_path, allow_pickle=False) as retained, np.load(
            RAW_ROOT / target.target_id / "raw_banks.npz", allow_pickle=False
        ) as original:
            retained_tm = score_bank(retained["coords"], refs, target.sequence)
            john_tm = score_bank(original["j_coords"], refs, target.sequence)
        rows.append(
            {
                "target_id": target.target_id,
                "cluster_id": target.mmseqs_sequence_similarity_cluster,
                "j_controlled_tbm": john_tm,
                "retained_thesis_tbm": retained_tm,
                "h1_delta": retained_tm - john_tm,
            }
        )
    frame = pd.DataFrame(rows)
    inference = inference_document("H1-retained-development", frame, "h1_delta")
    output = RESULTS / "rq1_tbm"
    frame.to_csv(output / "retained_h1_target_scores.csv", index=False)
    (output / "retained_h1_inference.json").write_text(json.dumps(inference, indent=2) + "\n")
    print(frame.mean(numeric_only=True).to_string())
    print(json.dumps(inference, indent=2))


def self_tm(coords: np.ndarray, sequence: str) -> tuple[float, float]:
    values = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            forward = score_bank(np.asarray([coords[left]]), [coords[right]], sequence)
            reverse = score_bank(np.asarray([coords[right]]), [coords[left]], sequence)
            values.append((forward + reverse) / 2.0)
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.mean(np.asarray(values) >= 0.95))


def cmd_score_bank(_: argparse.Namespace) -> None:
    manifest = read_manifest()
    labels = labels_for(manifest)
    rows = []
    for target in manifest.itertuples(index=False):
        refs = references_for(labels, target.target_id)
        with np.load(RAW_ROOT / target.target_id / "raw_banks.npz", allow_pickle=False) as payload, np.load(
            RAW_ROOT / target.target_id / "retained_tbm.npz", allow_pickle=False
        ) as retained:
            template = retained["coords"]
            deep = payload["deep_coords"]
            banks = {
                "5T": template[:5],
                "3T+2D": np.concatenate([template[:3], deep[:2]]),
                "2T": template[:2],
                "1T+1D": np.concatenate([template[:1], deep[:1]]),
                "2D": deep[:2],
            }
            for name, coords in banks.items():
                similarity, duplicate_fraction = self_tm(coords, target.sequence)
                rows.append(
                    {
                        "target_id": target.target_id,
                        "cluster_id": target.mmseqs_sequence_similarity_cluster,
                        "bank": name,
                        "n_candidates": len(coords),
                        "best_tm": score_bank(coords, refs, target.sequence),
                        "mean_pairwise_self_tm": similarity,
                        "near_duplicate_pair_fraction": duplicate_fraction,
                        "source_available": True,
                        "fallback_slots": 0,
                    }
                )
    output = RESULTS / "rq2_candidate_source"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "target_bank_scores.csv", index=False)
    pivot = frame.pivot(index=["target_id", "cluster_id"], columns="bank", values="best_tm").reset_index()
    pivot["h2_delta"] = pivot["3T+2D"] - pivot["5T"]
    h2 = inference_document("H2-development", pivot, "h2_delta")
    mechanism = []
    for bank in ["5T", "3T+2D", "2T", "1T+1D", "2D"]:
        group = frame[frame["bank"] == bank]
        mechanism.append(
            {
                "bank": bank,
                "target_n": len(group),
                "mean_best_tm": group["best_tm"].mean(),
                "mean_pairwise_self_tm": group["mean_pairwise_self_tm"].mean(),
                "near_duplicate_pair_fraction": group["near_duplicate_pair_fraction"].mean(),
                "availability": group["source_available"].mean(),
            }
        )
    pd.DataFrame(mechanism).to_csv(output / "bank_summary.csv", index=False)
    pivot.to_csv(output / "h2_target_deltas.csv", index=False)
    (output / "inference.json").write_text(json.dumps({"H2": h2}, indent=2) + "\n")
    print(pd.DataFrame(mechanism).to_string(index=False))
    print(json.dumps(h2, indent=2))


def best_raw_reference(raw: np.ndarray, refs: list[np.ndarray], sequence: str) -> tuple[int, float]:
    values = [score_bank(np.asarray([raw]), [reference], sequence) for reference in refs]
    index = int(np.argmax(values))
    return index, float(values[index])


def refined_cache_path(target_id: str, bank: str, index: int, setting: str, raw: np.ndarray, cfg: dict) -> Path:
    digest = hashlib.sha256(np.asarray(raw, np.float32).tobytes() + json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return REFINE_ROOT / target_id / f"{bank}__{index:02d}__{setting}__{digest}.npz"


def run_geometry_cached(
    target_id: str,
    bank: str,
    index: int,
    setting: str,
    raw: np.ndarray,
    sequence: str,
    confidence: np.ndarray,
    global_confidence: float,
    cfg: GeometryV2Config,
    priors_v1: dict,
    priors_v2: dict,
    device: str,
) -> tuple[np.ndarray, bool, str]:
    document = asdict(cfg)
    path = refined_cache_path(target_id, bank, index, setting, raw, document)
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            return payload["coords"], bool(payload["failed"].item()), str(payload["reason"].item())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        coords, _ = refine_structure_v2(
            raw,
            sequence,
            priors_v1,
            priors_v2,
            source_confidence=confidence,
            global_confidence=global_confidence,
            cfg=cfg,
            device=device,
            seed=stable_seed("v4-refine", target_id, bank, index, setting),
        )
        failed, reason = False, ""
        if not np.isfinite(coords).all():
            raise FloatingPointError("nonfinite output")
    except Exception as error:
        coords, failed, reason = raw.copy(), True, f"{type(error).__name__}:{error}"
    np.savez_compressed(path, coords=np.asarray(coords, np.float32), failed=np.asarray(failed), reason=np.asarray(reason))
    return coords, failed, reason


def cmd_score_refinement(args: argparse.Namespace) -> None:
    manifest = read_manifest()
    labels = labels_for(manifest)
    priors_v1 = json.loads(P0_DISTANCE.read_text())
    priors_v2 = json.loads(P0_GEOMETRY.read_text())
    production = GeometryV2Config(**json.loads(P0_AUDIT.read_text())["production"]["geometry_config"])
    simple = replace(
        production,
        adaptive_strength=False,
        fixed_strength=1.0,
        context_mode="global",
        w_clash=0.0,
        w_rg=0.0,
        w_angle=0.0,
        w_torsion=0.0,
        w_kink=0.0,
    )
    variants = {
        "Geometry": production,
        "Geometry-fixed-strength": replace(production, adaptive_strength=False, fixed_strength=1.0),
        "Geometry-global-prior": replace(production, context_mode="global"),
        "Geometry-no-Rg": replace(production, w_rg=0.0),
        "Simple": simple,
    }
    candidate_rows, bank_rows, failure_rows = [], [], []
    for number, target in enumerate(manifest.itertuples(index=False), start=1):
        started = time.time()
        refs = references_for(labels, target.target_id)
        with np.load(RAW_ROOT / target.target_id / "raw_banks.npz", allow_pickle=False) as payload, np.load(
            RAW_ROOT / target.target_id / "retained_tbm.npz", allow_pickle=False
        ) as retained:
            banks = {
                "J-controlled": (payload["j_coords"], payload["j_conf"], payload["j_global_conf"]),
                "Thesis-3T+2D": (
                    np.concatenate([retained["coords"][:3], payload["deep_coords"][:2]]),
                    np.concatenate([retained["confidence"][:3], payload["deep_conf"][:2]]),
                    np.concatenate([retained["global_confidence"][:3], payload["deep_global_conf"][:2]]),
                ),
            }
            for bank, (raw_coords, confidence, global_confidence) in banks.items():
                active_variants = (
                    variants
                    if bank == "Thesis-3T+2D"
                    else {name: variants[name] for name in ("Simple", "Geometry")}
                )
                setting_coords: dict[str, list[np.ndarray]] = {
                    "Raw": [],
                    "John-original": [],
                    "John-fixed": [],
                    **{name: [] for name in active_variants},
                }
                geometry_outputs: dict[tuple[int, str], tuple[np.ndarray, bool, str]] = {}
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {}
                    for index, raw in enumerate(raw_coords):
                        for setting, cfg in active_variants.items():
                            future = executor.submit(
                                run_geometry_cached,
                                target.target_id,
                                bank,
                                index,
                                setting,
                                raw,
                                target.sequence,
                                confidence[index],
                                float(global_confidence[index]),
                                cfg,
                                priors_v1,
                                priors_v2,
                                args.device,
                            )
                            futures[future] = (index, setting)
                    for future in as_completed(futures):
                        geometry_outputs[futures[future]] = future.result()
                for index, raw in enumerate(raw_coords):
                    ref_index, raw_tm = best_raw_reference(raw, refs, target.sequence)
                    reference = refs[ref_index]
                    outputs = {
                        "Raw": raw,
                        "John-original": refine_rule_based(raw, target.sequence, confidence=float(global_confidence[index])),
                        "John-fixed": refine_rule_based(raw, target.sequence, confidence=0.5),
                    }
                    for setting in active_variants:
                        coords, failed, reason = geometry_outputs[(index, setting)]
                        outputs[setting] = coords
                        failure_rows.append(
                            {
                                "target_id": target.target_id,
                                "bank": bank,
                                "candidate_index": index,
                                "setting": setting,
                                "failed": failed,
                                "reason": reason,
                            }
                        )
                    for setting, coords in outputs.items():
                        setting_coords[setting].append(coords)
                        independent = local_accuracy_metrics(coords, reference, windows=(9, 15))
                        diagnostics = geometry_v2_metrics(coords, target.sequence, priors_v1, priors_v2)
                        candidate_rows.append(
                            {
                                "target_id": target.target_id,
                                "cluster_id": target.mmseqs_sequence_similarity_cluster,
                                "seq_len": int(target.seq_len),
                                "bank": bank,
                                "candidate_index": index,
                                "setting": setting,
                                "raw_reference_index": ref_index,
                                "raw_candidate_tm": raw_tm,
                                "candidate_tm_same_reference": score_bank(np.asarray([coords]), [reference], target.sequence),
                                **independent,
                                **diagnostics,
                            }
                        )
                for setting, coords in setting_coords.items():
                    bank_rows.append(
                        {
                            "target_id": target.target_id,
                            "cluster_id": target.mmseqs_sequence_similarity_cluster,
                            "seq_len": int(target.seq_len),
                            "bank": bank,
                            "setting": setting,
                            "best5_tm": score_bank(np.stack(coords), refs, target.sequence),
                        }
                    )
        print(f"[{number:02d}/{len(manifest)} {target.target_id}] refinement sec={time.time()-started:.1f}", flush=True)

    output = RESULTS / "rq3_refinement"
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    banks = pd.DataFrame(bank_rows)
    failures = pd.DataFrame(failure_rows)
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    banks.to_csv(output / "bank_metrics.csv", index=False)
    failures.to_csv(output / "failures.csv", index=False)

    thesis_candidates = candidates[candidates["bank"] == "Thesis-3T+2D"]
    thesis_banks = banks[banks["bank"] == "Thesis-3T+2D"]
    cand_pivot = thesis_candidates.pivot_table(
        index=["target_id", "cluster_id"], columns="setting", values="sw_rmsd_9", aggfunc="mean"
    ).reset_index()
    cand_pivot["h3_delta"] = cand_pivot["Simple"] - cand_pivot["Geometry"]
    h3 = inference_document("H3-development", cand_pivot, "h3_delta")
    bank_pivot = thesis_banks.pivot(index=["target_id", "cluster_id"], columns="setting", values="best5_tm").reset_index()
    bank_pivot["geometry_vs_simple_tm_delta"] = bank_pivot["Geometry"] - bank_pivot["Simple"]
    tm_safeguard = inference_document("H3-TM-safeguard-development", bank_pivot, "geometry_vs_simple_tm_delta")
    h3_pass = h3["ci_lower"] > 0.0 and tm_safeguard["ci_lower"] > -0.005

    mechanisms = {}
    for name, comparator in {
        "confidence_adaptive_vs_fixed": "Geometry-fixed-strength",
        "candidate_context_vs_global": "Geometry-global-prior",
        "rg_on_vs_off": "Geometry-no-Rg",
    }.items():
        pivot = thesis_candidates.pivot_table(
            index=["target_id", "cluster_id"], columns="setting", values="sw_rmsd_9", aggfunc="mean"
        ).reset_index()
        pivot["delta"] = pivot[comparator] - pivot["Geometry"]
        mechanisms[name] = inference_document(name, pivot, "delta")
    length_bins = pd.cut(
        thesis_banks["seq_len"], bins=[29, 79, 149, 249, 400], labels=["30-79", "80-149", "150-249", "250-400"]
    )
    length_frame = thesis_banks.assign(length_bin=length_bins)
    rg_rows = []
    for length_bin, group in length_frame.groupby("length_bin", observed=True):
        pivot = group.pivot(index=["target_id", "cluster_id"], columns="setting", values="best5_tm").reset_index()
        pivot["delta"] = pivot["Geometry"] - pivot["Geometry-no-Rg"]
        rg_rows.append({"length_bin": str(length_bin), **inference_document(f"Rg-{length_bin}", pivot, "delta")})
    pd.DataFrame(rg_rows).to_csv(output / "rg_length_bins.csv", index=False)

    decision = {
        "H3": h3,
        "tm_safeguard": tm_safeguard,
        "development_gate_pass": h3_pass,
        "selected_final_refiner": "Geometry" if h3_pass else "Simple",
        "rule_applied": "Geometry requires SW-RMSD9 superiority CI lower > 0 and TM delta CI lower > -0.005; otherwise Simple",
        "mechanisms": mechanisms,
        "failure_count": int(failures["failed"].sum()),
    }
    (output / "inference_and_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    cand_pivot.to_csv(output / "h3_target_deltas.csv", index=False)
    bank_pivot.to_csv(output / "tm_safeguard_target_deltas.csv", index=False)
    print(json.dumps(decision, indent=2))


def cmd_freeze_selected_refiner(args: argparse.Namespace) -> None:
    """Apply mechanism gates and freeze one combined Geometry configuration."""
    freeze_path = DEV / "selected_refiner_freeze.json"
    if freeze_path.exists() and not args.replace:
        raise FileExistsError("selected refiner freeze exists")
    if (RESULTS / "rq3_refinement" / "selected_inference_and_decision.json").exists():
        raise RuntimeError("selected refiner has already been scored")
    output = RESULTS / "rq3_refinement"
    candidates = pd.read_csv(output / "candidate_metrics.csv")
    banks = pd.read_csv(output / "bank_metrics.csv")
    thesis_candidates = candidates[candidates["bank"] == "Thesis-3T+2D"]
    thesis_banks = banks[banks["bank"] == "Thesis-3T+2D"]
    sw = thesis_candidates.pivot_table(
        index=["target_id", "cluster_id"], columns="setting", values="sw_rmsd_9", aggfunc="mean"
    ).reset_index()
    tm = thesis_banks.pivot(index=["target_id", "cluster_id"], columns="setting", values="best5_tm").reset_index()

    controls = {
        "confidence": "Geometry-fixed-strength",
        "pair_context": "Geometry-global-prior",
        "rg": "Geometry-no-Rg",
    }
    mechanism = {}
    for name, control in controls.items():
        sw_frame = sw.assign(delta=sw["Geometry"] - sw[control])
        tm_frame = tm.assign(delta=tm[control] - tm["Geometry"])
        sw_result = inference_document(f"{name}-control-SW", sw_frame, "delta")
        tm_result = inference_document(f"{name}-control-TM", tm_frame, "delta")
        mechanism[name] = {
            "control": control,
            "control_improves_sw_point": sw_result["mean_delta"] > 0,
            "control_tm_preserved": tm_result["ci_lower"] > -0.005,
            "sw": sw_result,
            "tm": tm_result,
        }

    # Exact keep/drop interpretation from the preregistration: confidence and
    # pair context stay only when they beat the simpler control; Rg stays only
    # with a clear benefit. All three controls preserve TM here.
    use_fixed = mechanism["confidence"]["control_improves_sw_point"] and mechanism["confidence"]["control_tm_preserved"]
    use_global = mechanism["pair_context"]["control_improves_sw_point"] and mechanism["pair_context"]["control_tm_preserved"]
    use_no_rg = (
        not json.loads((output / "inference_and_decision.json").read_text())["mechanisms"]["rg_on_vs_off"]["ci_lower"] > 0
        and mechanism["rg"]["control_tm_preserved"]
    )
    production = GeometryV2Config(**json.loads(P0_AUDIT.read_text())["production"]["geometry_config"])
    selected = replace(
        production,
        adaptive_strength=not use_fixed,
        fixed_strength=1.0,
        context_mode="global" if use_global else "candidate_derived",
        w_rg=0.0 if use_no_rg else production.w_rg,
    )
    freeze = {
        "status": "FROZEN_BEFORE_SELECTED_REFINER_NATIVE_SCORING",
        "final_test_performance_accessed": False,
        "source_result_sha256": {
            "candidate_metrics": sha256(output / "candidate_metrics.csv"),
            "bank_metrics": sha256(output / "bank_metrics.csv"),
            "full_geometry_decision": sha256(output / "inference_and_decision.json"),
        },
        "mechanism_gates": mechanism,
        "decisions": {
            "confidence": "fixed strength" if use_fixed else "adaptive strength",
            "angle_torsion_context": "global unconditional prior" if use_global else "candidate-derived pair context",
            "rg": "off" if use_no_rg else "on",
        },
        "selected_config": asdict(selected),
        "next_gate": "Combined selected Geometry must pass SW-RMSD9 superiority plus TM preservation versus Simple; otherwise Simple is final.",
        "generation_code_sha256": sha256(Path(__file__)),
    }
    freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps(freeze, indent=2))


def cmd_score_selected_refiner(args: argparse.Namespace) -> None:
    freeze = json.loads((DEV / "selected_refiner_freeze.json").read_text())
    cfg = GeometryV2Config(**freeze["selected_config"])
    manifest = read_manifest()
    labels = labels_for(manifest)
    priors_v1 = json.loads(P0_DISTANCE.read_text())
    priors_v2 = json.loads(P0_GEOMETRY.read_text())
    candidate_rows, bank_rows, failure_rows = [], [], []
    for number, target in enumerate(manifest.itertuples(index=False), start=1):
        started = time.time()
        refs = references_for(labels, target.target_id)
        with np.load(RAW_ROOT / target.target_id / "raw_banks.npz", allow_pickle=False) as payload, np.load(
            RAW_ROOT / target.target_id / "retained_tbm.npz", allow_pickle=False
        ) as retained:
            banks = {
                "J-controlled": (payload["j_coords"], payload["j_conf"], payload["j_global_conf"]),
                "Thesis-3T+2D": (
                    np.concatenate([retained["coords"][:3], payload["deep_coords"][:2]]),
                    np.concatenate([retained["confidence"][:3], payload["deep_conf"][:2]]),
                    np.concatenate([retained["global_confidence"][:3], payload["deep_global_conf"][:2]]),
                ),
            }
            tasks = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for bank, (raw_coords, confidence, global_confidence) in banks.items():
                    for index, raw in enumerate(raw_coords):
                        future = executor.submit(
                            run_geometry_cached,
                            target.target_id,
                            bank,
                            index,
                            "Geometry-selected",
                            raw,
                            target.sequence,
                            confidence[index],
                            float(global_confidence[index]),
                            cfg,
                            priors_v1,
                            priors_v2,
                            args.device,
                        )
                        tasks[future] = (bank, index)
                outputs = {tasks[future]: future.result() for future in as_completed(tasks)}
            for bank, (raw_coords, _, _) in banks.items():
                refined_bank = []
                for index, raw in enumerate(raw_coords):
                    coords, failed, reason = outputs[(bank, index)]
                    refined_bank.append(coords)
                    ref_index, raw_tm = best_raw_reference(raw, refs, target.sequence)
                    reference = refs[ref_index]
                    candidate_rows.append(
                        {
                            "target_id": target.target_id,
                            "cluster_id": target.mmseqs_sequence_similarity_cluster,
                            "seq_len": int(target.seq_len),
                            "bank": bank,
                            "candidate_index": index,
                            "setting": "Geometry-selected",
                            "raw_reference_index": ref_index,
                            "raw_candidate_tm": raw_tm,
                            "candidate_tm_same_reference": score_bank(np.asarray([coords]), [reference], target.sequence),
                            **local_accuracy_metrics(coords, reference, windows=(9, 15)),
                            **geometry_v2_metrics(coords, target.sequence, priors_v1, priors_v2),
                        }
                    )
                    failure_rows.append(
                        {
                            "target_id": target.target_id,
                            "bank": bank,
                            "candidate_index": index,
                            "setting": "Geometry-selected",
                            "failed": failed,
                            "reason": reason,
                        }
                    )
                bank_rows.append(
                    {
                        "target_id": target.target_id,
                        "cluster_id": target.mmseqs_sequence_similarity_cluster,
                        "seq_len": int(target.seq_len),
                        "bank": bank,
                        "setting": "Geometry-selected",
                        "best5_tm": score_bank(np.stack(refined_bank), refs, target.sequence),
                    }
                )
        print(f"[{number:02d}/{len(manifest)} {target.target_id}] selected-refiner sec={time.time()-started:.1f}", flush=True)

    output = RESULTS / "rq3_refinement"
    selected_candidates = pd.DataFrame(candidate_rows)
    selected_banks = pd.DataFrame(bank_rows)
    selected_failures = pd.DataFrame(failure_rows)
    selected_candidates.to_csv(output / "selected_candidate_metrics.csv", index=False)
    selected_banks.to_csv(output / "selected_bank_metrics.csv", index=False)
    selected_failures.to_csv(output / "selected_failures.csv", index=False)

    original_candidates = pd.read_csv(output / "candidate_metrics.csv")
    original_banks = pd.read_csv(output / "bank_metrics.csv")
    simple_c = original_candidates[
        (original_candidates["bank"] == "Thesis-3T+2D") & (original_candidates["setting"] == "Simple")
    ]
    selected_c = selected_candidates[selected_candidates["bank"] == "Thesis-3T+2D"]
    keys = ["target_id", "cluster_id", "candidate_index"]
    paired_c = simple_c[keys + ["sw_rmsd_9"]].merge(
        selected_c[keys + ["sw_rmsd_9"]], on=keys, suffixes=("_simple", "_selected")
    )
    target_c = paired_c.groupby(["target_id", "cluster_id"], as_index=False)[["sw_rmsd_9_simple", "sw_rmsd_9_selected"]].mean()
    target_c["h3_delta"] = target_c["sw_rmsd_9_simple"] - target_c["sw_rmsd_9_selected"]
    h3 = inference_document("H3-selected-development", target_c, "h3_delta")

    simple_b = original_banks[
        (original_banks["bank"] == "Thesis-3T+2D") & (original_banks["setting"] == "Simple")
    ][["target_id", "cluster_id", "best5_tm"]]
    selected_b = selected_banks[selected_banks["bank"] == "Thesis-3T+2D"][["target_id", "cluster_id", "best5_tm"]]
    target_b = simple_b.merge(selected_b, on=["target_id", "cluster_id"], suffixes=("_simple", "_selected"))
    target_b["tm_delta"] = target_b["best5_tm_selected"] - target_b["best5_tm_simple"]
    tm_guard = inference_document("H3-selected-TM-safeguard-development", target_b, "tm_delta")

    h1 = json.loads((RESULTS / "rq1_tbm" / "retained_h1_inference.json").read_text())
    h2 = json.loads((RESULTS / "rq2_candidate_source" / "inference.json").read_text())["H2"]
    holm = holm_step_down({"H1": h1["raw_one_sided_p"], "H2": h2["raw_one_sided_p"], "H3": h3["raw_one_sided_p"]})
    gate = h3["ci_lower"] > 0 and tm_guard["ci_lower"] > -0.005 and bool(holm["H3"]["reject"])
    decision = {
        "H3": h3,
        "tm_safeguard": tm_guard,
        "development_holm": holm,
        "development_gate_pass": gate,
        "selected_final_refiner": "Geometry-selected" if gate else "Simple",
        "rule_applied": "No rescue after this combined keep/drop configuration; failure selects Simple.",
        "failure_count": int(selected_failures["failed"].sum()),
        "selected_config": asdict(cfg),
    }
    target_c.to_csv(output / "selected_h3_target_deltas.csv", index=False)
    target_b.to_csv(output / "selected_tm_safeguard_target_deltas.csv", index=False)
    (output / "selected_inference_and_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


def cmd_summarize(_: argparse.Namespace) -> None:
    h1 = json.loads((RESULTS / "rq1_tbm" / "retained_h1_inference.json").read_text())
    h2 = json.loads((RESULTS / "rq2_candidate_source" / "inference.json").read_text())["H2"]
    selected_path = RESULTS / "rq3_refinement" / "selected_inference_and_decision.json"
    h3_doc = json.loads(selected_path.read_text())
    refiner_freeze = json.loads((DEV / "selected_refiner_freeze.json").read_text())
    summary = {
        "scope": "V4 development only; all targets were previously exposed",
        "confirmatory_claims_allowed": False,
        "H1": h1,
        "H2": h2,
        "H3": h3_doc["H3"],
        "H3_tm_safeguard": h3_doc["tm_safeguard"],
        "holm_primary_family": h3_doc["development_holm"],
        "development_component_decisions": {
            "refiner": h3_doc["selected_final_refiner"],
            "geometry_gate_pass": h3_doc["development_gate_pass"],
            "geometry_config": h3_doc["selected_config"],
            "mechanism_decisions": refiner_freeze["decisions"],
        },
    }
    (RESULTS / "development_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    raw = commands.add_parser("build-raw")
    raw.add_argument("--refresh-mmseqs", action="store_true")
    raw.add_argument("--replace", action="store_true")
    raw.set_defaults(func=cmd_build_raw)
    commands.add_parser("score-tbm").set_defaults(func=cmd_score_tbm)
    selected = commands.add_parser("build-selected-tbm")
    selected.add_argument("--replace", action="store_true")
    selected.set_defaults(func=cmd_build_selected_tbm)
    commands.add_parser("score-selected-tbm").set_defaults(func=cmd_score_selected_tbm)
    retained = commands.add_parser("build-retained-tbm")
    retained.add_argument("--replace", action="store_true")
    retained.set_defaults(func=cmd_build_retained_tbm)
    commands.add_parser("score-retained-tbm").set_defaults(func=cmd_score_retained_tbm)
    commands.add_parser("score-bank").set_defaults(func=cmd_score_bank)
    refine = commands.add_parser("score-refinement")
    refine.add_argument("--device", default="cuda")
    refine.add_argument("--workers", type=int, default=4)
    refine.set_defaults(func=cmd_score_refinement)
    freeze_refiner = commands.add_parser("freeze-selected-refiner")
    freeze_refiner.add_argument("--replace", action="store_true")
    freeze_refiner.set_defaults(func=cmd_freeze_selected_refiner)
    selected_refiner = commands.add_parser("score-selected-refiner")
    selected_refiner.add_argument("--device", default="cuda")
    selected_refiner.add_argument("--workers", type=int, default=8)
    selected_refiner.set_defaults(func=cmd_score_selected_refiner)
    commands.add_parser("summarize").set_defaults(func=cmd_summarize)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
