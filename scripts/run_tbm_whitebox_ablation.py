#!/usr/bin/env python
"""White-box TBM component study on the 20 calibration RNAs.

This experiment separates retrieval-score components, template reranking,
distinct-PDB selection, and gap filling.  It is intentionally a development-set
mechanism study; the 20 newest RNAs remain the confirmatory Geometry-v2 set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.baselines.top1 import composite_similarity_components, rna_features
from rna3d.data import io
from rna3d.eval.local_metrics import sliding_window_c1_rmsd
from rna3d.eval.statistics import paired_target_summary
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, processed
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.template import db, mmseqs_search
from rna3d.template.align import align_and_transfer
from rna3d.template.confidence import template_confidence
from rna3d.template.gap_fill import fill_gaps, fill_gaps_linear


MANIFEST = processed() / "geofuse_real_oof_v2" / "medium_manifest.csv"
PRIORS = processed() / "geometry_v2_confirmatory" / "priors_v1_train60.json"
COMPONENT_CACHE = cache() / "tbm_whitebox" / "components"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "tables" / "tbm_whitebox"
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_notes" / "tbm_whitebox_ablation.md"

CORE_RETRIEVAL = {
    "G_only": (1.0, 0.0, 0.0, 0.0),
    "G_plus_L": (0.4, 0.3, 0.0, 0.0),
    "G_L_F": (0.4, 0.3, 0.2, 0.0),
    "G_L_F_K_equal": (0.25, 0.25, 0.25, 0.25),
    "full_weighted": (0.4, 0.3, 0.2, 0.1),
}
SENSITIVITY = {
    "wG_minus25pct": (0.30, 0.30, 0.20, 0.10),
    "wG_plus25pct": (0.50, 0.30, 0.20, 0.10),
    "wL_minus25pct": (0.40, 0.225, 0.20, 0.10),
    "wL_plus25pct": (0.40, 0.375, 0.20, 0.10),
}


def _manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST)
    return frame[frame["split"] == "calibration"].reset_index(drop=True)


def _labels(targets: pd.Series) -> pd.DataFrame:
    labels = io.load_labels("train_v2")
    tids = labels["ID"].map(io.target_id_of)
    return labels[tids.isin(set(targets))].copy()


def _length_allowed(query_len: int, template_len: int) -> bool:
    ratio = abs(template_len - query_len) / max(template_len, query_len)
    if query_len < 50 or template_len < 50:
        return ratio <= 0.6
    if query_len > 1000 or template_len > 1000:
        return ratio <= 0.2
    return ratio <= 0.4


def _component_frame(target, library: pd.DataFrame) -> pd.DataFrame:
    digest = hashlib.sha256(
        f"{target.target_id}|{target.sequence}|component-v1".encode()
    ).hexdigest()[:12]
    path = COMPONENT_CACHE / f"{target.target_id}__{digest}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    excluded = {
        value.strip().upper()
        for value in str(target.excluded_pdb_ids).replace(";", ",").split(",")
        if value.strip() and value.strip().lower() != "nan"
    }
    qfeat = rna_features(target.sequence)
    rows = []
    for template in library.itertuples(index=False):
        if str(template.pdb_id).upper() in excluded:
            continue
        if not _length_allowed(len(target.sequence), len(template.sequence)):
            continue
        values = composite_similarity_components(
            target.sequence, template.sequence, qfeat
        )
        rows.append(
            {
                "chain_key": template.target_id,
                "pdb_id": str(template.pdb_id).upper(),
                "sequence": template.sequence,
                "release_date": str(template.release_date),
                **values,
            }
        )
    result = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def _score_column(frame: pd.DataFrame, weights: tuple[float, ...]) -> pd.Series:
    return (
        weights[0] * frame["global"]
        + weights[1] * frame["local"]
        + weights[2] * frame["features"]
        + weights[3] * frame["kmer3"]
    )


def _select_keys(
    frame: pd.DataFrame,
    weights: tuple[float, ...],
    n: int,
    *,
    distinct_pdb: bool = True,
) -> list[str]:
    ranked = frame.assign(_score=_score_column(frame, weights)).sort_values(
        ["_score", "chain_key"], ascending=[False, True]
    )
    keys, used = [], set()
    for row in ranked.itertuples(index=False):
        if distinct_pdb and row.pdb_id in used:
            continue
        keys.append(row.chain_key)
        used.add(row.pdb_id)
        if len(keys) == n:
            break
    return keys


def _materialize(
    chain_key: str,
    query: str,
    meta_index: pd.DataFrame,
    coordinates: dict,
    adj_dist: float,
) -> dict:
    row = meta_index.loc[chain_key]
    payload = coordinates[chain_key]
    transfer = align_and_transfer(
        query,
        {"seq": str(row["sequence"]), "coords": np.asarray(payload["coords"], float)},
        chain_key,
    )
    completeness = transfer.template_resolved / max(transfer.template_len, 1)
    current, _ = fill_gaps(transfer.coords, transfer.mask, adj_dist=adj_dist)
    linear, _ = fill_gaps_linear(transfer.coords, transfer.mask, adj_dist=adj_dist)
    return {
        "chain_key": chain_key,
        "pdb_id": str(row["pdb_id"]).upper(),
        "identity": transfer.identity,
        "coverage": transfer.coverage,
        "completeness": completeness,
        "confidence": template_confidence(
            transfer.identity, transfer.coverage, completeness
        ),
        "mask": transfer.mask,
        "current": current,
        "linear": linear,
    }


def _distinct_rank(items: list[dict], score_name: str, n: int, distinct: bool) -> list[dict]:
    ranked = sorted(items, key=lambda item: (-float(item[score_name]), item["chain_key"]))
    selected, used = [], set()
    for item in ranked:
        if distinct and item["pdb_id"] in used:
            continue
        selected.append(item)
        used.add(item["pdb_id"])
        if len(selected) == n:
            break
    if len(selected) < n:
        selected.extend(item for item in ranked if item not in selected)
    return selected[:n]


def _tm(coords: list[np.ndarray], references: list[np.ndarray], sequence: str) -> float:
    if not coords:
        return float("nan")
    return float(score_target(coords, references, list(sequence)))


def _best_reference(coords: np.ndarray, references: list[np.ndarray], sequence: str) -> int:
    scores = [_tm([coords], [reference], sequence) for reference in references]
    return int(np.nanargmax(scores))


def _gap_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    missing = ~np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(np.r_[missing, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _gap_bin(length: int) -> str:
    if length <= 3:
        return "1-3"
    if length <= 8:
        return "4-8"
    if length <= 20:
        return "9-20"
    return ">20"


def _batch_mmseqs(sequences: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    query = cache() / "tbm_whitebox" / "calibration_queries.fasta"
    output = cache() / "tbm_whitebox" / "calibration_hits.m8"
    query.parent.mkdir(parents=True, exist_ok=True)
    query.write_text(
        "".join(f">{row.target_id}\n{row.sequence}\n" for row in sequences.itertuples())
    )
    if output.exists() and not refresh:
        hits = mmseqs_search.read_m8(output)
        if set(sequences["target_id"]).issubset(set(hits["query"])):
            return hits
    return mmseqs_search.search(query, output)


def _summarize_gap(gaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gaps.empty:
        return gaps, gaps
    target = gaps.groupby(["target_id", "gap_bin"])[
        ["linear_sw9", "current_sw9"]
    ].mean().reset_index()
    rows = []
    for name, group in target.groupby("gap_bin"):
        result = paired_target_summary(
            group["linear_sw9"].to_numpy(),
            group["current_sw9"].to_numpy(),
            higher_is_better=False,
        )
        rows.append(
            {
                "gap_bin": name,
                "linear_sw9": float(group["linear_sw9"].mean()),
                "current_sw9": float(group["current_sw9"].mean()),
                **result,
            }
        )
    return target, pd.DataFrame(rows)


def _paired_effect(
    frame: pd.DataFrame,
    setting_column: str,
    metric: str,
    baseline: str,
    method: str,
    effect_name: str,
) -> dict:
    pivot = frame.pivot(index="target_id", columns=setting_column, values=metric)
    shared = pivot[[baseline, method]].dropna()
    result = paired_target_summary(
        shared[baseline].to_numpy(),
        shared[method].to_numpy(),
        higher_is_better=True,
    )
    return {
        "effect": effect_name,
        "metric": metric,
        "baseline": baseline,
        "method": method,
        **result,
    }


def run(args: argparse.Namespace) -> None:
    sequences = _manifest()
    labels = _labels(sequences["target_id"])
    library = pd.read_parquet(processed() / "top1_template_meta.parquet")
    with (cache() / "top1_template_coords.pkl").open("rb") as handle:
        coordinates = pickle.load(handle)
    meta_index = library.set_index("target_id")
    adj_dist = float(json.loads(PRIORS.read_text())["adjacent_c1"]["mean"])
    mm_meta = db.load_meta()
    db.load_coords()
    hits = _batch_mmseqs(sequences, args.refresh_mmseqs)

    retrieval_rows, ranking_rows, source_rows, gap_rows = [], [], [], []
    all_variants = {**CORE_RETRIEVAL, **SENSITIVITY}
    for number, target in enumerate(sequences.itertuples(index=False), 1):
        started = time.time()
        component_all = _component_frame(target, library)
        component_safe = component_all[
            component_all["release_date"].astype(str) < str(target.date)
        ].copy()
        references = io.get_reference_coords(labels, target.target_id)
        materialized: dict[str, dict] = {}

        def get(key: str) -> dict:
            if key not in materialized:
                materialized[key] = _materialize(
                    key, target.sequence, meta_index, coordinates, adj_dist
                )
            return materialized[key]

        selected_by_variant = {
            name: _select_keys(component_safe, weights, 5)
            for name, weights in all_variants.items()
        }
        selected_by_variant["full_weighted_unsafe_dates"] = _select_keys(
            component_all, CORE_RETRIEVAL["full_weighted"], 5
        )
        individual_tm: dict[str, float] = {}
        for key in set().union(*map(set, selected_by_variant.values())):
            individual_tm[key] = _tm([get(key)["current"]], references, target.sequence)
        useful = {key for key, value in individual_tm.items() if value >= 0.45}

        for name, keys in selected_by_variant.items():
            items = [get(key) for key in keys]
            retrieval_rows.append(
                {
                    "target_id": target.target_id,
                    "variant": name,
                    "top1_tm": individual_tm[keys[0]] if keys else np.nan,
                    "best5_tm": _tm(
                        [item["current"] for item in items], references, target.sequence
                    ),
                    "mean_coverage": float(np.mean([item["coverage"] for item in items])),
                    "useful_hit_045": int(any(individual_tm[key] >= 0.45 for key in keys)),
                    "useful_recall_045": (
                        len(set(keys) & useful) / len(useful) if useful else np.nan
                    ),
                    "n_distinct_pdb": len({item["pdb_id"] for item in items}),
                }
            )

        # Ranking study uses one common high-recall pool, so only the ranking
        # formula and distinct-PDB rule change.
        pool_keys = _select_keys(
            component_safe, CORE_RETRIEVAL["full_weighted"], 50, distinct_pdb=False
        )
        pool = [get(key) for key in pool_keys]
        for item in pool:
            item["identity_only"] = item["identity"]
            item["identity_coverage"] = item["identity"] * item["coverage"]
            item["full_rank"] = item["confidence"]
        for name, score_name, distinct in (
            ("identity_only", "identity_only", False),
            ("identity_x_coverage", "identity_coverage", False),
            ("identity_x_coverage_x_completeness", "full_rank", False),
            ("full_plus_distinct_pdb", "full_rank", True),
        ):
            chosen = _distinct_rank(pool, score_name, 5, distinct)
            ranking_rows.append(
                {
                    "target_id": target.target_id,
                    "ranking": name,
                    "top1_tm": _tm([chosen[0]["current"]], references, target.sequence),
                    "top3_tm": _tm(
                        [item["current"] for item in chosen[:3]], references, target.sequence
                    ),
                    "top5_tm": _tm(
                        [item["current"] for item in chosen], references, target.sequence
                    ),
                    "mean_coverage": float(np.mean([item["coverage"] for item in chosen])),
                    "n_distinct_pdb": len({item["pdb_id"] for item in chosen}),
                }
            )

        excluded = tuple(
            value.strip().upper()
            for value in str(target.excluded_pdb_ids).replace(";", ",").split(",")
            if value.strip() and value.strip().lower() != "nan"
        )
        mm_candidates = build_tbm_candidates(
            target.target_id,
            target.sequence,
            str(target.date),
            hits[hits["query"] == target.target_id],
            mm_meta,
            adj_dist=adj_dist,
            max_candidates=5,
            exclude_pdb_ids=excluded,
            composite_fallback=False,
            rng=np.random.default_rng(0),
        )
        composite_items = [get(key) for key in selected_by_variant["full_weighted"]]
        mm_items = [
            {
                "chain_key": item.chain_key,
                "pdb_id": item.meta["pdb_id"],
                "confidence": item.confidence,
                "coverage": item.coverage,
                "current": item.coords,
            }
            for item in mm_candidates
        ]
        union = {item["chain_key"]: item for item in mm_items + composite_items}
        union_ranked = _distinct_rank(list(union.values()), "confidence", 5, True)
        for name, items in (
            ("MMseqs_only", mm_items[:5]),
            ("composite_only", composite_items),
            ("MMseqs_plus_composite", union_ranked),
        ):
            source_rows.append(
                {
                    "target_id": target.target_id,
                    "source_variant": name,
                    "n_candidates": len(items),
                    "top1_tm": _tm([items[0]["current"]], references, target.sequence)
                    if items
                    else np.nan,
                    "best5_tm": _tm(
                        [item["current"] for item in items], references, target.sequence
                    ),
                    "mean_coverage": float(np.mean([item["coverage"] for item in items]))
                    if items
                    else np.nan,
                }
            )

        for item in composite_items:
            reference_index = _best_reference(item["current"], references, target.sequence)
            current_local = sliding_window_c1_rmsd(
                item["current"], references[reference_index], window=9
            )["per_residue"]
            linear_local = sliding_window_c1_rmsd(
                item["linear"], references[reference_index], window=9
            )["per_residue"]
            for start, stop in _gap_runs(item["mask"]):
                length = stop - start
                finite = np.isfinite(current_local[start:stop]) & np.isfinite(
                    linear_local[start:stop]
                )
                if not finite.any():
                    continue
                gap_rows.append(
                    {
                        "target_id": target.target_id,
                        "chain_key": item["chain_key"],
                        "gap_length": length,
                        "gap_bin": _gap_bin(length),
                        "region": "terminal"
                        if start == 0 or stop == len(item["mask"])
                        else "internal",
                        "linear_sw9": float(linear_local[start:stop][finite].mean()),
                        "current_sw9": float(current_local[start:stop][finite].mean()),
                    }
                )
        full_row = next(
            row
            for row in reversed(retrieval_rows)
            if row["target_id"] == target.target_id and row["variant"] == "full_weighted"
        )
        print(
            f"[{number:02d}/{len(sequences)} {target.target_id}] "
            f"library={len(component_safe)} mmseqs={len(mm_items)} "
            f"fullTM={full_row['best5_tm']:.4f} sec={time.time()-started:.1f}",
            flush=True,
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    retrieval = pd.DataFrame(retrieval_rows)
    ranking = pd.DataFrame(ranking_rows)
    sources = pd.DataFrame(source_rows)
    gaps = pd.DataFrame(gap_rows)
    gap_targets, gap_summary = _summarize_gap(gaps)
    retrieval.to_csv(output / "retrieval_components.csv", index=False)
    ranking.to_csv(output / "ranking_ablation.csv", index=False)
    sources.to_csv(output / "search_source_ablation.csv", index=False)
    gaps.to_csv(output / "gap_instances.csv", index=False)
    gap_targets.to_csv(output / "gap_target_metrics.csv", index=False)
    gap_summary.to_csv(output / "gap_summary.csv", index=False)

    retrieval_summary = retrieval.groupby("variant")[
        ["top1_tm", "best5_tm", "mean_coverage", "useful_hit_045", "useful_recall_045"]
    ].mean()
    ranking_summary = ranking.groupby("ranking")[
        ["top1_tm", "top3_tm", "top5_tm", "mean_coverage", "n_distinct_pdb"]
    ].mean()
    source_summary_rows = []
    for name, group in sources.groupby("source_variant"):
        available = group["n_candidates"] > 0
        source_summary_rows.append(
            {
                "source_variant": name,
                "n_targets": int(len(group)),
                "available_targets": int(available.sum()),
                "availability": float(available.mean()),
                "mean_candidates_all_targets": float(group["n_candidates"].mean()),
                "top1_tm_when_available": float(group.loc[available, "top1_tm"].mean()),
                "best5_tm_when_available": float(group.loc[available, "best5_tm"].mean()),
                "mean_coverage_when_available": float(
                    group.loc[available, "mean_coverage"].mean()
                ),
            }
        )
    source_summary = pd.DataFrame(source_summary_rows).set_index("source_variant")
    effects = pd.DataFrame(
        [
            _paired_effect(
                retrieval,
                "variant",
                "best5_tm",
                "full_weighted",
                "G_only",
                "G-only versus full weighted",
            ),
            _paired_effect(
                retrieval,
                "variant",
                "best5_tm",
                "full_weighted",
                "full_weighted_unsafe_dates",
                "temporal leakage (unsafe minus safe)",
            ),
            _paired_effect(
                ranking,
                "ranking",
                "top5_tm",
                "identity_only",
                "identity_x_coverage",
                "add target coverage to identity",
            ),
            _paired_effect(
                ranking,
                "ranking",
                "top5_tm",
                "identity_x_coverage",
                "identity_x_coverage_x_completeness",
                "add template completeness",
            ),
            _paired_effect(
                ranking,
                "ranking",
                "top5_tm",
                "identity_x_coverage_x_completeness",
                "full_plus_distinct_pdb",
                "enforce distinct PDB",
            ),
            _paired_effect(
                sources,
                "source_variant",
                "best5_tm",
                "composite_only",
                "MMseqs_plus_composite",
                "add MMseqs candidates to composite",
            ),
        ]
    )
    retrieval_summary.to_csv(output / "retrieval_summary.csv")
    ranking_summary.to_csv(output / "ranking_summary.csv")
    source_summary.to_csv(output / "search_source_summary.csv")
    effects.to_csv(output / "paired_effects.csv", index=False)
    report = [
        "# TBM white-box component ablation (20 calibration RNA)",
        "",
        "This mechanism study uses only the 20 calibration RNAs. Every template must be "
        "released strictly before its target and the target PDB itself is excluded. The "
        "`unsafe_dates` row is shown only to quantify temporal leakage and is never a method.",
        "",
        "## Search source",
        "",
        "TM values for MMseqs are conditional on finding at least one template; availability "
        "is reported separately so 12 missing targets cannot silently disappear from the mean.",
        "",
        source_summary.round(6).to_markdown(),
        "",
        "## Composite retrieval-score components",
        "",
        "`useful_hit_045` is the fraction of targets whose retrieved top five contain a "
        "raw template candidate with TM ≥ 0.45. `useful_recall_045` is recall inside the "
        "union of candidates scored by this component study; it is a diagnostic, not an "
        "official biological metric.",
        "",
        retrieval_summary.round(6).to_markdown(),
        "",
        "## Reranking and distinct-PDB selection",
        "",
        ranking_summary.round(6).to_markdown(),
        "",
        "## Paired target effects",
        "",
        "Every delta is oriented so positive means the named method is better. Intervals "
        "bootstrap RNA targets (10,000 samples), not templates or residues.",
        "",
        effects.round(6).to_markdown(index=False),
        "",
        "## Gap filling on unsupported residues",
        "",
        "Values are 9-residue sliding-window C1′ RMSD at unsupported residues. Positive "
        "paired deltas mean the current curved-gap heuristic is better than linear filling. "
        "Candidate gaps are averaged within each RNA before the target bootstrap.",
        "",
        gap_summary.round(6).to_markdown(index=False),
        "",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report))
    print(retrieval_summary.round(6).to_string())
    print(ranking_summary.round(6).to_string())
    print(gap_summary.round(6).to_string(index=False))
    print(f"[report] {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-mmseqs", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
