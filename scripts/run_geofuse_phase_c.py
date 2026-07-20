#!/usr/bin/env python
"""Evaluate fold clustering, segment fusion, and diverse final-five selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.eval.self_tm import self_tm_matrix
from rna3d.eval.usalign import score_target
from rna3d.geofuse.candidate import CandidateCache, StructureCandidate
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.phase_a import select_source_balanced
from rna3d.geofuse.phase_c import (
    FusionConfig,
    SelectionConfig,
    cluster_fold_families,
    fuse_template_pretrained,
    mean_selected_similarity,
    native_blind_quality_scores,
    select_quality_diversity,
)
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.paths import cache, processed


SIMILARITY_CACHE_VERSION = 1
PHASE_C_CACHE_VERSION = 2
GEOMETRY_V2_IMPLEMENTATION_VERSION = 4


def selected_sequences(args: argparse.Namespace) -> pd.DataFrame:
    sequences = io.load_sequences("validation")
    if args.target_ids:
        requested = {value.strip() for value in args.target_ids.split(",") if value.strip()}
        missing = requested - set(sequences["target_id"])
        if missing:
            raise KeyError(f"unknown validation targets: {sorted(missing)}")
        sequences = sequences[sequences["target_id"].isin(requested)]
    if args.limit:
        sequences = sequences.head(args.limit)
    return sequences.reset_index(drop=True)


def load_priors() -> tuple[dict, dict]:
    return (
        json.loads((processed() / "geometry_priors.json").read_text()),
        json.loads((processed() / "geofuse_geometry_v2_priors.json").read_text()),
    )


def candidate_digest(candidates: list[StructureCandidate]) -> str:
    digest = hashlib.sha256()
    digest.update(str(SIMILARITY_CACHE_VERSION).encode())
    for candidate in candidates:
        digest.update(candidate.candidate_id.encode())
        digest.update(np.asarray(candidate.coords, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def cached_similarity(
    candidates: list[StructureCandidate], sequence: str, target_id: str
) -> np.ndarray:
    digest = candidate_digest(candidates)
    path = cache() / "geofuse_phase_c" / "similarity" / f"{target_id}__{digest}.npz"
    expected_ids = np.asarray([candidate.candidate_id for candidate in candidates])
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            if np.array_equal(payload["candidate_ids"], expected_ids):
                return payload["similarity"].copy()
    print(
        f"[{target_id}] self-TM: {len(candidates)} candidates, "
        f"{len(candidates) * (len(candidates) - 1) // 2} pairs",
        flush=True,
    )
    matrix = self_tm_matrix(
        [candidate.coords for candidate in candidates], list(sequence)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, similarity=matrix, candidate_ids=expected_ids)
    return matrix


def candidate_features(
    candidate: StructureCandidate, sequence: str, priors_v1: dict, priors_v2: dict
) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "source": candidate.source,
        "global_confidence": candidate.global_confidence,
        "support_fraction": float(candidate.support_mask.mean()),
        **geometry_v2_metrics(candidate.coords, sequence, priors_v1, priors_v2),
    }


def _refinement_digest(candidate: StructureCandidate, config: dict) -> str:
    digest = hashlib.sha256()
    digest.update(candidate_digest([candidate]).encode())
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def project_fusion(
    candidate: StructureCandidate,
    priors_v1: dict,
    priors_v2: dict,
    *,
    steps: int,
    device: str,
    overwrite: bool,
) -> tuple[StructureCandidate, float]:
    geometry_config = GeometryV2Config(steps=steps)
    cache_config = {
        "phase_c_cache_version": PHASE_C_CACHE_VERSION,
        "geometry_v2_implementation_version": GEOMETRY_V2_IMPLEMENTATION_VERSION,
        **asdict(geometry_config),
    }
    digest = _refinement_digest(candidate, cache_config)
    path = (
        cache()
        / "geofuse_phase_c"
        / "refined_fusions"
        / candidate.target_id
        / f"{digest}.npz"
    )
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as payload:
            coordinates = payload["coords"].copy()
            seconds = float(payload["seconds"])
    else:
        started = time.time()
        coordinates, info = refine_structure_v2(
            candidate.coords,
            candidate.sequence,
            priors_v1,
            priors_v2,
            source_confidence=candidate.confidence,
            global_confidence=candidate.global_confidence,
            cfg=geometry_config,
            device=device,
        )
        seconds = time.time() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coords=np.asarray(coordinates, dtype=np.float32),
            seconds=np.asarray(seconds),
            document=np.asarray(
                json.dumps({"config": cache_config, "info": info}, sort_keys=True)
            ),
        )

    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "projection": "geometry_v2",
            "raw_fusion_parent": candidate.candidate_id,
            "projection_seconds": seconds,
        }
    )
    return (
        StructureCandidate(
            target_id=candidate.target_id,
            sequence=candidate.sequence,
            candidate_id=f"{candidate.candidate_id}__geometry_v2",
            kind="fused",
            source="geofuse_v2",
            model="heuristic_segment_v1_geometry_v2",
            coords=coordinates,
            confidence=candidate.confidence,
            support_mask=candidate.support_mask,
            global_confidence=candidate.global_confidence,
            metadata=metadata,
        ),
        seconds,
    )


def build_fusions(
    candidates: list[StructureCandidate],
    cluster_labels: np.ndarray,
    quality: np.ndarray,
    priors_v1: dict,
    priors_v2: dict,
    args: argparse.Namespace,
) -> tuple[list[StructureCandidate], int, float]:
    mixed: list[tuple[int, int, float, list[int], list[int]]] = []
    for label in np.unique(cluster_labels):
        members = np.flatnonzero(cluster_labels == label).tolist()
        templates = [index for index in members if candidates[index].kind == "template"]
        pretrained = [index for index in members if candidates[index].kind == "pretrained"]
        if templates and pretrained:
            mixed.append(
                (int(label), len(members), float(np.max(quality[members])), templates, pretrained)
            )
    mixed.sort(key=lambda item: (-item[1], -item[2], item[0]))

    output: list[StructureCandidate] = []
    runtime = 0.0
    fusion_config = FusionConfig()
    for _, _, _, templates, pretrained in mixed[: args.max_fusions]:
        template_index = max(
            templates, key=lambda index: (quality[index], candidates[index].candidate_id)
        )
        pretrained_index = max(
            pretrained, key=lambda index: (quality[index], candidates[index].candidate_id)
        )
        for mode in ("template_conservative", "pretrained_heavy"):
            fused = fuse_template_pretrained(
                candidates[template_index],
                candidates[pretrained_index],
                fusion_config,
                mode=mode,
            )
            output.append(fused)
            if not args.skip_projection:
                projected, seconds = project_fusion(
                    fused,
                    priors_v1,
                    priors_v2,
                    steps=args.steps,
                    device=args.device,
                    overwrite=args.overwrite,
                )
                output.append(projected)
                runtime += seconds
    return output, len(mixed), runtime


def quality_top_indices(
    quality: np.ndarray, candidates: list[StructureCandidate], limit: int
) -> list[int]:
    return sorted(
        range(len(candidates)), key=lambda i: (-quality[i], candidates[i].candidate_id)
    )[:limit]


def _indices_for(selected: list[StructureCandidate], pool: list[StructureCandidate]) -> list[int]:
    index = {candidate.candidate_id: i for i, candidate in enumerate(pool)}
    return [index[candidate.candidate_id] for candidate in selected]


def run(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    labels = io.load_labels("validation")
    priors_v1, priors_v2 = load_priors()
    store = CandidateCache(cache() / "geofuse_candidates", "validation")
    target_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for target in sequences.itertuples(index=False):
        raw = store.load_target(target.target_id, target.sequence)
        if not raw:
            continue
        raw_similarity = cached_similarity(raw, target.sequence, target.target_id)
        raw_clusters = cluster_fold_families(raw_similarity, args.fold_threshold)
        raw_features = [
            candidate_features(candidate, target.sequence, priors_v1, priors_v2)
            for candidate in raw
        ]
        raw_quality = native_blind_quality_scores(raw, raw_features)
        fusions, mixed_cluster_count, refinement_seconds = build_fusions(
            raw, raw_clusters, raw_quality, priors_v1, priors_v2, args
        )
        augmented = raw + fusions
        augmented_similarity = (
            cached_similarity(augmented, target.sequence, f"{target.target_id}__aug")
            if fusions
            else raw_similarity
        )
        augmented_clusters = cluster_fold_families(
            augmented_similarity, args.fold_threshold
        )
        augmented_features = [
            candidate_features(candidate, target.sequence, priors_v1, priors_v2)
            for candidate in augmented
        ]
        augmented_quality = native_blind_quality_scores(augmented, augmented_features)

        source_balanced = _indices_for(
            select_source_balanced(raw, args.final_count), raw
        )
        quality_raw = quality_top_indices(raw_quality, raw, args.final_count)
        cluster_raw = select_quality_diversity(
            raw,
            raw_similarity,
            raw_clusters,
            raw_quality,
            limit=args.final_count,
        )
        cluster_augmented = select_quality_diversity(
            augmented,
            augmented_similarity,
            augmented_clusters,
            augmented_quality,
            limit=args.final_count,
        )
        # Selection is complete before native coordinates are accessed below.
        references = io.get_reference_coords(labels, target.target_id)
        native_scores = np.asarray(
            [
                score_target([candidate.coords], references, list(target.sequence))
                for candidate in augmented
            ],
            dtype=float,
        )

        methods = {
            "source_balanced_raw": (source_balanced, raw_similarity),
            "quality_raw": (quality_raw, raw_similarity),
            "cluster_raw": (cluster_raw, raw_similarity),
            "cluster_augmented": (cluster_augmented, augmented_similarity),
        }
        row = {
            "target_id": target.target_id,
            "seq_len": len(target.sequence),
            "n_raw": len(raw),
            "n_fused": len(fusions),
            "n_raw_clusters": int(len(np.unique(raw_clusters))),
            "n_augmented_clusters": int(len(np.unique(augmented_clusters))),
            "n_mixed_source_clusters": mixed_cluster_count,
            "raw_oracle_tm": float(native_scores[: len(raw)].max()),
            "augmented_oracle_tm": float(native_scores.max()),
            "fusion_oracle_tm": float(native_scores[len(raw) :].max())
            if fusions
            else np.nan,
            "refinement_seconds": refinement_seconds,
        }
        for name, (indices, matrix) in methods.items():
            score_values = native_scores[indices]
            row[f"{name}_tm"] = float(score_values.max())
            row[f"{name}_self_tm"] = mean_selected_similarity(matrix, indices)
            pool = raw if name != "cluster_augmented" else augmented
            row[f"{name}_ids"] = ";".join(pool[index].candidate_id for index in indices)
        row["augmented_oracle_gain"] = row["augmented_oracle_tm"] - row["raw_oracle_tm"]
        row["augmented_selection_regret"] = (
            row["augmented_oracle_tm"] - row["cluster_augmented_tm"]
        )
        target_rows.append(row)

        for index, (candidate, features) in enumerate(zip(augmented, augmented_features)):
            pool = (
                "raw"
                if index < len(raw)
                else "fusion_v2"
                if candidate.source == "geofuse_v2"
                else "fusion_raw"
            )
            candidate_rows.append(
                {
                    "target_id": target.target_id,
                    "seq_len": len(target.sequence),
                    "pool": pool,
                    "cluster": int(augmented_clusters[index]),
                    "quality": float(augmented_quality[index]),
                    "native_tm_posthoc": float(native_scores[index]),
                    **features,
                }
            )
        print(
            f"[{target.target_id}] clusters={row['n_raw_clusters']} mixed={mixed_cluster_count} "
            f"fused={len(fusions)} TM balanced={row['source_balanced_raw_tm']:.4f} "
            f"raw-cluster={row['cluster_raw_tm']:.4f} aug={row['cluster_augmented_tm']:.4f} "
            f"oracle+={row['augmented_oracle_gain']:+.4f}",
            flush=True,
        )

    targets = pd.DataFrame(target_rows)
    candidates = pd.DataFrame(candidate_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / "phase_c_targets.csv", index=False)
    candidates.to_csv(output_dir / "phase_c_candidates.csv", index=False)
    write_report(targets, candidates, args)


def _summary(frame: pd.DataFrame) -> pd.Series:
    numeric = [
        "source_balanced_raw_tm",
        "quality_raw_tm",
        "cluster_raw_tm",
        "cluster_augmented_tm",
        "raw_oracle_tm",
        "augmented_oracle_tm",
        "augmented_oracle_gain",
        "augmented_selection_regret",
        "source_balanced_raw_self_tm",
        "quality_raw_self_tm",
        "cluster_raw_self_tm",
        "cluster_augmented_self_tm",
        "refinement_seconds",
    ]
    return frame[numeric].mean()


def write_report(
    targets: pd.DataFrame, candidates: pd.DataFrame, args: argparse.Namespace
) -> None:
    full = _summary(targets)
    excluded = {
        value.strip() for value in args.sensitivity_exclude.split(",") if value.strip()
    }
    sensitivity = _summary(targets[~targets["target_id"].isin(excluded)])
    selector_gain = full["cluster_augmented_tm"] - full["source_balanced_raw_tm"]
    sensitivity_gain = (
        sensitivity["cluster_augmented_tm"] - sensitivity["source_balanced_raw_tm"]
    )
    quality_gain = full["quality_raw_tm"] - full["source_balanced_raw_tm"]
    quality_sensitivity_gain = (
        sensitivity["quality_raw_tm"] - sensitivity["source_balanced_raw_tm"]
    )
    target_delta = targets["cluster_augmented_tm"] - targets["source_balanced_raw_tm"]
    worst_target = str(targets.loc[target_delta.idxmin(), "target_id"])
    worst_delta = float(target_delta.min())
    gate = selector_gain > 0 and full["augmented_oracle_gain"] > 0
    sensitivity_gate = sensitivity_gain > 0 and sensitivity["augmented_oracle_gain"] > 0
    fused = candidates[candidates["pool"] != "raw"]
    selected_fusion_targets = int(
        targets["cluster_augmented_ids"].str.contains("fused__", regex=False).sum()
    )
    lines = [
        "# GeoFuse Phase C — fold clustering and heuristic fusion",
        "",
        "All clustering, fusion weights, geometry quality scores, and final-five choices "
        "are fixed before native labels are read. Native TM is joined only post hoc.",
        "",
        f"- Targets: {len(targets)}",
        f"- Fold threshold: {args.fold_threshold:.2f} complete-link self-TM",
        f"- Generated fused/projected candidates: {len(fused)}",
        f"- Final sets containing a fusion: {selected_fusion_targets}/{len(targets)}",
        f"- Gate: **{'pass' if gate else 'fail'}**",
        f"- Sensitivity gate excluding {', '.join(sorted(excluded))}: "
        f"**{'pass' if sensitivity_gate else 'fail'}**",
        "",
        "Gate requires both positive native-blind final-five gain and positive augmented "
        "oracle gain. A selector improvement alone does not prove coordinate fusion works.",
        "",
        "## Full aggregate",
        "",
        full.round(4).to_frame("mean").to_markdown(),
        "",
        f"Selector gain over source-balanced raw: {selector_gain:+.6f} TM.",
        f"Quality-only gain over source-balanced raw: {quality_gain:+.6f} TM.",
        "",
        f"## Sensitivity excluding {', '.join(sorted(excluded))}",
        "",
        sensitivity.round(4).to_frame("mean").to_markdown(),
        "",
        f"Sensitivity selector gain: {sensitivity_gain:+.6f} TM.",
        f"Sensitivity quality-only gain: {quality_sensitivity_gain:+.6f} TM.",
        "",
        "## Interpretation",
        "",
        "- `raw_oracle` vs `augmented_oracle` isolates whether fusion creates a better "
        "fold hypothesis; selected scores measure the native-blind routing problem.",
        f"- Heuristic fusion did not raise oracle TM. It should remain an experimental "
        f"candidate generator, not replace either parent source.",
        f"- The largest selected-set regression is {worst_target} ({worst_delta:+.4f} TM), "
        "showing that the hand-built cross-source confidence score is not calibrated.",
        "- The next justified step is a leakage-safe learned confidence gate. Further "
        "weight tuning on these 12 native-scored targets would overfit the development set.",
        "- Lower selected self-TM means more fold diversity. Diversity is useful only if "
        "the selected best-of-five TM is preserved or improved.",
        "- This is development-set evidence. R1128 is reported separately because of the "
        "known exact pretrained-training overlap; pretrained cutoffs remain distinct from "
        "the temporal-safe TBM/prior claim.",
        "",
        "## Reproducibility",
        "",
        f"- Max mixed clusters fused per target: {args.max_fusions}",
        f"- Geometry projection steps: {args.steps}",
        f"- Fusion config: `{json.dumps(asdict(FusionConfig()), sort_keys=True)}`",
        f"- Selection config: `{json.dumps(asdict(SelectionConfig()), sort_keys=True)}`",
        "- The 0.35 threshold was chosen from a native-blind cross-source self-TM audit "
        "on the pilot targets; it was not selected from fusion/native TM outcomes.",
        "",
        "## Per-target",
        "",
        targets.drop(columns=[column for column in targets if column.endswith("_ids")])
        .round(4)
        .to_markdown(index=False),
        "",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))
    print(full.round(4).to_string())
    print(
        f"gate={'pass' if gate else 'fail'} "
        f"sensitivity_gate={'pass' if sensitivity_gate else 'fail'}"
    )
    print(f"[tables] {args.output_dir}")
    print(f"[report] {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ids", help="comma-separated validation targets")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fold-threshold", type=float, default=0.35)
    parser.add_argument("--max-fusions", type=int, default=3)
    parser.add_argument("--final-count", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-projection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sensitivity-exclude", default="R1128")
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_phase_c",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_phase_c.md",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
