#!/usr/bin/env python
"""Run frozen global/local clustering and selective-fusion ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.data import io
from rna3d.eval.local_metrics import c1_lddt
from rna3d.eval.statistics import paired_target_summary
from rna3d.eval.usalign import score_target
from rna3d.geofuse.candidate import CandidateCache, StructureCandidate
from rna3d.geofuse.phase_c import (
    cluster_fold_families,
    fuse_template_pretrained,
    native_blind_quality_scores,
    select_quality_diversity,
)
from rna3d.geofuse.phase_d import (
    load_gate_checkpoint,
    pair_gate_features,
    predict_pretrained_probability,
)
from rna3d.geofuse.real_oof import audit_pretrained_oof, audit_template_oof
from rna3d.geofuse.selective import (
    local_source_disagreement,
    oracle_source_fusion,
    selective_quality_fusion,
)
from rna3d.paths import cache, processed
from run_geofuse_phase_c import (
    cached_similarity,
    candidate_features,
    load_priors,
    project_fusion,
)


VARIANTS = ("F0_raw", "F1_heuristic", "F2_quality", "F3_quality_geometry", "F4_oracle")


def _audited_candidates(row, store: CandidateCache) -> list[StructureCandidate]:
    excluded = set(str(row.excluded_pdb_ids).split(";")) - {"", "nan"}
    templates = []
    pretrained = []
    for candidate in store.load_target(row.target_id, row.sequence):
        try:
            if candidate.kind == "template":
                audit_template_oof(candidate, row.date, excluded)
                templates.append(candidate)
            elif candidate.kind == "pretrained":
                audit_pretrained_oof(candidate, row.date)
                pretrained.append(candidate)
        except ValueError:
            continue
    templates.sort(key=lambda value: (-value.global_confidence, value.candidate_id))
    pretrained.sort(key=lambda value: (-value.global_confidence, value.candidate_id))
    return templates[:3] + pretrained[:2]


def _load_router(selection_path: Path, manifest_path: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if selection.get("manifest_sha256") != digest:
        raise ValueError("quality-estimator selection was trained on another manifest")
    checkpoint_path = Path(selection["checkpoint"])
    if selection["model_name"] == "conv1d":
        checkpoint = load_gate_checkpoint(str(checkpoint_path))
        return {**selection, "checkpoint_data": checkpoint}
    checkpoint = joblib.load(checkpoint_path)
    if checkpoint.get("feature_names") is None:
        raise ValueError("invalid sklearn quality-estimator checkpoint")
    return {**selection, "checkpoint_data": checkpoint}


def _router_probability(
    router: dict,
    template: StructureCandidate,
    pretrained: StructureCandidate,
    priors_v1: dict,
    priors_v2: dict,
    device: str,
) -> np.ndarray:
    features, _, _ = pair_gate_features(template, pretrained, priors_v1, priors_v2)
    if router["model_name"] == "conv1d":
        checkpoint = router["checkpoint_data"]
        return predict_pretrained_probability(
            checkpoint["model"],
            features,
            np.asarray(checkpoint["feature_mean"], dtype=np.float32),
            np.asarray(checkpoint["feature_std"], dtype=np.float32),
            device=device,
        )
    return router["checkpoint_data"]["model"].predict_proba(features)[:, 1]


def _mixed_pairs(
    raw: list[StructureCandidate],
    clusters: np.ndarray,
    quality: np.ndarray,
) -> list[tuple[StructureCandidate, StructureCandidate, int]]:
    output = []
    for label in np.unique(clusters):
        members = np.flatnonzero(clusters == label)
        templates = [index for index in members if raw[index].kind == "template"]
        pretrained = [index for index in members if raw[index].kind == "pretrained"]
        if not templates or not pretrained:
            continue
        template_index = max(templates, key=lambda index: quality[index])
        pretrained_index = max(pretrained, key=lambda index: quality[index])
        output.append((raw[template_index], raw[pretrained_index], int(label)))
    return output


def _native_metrics(
    candidate: StructureCandidate, references: list[np.ndarray]
) -> tuple[float, float, int, np.ndarray]:
    tm = np.asarray(
        [
            score_target(
                [candidate.coords], [reference], list(candidate.sequence)
            )
            for reference in references
        ],
        dtype=float,
    )
    reference_index = int(np.nanargmax(tm))
    local = c1_lddt(candidate.coords, references[reference_index])
    return (
        float(tm[reference_index]),
        float(local["score"]),
        reference_index,
        np.asarray(local["per_residue"], dtype=float),
    )


def _oracle_for_pair(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    references: list[np.ndarray],
) -> StructureCandidate:
    template_values = [_native_metrics(template, [reference]) for reference in references]
    pretrained_values = [_native_metrics(pretrained, [reference]) for reference in references]
    reference_index = max(
        range(len(references)),
        key=lambda index: max(
            template_values[index][0], pretrained_values[index][0]
        ),
    )
    template_lddt = c1_lddt(
        template.coords, references[reference_index]
    )["per_residue"]
    pretrained_lddt = c1_lddt(
        pretrained.coords, references[reference_index]
    )["per_residue"]
    choice = np.nan_to_num(pretrained_lddt, nan=-1.0) > np.nan_to_num(
        template_lddt, nan=-1.0
    )
    return oracle_source_fusion(template, pretrained, choice)


def _evaluate_bank(
    target_id: str,
    sequence: str,
    variant: str,
    bank: list[StructureCandidate],
    fold_threshold: float,
    priors_v1: dict,
    priors_v2: dict,
    native: dict[str, tuple[float, float, int, np.ndarray]],
) -> dict:
    similarity = cached_similarity(bank, sequence, f"{target_id}__confirm__{variant}")
    clusters = cluster_fold_families(similarity, fold_threshold)
    features = [
        candidate_features(candidate, sequence, priors_v1, priors_v2)
        for candidate in bank
    ]
    quality = native_blind_quality_scores(bank, features)
    selected = select_quality_diversity(
        bank, similarity, clusters, quality, limit=5
    )
    tm_values = np.asarray([native[candidate.candidate_id][0] for candidate in bank])
    lddt_values = np.asarray([native[candidate.candidate_id][1] for candidate in bank])
    return {
        "variant": variant,
        "bank_size": len(bank),
        "bank_clusters": int(len(np.unique(clusters))),
        "selected_tm": float(tm_values[selected].max()),
        "selected_lddt": float(lddt_values[selected].max()),
        "oracle_tm": float(tm_values.max()),
        "oracle_lddt": float(lddt_values.max()),
        "selected_ids": ";".join(bank[index].candidate_id for index in selected),
    }


def _target_threshold(
    row,
    raw: list[StructureCandidate],
    fold_threshold: float,
    router: dict,
    priors_v1: dict,
    priors_v2: dict,
    references: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    raw_similarity = cached_similarity(raw, row.sequence, row.target_id)
    raw_clusters = cluster_fold_families(raw_similarity, fold_threshold)
    raw_features = [
        candidate_features(candidate, row.sequence, priors_v1, priors_v2)
        for candidate in raw
    ]
    raw_quality = native_blind_quality_scores(raw, raw_features)
    pairs = _mixed_pairs(raw, raw_clusters, raw_quality)
    heuristic: list[StructureCandidate] = []
    selective: list[StructureCandidate] = []
    projected: list[StructureCandidate] = []
    oracle: list[StructureCandidate] = []
    disagreements = []
    for template, pretrained, cluster in pairs[: args.max_fusions]:
        _, window_disagreement, alignment = local_source_disagreement(
            template, pretrained, window=args.disagreement_window
        )
        probability = _router_probability(
            router, template, pretrained, priors_v1, priors_v2, args.device
        )
        disagreements.append(
            {
                "target_id": row.target_id,
                "fold_threshold": fold_threshold,
                "cluster": cluster,
                "template_id": template.candidate_id,
                "pretrained_id": pretrained.candidate_id,
                "mean_window_disagreement": float(window_disagreement.mean()),
                "p90_window_disagreement": float(
                    np.quantile(window_disagreement, 0.90)
                ),
                "mean_pretrained_probability": float(probability.mean()),
                **alignment,
            }
        )
        for mode in ("template_conservative", "pretrained_heavy"):
            heuristic.append(
                fuse_template_pretrained(template, pretrained, mode=mode)
            )
        fused = selective_quality_fusion(
            template,
            pretrained,
            probability,
            decision_threshold=float(router["decision_threshold"]),
            probability_margin=args.probability_margin,
            minimum_segment=args.minimum_segment,
            disagreement_window=args.disagreement_window,
            minimum_disagreement=args.minimum_disagreement,
            maximum_disagreement=args.maximum_disagreement,
        )
        if fused is not None:
            selective.append(fused)
            refined, _ = project_fusion(
                fused,
                priors_v1,
                priors_v2,
                steps=args.steps,
                device=args.device,
                overwrite=False,
            )
            projected.append(refined)
        oracle.append(_oracle_for_pair(template, pretrained, references))

    banks = {
        "F0_raw": raw,
        "F1_heuristic": raw + heuristic,
        "F2_quality": raw + selective,
        "F3_quality_geometry": raw + projected,
        "F4_oracle": raw + oracle,
    }
    native = {}
    unique = {
        candidate.candidate_id: candidate
        for bank in banks.values()
        for candidate in bank
    }
    for candidate_id, candidate in unique.items():
        native[candidate_id] = _native_metrics(candidate, references)
    rows = []
    for variant in VARIANTS:
        result = _evaluate_bank(
            row.target_id,
            row.sequence,
            variant,
            banks[variant],
            fold_threshold,
            priors_v1,
            priors_v2,
            native,
        )
        result.update(
            {
                "target_id": row.target_id,
                "split": row.split,
                "fold_threshold": fold_threshold,
                "raw_clusters": int(len(np.unique(raw_clusters))),
                "mixed_clusters": len(pairs),
                "heuristic_candidates": len(heuristic),
                "selective_candidates": len(selective),
                "projected_candidates": len(projected),
                "oracle_candidates": len(oracle),
            }
        )
        rows.append(result)
    return rows, disagreements


def _variant_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["fold_threshold", "variant"])[
        ["selected_tm", "selected_lddt", "oracle_tm", "oracle_lddt", "mixed_clusters"]
    ].mean().reset_index()


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, dtype={"target_id": str})
    manifest = manifest[manifest["split"].isin(["calibration", "validation"])].copy()
    router = _load_router(Path(args.selection), manifest_path)
    priors_v1, priors_v2 = load_priors()
    store = CandidateCache(Path(args.cache_root), "train_v2")
    labels = io.load_labels("train_v2")
    calibration_rows = []
    disagreement_rows = []
    calibration = manifest[manifest["split"] == "calibration"].sort_values(
        ["date", "target_id"]
    )
    for row in calibration.itertuples(index=False):
        raw = _audited_candidates(row, store)
        if len(raw) < 2 or not {candidate.kind for candidate in raw} >= {
            "template",
            "pretrained",
        }:
            raise RuntimeError(f"{row.target_id}: audited candidate bank incomplete")
        references = io.get_reference_coords(labels, row.target_id)
        for threshold in args.fold_thresholds:
            target_rows, disagreements = _target_threshold(
                row,
                raw,
                threshold,
                router,
                priors_v1,
                priors_v2,
                references,
                args,
            )
            calibration_rows.extend(target_rows)
            disagreement_rows.extend(disagreements)
        print(f"[calibration] {row.target_id}", flush=True)
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_summary = _variant_summary(calibration_frame)
    f2 = calibration_summary[
        calibration_summary["variant"] == "F2_quality"
    ].sort_values(
        ["selected_lddt", "selected_tm", "fold_threshold"],
        ascending=[False, False, True],
    )
    selected_threshold = float(f2.iloc[0]["fold_threshold"])
    at_threshold = calibration_summary[
        calibration_summary["fold_threshold"] == selected_threshold
    ]
    selected_variant = str(
        at_threshold[
            at_threshold["variant"].isin(["F2_quality", "F3_quality_geometry"])
        ]
        .sort_values(
            ["selected_lddt", "selected_tm", "variant"],
            ascending=[False, False, True],
        )
        .iloc[0]["variant"]
    )
    print(
        f"[freeze] threshold={selected_threshold:.2f} variant={selected_variant}",
        flush=True,
    )

    validation_rows = []
    validation = manifest[manifest["split"] == "validation"].sort_values(
        ["date", "target_id"]
    )
    for row in validation.itertuples(index=False):
        raw = _audited_candidates(row, store)
        references = io.get_reference_coords(labels, row.target_id)
        target_rows, disagreements = _target_threshold(
            row,
            raw,
            selected_threshold,
            router,
            priors_v1,
            priors_v2,
            references,
            args,
        )
        validation_rows.extend(target_rows)
        disagreement_rows.extend(disagreements)
        print(f"[validation] {row.target_id}", flush=True)
    validation_frame = pd.DataFrame(validation_rows)
    validation_summary = _variant_summary(validation_frame)
    target_pivot_tm = validation_frame.pivot(
        index="target_id", columns="variant", values="selected_tm"
    )
    target_pivot_lddt = validation_frame.pivot(
        index="target_id", columns="variant", values="selected_lddt"
    )
    tm_bootstrap = paired_target_summary(
        target_pivot_tm["F0_raw"].to_numpy(),
        target_pivot_tm[selected_variant].to_numpy(),
        higher_is_better=True,
    )
    lddt_bootstrap = paired_target_summary(
        target_pivot_lddt["F0_raw"].to_numpy(),
        target_pivot_lddt[selected_variant].to_numpy(),
        higher_is_better=True,
    )
    validation_lookup = validation_summary.set_index("variant")
    oracle_gain = (
        validation_lookup.loc[selected_variant, "oracle_tm"]
        - validation_lookup.loc["F0_raw", "oracle_tm"]
    )
    selected_tm_delta = (
        validation_lookup.loc[selected_variant, "selected_tm"]
        - validation_lookup.loc["F0_raw", "selected_tm"]
    )
    selected_lddt_delta = (
        validation_lookup.loc[selected_variant, "selected_lddt"]
        - validation_lookup.loc["F0_raw", "selected_lddt"]
    )
    material = (
        (target_pivot_tm[selected_variant] - target_pivot_tm["F0_raw"] < -0.05)
        | (
            target_pivot_lddt[selected_variant]
            - target_pivot_lddt["F0_raw"]
            < -0.05
        )
    )
    passed = bool(
        oracle_gain > 0
        and selected_tm_delta >= 0
        and selected_lddt_delta >= 0
        and not material.any()
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibration_frame.to_csv(output / "calibration_targets.csv", index=False)
    calibration_summary.to_csv(output / "calibration_thresholds.csv", index=False)
    validation_frame.to_csv(output / "validation_targets.csv", index=False)
    validation_summary.to_csv(output / "validation_summary.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(
        output / "local_disagreement.csv", index=False
    )
    lines = [
        "# GeoFuse confirmatory clustering and selective fusion",
        "",
        "This combines E13 (global/local clustering) and E14 (selective fusion). "
        "Threshold and deployable variant are selected only on calibration. The "
        "newest 20 targets are evaluated once after freezing. Every augmented bank "
        "retains all raw parents.",
        "",
        f"- Quality estimator: {router['model_name']} ({router['supervision']})",
        f"- Router decision threshold: {router['decision_threshold']:.4f}",
        f"- Calibration-selected fold threshold: {selected_threshold:.2f}",
        f"- Calibration-selected deployable variant: **{selected_variant}**",
        f"- Confirmatory fusion gate: **{'pass' if passed else 'fail'}**",
        f"- Validation material regressions (>0.05 TM or lDDT): "
        f"{int(material.sum())}",
        "",
        "## Calibration: global-cluster threshold ablation",
        "",
        calibration_summary.round(6).to_markdown(index=False),
        "",
        "Threshold selection maximizes F2 selected C1′-lDDT, then selected TM. F2 "
        "versus F3 is then frozen by the same calibration ordering.",
        "",
        "## Final newest-target variants",
        "",
        validation_summary.round(6).to_markdown(index=False),
        "",
        "F0 is raw parents; F1 is the old heuristic; F2 is quality-gated fusion with "
        "abstention; F3 projects F2 with geometry v2; F4 reads native local lDDT and "
        "is a non-deployable upper bound.",
        "",
        f"- Selected TM delta over F0: {selected_tm_delta:+.6f}",
        f"- Selected C1′-lDDT delta over F0: {selected_lddt_delta:+.6f}",
        f"- Augmented oracle TM gain over F0: {oracle_gain:+.6f}",
        "",
        "## Target bootstrap versus F0",
        "",
        "### Selected TM",
        "",
        pd.Series(tm_bootstrap, name="value").to_frame().round(6).to_markdown(),
        "",
        "### Selected C1′-lDDT",
        "",
        pd.Series(lddt_bootstrap, name="value").to_frame().round(6).to_markdown(),
        "",
        "## Interpretation",
        "",
        (
            "- Selective fusion passes the frozen gate."
            if passed
            else "- Selective fusion fails the frozen gate. Raw parents remain the "
            "deployable choice; F4 quantifies headroom but is not a method."
        ),
        "- Local disagreement is descriptive and native-blind. A same-fold cluster "
        "does not imply that either source is locally correct.",
        "",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(validation_summary.round(6).to_string(index=False))
    print(f"gate={'pass' if passed else 'fail'}")
    print(f"[report] {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=processed() / "geofuse_real_oof_v2" / "medium_manifest.csv",
    )
    parser.add_argument(
        "--selection",
        default=processed() / "geofuse_quality_estimator_selection_v2.json",
    )
    parser.add_argument("--cache-root", default=cache() / "geofuse_candidates")
    parser.add_argument(
        "--fold-thresholds",
        type=float,
        nargs="+",
        default=[0.35, 0.45, 0.55],
    )
    parser.add_argument("--max-fusions", type=int, default=3)
    parser.add_argument("--probability-margin", type=float, default=0.15)
    parser.add_argument("--minimum-segment", type=int, default=7)
    parser.add_argument("--disagreement-window", type=int, default=15)
    parser.add_argument("--minimum-disagreement", type=float, default=0.5)
    parser.add_argument("--maximum-disagreement", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_confirmatory_fusion",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT
        / "reports"
        / "thesis_notes"
        / "geofuse_confirmatory_fusion.md",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
