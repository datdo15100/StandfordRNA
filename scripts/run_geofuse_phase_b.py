#!/usr/bin/env python
"""Diagnose and ablate GeoFuse Geometry v2 on the normalized candidate bank."""
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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.eval.usalign import score_target
from rna3d.geofuse.candidate import CandidateCache, StructureCandidate, safe_name
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.phase_a import select_source_balanced
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.paths import cache, processed
from rna3d.refine.optimizer import RefineConfig, refine_structure


FEATURES = [
    "global_confidence",
    "mean_residue_confidence",
    "support_fraction",
    "clash_per_res",
    "bb_dev",
    "rg_err",
    "sharp_kinks",
    "angle_nll",
    "torsion_nll",
    "pair_like_fraction",
]
GEOMETRY_FEATURES = [
    "clash_per_res",
    "bb_dev",
    "rg_err",
    "sharp_kinks",
    "angle_nll",
    "torsion_nll",
]
REFINER_IMPLEMENTATION_VERSION = {"v1": 1, "v2": 4}


def load_priors() -> tuple[dict, dict]:
    return (
        json.loads((processed() / "geometry_priors.json").read_text()),
        json.loads((processed() / "geofuse_geometry_v2_priors.json").read_text()),
    )


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


def candidate_features(
    candidate: StructureCandidate, sequence: str, priors_v1: dict, priors_v2: dict
) -> dict:
    geometry = geometry_v2_metrics(candidate.coords, sequence, priors_v1, priors_v2)
    return {
        "target_id": candidate.target_id,
        "seq_len": len(sequence),
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "source": candidate.source,
        "model": candidate.model,
        "global_confidence": candidate.global_confidence,
        "mean_residue_confidence": float(candidate.confidence.mean()),
        "min_residue_confidence": float(candidate.confidence.min()),
        "support_fraction": float(candidate.support_mask.mean()),
        **geometry,
    }


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    centered = frame.copy()
    centered["tm_centered"] = frame["tm_score"] - frame.groupby("target_id")[
        "tm_score"
    ].transform("mean")
    for feature in FEATURES:
        finite = np.isfinite(frame[feature]) & np.isfinite(frame["tm_score"])
        pooled = spearmanr(frame.loc[finite, feature], frame.loc[finite, "tm_score"])
        centered_feature = frame[feature] - frame.groupby("target_id")[feature].transform("mean")
        centered_finite = np.isfinite(centered_feature) & np.isfinite(centered["tm_centered"])
        target_centered = spearmanr(
            centered_feature[centered_finite], centered.loc[centered_finite, "tm_centered"]
        )
        within = []
        for _, target in frame.groupby("target_id"):
            if target[feature].nunique() > 1 and target["tm_score"].nunique() > 1:
                value = spearmanr(target[feature], target["tm_score"]).statistic
                if np.isfinite(value):
                    within.append(float(value))
        rows.append(
            {
                "feature": feature,
                "n": int(finite.sum()),
                "pooled_rho": float(pooled.statistic),
                "pooled_p": float(pooled.pvalue),
                "target_centered_rho": float(target_centered.statistic),
                "mean_within_target_rho": float(np.mean(within)) if within else np.nan,
                "n_within_targets": len(within),
            }
        )
    return pd.DataFrame(rows)


def _safe_rho(x: pd.Series, y: pd.Series) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 4 or x[finite].nunique() < 2 or y[finite].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _source_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in frame.groupby("source"):
        if len(group) < 5:
            continue
        for feature in FEATURES:
            centered_feature = group[feature] - group.groupby("target_id")[feature].transform(
                "mean"
            )
            centered_tm = group["tm_score"] - group.groupby("target_id")[
                "tm_score"
            ].transform("mean")
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "n": len(group),
                    "pooled_rho": _safe_rho(group[feature], group["tm_score"]),
                    "target_centered_rho": _safe_rho(centered_feature, centered_tm),
                }
            )
    return pd.DataFrame(rows)


def diagnose(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    labels = io.load_labels("validation")
    priors_v1, priors_v2 = load_priors()
    store = CandidateCache(cache() / "geofuse_candidates", "validation")
    rows = []
    for target in sequences.itertuples(index=False):
        references = io.get_reference_coords(labels, target.target_id)
        candidates = store.load_target(target.target_id, target.sequence)
        for candidate in candidates:
            row = candidate_features(candidate, target.sequence, priors_v1, priors_v2)
            row["tm_score"] = float(
                score_target([candidate.coords], references, list(target.sequence))
            )
            rows.append(row)
        print(f"[{target.target_id}] diagnosed {len(candidates)} candidates")
    frame = pd.DataFrame(rows)
    correlations = _correlations(frame)
    excluded = {
        value.strip() for value in args.sensitivity_exclude.split(",") if value.strip()
    }
    sensitivity_frame = frame[~frame["target_id"].isin(excluded)]
    sensitivity_correlations = _correlations(sensitivity_frame)
    source_correlations = _source_correlations(frame)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "candidate_geometry.csv", index=False)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)
    sensitivity_correlations.to_csv(
        output_dir / "feature_correlations_sensitivity.csv", index=False
    )
    source_correlations.to_csv(output_dir / "feature_correlations_by_source.csv", index=False)

    useful = correlations.sort_values("target_centered_rho", ascending=False)
    lookup = correlations.set_index("feature")
    sensitivity_lookup = sensitivity_correlations.set_index("feature")
    lines = [
        "# GeoFuse Phase B — native-blind geometry diagnostics",
        "",
        f"Candidates: {len(frame)} across {frame['target_id'].nunique()} validation targets.",
        "Native TM-score is joined only for this post-hoc signal audit; none of the "
        "features reads validation labels.",
        "",
        "Positive rho means larger feature values associate with better TM. Geometry "
        "violations are expected to have negative rho if they are useful for routing.",
        "Target-centered rho removes between-target difficulty/length effects.",
        "",
        useful.round(4).to_markdown(index=False),
        "",
        f"## Sensitivity excluding {', '.join(sorted(excluded)) or 'nothing'}",
        "",
        sensitivity_correlations.sort_values("target_centered_rho", ascending=False)
        .round(4)
        .to_markdown(index=False),
        "",
        "## Source-specific correlations",
        "",
        "These expose whether a feature ranks candidates within a generator or mainly "
        "calibrates differences between generators.",
        "",
        source_correlations.round(4).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- Angle NLL is the strongest violation signal after target centering "
        f"(rho={lookup.loc['angle_nll', 'target_centered_rho']:.3f}; excluding the "
        f"overlap target: {sensitivity_lookup.loc['angle_nll', 'target_centered_rho']:.3f}).",
        f"- Pair-like fraction has the strongest positive association "
        f"(rho={lookup.loc['pair_like_fraction', 'target_centered_rho']:.3f}), but it "
        "is a candidate-derived topology proxy, not true secondary-structure accuracy.",
        "- Source-specific target-centered correlations are weaker or can reverse. "
        "These features currently calibrate heterogeneous generators better than they "
        "rank samples from one generator.",
        "- Raw confidence is not calibrated across sources, so it is unsafe as a single "
        "whole-bank ranking score.",
        "",
        "This table is a gate, not a trained selector. Features with weak or reversed "
        "within-target signal must not be assigned large heuristic routing weights.",
        "",
    ]
    Path(args.report).write_text("\n".join(lines))
    print(correlations.round(4).to_string(index=False))
    print(f"[tables] {output_dir}")
    print(f"[report] {args.report}")


def _config_digest(name: str, config: dict) -> str:
    document = json.dumps({"method": name, **config}, sort_keys=True)
    return hashlib.sha256(document.encode()).hexdigest()[:12]


def _cached_refinement(
    candidate: StructureCandidate,
    sequence: str,
    method: str,
    priors_v1: dict,
    priors_v2: dict,
    steps: int,
    device: str,
    overwrite: bool,
) -> tuple[np.ndarray, float]:
    if method == "v1":
        config = asdict(RefineConfig(steps=steps))
    elif method == "v2":
        config = asdict(GeometryV2Config(steps=steps))
    else:
        raise ValueError(method)
    cache_config = {
        "implementation_version": REFINER_IMPLEMENTATION_VERSION[method],
        **config,
    }
    digest = _config_digest(method, cache_config)
    path = (
        cache()
        / "geofuse_phase_b"
        / candidate.target_id
        / f"{safe_name(candidate.candidate_id)}__{method}__{digest}.npz"
    )
    if path.exists() and not overwrite:
        with np.load(path, allow_pickle=False) as payload:
            return payload["coords"].copy(), float(payload["seconds"])

    started = time.time()
    if method == "v1":
        coords, info = refine_structure(
            candidate.coords,
            priors_v1,
            template_coords=candidate.coords,
            conf_residue=candidate.confidence,
            template_confidence=candidate.global_confidence,
            cfg=RefineConfig(steps=steps),
            device=device,
        )
    else:
        coords, info = refine_structure_v2(
            candidate.coords,
            sequence,
            priors_v1,
            priors_v2,
            source_confidence=candidate.confidence,
            global_confidence=candidate.global_confidence,
            cfg=GeometryV2Config(steps=steps),
            device=device,
        )
    seconds = time.time() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(
        {"method": method, "config": cache_config, "info": info}, sort_keys=True
    )
    np.savez_compressed(
        path,
        coords=np.asarray(coords, dtype=np.float32),
        seconds=np.asarray(seconds),
        document=np.asarray(document),
    )
    return coords, seconds


def _mean_metrics(
    candidates: list[StructureCandidate],
    coordinates: list[np.ndarray],
    sequence: str,
    priors_v1: dict,
    priors_v2: dict,
) -> dict:
    metrics = [geometry_v2_metrics(x, sequence, priors_v1, priors_v2) for x in coordinates]
    output = {
        name: float(np.mean([row[name] for row in metrics]))
        for name in metrics[0]
    }
    displacements = [
        np.linalg.norm(x - candidate.coords, axis=1)
        for candidate, x in zip(candidates, coordinates)
    ]
    output["mean_drift"] = float(np.mean([value.mean() for value in displacements]))
    output["confidence_weighted_drift"] = float(
        np.mean(
            [
                np.average(value, weights=np.maximum(candidate.confidence, 1e-3))
                for candidate, value in zip(candidates, displacements)
            ]
        )
    )
    return output


def ablate(args: argparse.Namespace) -> None:
    sequences = selected_sequences(args)
    labels = io.load_labels("validation")
    priors_v1, priors_v2 = load_priors()
    store = CandidateCache(cache() / "geofuse_candidates", "validation")
    rows = []
    for target in sequences.itertuples(index=False):
        candidates = select_source_balanced(
            store.load_target(target.target_id, target.sequence), args.candidates
        )
        if not candidates:
            continue
        coordinates = {"raw": [candidate.coords for candidate in candidates]}
        runtimes = {"raw": 0.0}
        for method in ("v1", "v2"):
            outputs = [
                _cached_refinement(
                    candidate,
                    target.sequence,
                    method,
                    priors_v1,
                    priors_v2,
                    args.steps,
                    args.device,
                    args.overwrite,
                )
                for candidate in candidates
            ]
            coordinates[method] = [value[0] for value in outputs]
            runtimes[method] = sum(value[1] for value in outputs)

        references = io.get_reference_coords(labels, target.target_id)
        for method, structures in coordinates.items():
            row = {
                "target_id": target.target_id,
                "seq_len": len(target.sequence),
                "setting": method,
                "n_candidates": len(candidates),
                "best5_tm": float(score_target(structures, references, list(target.sequence))),
                "runtime_seconds": runtimes[method],
                **_mean_metrics(
                    candidates, structures, target.sequence, priors_v1, priors_v2
                ),
            }
            rows.append(row)
        recent = {row["setting"]: row for row in rows[-3:]}
        print(
            f"[{target.target_id}] TM raw={recent['raw']['best5_tm']:.4f} "
            f"v1={recent['v1']['best5_tm']:.4f} v2={recent['v2']['best5_tm']:.4f}; "
            f"kink {recent['raw']['sharp_kinks']:.3f}->{recent['v2']['sharp_kinks']:.3f}"
        )

    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "geometry_v2_ablation.csv", index=False)
    columns = [
        "best5_tm",
        "clash_per_res",
        "bb_dev",
        "rg_err",
        "sharp_kinks",
        "angle_nll",
        "torsion_nll",
        "mean_drift",
        "runtime_seconds",
    ]

    def aggregate_for(data: pd.DataFrame) -> pd.DataFrame:
        return data.groupby("setting")[columns].mean().reindex(["raw", "v1", "v2"])

    def gate(result: pd.DataFrame) -> bool:
        raw = result.loc["raw"]
        v2 = result.loc["v2"]
        return bool(
            v2["best5_tm"] >= raw["best5_tm"] - args.tm_tolerance
            and v2["sharp_kinks"] <= raw["sharp_kinks"] + 1e-8
            and v2["angle_nll"] < raw["angle_nll"]
            and v2["torsion_nll"] < raw["torsion_nll"]
        )

    aggregate = aggregate_for(frame)
    passed = gate(aggregate)
    excluded = {
        value.strip() for value in args.sensitivity_exclude.split(",") if value.strip()
    }
    sensitivity = aggregate_for(frame[~frame["target_id"].isin(excluded)])
    sensitivity_passed = gate(sensitivity)
    tm_by_target = frame.pivot(index="target_id", columns="setting", values="best5_tm")
    tm_delta = tm_by_target["v2"] - tm_by_target["raw"]
    improved = int((tm_delta > 0).sum())
    regressed = int((tm_delta < 0).sum())
    material_regressions = tm_delta[tm_delta < -args.tm_tolerance].sort_values()
    config_document = json.dumps(asdict(GeometryV2Config(steps=args.steps)), sort_keys=True)
    lines = [
        "# GeoFuse Phase B — Geometry v2 ablation",
        "",
        "The same native-blind source-balanced candidate set is used for raw, "
        "gradient-v1 and geometry-v2. Hyperparameters are fixed before native TM scoring.",
        "",
        f"- Targets: {frame['target_id'].nunique()}",
        f"- Candidates per target: up to {args.candidates}",
        f"- Optimization steps: {args.steps}",
        f"- Gate: **{'pass' if passed else 'fail'}**",
        f"- TM preservation tolerance: {args.tm_tolerance:.4f}",
        "",
        aggregate.round(4).to_markdown(),
        "",
        "Gate requires v2 to preserve mean best-of-5 TM within tolerance, avoid the "
        "raw sharp-kink regression, and improve both empirical angle and signed-torsion NLL.",
        "",
        f"## Sensitivity excluding {', '.join(sorted(excluded)) or 'nothing'}",
        "",
        f"Sensitivity gate: **{'pass' if sensitivity_passed else 'fail'}**",
        "",
        sensitivity.round(4).to_markdown(),
        "",
        "## Interpretation",
        "",
        f"- Mean TM delta (v2 - raw): "
        f"{aggregate.loc['v2', 'best5_tm'] - aggregate.loc['raw', 'best5_tm']:+.6f}.",
        f"- Per target: {improved} improved and {regressed} regressed; material "
        f"regressions beyond the {args.tm_tolerance:.3f} tolerance: "
        f"{', '.join(material_regressions.index) if len(material_regressions) else 'none'}.",
        "- Geometry v2 is a safer projection than v1, but it is not an accuracy gain by "
        "itself. Raw and projected candidates should both remain available to the "
        "Phase-C selector; refinement must be gated rather than applied universally.",
        "- Pilot weights were engineered on R1107/R1116/R1156 geometry behavior. The "
        "full table is development-set evidence, not a final untouched test result.",
        "",
        "Sharp-kink, angle NLL and torsion NLL are explicit v2 objectives, so they are "
        "optimization endpoints rather than independent validation. Native TM is never "
        "read by the refiner and remains the independent preservation check.",
        "",
        "## Reproducibility",
        "",
        f"`{config_document}`",
        "",
        "## Per-target",
        "",
        frame.round(4).to_markdown(index=False),
        "",
    ]
    Path(args.report).write_text("\n".join(lines))
    print(aggregate.round(4).to_string())
    print(f"gate={'pass' if passed else 'fail'}")
    print(f"sensitivity_gate={'pass' if sensitivity_passed else 'fail'}")
    print(f"[tables] {output_dir}")
    print(f"[report] {args.report}")


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-ids", help="comma-separated validation targets")
    parser.add_argument("--limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    diagnosis = commands.add_parser("diagnose")
    common(diagnosis)
    diagnosis.add_argument(
        "--output-dir", default=REPO_ROOT / "reports" / "tables" / "geofuse_phase_b"
    )
    diagnosis.add_argument(
        "--report",
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_phase_b_diagnostics.md",
    )
    diagnosis.add_argument(
        "--sensitivity-exclude",
        default="R1128",
        help="comma-separated exact-overlap targets excluded in the sensitivity table",
    )
    diagnosis.set_defaults(func=diagnose)

    ablation = commands.add_parser("ablate")
    common(ablation)
    ablation.add_argument("--candidates", type=int, default=5)
    ablation.add_argument("--steps", type=int, default=300)
    ablation.add_argument("--device", default="cuda")
    ablation.add_argument("--overwrite", action="store_true")
    ablation.add_argument("--tm-tolerance", type=float, default=0.005)
    ablation.add_argument(
        "--sensitivity-exclude",
        default="R1128",
        help="comma-separated exact-overlap targets excluded in sensitivity analysis",
    )
    ablation.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_phase_b_ablation",
    )
    ablation.add_argument(
        "--report",
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_phase_b_ablation.md",
    )
    ablation.set_defaults(func=ablate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
