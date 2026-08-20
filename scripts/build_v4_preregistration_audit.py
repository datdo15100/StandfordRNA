#!/usr/bin/env python
"""Build data-only audit artifacts for the V4 preregistration.

This script deliberately does not run a structure predictor and does not calculate
native performance.  It only reconstructs cohort counts and inventories target IDs
that already occur in historical result artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "thesis_v4" / "preregistration"
TRAIN_SEQUENCES = (
    REPO_ROOT / "data" / "stanford-rna-3d-folding" / "train_sequences.v2.csv"
)
TRAIN_LABELS = (
    REPO_ROOT / "data" / "stanford-rna-3d-folding" / "train_labels.v2.csv"
)
CASP_SEQUENCES = (
    REPO_ROOT / "data" / "stanford-rna-3d-folding" / "validation_sequences.csv"
)
FULL_MANIFEST = REPO_ROOT / "data" / "processed" / "geofuse_real_oof_v2" / "manifest.csv"
MEDIUM_MANIFEST = (
    REPO_ROOT / "data" / "processed" / "geofuse_real_oof_v2" / "medium_manifest.csv"
)
PILOT_MANIFEST = (
    REPO_ROOT / "data" / "processed" / "geofuse_real_oof_v2" / "pilot_manifest.csv"
)
FAMILY_CLUSTERS = (
    REPO_ROOT / "data" / "processed" / "geofuse_real_oof_v2" / "family80_cluster.tsv"
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_fraction_ids(requested: set[str]) -> set[str]:
    """Return targets having at least 3 residues and >=80% resolved C1'."""
    counts = {target_id: 0 for target_id in requested}
    resolved = {target_id: 0 for target_id in requested}
    for chunk in pd.read_csv(TRAIN_LABELS, usecols=["ID", "x_1"], chunksize=500_000):
        target = chunk["ID"].str.rsplit("_", n=1).str[0]
        mask = target.isin(requested)
        if not mask.any():
            continue
        selected = pd.DataFrame(
            {
                "target_id": target[mask].to_numpy(),
                "resolved": chunk.loc[mask, "x_1"].to_numpy(float) > -1e17,
            }
        )
        for target_id, group in selected.groupby("target_id"):
            counts[target_id] += len(group)
            resolved[target_id] += int(group["resolved"].sum())
    return {
        target_id
        for target_id in requested
        if counts[target_id] >= 3
        and resolved[target_id] / counts[target_id] >= 0.8
    }


def result_artifact_targets() -> dict[str, set[str]]:
    """Map historical target IDs to tracked evidence files containing them."""
    evidence: dict[str, set[str]] = {}
    for path in sorted((REPO_ROOT / "reports" / "tables").rglob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
            if "target_id" not in header.columns:
                continue
            values = pd.read_csv(path, usecols=["target_id"])["target_id"]
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        for target_id in values.dropna().astype(str).unique():
            evidence.setdefault(target_id, set()).add(relative(path))
    return evidence


def add_reason(store: dict[str, set[str]], target_id: str, reason: str) -> None:
    store.setdefault(target_id, set()).add(reason)


def build_exposure_ledger(output_dir: Path) -> tuple[pd.DataFrame, dict]:
    train = pd.read_csv(TRAIN_SEQUENCES, dtype=str)
    train["seq_len"] = train["sequence"].str.len()
    casp = pd.read_csv(CASP_SEQUENCES, dtype=str)
    casp["seq_len"] = casp["sequence"].str.len()
    full = pd.read_csv(FULL_MANIFEST, dtype=str)
    medium = pd.read_csv(MEDIUM_MANIFEST, dtype=str)
    pilot = pd.read_csv(PILOT_MANIFEST, dtype=str)
    artifact_evidence = result_artifact_targets()

    metadata: dict[str, dict] = {}
    for source_name, frame in (("kaggle_train_v2", train), ("CASP15_validation", casp)):
        for row in frame.itertuples(index=False):
            metadata[str(row.target_id)] = {
                "sequence_length": int(row.seq_len),
                "temporal_cutoff": str(row.temporal_cutoff),
                "source_dataset": source_name,
            }
    full_by_id = full.set_index("target_id").to_dict("index")
    medium_by_id = medium.set_index("target_id").to_dict("index")
    pilot_by_id = pilot.set_index("target_id").to_dict("index")

    reasons: dict[str, set[str]] = {}
    native: dict[str, bool] = {}
    decision: dict[str, bool] = {}
    failure: dict[str, bool] = {}

    train_dates = pd.to_datetime(train["temporal_cutoff"], errors="coerce")
    p0_ids = set(
        train.loc[train_dates < pd.Timestamp("2022-05-27"), "target_id"].astype(str)
    )
    for target_id in p0_ids:
        add_reason(reasons, target_id, "P0 production geometry-prior training from native C1' coordinates")
        native[target_id] = True
        decision[target_id] = True

    for target_id in casp["target_id"].astype(str):
        add_reason(reasons, target_id, "CASP15 native scoring and repeated method analysis")
        native[target_id] = True
        decision[target_id] = True

    for target_id, row in medium_by_id.items():
        split = row["split"]
        if split == "train":
            add_reason(reasons, target_id, "60-RNA prior/gate training and native supervision")
        elif split == "calibration":
            add_reason(reasons, target_id, "20-RNA calibration and hyperparameter selection")
        else:
            add_reason(reasons, target_id, "20-RNA historical validation and per-target scoring")
        native[target_id] = True
        decision[target_id] = True

    medium_ids = set(medium_by_id)
    for target_id, row in pilot_by_id.items():
        if target_id not in medium_ids:
            add_reason(reasons, target_id, "earlier 15-RNA pilot native supervision/analysis")
            native[target_id] = True
            decision[target_id] = True

    for target_id, paths in artifact_evidence.items():
        add_reason(reasons, target_id, "appears in a historical result artifact")
        # All artifact-only IDs in the current repository occur in native-supervised
        # pilot tables.  Keep this conservative default explicit in the ledger.
        native[target_id] = True
        decision[target_id] = True

    # This target failed Arena and was replaced before native scoring.  The failure
    # nevertheless influenced a technical cohort decision, so it is not untouched.
    failed_target = "8YUR_X"
    add_reason(reasons, failed_target, "predictor/Arena failure inspected before technical replacement")
    native[failed_target] = False
    decision[failed_target] = True
    failure[failed_target] = True

    exposed_ids = sorted(reasons)
    rows = []
    for target_id in exposed_ids:
        base = metadata.get(target_id, {})
        historical = medium_by_id.get(target_id) or pilot_by_id.get(target_id) or full_by_id.get(target_id)
        evidence = set(artifact_evidence.get(target_id, set()))
        if target_id in casp["target_id"].astype(str).values:
            evidence.add(relative(CASP_SEQUENCES))
        if target_id in medium_by_id:
            evidence.add(relative(MEDIUM_MANIFEST))
        elif target_id in pilot_by_id:
            evidence.add(relative(PILOT_MANIFEST))
        elif target_id in full_by_id:
            evidence.add(relative(FULL_MANIFEST))
        if target_id == failed_target:
            evidence.add("reports/thesis_notes/geofuse_real_oof_medium_preparation.md")
            evidence.add("reports/thesis_notes/geofuse_experiment_log.md")
        if target_id in p0_ids:
            evidence.add(relative(TRAIN_SEQUENCES))
            evidence.add("data/processed/geometry_priors.json")
            evidence.add("data/processed/geofuse_geometry_v2_priors.json")
            evidence.add("scripts/run_phase2_priors.py")
            evidence.add("scripts/build_geofuse_geometry_v2_priors.py")
        rows.append(
            {
                "target_id": target_id,
                "sequence_length": base.get(
                    "sequence_length",
                    int(historical["seq_len"]) if historical is not None else "UNKNOWN",
                ),
                "temporal_cutoff": base.get(
                    "temporal_cutoff", historical["date"] if historical is not None else "UNKNOWN"
                ),
                "source_dataset": base.get("source_dataset", "UNKNOWN"),
                "historical_split": (
                    historical["split"]
                    if historical is not None
                    else (
                        "CASP15_DEVELOPMENT"
                        if base.get("source_dataset") == "CASP15_validation"
                        else "NOT_APPLICABLE"
                    )
                ),
                "mmseqs_sequence_similarity_cluster": (
                    historical["family_group"] if historical is not None else "UNKNOWN"
                ),
                "native_scored_or_supervised": bool(native.get(target_id, False)),
                "affected_tuning_or_method_decision": bool(decision.get(target_id, False)),
                "failure_case_inspected": bool(failure.get(target_id, False)),
                "exposure_status": "EXPOSED_DO_NOT_USE_AS_FINAL_UNTOUCHED",
                "exposure_reasons": " | ".join(sorted(reasons[target_id])),
                "evidence_paths": " | ".join(sorted(evidence)),
                "v4_final_eligible": False,
            }
        )
    ledger = pd.DataFrame(rows).sort_values("target_id")
    ledger.to_csv(output_dir / "development_exposure_ledger.csv", index=False)

    # The historical 354-row manifest is not the V4 universe: grouped temporal
    # splitting dropped 65 otherwise valid RNAs.  Rebuild the known-clean count
    # from all 419 valid members recorded by the pre-split MMseqs output.
    clusters = pd.read_csv(
        FAMILY_CLUSTERS,
        sep="\t",
        names=["representative", "member"],
        dtype=str,
    )
    exposed_later = set(ledger["target_id"].astype(str)) & set(clusters["member"])
    exposed_clusters = set(
        clusters.loc[clusters["member"].isin(exposed_later), "representative"]
    )
    cluster_disjoint = clusters[~clusters["representative"].isin(exposed_clusters)]
    historical_remaining = full[
        full["target_id"].isin(set(cluster_disjoint["member"]))
    ]
    summary = {
        "ledger_targets": int(len(ledger)),
        "p0_production_prior_training_targets": int(len(p0_ids)),
        "native_scored_or_supervised": int(ledger["native_scored_or_supervised"].sum()),
        "technical_failure_without_native_score": int(
            ((~ledger["native_scored_or_supervised"]) & ledger["failure_case_inspected"]).sum()
        ),
        "later_technical_valid_targets": int(clusters["member"].nunique()),
        "historical_post_split_manifest_targets": int(len(full)),
        "later_exposed_targets": int(len(exposed_later)),
        "later_exposed_mmseqs_clusters": int(len(exposed_clusters)),
        "later_targets_remaining_after_known_exposure_cluster_exclusion": int(
            cluster_disjoint["member"].nunique()
        ),
        "remaining_mmseqs_clusters": int(cluster_disjoint["representative"].nunique()),
        "remaining_targets_visible_in_historical_354_manifest": int(
            len(historical_remaining)
        ),
        "warning": (
            "This is a repository-artifact audit, not a final eligibility manifest. "
            "External/manual exposure and pretrained overlap remain to be audited."
        ),
    }
    (output_dir / "development_exposure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return ledger, summary


def build_data_flow(output_dir: Path, exposure_summary: dict) -> pd.DataFrame:
    sequences = pd.read_csv(TRAIN_SEQUENCES, dtype=str)
    dates = pd.to_datetime(sequences["temporal_cutoff"], errors="coerce")
    lengths = sequences["sequence"].str.len()
    p0 = sequences[dates < pd.Timestamp("2022-05-27")]
    later_date = sequences[dates > pd.Timestamp("2023-12-31")]
    later_bounded = later_date[dates.loc[later_date.index] <= pd.Timestamp("2025-03-26")]
    length_scope = later_bounded[lengths.loc[later_bounded.index].between(30, 400)]
    resolved = resolved_fraction_ids(set(length_scope["target_id"].astype(str)))
    historical_manifest = pd.read_csv(FULL_MANIFEST, dtype=str)
    medium = pd.read_csv(MEDIUM_MANIFEST, dtype=str)

    rows = [
        ("D0", "all", "-", len(sequences), "Kaggle train_sequences.v2.csv", "VERIFIED"),
        (
            "P0-input",
            "production-prior",
            "D0",
            len(p0),
            "temporal_cutoff < 2022-05-27",
            "VERIFIED",
        ),
        (
            "P0-artifact",
            "production-prior",
            "P0-input",
            3397,
            "current geometry_priors.json metadata; exact rebuild pending",
            "ARTIFACT_VERIFIED_REBUILD_PENDING",
        ),
        (
            "L1",
            "later-evaluation",
            "D0",
            len(later_date),
            "temporal_cutoff > 2023-12-31",
            "VERIFIED",
        ),
        (
            "L2",
            "later-evaluation",
            "L1",
            len(later_bounded),
            "temporal_cutoff <= 2025-03-26",
            "VERIFIED",
        ),
        (
            "L3",
            "later-evaluation",
            "L2",
            len(length_scope),
            "30 <= sequence length <= 400",
            "VERIFIED",
        ),
        (
            "L4",
            "later-evaluation",
            "L3",
            len(resolved),
            "at least 3 residues and >=80% native C1' resolved",
            "VERIFIED",
        ),
        (
            "L5-historical",
            "historical-only",
            "L4",
            len(historical_manifest),
            "historical MMseqs 80% identity/80% coverage grouped temporal split",
            "VERIFIED_HISTORICAL_NOT_V4_FINAL",
        ),
        (
            "H100",
            "historical-only",
            "L5-historical",
            len(medium),
            "old deterministic 60/20/20 cohort; appendix only",
            "VERIFIED_HISTORICAL_NOT_V4_FINAL",
        ),
        (
            "V4-known-clean",
            "provisional-v4",
            "L4",
            exposure_summary[
                "later_targets_remaining_after_known_exposure_cluster_exclusion"
            ],
            "from all 419 valid RNAs, remove every known exposed target and its MMseqs sequence-similarity cluster",
            "PROVISIONAL_NOT_A_FINAL_MANIFEST",
        ),
        (
            "V4-final",
            "final-untouched",
            "V4-known-clean",
            "UNKNOWN",
            "also pass duplicate, external-exposure, pretrained-overlap and technical audits",
            "LOCKED_UNTIL_PREREGISTRATION_REVIEW",
        ),
    ]
    flow = pd.DataFrame(
        rows, columns=["stage_id", "branch", "parent", "target_count", "rule", "status"]
    )
    flow["evidence"] = ""
    flow.loc[flow["stage_id"].isin(["D0", "P0-input", "L1", "L2", "L3", "L4"]), "evidence"] = relative(TRAIN_SEQUENCES)
    flow.loc[flow["stage_id"] == "P0-artifact", "evidence"] = "data/processed/geometry_priors.json"
    flow.loc[flow["stage_id"] == "L5-historical", "evidence"] = relative(FULL_MANIFEST)
    flow.loc[flow["stage_id"] == "V4-known-clean", "evidence"] = relative(FAMILY_CLUSTERS)
    flow.loc[flow["stage_id"] == "H100", "evidence"] = relative(MEDIUM_MANIFEST)
    flow.to_csv(output_dir / "data_flow_5135.csv", index=False)
    return flow


def write_snapshot(output_dir: Path, ledger: pd.DataFrame, flow: pd.DataFrame) -> None:
    paths = [
        TRAIN_SEQUENCES,
        TRAIN_LABELS,
        CASP_SEQUENCES,
        FULL_MANIFEST,
        MEDIUM_MANIFEST,
        PILOT_MANIFEST,
        FAMILY_CLUSTERS,
        REPO_ROOT / "data" / "processed" / "geometry_priors.json",
        REPO_ROOT / "data" / "processed" / "geofuse_geometry_v2_priors.json",
        REPO_ROOT / "utilities" / "top1_tbm.py",
        REPO_ROOT / "utilities" / "top1_4_4_hybrid_final_take.py",
        REPO_ROOT / "scripts" / "build_v4_preregistration_audit.py",
    ]
    snapshot = {
        "purpose": "data-only V4 preregistration audit; no predictor or native scoring",
        "ledger_rows": int(len(ledger)),
        "flow_rows": int(len(flow)),
        "inputs": [
            {
                "path": relative(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else "UNKNOWN",
            }
            for path in paths
        ],
    }
    (output_dir / "audit_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger, summary = build_exposure_ledger(args.output_dir)
    flow = build_data_flow(args.output_dir, summary)
    write_snapshot(args.output_dir, ledger, flow)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
