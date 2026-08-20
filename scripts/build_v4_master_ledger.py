#!/usr/bin/env python
"""Build the 5,135-row V4 master ledger and regenerate MMseqs clusters.

This is a data-only preregistration step. It reads sequence/native-validity metadata
but never generates structures and never calculates a performance metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPETITION = REPO_ROOT / "data" / "stanford-rna-3d-folding"
TRAIN_SEQUENCES = COMPETITION / "train_sequences.v2.csv"
TRAIN_LABELS = COMPETITION / "train_labels.v2.csv"
CASP_SEQUENCES = COMPETITION / "validation_sequences.csv"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "thesis_v4" / "preregistration"
DEFAULT_EXPOSURE = DEFAULT_OUTPUT / "development_exposure_ledger.csv"
MODEL_CUTOFF_DECLARED = pd.Timestamp("2023-12-31")
LAST_SNAPSHOT_DATE = pd.Timestamp("2025-03-26")
P0_CUTOFF = pd.Timestamp("2022-05-27")


def digest_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            result.update(chunk)
    return result.hexdigest()


def normalized_sequence(sequence: str) -> str:
    return str(sequence).upper().replace("T", "U")


def native_validity() -> pd.DataFrame:
    counts: dict[str, int] = {}
    resolved: dict[str, int] = {}
    for chunk in pd.read_csv(TRAIN_LABELS, usecols=["ID", "x_1"], chunksize=500_000):
        target = chunk["ID"].str.rsplit("_", n=1).str[0]
        finite = chunk["x_1"].to_numpy(float) > -1e17
        temporary = pd.DataFrame({"target_id": target.to_numpy(), "resolved": finite})
        grouped = temporary.groupby("target_id")["resolved"].agg(["size", "sum"])
        for target_id, row in grouped.iterrows():
            counts[target_id] = counts.get(target_id, 0) + int(row["size"])
            resolved[target_id] = resolved.get(target_id, 0) + int(row["sum"])
    rows = []
    for target_id, count in counts.items():
        n_resolved = resolved[target_id]
        fraction = n_resolved / count if count else 0.0
        rows.append(
            {
                "target_id": target_id,
                "native_residue_count": count,
                "native_resolved_count": n_resolved,
                "native_resolved_fraction": fraction,
                "native_valid": count >= 3 and fraction >= 0.8,
            }
        )
    return pd.DataFrame(rows)


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.target_id}\n{row.normalized_sequence}\n")


def regenerate_clusters(
    representatives: pd.DataFrame,
    output_dir: Path,
    mmseqs: Path,
    threads: int,
) -> tuple[pd.DataFrame, dict]:
    fasta = output_dir / "v4_mmseqs_input.fasta"
    cluster_tsv = output_dir / "v4_mmseqs_sequence_similarity_clusters.tsv"
    write_fasta(representatives, fasta)
    with tempfile.TemporaryDirectory(prefix="v4_mmseqs_", dir=output_dir) as temporary:
        temporary = Path(temporary)
        prefix = temporary / "cluster80"
        workspace = temporary / "tmp"
        command = [
            str(mmseqs),
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(workspace),
            "--min-seq-id",
            "0.8",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--threads",
            str(threads),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        shutil.copy2(f"{prefix}_cluster.tsv", cluster_tsv)
    clusters = pd.read_csv(
        cluster_tsv,
        sep="\t",
        names=["representative", "member"],
        dtype=str,
    )
    missing = set(representatives["target_id"]) - set(clusters["member"])
    if missing:
        raise RuntimeError(f"MMseqs omitted {len(missing)} representatives")
    version = subprocess.run(
        [str(mmseqs), "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    provenance = {
        "purpose": "V4 sequence-similarity clusters for dependence-aware inference",
        "input_targets": int(len(representatives)),
        "clusters": int(clusters["representative"].nunique()),
        "threshold_identity": 0.8,
        "threshold_coverage": 0.8,
        "coverage_mode": 0,
        "threads": threads,
        "mmseqs_path": str(mmseqs),
        "mmseqs_version": version,
        "mmseqs_sha256": sha256(mmseqs),
        "input_fasta": fasta.name,
        "input_fasta_sha256": sha256(fasta),
        "cluster_tsv": cluster_tsv.name,
        "cluster_tsv_sha256": sha256(cluster_tsv),
        "command_template": [
            "<MMSEQS_BINARY>",
            "easy-cluster",
            "v4_mmseqs_input.fasta",
            "<TEMP>/cluster80",
            "<TEMP>/tmp",
            "--min-seq-id",
            "0.8",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--threads",
            str(threads),
        ],
        "performance_or_predictor_access": False,
    }
    (output_dir / "v4_mmseqs_cluster_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return clusters, provenance


def exact_representatives(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Keep newest target per exact sequence; tie-break by lexical target ID."""
    ordered = frame.sort_values(
        ["normalized_sequence_sha256", "date", "target_id"],
        ascending=[True, False, True],
    )
    representatives = ordered.drop_duplicates("normalized_sequence_sha256", keep="first")
    representative_of = dict(
        zip(
            ordered["target_id"],
            ordered["normalized_sequence_sha256"].map(
                representatives.set_index("normalized_sequence_sha256")["target_id"]
            ),
        )
    )
    return representatives.sort_values("target_id"), representative_of


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_csv(TRAIN_SEQUENCES, dtype=str)
    if len(sequences) != 5135 or sequences["target_id"].nunique() != 5135:
        raise RuntimeError("train V2 no longer contains exactly 5,135 unique targets")
    sequences["normalized_sequence"] = sequences["sequence"].map(normalized_sequence)
    sequences["normalized_sequence_sha256"] = sequences["normalized_sequence"].map(
        digest_bytes
    )
    sequences["sequence_length"] = sequences["normalized_sequence"].str.len()
    sequences["date"] = pd.to_datetime(sequences["temporal_cutoff"], errors="coerce")
    sequences["canonical_sequence"] = sequences["normalized_sequence"].str.fullmatch(
        r"[AUGC]+"
    )
    sequences = sequences.merge(native_validity(), on="target_id", how="left")
    for column in ["native_residue_count", "native_resolved_count"]:
        sequences[column] = sequences[column].fillna(0).astype(int)
    sequences["native_resolved_fraction"] = sequences[
        "native_resolved_fraction"
    ].fillna(0.0)
    sequences["native_valid"] = sequences["native_valid"].fillna(False).astype(bool)

    exposure = pd.read_csv(args.exposure_ledger, dtype=str).fillna("")
    reason_map = dict(zip(exposure["target_id"], exposure["exposure_reasons"]))
    non_p0_exposed = {
        target_id
        for target_id, reason in reason_map.items()
        if reason
        and reason != "P0 production geometry-prior training from native C1' coordinates"
    }

    sequences["source_dataset"] = "kaggle_train_v2"
    sequences["used_for_p0_prior"] = sequences["date"] < P0_CUTOFF
    sequences["development_exposed"] = sequences["target_id"].isin(non_p0_exposed)
    sequences["native_score_or_supervision_exposed"] = sequences["target_id"].isin(
        set(exposure.loc[exposure["native_scored_or_supervised"] == "True", "target_id"])
    )
    sequences["failure_case_exposed"] = sequences["target_id"].isin(
        set(exposure.loc[exposure["failure_case_inspected"] == "True", "target_id"])
    )
    sequences["known_exposure_reason"] = sequences["target_id"].map(reason_map).fillna("")
    sequences["later_date_scope"] = (
        (sequences["date"] > MODEL_CUTOFF_DECLARED)
        & (sequences["date"] <= LAST_SNAPSHOT_DATE)
    )
    sequences["length_scope"] = sequences["sequence_length"].between(30, 400)
    sequences["technical_valid"] = (
        sequences["canonical_sequence"] & sequences["native_valid"]
    )
    sequences["later_eligible_pre_duplicate"] = (
        sequences["later_date_scope"]
        & sequences["length_scope"]
        & sequences["technical_valid"]
    )

    eligible = sequences[sequences["later_eligible_pre_duplicate"]].copy()
    representatives, representative_of = exact_representatives(eligible)
    sequences["exact_sequence_representative"] = sequences["target_id"].map(
        representative_of
    ).fillna("")
    sequences["exact_duplicate_excluded"] = (
        sequences["later_eligible_pre_duplicate"]
        & (sequences["target_id"] != sequences["exact_sequence_representative"])
    )
    sequences["later_eligible_exact_deduplicated"] = (
        sequences["later_eligible_pre_duplicate"]
        & ~sequences["exact_duplicate_excluded"]
    )

    clusters, cluster_provenance = regenerate_clusters(
        representatives,
        output_dir,
        args.mmseqs.resolve(),
        args.threads,
    )
    representative_cluster = dict(zip(clusters["member"], clusters["representative"]))
    sequences["mmseqs_sequence_similarity_cluster"] = sequences[
        "exact_sequence_representative"
    ].map(representative_cluster).fillna("")
    exposed_clusters = set(
        sequences.loc[
            sequences["development_exposed"]
            & sequences["later_eligible_pre_duplicate"],
            "mmseqs_sequence_similarity_cluster",
        ]
    ) - {""}
    sequences["exposed_cluster"] = sequences[
        "mmseqs_sequence_similarity_cluster"
    ].isin(exposed_clusters)

    # These columns are intentionally not inferred from release dates. They remain
    # pending until the model-specific provenance audits are complete.
    sequences["drfold_structural_overlap_status"] = np.where(
        sequences["later_eligible_exact_deduplicated"] & ~sequences["exposed_cluster"],
        "AUDIT_PENDING",
        "NOT_IN_CANDIDATE_POOL",
    )
    sequences["drfold_language_model_provenance_status"] = np.where(
        sequences["later_eligible_exact_deduplicated"] & ~sequences["exposed_cluster"],
        "AUDIT_PENDING",
        "NOT_IN_CANDIDATE_POOL",
    )
    sequences["boltz_overlap_status"] = np.where(
        sequences["later_eligible_exact_deduplicated"] & ~sequences["exposed_cluster"],
        "AUDIT_PENDING",
        "NOT_IN_CANDIDATE_POOL",
    )
    sequences["external_exposure_status"] = np.where(
        sequences["development_exposed"], "CONFIRMED_EXPOSED", "AUDIT_PENDING"
    )
    sequences["provisional_after_repository_audit"] = (
        sequences["later_eligible_exact_deduplicated"]
        & ~sequences["exposed_cluster"]
    )
    # No target is called untouched before model and external exposure audits pass.
    sequences["provisional_untouched"] = False
    sequences["final_included"] = False

    def state(row: pd.Series) -> str:
        if row["used_for_p0_prior"]:
            return "P0_PRIOR"
        if row["development_exposed"]:
            return "DEVELOPMENT_EXPOSED"
        if not row["later_date_scope"]:
            return "OUTSIDE_LATER_DATE_SCOPE"
        if not row["length_scope"]:
            return "OUTSIDE_LENGTH_SCOPE"
        if not row["canonical_sequence"]:
            return "TECHNICAL_INVALID_SEQUENCE"
        if not row["native_valid"]:
            return "TECHNICAL_INVALID_NATIVE"
        if row["exact_duplicate_excluded"]:
            return "EXACT_DUPLICATE_EXCLUDED"
        if row["exposed_cluster"]:
            return "EXPOSED_CLUSTER"
        return "PRETRAINED_AND_EXTERNAL_AUDIT_PENDING"

    sequences["eligibility_state"] = sequences.apply(state, axis=1)
    evidence = {
        "P0_PRIOR": "scripts/run_phase2_priors.py",
        "DEVELOPMENT_EXPOSED": "development_exposure_ledger.csv",
        "EXPOSED_CLUSTER": "v4_mmseqs_sequence_similarity_clusters.tsv",
        "PRETRAINED_AND_EXTERNAL_AUDIT_PENDING": "v4_mmseqs_sequence_similarity_clusters.tsv",
    }
    sequences["evidence"] = sequences["eligibility_state"].map(evidence).fillna(
        "train_sequences.v2.csv; train_labels.v2.csv"
    )

    columns = [
        "target_id",
        "source_dataset",
        "temporal_cutoff",
        "sequence_length",
        "normalized_sequence_sha256",
        "canonical_sequence",
        "native_residue_count",
        "native_resolved_count",
        "native_resolved_fraction",
        "native_valid",
        "used_for_p0_prior",
        "development_exposed",
        "native_score_or_supervision_exposed",
        "failure_case_exposed",
        "known_exposure_reason",
        "later_date_scope",
        "length_scope",
        "technical_valid",
        "later_eligible_pre_duplicate",
        "exact_sequence_representative",
        "exact_duplicate_excluded",
        "later_eligible_exact_deduplicated",
        "mmseqs_sequence_similarity_cluster",
        "exposed_cluster",
        "drfold_structural_overlap_status",
        "drfold_language_model_provenance_status",
        "boltz_overlap_status",
        "external_exposure_status",
        "provisional_after_repository_audit",
        "provisional_untouched",
        "final_included",
        "eligibility_state",
        "evidence",
    ]
    master = sequences[columns].sort_values("target_id")
    master_path = output_dir / "v4_master_rna_ledger.csv"
    master.to_csv(master_path, index=False)

    casp = pd.read_csv(CASP_SEQUENCES, dtype=str)
    casp_ledger = pd.DataFrame(
        {
            "target_id": casp["target_id"],
            "source_dataset": "CASP15_validation",
            "temporal_cutoff": casp["temporal_cutoff"],
            "sequence_length": casp["sequence"].str.len(),
            "development_exposed": True,
            "eligibility_state": "CASP15_DEVELOPMENT",
            "final_included": False,
            "evidence": "reports/tables/reproduce_top1.csv and historical CASP15 analyses",
        }
    )
    casp_ledger.to_csv(output_dir / "casp15_development_ledger.csv", index=False)

    summary = {
        "master_rows": int(len(master)),
        "p0_prior_targets": int(master["used_for_p0_prior"].sum()),
        "known_development_exposed_train_v2": int(master["development_exposed"].sum()),
        "later_date_scope": int(master["later_date_scope"].sum()),
        "later_length_scope": int(
            (master["later_date_scope"] & master["length_scope"]).sum()
        ),
        "later_technical_valid_pre_duplicate": int(
            master["later_eligible_pre_duplicate"].sum()
        ),
        "exact_sequence_representatives": int(
            master["later_eligible_exact_deduplicated"].sum()
        ),
        "regenerated_mmseqs_clusters": int(cluster_provenance["clusters"]),
        "known_exposed_clusters": int(len(exposed_clusters)),
        "provisional_after_repository_audit": int(
            master["provisional_after_repository_audit"].sum()
        ),
        "provisional_clusters_after_repository_audit": int(
            master.loc[
                master["provisional_after_repository_audit"],
                "mmseqs_sequence_similarity_cluster",
            ].nunique()
        ),
        "provisional_untouched": int(master["provisional_untouched"].sum()),
        "final_included": int(master["final_included"].sum()),
        "casp15_development_targets": int(len(casp_ledger)),
        "warning": (
            "No RNA is called untouched yet. Pretrained overlap and external/manual "
            "exposure audits remain pending."
        ),
        "master_ledger_sha256": sha256(master_path),
    }
    (output_dir / "v4_master_ledger_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    data_flow = pd.DataFrame(
        [
            ("D0", "target", 5135, "Kaggle train V2 master universe"),
            ("P0", "target", summary["p0_prior_targets"], "temporal cutoff before 2022-05-27; geometry-prior training role only"),
            ("L1", "target", summary["later_date_scope"], "declared pretrained structural cutoff scope; provenance audit pending"),
            ("L2", "target", summary["later_length_scope"], "sequence length 30 to 400"),
            ("L3", "target", summary["later_technical_valid_pre_duplicate"], "canonical AUGC and valid native C1' coverage"),
            ("L4", "target", summary["exact_sequence_representatives"], "newest deterministic representative per exact normalized sequence"),
            ("L5", "cluster", summary["regenerated_mmseqs_clusters"], "MMseqs 80% identity and 80% coverage sequence-similarity clusters"),
            ("L6-excluded", "cluster", summary["known_exposed_clusters"], "cluster contains at least one repository-known development-exposed target"),
            ("L7", "target", summary["provisional_after_repository_audit"], "repository-audit provisional pool; not yet untouched"),
            ("L7", "cluster", summary["provisional_clusters_after_repository_audit"], "dependence blocks in repository-audit provisional pool"),
            ("FINAL", "target", "UNKNOWN", "requires pretrained-overlap and external/manual exposure audits"),
        ],
        columns=["stage_id", "unit", "count", "rule"],
    )
    data_flow["performance_accessed"] = False
    data_flow.to_csv(output_dir / "data_flow_5135.csv", index=False)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--exposure-ledger", type=Path, default=DEFAULT_EXPOSURE)
    default_mmseqs = Path(shutil.which("mmseqs") or "mmseqs")
    result.add_argument("--mmseqs", type=Path, default=default_mmseqs)
    result.add_argument("--threads", type=int, default=4)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
