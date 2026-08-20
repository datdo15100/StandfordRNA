#!/usr/bin/env python
"""Hash pretrained artifacts and audit V4 candidate overlap without scoring structures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DRFOLD = Path("/home/datdo/.cache/rna3d/external/DRfold2")
CLEAN_DRFOLD = Path("/home/datdo/.cache/rna3d/external/DRfold2_clean_3667d6a")
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_v4" / "phase1_pretrained"
DEFAULT_WORK = REPO_ROOT / "data" / "processed" / "v4_pretrained_audit"
MASTER = REPO_ROOT / "reports" / "thesis_v4" / "preregistration" / "v4_master_rna_ledger.csv"
SEQUENCES = REPO_ROOT / "data" / "stanford-rna-3d-folding" / "train_sequences.v2.csv"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(sequence: str) -> str:
    return str(sequence).upper().replace("T", "U")


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, normalize("".join(sequence))))
            header, sequence = line[1:].strip(), []
        else:
            sequence.append(line.strip())
    if header is not None:
        records.append((header, normalize("".join(sequence))))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(DRFOLD), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def source_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    report = args.report_dir.resolve()
    work = args.work_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    mmseqs = args.mmseqs.resolve()

    commit = git_output("rev-parse", "HEAD").strip()
    tree = git_output("rev-parse", f"{commit}^{{tree}}").strip()
    if commit != "3667d6ac44c5b14cff94c36e0371224acb2cdae5":
        raise RuntimeError(f"unexpected DRfold2 commit {commit}")
    if not (CLEAN_DRFOLD / "README.md").is_file():
        raise RuntimeError("clean DRfold2 git-archive snapshot is missing")
    clean_manifest = source_manifest(CLEAN_DRFOLD)
    clean_manifest_path = report / "drfold2_clean_source_manifest.csv"
    clean_manifest.to_csv(clean_manifest_path, index=False)
    tracked_diff = git_output(
        "diff",
        "--ignore-space-at-eol",
        "--",
        "PotentialFold/Optimization.py",
        "PotentialFold/Selection.py",
    )
    tracked_diff = "\n".join(line.rstrip() for line in tracked_diff.splitlines()) + "\n"
    (report / "drfold2_local_tracked_diff.patch").write_text(tracked_diff)
    (report / "drfold2_local_status.txt").write_text(git_output("status", "--short"))

    checkpoint_dir = DRFOLD / "model_hub" / "cfg_97"
    checkpoints = sorted(
        checkpoint_dir.glob("model_*"), key=lambda path: int(path.name.split("_")[1])
    )
    if len(checkpoints) != 20:
        raise RuntimeError(f"expected 20 cfg_97 checkpoints, found {len(checkpoints)}")
    checkpoint_manifest = pd.DataFrame(
        [
            {"checkpoint": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in checkpoints
        ]
    )
    checkpoint_manifest_path = report / "drfold2_cfg97_checkpoint_manifest.csv"
    checkpoint_manifest.to_csv(checkpoint_manifest_path, index=False)
    rclm = DRFOLD / "model_hub" / "RCLM" / "epoch_67000"

    training_fasta = DRFOLD / "data" / "train.fasta"
    training_records = parse_fasta(training_fasta)
    unique_sequences: dict[str, str] = {}
    for header, sequence in training_records:
        unique_sequences.setdefault(sequence, header)
    deduplicated_training = [
        (f"train_{index:05d}|{header}", sequence)
        for index, (sequence, header) in enumerate(sorted(unique_sequences.items()))
    ]
    training_dedup_path = work / "drfold2_train_unique.fasta"
    write_fasta(deduplicated_training, training_dedup_path)

    master = pd.read_csv(MASTER)
    target_ids = set(
        master.loc[master["provisional_after_repository_audit"], "target_id"].astype(str)
    )
    sequences = pd.read_csv(SEQUENCES, dtype=str)
    queries = sequences[sequences["target_id"].isin(target_ids)].copy()
    queries["sequence"] = queries["sequence"].map(normalize)
    queries = queries.sort_values("target_id")
    if len(queries) != 172:
        raise RuntimeError(f"expected 172 repository-provisional targets, found {len(queries)}")
    query_path = work / "v4_repository_provisional_queries.fasta"
    write_fasta(list(queries[["target_id", "sequence"]].itertuples(index=False, name=None)), query_path)

    exact_sequences = set(unique_sequences)
    exact = dict(zip(queries["target_id"], queries["sequence"].isin(exact_sequences)))
    result_path = work / "drfold2_homolog_hits.tsv"
    with tempfile.TemporaryDirectory(prefix="mmseqs_", dir=work) as temporary:
        command = [
            str(mmseqs),
            "easy-search",
            str(query_path),
            str(training_dedup_path),
            str(result_path),
            str(Path(temporary) / "tmp"),
            "--min-seq-id",
            "0.8",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--max-seqs",
            "1000",
            "--search-type",
            "3",
            "--threads",
            str(args.threads),
            "--format-output",
            "query,target,pident,qcov,tcov,evalue,bits",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
    if result_path.stat().st_size:
        hits = pd.read_csv(
            result_path,
            sep="\t",
            names=["target_id", "training_id", "pident", "qcov", "tcov", "evalue", "bits"],
        )
    else:
        hits = pd.DataFrame(
            columns=["target_id", "training_id", "pident", "qcov", "tcov", "evalue", "bits"]
        )
    grouped = hits.groupby("target_id") if len(hits) else None
    rows = []
    for target_id in queries["target_id"]:
        group = grouped.get_group(target_id) if grouped is not None and target_id in grouped.groups else None
        rows.append(
            {
                "target_id": target_id,
                "exact_sequence_overlap": bool(exact[target_id]),
                "mmseqs_homolog_overlap_80_80": group is not None,
                "homolog_hit_count": 0 if group is None else len(group),
                "maximum_percent_identity": None if group is None else float(group["pident"].max()),
                "maximum_query_coverage": None if group is None else float(group["qcov"].max()),
                "maximum_target_coverage": None if group is None else float(group["tcov"].max()),
                "v4_status": "EXCLUDE_MODEL_OVERLAP" if group is not None else "STRUCTURAL_OVERLAP_PASS",
            }
        )
    overlap = pd.DataFrame(rows)
    overlap_path = report / "drfold2_overlap_by_target.csv"
    overlap.to_csv(overlap_path, index=False)

    boltz_candidates = sorted(
        (REPO_ROOT / "data" / "cache" / "geofuse_candidates").glob("**/boltz*.npz")
    )
    boltz_checkpoint = list(Path("/home/datdo/.cache").glob("**/boltz1_conf.ckpt"))
    provenance = {
        "phase": "V4 Phase 1 pretrained provenance and overlap audit",
        "performance_accessed": False,
        "audit_driver": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": sha256(Path(__file__))},
        "drfold2": {
            "remote": "https://github.com/leeyang/DRfold2.git",
            "commit": commit,
            "tree": tree,
            "clean_source_path": str(CLEAN_DRFOLD),
            "clean_source_manifest": {"path": str(clean_manifest_path.relative_to(REPO_ROOT)), "sha256": sha256(clean_manifest_path)},
            "local_checkout_dirty": bool(git_output("status", "--porcelain").strip()),
            "tracked_patch": str((report / "drfold2_local_tracked_diff.patch").relative_to(REPO_ROOT)),
            "tracked_patch_role": "removes deprecated SciPy iprint arguments; scientific equivalence still requires smoke validation",
            "cfg97_checkpoints": {"count": len(checkpoints), "manifest_sha256": sha256(checkpoint_manifest_path)},
            "rclm_checkpoint": {"path": str(rclm), "bytes": rclm.stat().st_size, "sha256": sha256(rclm)},
            "local_structural_training_fasta": {
                "path": str(training_fasta),
                "sha256": sha256(training_fasta),
                "records": len(training_records),
                "unique_headers": len({header for header, _ in training_records}),
                "unique_sequences": len(unique_sequences),
                "minimum_length": min(map(len, unique_sequences)),
                "maximum_length": max(map(len, unique_sequences)),
                "identity_to_published_data_zip": "UNVERIFIED",
            },
            "published_provenance": {
                "paper": "https://doi.org/10.1371/journal.pbio.3003659",
                "structural_training_release_rule": "RNA structures released before 2024",
                "paper_test_exclusion_rule": "more than 80% sequence identity to test RNA excluded",
                "rclm_corpus": "approximately 30M RNAcentral Release 22 sequences",
                "rclm_training_batches": 67000,
            },
            "v4_overlap": {
                "candidate_targets": len(overlap),
                "exact_sequence_overlaps": int(overlap["exact_sequence_overlap"].sum()),
                "mmseqs_80_80_homolog_overlaps": int(overlap["mmseqs_homolog_overlap_80_80"].sum()),
                "structural_overlap_pass": int(overlap["v4_status"].eq("STRUCTURAL_OVERLAP_PASS").sum()),
                "overlap_table": str(overlap_path.relative_to(REPO_ROOT)),
                "overlap_table_sha256": sha256(overlap_path),
                "mmseqs_version": subprocess.run([str(mmseqs), "version"], check=True, capture_output=True, text=True).stdout.strip(),
                "mmseqs_sha256": sha256(mmseqs),
                "command_template": "easy-search query structural_train --search-type 3 --min-seq-id 0.8 -c 0.8 --cov-mode 0 --max-seqs 1000",
            },
            "language_model_overlap": "RNAcentral Release 22 corpus manifest unavailable locally; exact/homolog target membership remains UNVERIFIED and no whole-model time-safe claim is allowed",
        },
        "boltz": {
            "local_checkpoint_count": len(boltz_checkpoint),
            "local_checkpoint_status": "UNAVAILABLE" if not boltz_checkpoint else "AVAILABLE",
            "historical_candidate_count": len(boltz_candidates),
            "historical_candidate_paths": [str(path.relative_to(REPO_ROOT)) for path in boltz_candidates],
            "source_commit": "UNKNOWN",
            "checkpoint_hash": "UNKNOWN" if not boltz_checkpoint else sha256(boltz_checkpoint[0]),
            "decision": "NOT ADMISSIBLE FOR PRIMARY OR MECHANISM EVIDENCE until source/checkpoint/training provenance is recovered",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    provenance_path = report / "pretrained_provenance_audit.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    summary = provenance["drfold2"]["v4_overlap"]
    readme = f"""# V4 Phase 1: pretrained provenance audit

- Native performance accessed: **No**.
- DRfold2 clean source: commit `{commit}`, tree `{tree}`.
- `cfg_97`: {len(checkpoints)} checkpoint files, all individually hashed.
- RCLM checkpoint: `{sha256(rclm)}`.
- Local structural FASTA: {len(training_records):,} records, {len(unique_sequences):,} unique sequences.
- Published structural rule: structures released before 2024.
- Published RCLM corpus: approximately 30 million RNAcentral Release 22 sequences,
  trained for 67,000 batches.
- Repository-provisional V4 pool audited: {summary['candidate_targets']} targets.
- Exact structural-training sequence overlaps: {summary['exact_sequence_overlaps']}.
- MMseqs 80% identity and 80% coverage overlaps: {summary['mmseqs_80_80_homolog_overlaps']}.
- Structural-overlap pass at this rule: {summary['structural_overlap_pass']}.
- RCLM exact membership remains **UNVERIFIED** because the Release 22 training manifest
  is not present locally. DRfold2 is therefore not described as wholly time-safe.
- Boltz source commit and checkpoint are unavailable locally. Historical Boltz output
  is not admissible as V4 evidence until provenance is recovered.
"""
    (report / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    result.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    result.add_argument(
        "--mmseqs",
        type=Path,
        default=Path(shutil.which("mmseqs") or "/home/datdo/miniforge3/envs/rna-fold/bin/mmseqs"),
    )
    result.add_argument("--threads", type=int, default=4)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
