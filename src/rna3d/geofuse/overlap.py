"""Sequence-overlap auditing for pretrained candidate generators."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_rna_sequence(sequence: str) -> str:
    """Normalize RNA/DNA spelling for conservative exact-overlap checks."""
    return "".join(str(sequence).upper().split()).replace("T", "U").replace("-", "")


def read_fasta_sequences(path: str | Path) -> list[tuple[str, str]]:
    """Read a FASTA without requiring model-specific parser dependencies."""
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    chunks: list[str] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, normalize_rna_sequence("".join(chunks))))
            identifier = line[1:].split()[0]
            chunks = []
        elif identifier is None:
            raise ValueError(f"{path}: sequence data found before a FASTA header")
        else:
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, normalize_rna_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def audit_exact_sequence_overlap(
    targets: pd.DataFrame,
    model_fastas: Iterable[tuple[str, str | Path]],
) -> pd.DataFrame:
    """Report exact target matches in one or more model training manifests.

    This deliberately makes a narrow, defensible claim: an exact normalized
    sequence is either present or absent. It does not infer what a checkpoint
    saw unless the supplied FASTA documents that checkpoint's training set.
    """
    required = {"target_id", "sequence"}
    missing = required - set(targets.columns)
    if missing:
        raise KeyError(f"target table is missing columns: {sorted(missing)}")

    rows = []
    for model, fasta_path in model_fastas:
        index: dict[str, list[str]] = defaultdict(list)
        records = read_fasta_sequences(fasta_path)
        for identifier, sequence in records:
            index[sequence].append(identifier)
        for target in targets.itertuples(index=False):
            sequence = normalize_rna_sequence(target.sequence)
            matches = sorted(index.get(sequence, []))
            rows.append(
                {
                    "target_id": str(target.target_id),
                    "seq_len": len(sequence),
                    "model": str(model),
                    "training_fasta": str(Path(fasta_path).resolve()),
                    "training_records": len(records),
                    "exact_overlap": bool(matches),
                    "matching_training_ids": ";".join(matches),
                }
            )
    return pd.DataFrame(rows)
