"""Loading competition CSVs and writing/validating submissions.

Coordinate sentinel
-------------------
Unresolved residues in the label CSVs are stored as a large negative value
(-1e18 in the data). The official scorer treats a coordinate as *resolved* iff
``x > -1e17``. We follow that exact convention via :data:`RESOLVED_THRESHOLD`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import comp_file

# A coordinate is "resolved" iff it is greater than this threshold (official rule).
RESOLVED_THRESHOLD = -1e17


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
def load_sequences(split: str) -> pd.DataFrame:
    """Load a *_sequences.csv split: 'train', 'train_v2', 'validation', 'test'."""
    key = {
        "train": "train_sequences",
        "train_v2": "train_sequences_v2",
        "validation": "validation_sequences",
        "test": "test_sequences",
    }[split]
    df = pd.read_csv(comp_file(key), dtype=str)
    df["seq_len"] = df["sequence"].str.len().astype(int)
    return df


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def load_labels(split: str) -> pd.DataFrame:
    """Load a *_labels.csv split: 'train', 'train_v2', 'validation'."""
    key = {
        "train": "train_labels",
        "train_v2": "train_labels_v2",
        "validation": "validation_labels",
    }[split]
    return pd.read_csv(comp_file(key))


def n_reference_structures(labels_df: pd.DataFrame) -> int:
    """How many reference conformations (x_1..x_n) the label frame carries."""
    return sum(1 for c in labels_df.columns if c.startswith("x_"))


def target_id_of(label_id: str) -> str:
    """``R1107_1`` -> ``R1107``;  ``1SCL_A_3`` -> ``1SCL_A`` (strip trailing resid)."""
    return label_id.rsplit("_", 1)[0]


def get_reference_coords(labels_df: pd.DataFrame, target_id: str) -> list[np.ndarray]:
    """Return a list of (L, 3) coordinate arrays, one per reference conformation.

    Unresolved residues are returned as ``np.nan`` rows. Conformations that are
    entirely unresolved (all-NaN) are dropped, matching the scorer's behaviour of
    skipping references with zero resolved residues.
    """
    tid = labels_df["ID"].map(target_id_of)
    sub = labels_df[tid == target_id].sort_values("resid")
    if sub.empty:
        raise KeyError(f"target_id {target_id!r} not found in labels")

    n_ref = n_reference_structures(labels_df)
    out: list[np.ndarray] = []
    for k in range(1, n_ref + 1):
        cols = [f"x_{k}", f"y_{k}", f"z_{k}"]
        if not all(c in sub.columns for c in cols):
            break
        xyz = sub[cols].to_numpy(dtype=float).copy()
        resolved = xyz[:, 0] > RESOLVED_THRESHOLD
        xyz[~resolved] = np.nan
        if resolved.any():
            out.append(xyz)
    return out


def get_sequence_from_labels(labels_df: pd.DataFrame, target_id: str) -> str:
    tid = labels_df["ID"].map(target_id_of)
    sub = labels_df[tid == target_id].sort_values("resid")
    return "".join(sub["resname"].tolist())


# --------------------------------------------------------------------------- #
# Submission writing / validation
# --------------------------------------------------------------------------- #
SUBMISSION_COORD_COLS = [f"{a}_{k}" for k in range(1, 6) for a in ("x", "y", "z")]
SUBMISSION_COLUMNS = ["ID", "resname", "resid", *SUBMISSION_COORD_COLS]


def build_submission(
    predictions: dict[str, np.ndarray],
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a submission frame.

    ``predictions[target_id]`` must be an array of shape (5, L, 3) — five
    structures of L residues each. ``sequences`` provides target_id -> sequence.
    """
    seq_map = dict(zip(sequences["target_id"], sequences["sequence"]))
    rows = []
    for target_id, preds in predictions.items():
        preds = np.asarray(preds, dtype=float)
        assert preds.ndim == 3 and preds.shape[0] == 5, (
            f"{target_id}: expected (5, L, 3), got {preds.shape}"
        )
        seq = seq_map[target_id]
        L = preds.shape[1]
        assert L == len(seq), f"{target_id}: {L} coords vs {len(seq)} residues"
        for i in range(L):
            row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
            for k in range(5):
                row[f"x_{k + 1}"] = preds[k, i, 0]
                row[f"y_{k + 1}"] = preds[k, i, 1]
                row[f"z_{k + 1}"] = preds[k, i, 2]
            rows.append(row)
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def validate_submission(sub: pd.DataFrame, sequences: pd.DataFrame) -> None:
    """Raise if the submission frame is malformed relative to the sequence set."""
    missing_cols = set(SUBMISSION_COLUMNS) - set(sub.columns)
    if missing_cols:
        raise ValueError(f"submission missing columns: {sorted(missing_cols)}")
    if sub[SUBMISSION_COORD_COLS].isna().any().any():
        raise ValueError("submission contains NaN coordinates")
    for _, r in sequences.iterrows():
        tid, seq = r["target_id"], r["sequence"]
        got = (sub["ID"].map(target_id_of) == tid).sum()
        if got != len(seq):
            raise ValueError(f"{tid}: {got} rows vs {len(seq)} residues")


def write_submission(sub: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path
