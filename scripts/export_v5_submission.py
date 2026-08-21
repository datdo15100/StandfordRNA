#!/usr/bin/env python
"""Export the CASP15-selected V5 3T+2D Raw bank in Kaggle CSV format."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.data import io


CACHE = REPO / "data" / "cache" / "v5_casp15"
OUT = REPO / "data" / "interim" / "v5_final_raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sequences = io.load_sequences("test")
    predictions = {}
    banks = []
    for target in sequences.itertuples(index=False):
        path = CACHE / target.target_id / "refined_banks.npz"
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["V3_3T2D__Raw"], dtype=np.float32)
        if coords.shape != (5, len(target.sequence), 3) or not np.isfinite(coords).all():
            raise ValueError(f"{target.target_id}: invalid frozen V5 raw bank")
        predictions[target.target_id] = coords
        banks.append(
            {
                "target_id": target.target_id,
                "length": len(target.sequence),
                "bank_array_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
                "refined_artifact_path": str(path.relative_to(REPO)),
                "refined_artifact_sha256": sha256(path),
            }
        )
    submission = io.build_submission(predictions, sequences)
    io.validate_submission(submission, sequences)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "submission.csv"
    submission.to_csv(path, index=False)
    pd.DataFrame(banks).to_csv(OUT / "candidate_bank_manifest.csv", index=False)
    receipt = {
        "status": "V5_SELECTED_RAW_SUBMISSION_EXPORTED_BEFORE_KAGGLE_SCORE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "exact reconstructed V3 TBM first 3 plus direct DRfold2 first 2; Raw; template fallback for unavailable D slots",
        "target_n": int(sequences["target_id"].nunique()),
        "row_n": int(len(submission)),
        "test_sequences_sha256": sha256(
            REPO / "data" / "stanford-rna-3d-folding" / "test_sequences.csv"
        ),
        "submission_path": str(path.relative_to(REPO)),
        "submission_sha256": sha256(path),
        "candidate_bank_manifest_sha256": sha256(OUT / "candidate_bank_manifest.csv"),
        "export_code_sha256": sha256(Path(__file__)),
        "kaggle_score_observed": False,
    }
    (OUT / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
