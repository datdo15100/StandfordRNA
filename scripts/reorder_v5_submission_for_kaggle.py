#!/usr/bin/env python
"""Repair only the row ordering of the frozen V5 Kaggle CSV.

The first deployment attempt correctly contained every frozen coordinate but
used target-manifest order instead of the competition sample-submission order.
This technical adapter changes no ID, residue, coordinate, candidate, or method.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rna3d.data.io import order_submission_like


SOURCE = REPO / "data" / "interim" / "v5_final_raw" / "submission.csv"
SAMPLE = REPO / "data" / "stanford-rna-3d-folding" / "sample_submission.csv"
OUT = REPO / "data" / "interim" / "v5_final_raw_kaggle_order"
ORIGINAL_FREEZE = REPO / "reports" / "thesis_v5" / "experiments" / "V5_FINAL_PIPELINE_FREEZE.json"
EXPECTED_SOURCE = "e819f68d90696b6c08ce05b52f47c0ee2ed59039cc5e8dfa92a4a1d7d4e79c21"
EXPECTED_FREEZE = "14b5929cd6dbf21bc4c272fa3a51059b6909691c68dd1974daf5f202743f3b41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE or sha256(ORIGINAL_FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("the frozen scientific artifact changed before technical reorder")
    source = pd.read_csv(SOURCE)
    sample = pd.read_csv(SAMPLE)
    if source.shape != sample.shape or source.columns.tolist() != sample.columns.tolist():
        raise RuntimeError("frozen submission and sample schema differ")
    ordered = order_submission_like(source, sample)
    if ordered["ID"].tolist() != sample["ID"].tolist():
        raise RuntimeError("technical reorder failed")
    # Prove that the adapter changed row order only.
    columns = source.columns.tolist()
    left = source.sort_values("ID").reset_index(drop=True)[columns]
    right = ordered.sort_values("ID").reset_index(drop=True)[columns]
    pd.testing.assert_frame_equal(left, right, check_exact=True)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "submission.csv"
    ordered.to_csv(output, index=False)
    receipt = {
        "status": "V5_KAGGLE_TECHNICAL_ROW_ORDER_AMENDMENT_BEFORE_SCORE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_method_changed": False,
        "candidate_or_coordinate_changed": False,
        "reason": "First notebook attempt failed before scoring because the frozen CSV had the correct ID set but not sample-submission row order.",
        "original_submission_sha256": EXPECTED_SOURCE,
        "method_freeze_sha256": EXPECTED_FREEZE,
        "sample_submission_sha256": sha256(SAMPLE),
        "ordered_submission_sha256": sha256(output),
        "row_n": len(ordered),
        "id_set_equal": set(source["ID"]) == set(ordered["ID"]),
        "all_values_equal_after_sort_by_id": True,
        "kaggle_score_observed_before_amendment": False,
        "adapter_code_sha256": sha256(Path(__file__)),
    }
    (OUT / "technical_amendment_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
