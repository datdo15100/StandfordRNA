#!/usr/bin/env python
"""Package the frozen V5 Raw CSV as a private Kaggle notebook input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data" / "interim" / "v5_final_raw_kaggle_order" / "submission.csv"
AMENDMENT = SOURCE.with_name("technical_amendment_receipt.json")
FREEZE = REPO / "reports" / "thesis_v5" / "experiments" / "V5_FINAL_PIPELINE_FREEZE.json"
OUT = REPO / "data" / "interim" / "kaggle_v5_frozen_raw_upload"
EXPECTED_SUBMISSION = "edb5ac4b1d484ea7292eea6747a87b8e64a8c174227eaaaa116adb59970c6109"
ORIGINAL_SUBMISSION = "e819f68d90696b6c08ce05b52f47c0ee2ed59039cc5e8dfa92a4a1d7d4e79c21"
EXPECTED_FREEZE = "14b5929cd6dbf21bc4c272fa3a51059b6909691c68dd1974daf5f202743f3b41"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SUBMISSION:
        raise RuntimeError("selected V5 submission changed after freeze")
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("V5 final method freeze changed")
    amendment = json.loads(AMENDMENT.read_text())
    if (
        amendment.get("original_submission_sha256") != ORIGINAL_SUBMISSION
        or amendment.get("ordered_submission_sha256") != EXPECTED_SUBMISSION
        or amendment.get("scientific_method_changed") is not False
        or amendment.get("kaggle_score_observed_before_amendment") is not False
    ):
        raise RuntimeError("invalid pre-score technical row-order amendment")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copy2(SOURCE, OUT / "submission.csv")
    manifest = {
        "status": "V5_FROZEN_RAW_NOTEBOOK_INPUT",
        "scientific_method": "exact reconstructed V3 3T+2D Raw",
        "native_labels_loaded_during_candidate_generation": False,
        "method_selected_on_casp15_validation": True,
        "submission_sha256": EXPECTED_SUBMISSION,
        "original_frozen_submission_sha256": ORIGINAL_SUBMISSION,
        "technical_change": "row order only; all values are exact after sorting by ID",
        "technical_amendment_sha256": sha256(AMENDMENT),
        "method_freeze_sha256": EXPECTED_FREEZE,
        "source_code_commit": "64c157344c76c0ab48d649696e0018aa71fcdf67",
        "freeze_commit": "3259a4b7ff9adf84afeb94868ac16d4ffba48507",
        "kaggle_score_observed_at_package_time": False,
        "independence_limitation": "Kaggle test sequences are byte-identical to the CASP15 validation sequences used for method development; late score is a deployment compatibility check."
    }
    (OUT / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (OUT / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "RNA3D V5 Frozen Raw Submission",
                "id": "datdo151000/rna3d-v5-frozen-raw-submission",
                "licenses": [{"name": "CC0-1.0"}]
            },
            indent=2,
        )
        + "\n"
    )
    print(OUT)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
