"""Validate and publish the frozen native-blind hybrid Geometry-v2 output."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd


INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
EXPECTED_COLUMNS = [
    "ID",
    "resname",
    "resid",
    *[f"{axis}_{rank}" for rank in range(1, 6) for axis in ("x", "y", "z")],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_matches = list(INPUT_ROOT.glob("**/inference_manifest.json"))
sample_matches = list(INPUT_ROOT.glob("**/sample_submission.csv"))
if len(manifest_matches) != 1:
    raise RuntimeError(f"expected one frozen inference manifest, found {manifest_matches}")
if len(sample_matches) != 1:
    raise RuntimeError(f"expected one competition sample submission, found {sample_matches}")

manifest_path = manifest_matches[0]
source_path = manifest_path.parent / "submission.csv"
if not source_path.exists():
    raise FileNotFoundError(f"submission.csv is missing beside {manifest_path}")
manifest = json.loads(manifest_path.read_text())
if manifest.get("native_labels_used") is not False:
    raise ValueError("artifact provenance does not certify native-blind inference")
if sha256(source_path) != manifest.get("submission_sha256"):
    raise ValueError("frozen submission checksum differs from its manifest")

submission = pd.read_csv(source_path)
sample = pd.read_csv(sample_matches[0])
if list(submission.columns) != EXPECTED_COLUMNS:
    raise ValueError("submission columns differ from the competition schema")
if list(submission.columns) != list(sample.columns):
    raise ValueError("submission and sample columns differ")
if len(submission) != len(sample):
    raise ValueError(f"row count differs: {len(submission)} != {len(sample)}")
if submission["ID"].duplicated().any():
    raise ValueError("submission contains duplicate IDs")
if submission["ID"].tolist() != sample["ID"].tolist():
    raise ValueError("submission IDs are not in exact competition order")
coordinates = submission[EXPECTED_COLUMNS[3:]].to_numpy(dtype=float)
if not np.isfinite(coordinates).all():
    raise ValueError("submission contains NaN or infinite coordinates")

output_path = WORKING / "submission.csv"
shutil.copyfile(source_path, output_path)
if sha256(output_path) != sha256(source_path):
    raise ValueError("published CSV bytes differ from the frozen artifact")
print(f"validated {len(submission)} rows; sha256={sha256(output_path)}")
print(output_path)
