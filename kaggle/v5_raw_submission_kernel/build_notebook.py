#!/usr/bin/env python
"""Build the minimal notebook that publishes the frozen V5 Raw CSV."""
from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path(__file__).with_name("v5_raw_submission.ipynb")


def lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


code = """from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

INPUT = Path('/kaggle/input')
WORKING = Path('/kaggle/working')
EXPECTED_SUBMISSION = 'edb5ac4b1d484ea7292eea6747a87b8e64a8c174227eaaaa116adb59970c6109'
EXPECTED_FREEZE = '14b5929cd6dbf21bc4c272fa3a51059b6909691c68dd1974daf5f202743f3b41'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifests = []
for path in INPUT.rglob('inference_manifest.json'):
    try:
        value = json.loads(path.read_text())
    except Exception:
        continue
    if value.get('status') == 'V5_FROZEN_RAW_NOTEBOOK_INPUT':
        manifests.append((path, value))
if len(manifests) != 1:
    raise RuntimeError(f'expected one V5 frozen input manifest, found {[str(x[0]) for x in manifests]}')
manifest_path, manifest = manifests[0]
source = manifest_path.parent / 'submission.csv'
if manifest.get('submission_sha256') != EXPECTED_SUBMISSION or sha256(source) != EXPECTED_SUBMISSION:
    raise RuntimeError('V5 frozen submission hash mismatch')
if manifest.get('method_freeze_sha256') != EXPECTED_FREEZE:
    raise RuntimeError('V5 method freeze hash mismatch')

competition_roots = [
    path.parent for path in INPUT.rglob('sample_submission.csv')
    if (path.parent / 'test_sequences.csv').is_file()
]
if len(competition_roots) != 1:
    raise RuntimeError(f'expected one competition input, found {competition_roots}')
sample = pd.read_csv(competition_roots[0] / 'sample_submission.csv')
submission = pd.read_csv(source)
if submission.shape != sample.shape:
    raise RuntimeError(f'shape mismatch: {submission.shape} vs {sample.shape}')
if submission.columns.tolist() != sample.columns.tolist():
    raise RuntimeError('column mismatch')
if submission['ID'].tolist() != sample['ID'].tolist():
    raise RuntimeError('ID/order mismatch')
if not np.isfinite(submission.iloc[:, 3:].to_numpy(dtype=float)).all():
    raise RuntimeError('non-finite coordinate')

output = WORKING / 'submission.csv'
shutil.copyfile(source, output)
if sha256(output) != EXPECTED_SUBMISSION:
    raise RuntimeError('published bytes differ from frozen V5 output')
receipt = {
    'status': 'V5_FROZEN_RAW_PUBLISHED',
    'submission_sha256': EXPECTED_SUBMISSION,
    'method_freeze_sha256': EXPECTED_FREEZE,
    'rows': len(submission),
    'late_score_is_independent_hidden_test': False,
}
(WORKING / 'v5_publish_receipt.json').write_text(json.dumps(receipt, indent=2) + '\\n')
print(json.dumps(receipt, indent=2))
submission.head(3)
"""

document = {
    "cells": [
        {
            "id": "v5-description",
            "cell_type": "markdown",
            "metadata": {},
            "source": lines(
                "# RNA3D V5 frozen Raw deployment\n\n"
                "Publishes the method frozen before its late Kaggle scorer check. "
                "The score is treated as deployment compatibility because the downloaded "
                "test sequences equal the CASP15 development sequences.\n"
            ),
        },
        {
            "id": "v5-validate-publish",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines(code),
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
print(OUTPUT)
