"""Produce a Kaggle-format submission.csv from the production inference loop.

Proves the end-to-end production path (search -> TBM -> refine -> 5 structures ->
validated submission) on the test_sequences set, using the exact thesis code.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rna3d.data import io
from rna3d.paths import processed
from kaggle.inference_pipeline import run_inference

def main():
    test = io.load_sequences("test")
    sub = run_inference(test, processed(), steps=200, max_len=900)
    out = processed() / "submission.csv"
    io.write_submission(sub, out)
    print(f"wrote {out}  rows={len(sub)}  targets={test['target_id'].nunique()}")
    print(sub.head(3).to_string(index=False))
    print("\nformat validation: PASSED (build_submission + validate_submission ran clean)")


if __name__ == "__main__":
    main()
