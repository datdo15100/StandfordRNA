#!/usr/bin/env python
"""Reproduce the controlled R1108 local-vs-global metric sanity check.

This is an evaluation-only diagnostic. It deliberately reads validation native
coordinates, creates synthetic perturbations, and scores them with the same C1'-only
US-align wrapper used by the offline competition mirror.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data.io import (  # noqa: E402
    get_reference_coords,
    get_sequence_from_labels,
    load_labels,
)
from rna3d.eval.local_metrics import local_accuracy_metrics  # noqa: E402
from rna3d.eval.usalign import score_target  # noqa: E402


OUTPUT = REPO_ROOT / "reports/tables/tm_score_sanity/r1108_controlled_perturbations.csv"


def variants(reference: np.ndarray) -> dict[str, np.ndarray]:
    """Return fixed, interpretable deformations of one complete native trace."""
    output = {"native_copied": reference.copy()}

    local = reference.copy()
    local[30:35] += np.asarray([8.0, 0.0, 0.0])
    output["residues_31_35_shift_x_8A"] = local

    domain = reference.copy()
    domain[20:50] += np.asarray([8.0, 0.0, 0.0])
    output["residues_21_50_shift_x_8A"] = domain

    extended = np.zeros_like(reference)
    extended[:, 0] = np.arange(len(reference), dtype=float) * 5.9
    output["straight_extended_chain_5p9A"] = extended
    return output


def main() -> None:
    labels = load_labels("validation")
    references = get_reference_coords(labels, "R1108")
    sequence = get_sequence_from_labels(labels, "R1108")
    # Both references are complete. Reference 2 is fixed to make the experiment
    # deterministic and to match the worked example in the defense guide.
    reference = references[1]
    if reference.shape != (69, 3) or not np.isfinite(reference).all():
        raise RuntimeError("R1108 reference 2 is not the expected complete 69-residue trace")

    rows = []
    for name, prediction in variants(reference).items():
        metrics = local_accuracy_metrics(prediction, reference, windows=(9, 15, 31))
        rows.append(
            {
                "target_id": "R1108",
                "reference_conformation": 2,
                "variant": name,
                "tm_score": score_target([prediction], [reference], list(sequence)),
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(frame.to_string(index=False))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
