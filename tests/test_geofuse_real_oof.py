"""Data-free tests for real-OOF leakage guards and supervision."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rna3d.geofuse.candidate import StructureCandidate
from rna3d.geofuse.real_oof import (
    audit_pretrained_oof,
    audit_template_oof,
    grouped_temporal_split,
    make_real_example,
)
from tests.test_geofuse_phase_d import native_chain, priors
from scripts.train_geofuse_real_gate import real_gate_passed


def candidate(kind: str, coords: np.ndarray, metadata: dict) -> StructureCandidate:
    length = len(coords)
    return StructureCandidate(
        target_id="T",
        sequence=("ACGU" * 20)[:length],
        candidate_id=f"{kind}_1",
        kind=kind,
        source="tbm" if kind == "template" else "drfold2",
        model="test",
        coords=coords,
        confidence=np.linspace(0.2, 0.9, length),
        support_mask=np.ones(length, dtype=bool),
        global_confidence=0.6,
        metadata=metadata,
    )


class RealOOFTests(unittest.TestCase):
    def test_gate_must_beat_whole_source_baselines(self) -> None:
        metrics = {
            "learned_gate_error": 7.6,
            "template_error": 7.7,
            "pretrained_error": 7.0,
            "gap_rule_error": 7.9,
            "confidence_rule_error": 7.8,
        }
        self.assertFalse(real_gate_passed(metrics))
        metrics["learned_gate_error"] = 6.9
        self.assertTrue(real_gate_passed(metrics))

    def test_grouped_temporal_split_keeps_families_together(self) -> None:
        frame = pd.DataFrame(
            {
                "target_id": [f"T{i}" for i in range(20)],
                "date": pd.date_range("2024-01-01", periods=20),
                "sequence_group": ["duplicate" if i in (2, 18) else f"g{i}" for i in range(20)],
            }
        )
        split = grouped_temporal_split(frame, 0.15, 0.20)
        self.assertEqual(split.loc[split.sequence_group == "duplicate", "split"].nunique(), 1)
        dates = split.groupby("split")["date"].agg(["min", "max"])
        self.assertLess(dates.loc["train", "max"], dates.loc["calibration", "min"])
        self.assertLess(dates.loc["calibration", "max"], dates.loc["validation", "min"])

    def test_provenance_audits_reject_leakage(self) -> None:
        xyz = native_chain(32)
        template = candidate(
            "template", xyz, {"release_date": "2023-01-01", "pdb_id": "1ABC"}
        )
        prediction = candidate(
            "pretrained", xyz, {"model_training_cutoff": "2023-12-31"}
        )
        self.assertEqual(audit_template_oof(template, "2024-02-01")["template_pdb_id"], "1ABC")
        self.assertEqual(audit_pretrained_oof(prediction, "2024-02-01")["oof_mode"], "date")
        with self.assertRaises(ValueError):
            audit_template_oof(template, "2024-02-01", {"1ABC"})
        prediction.metadata["model_training_cutoff"] = "2025-01-01"
        with self.assertRaises(ValueError):
            audit_pretrained_oof(prediction, "2024-02-01")
        with tempfile.TemporaryDirectory() as directory:
            exclusions = Path(directory) / "excluded.txt"
            exclusions.write_text("T\n")
            prediction.metadata["oof_exclusion_manifest"] = str(exclusions)
            self.assertEqual(
                audit_pretrained_oof(prediction, "2024-02-01")["oof_mode"],
                "explicit_exclusion",
            )

    def test_real_example_uses_only_resolved_native_rows(self) -> None:
        native = native_chain(40).astype(float)
        template = candidate(
            "template", native + 0.2, {"release_date": "2023-01-01", "pdb_id": "2ABC"}
        )
        prediction = candidate(
            "pretrained", native + np.linspace(0, 2, 40)[:, None],
            {"model_training_cutoff": "2023-12-31"},
        )
        native[10] = np.nan
        v1, v2 = priors()
        example = make_real_example(template, prediction, [native], v1, v2)
        self.assertEqual(example["features"].shape[0], 40)
        self.assertFalse(example["resolved_mask"][10])
        self.assertEqual(int(example["resolved_mask"].sum()), 39)
        self.assertTrue(np.isfinite(example["target"]).all())


if __name__ == "__main__":
    unittest.main()
