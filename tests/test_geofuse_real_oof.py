"""Data-free tests for real-OOF leakage guards and supervision."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
from scripts.run_geofuse_real_oof import cmd_replace_failed
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
        self.assertFalse(example["lddt_resolved_mask"][10])
        self.assertFalse(example["window_resolved_mask"][10])
        self.assertTrue(np.isfinite(example["lddt_target"]).all())
        self.assertTrue(np.isfinite(example["window_target"]).all())
        self.assertEqual(example["template_lddt"].shape, (40,))
        self.assertEqual(example["pretrained_window_rmsd"].shape, (40,))

    def test_failed_target_replacement_uses_first_unused_later_reserve(self) -> None:
        full = pd.DataFrame(
            {
                "target_id": ["A", "B", "C", "D", "E"],
                "sequence": ["ACGU"] * 5,
                "seq_len": [4] * 5,
                "date": pd.date_range("2024-01-01", periods=5).astype(str),
                "split": ["train", "train", "train", "train", "calibration"],
                "sequence_group": ["sa", "sb", "sc", "sd", "se"],
                "family_group": ["fa", "fb", "fc", "fd", "fe"],
                "excluded_pdb_ids": [""] * 5,
                "model_training_cutoff": ["2023-12-31"] * 5,
                "model_training_data": ["frozen"] * 5,
            }
        )
        cohort = full[full["target_id"].isin(["A", "B", "D", "E"])]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            full_path = directory / "manifest.csv"
            cohort_path = directory / "medium_manifest.csv"
            output_path = directory / "repaired.csv"
            full.to_csv(full_path, index=False)
            cohort.to_csv(cohort_path, index=False)
            cmd_replace_failed(
                SimpleNamespace(
                    manifest=full_path,
                    cohort=cohort_path,
                    failed_target="B",
                    max_len=100,
                    output=output_path,
                )
            )
            repaired = pd.read_csv(output_path)
            self.assertEqual(set(repaired["target_id"]), {"A", "C", "D", "E"})
            self.assertNotIn("B", repaired["target_id"].tolist())
            self.assertTrue((directory / "repaired_targets.txt").exists())
            self.assertTrue((directory / "repaired.fasta").exists())


if __name__ == "__main__":
    unittest.main()
