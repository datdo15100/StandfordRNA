"""Fast, data-free smoke tests for core numerical helpers."""

from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rna3d.data.io import build_submission, order_submission_like, validate_submission
from rna3d.geometry.transforms import apply_rigid, kabsch, random_rotation, rmsd
from rna3d.template.confidence import template_confidence, temporal_valid
from rna3d.paths import processed
from rna3d.template.mmseqs_search import mmseqs_bin


class TransformTests(unittest.TestCase):
    def test_random_rotation_is_proper_and_orthogonal(self) -> None:
        rotation = random_rotation(np.random.default_rng(7))

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_kabsch_recovers_rigid_transform(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(20, 3))
        expected = apply_rigid(
            source,
            random_rotation(rng),
            np.array([4.0, -2.5, 0.75]),
        )

        rotation, translation = kabsch(source, expected)
        actual = apply_rigid(source, rotation, translation)

        self.assertLess(rmsd(actual, expected), 1e-10)


class TemplateConfidenceTests(unittest.TestCase):
    def test_temporal_gate_is_strict(self) -> None:
        self.assertTrue(temporal_valid("2022-05-26", "2022-05-27"))
        self.assertFalse(temporal_valid("2022-05-27", "2022-05-27"))
        self.assertFalse(temporal_valid("9999-12-31", "2022-05-27"))

    def test_confidence_multiplies_components(self) -> None:
        self.assertAlmostEqual(template_confidence(0.8, 0.5, 0.75), 0.3)


class PortablePathTests(unittest.TestCase):
    def test_processed_directory_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"RNA3D_PROCESSED": tmp}
        ):
            self.assertEqual(processed(), Path(tmp))

    def test_mmseqs_binary_can_be_overridden(self) -> None:
        with patch.dict(os.environ, {"RNA3D_MMSEQS": "/opt/rna3d/mmseqs"}):
            self.assertEqual(mmseqs_bin(), "/opt/rna3d/mmseqs")


class SubmissionTests(unittest.TestCase):
    def test_submission_can_follow_sample_row_order(self) -> None:
        sequences = pd.DataFrame(
            {"target_id": ["A", "B"], "sequence": ["AC", "G"]}
        )
        predictions = {
            "A": np.zeros((5, 2, 3)),
            "B": np.ones((5, 1, 3)),
        }
        sub = build_submission(predictions, sequences)
        sample = pd.DataFrame({"ID": ["B_1", "A_1", "A_2"]})

        ordered = order_submission_like(sub, sample)

        self.assertEqual(ordered["ID"].tolist(), sample["ID"].tolist())
        validate_submission(ordered, sequences)

    def test_validation_rejects_duplicate_ids(self) -> None:
        sequences = pd.DataFrame({"target_id": ["A"], "sequence": ["AC"]})
        sub = build_submission({"A": np.zeros((5, 2, 3))}, sequences)
        sub.loc[1, "ID"] = "A_1"

        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            validate_submission(sub, sequences)


if __name__ == "__main__":
    unittest.main()
