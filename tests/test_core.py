"""Fast, data-free smoke tests for core numerical helpers."""

from __future__ import annotations

import unittest

import numpy as np

from rna3d.geometry.transforms import apply_rigid, kabsch, random_rotation, rmsd
from rna3d.template.confidence import template_confidence, temporal_valid


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


if __name__ == "__main__":
    unittest.main()
