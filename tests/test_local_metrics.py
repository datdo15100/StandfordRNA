"""Data-free tests for independent C1'-trace accuracy metrics."""
from __future__ import annotations

import unittest

import numpy as np

from rna3d.eval.local_metrics import (
    c1_lddt,
    c1_rmsd,
    local_accuracy_metrics,
    sliding_window_c1_rmsd,
)
from rna3d.eval.statistics import paired_target_summary
from rna3d.geometry.transforms import apply_rigid, random_rotation


class C1MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        steps = rng.normal(size=(40, 3))
        self.reference = np.cumsum(steps, axis=0)

    def test_exact_and_rigid_transform_are_perfect(self) -> None:
        rng = np.random.default_rng(17)
        transformed = apply_rigid(
            self.reference, random_rotation(rng), np.array([8.0, -3.0, 2.0])
        )
        self.assertLess(c1_rmsd(transformed, self.reference), 1e-10)
        result = c1_lddt(transformed, self.reference)
        self.assertAlmostEqual(result["score"], 1.0)
        metrics = local_accuracy_metrics(transformed, self.reference)
        self.assertLess(metrics["sw_rmsd_9"], 1e-10)
        self.assertLess(metrics["sw_rmsd_15"], 1e-10)
        self.assertLess(metrics["sw_rmsd_31"], 1e-10)

    def test_local_perturbation_lowers_local_scores(self) -> None:
        prediction = self.reference.copy()
        prediction[18:22] += np.array([8.0, -4.0, 3.0])
        lddt = c1_lddt(prediction, self.reference)
        local = sliding_window_c1_rmsd(prediction, self.reference, window=9)
        self.assertLess(lddt["score"], 1.0)
        self.assertLess(np.nanmean(lddt["per_residue"][18:22]), 1.0)
        self.assertGreater(local["mean"], 0.0)
        self.assertGreater(
            np.nanmean(local["per_residue"][18:22]),
            np.nanmean(local["per_residue"][:5]),
        )

    def test_unresolved_native_is_excluded_and_missing_prediction_fails_pairs(self) -> None:
        reference = self.reference.copy()
        reference[0] = np.nan
        exact = c1_lddt(self.reference, reference)
        self.assertTrue(np.isnan(exact["per_residue"][0]))
        self.assertAlmostEqual(exact["score"], 1.0)

        prediction = self.reference.copy()
        prediction[10] = np.nan
        missing = c1_lddt(prediction, reference)
        self.assertEqual(missing["per_residue"][10], 0.0)
        self.assertLess(missing["score"], 1.0)

    def test_short_sequence_uses_one_complete_window(self) -> None:
        reference = self.reference[:7]
        result = sliding_window_c1_rmsd(reference, reference, window=31)
        self.assertEqual(result["effective_window"], 7)
        self.assertEqual(result["n_windows"], 1)
        self.assertAlmostEqual(result["mean"], 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            c1_rmsd(np.zeros((3, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            c1_lddt(self.reference, self.reference, thresholds=(0.0, 1.0))
        with self.assertRaises(ValueError):
            sliding_window_c1_rmsd(self.reference, self.reference, window=0)


class TargetStatisticsTests(unittest.TestCase):
    def test_lower_metric_is_oriented_as_positive_improvement(self) -> None:
        result = paired_target_summary(
            np.array([3.0, 4.0, 5.0]),
            np.array([2.0, 4.0, 6.0]),
            higher_is_better=False,
            bootstrap_samples=1000,
        )
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["tied"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertAlmostEqual(result["mean_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
