"""Data-free tests for the preregistered V4 cluster-aware inference."""
from __future__ import annotations

import unittest

import numpy as np

from rna3d.eval.v4_statistics import (
    cluster_bootstrap_means,
    cluster_sign_flip_pvalue,
    holm_step_down,
    primary_inference,
)


class V4StatisticsTests(unittest.TestCase):
    def test_cluster_bootstrap_is_repeatable_and_carries_members(self) -> None:
        deltas = [1.0, 3.0, -2.0]
        clusters = ["A", "A", "B"]
        first = cluster_bootstrap_means(deltas, clusters, replicates=50, seed=7)
        second = cluster_bootstrap_means(deltas, clusters, replicates=50, seed=7)

        np.testing.assert_array_equal(first, second)
        possible = {2.0, -2.0, 2.0 / 3.0}
        self.assertTrue(set(first).issubset(possible))

    def test_sign_flip_uses_reproducible_one_sided_test(self) -> None:
        deltas = [0.4, 0.2, 0.3, 0.1]
        clusters = ["A", "A", "B", "C"]
        first = cluster_sign_flip_pvalue(deltas, clusters, permutations=999, seed=11)
        second = cluster_sign_flip_pvalue(deltas, clusters, permutations=999, seed=11)

        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_holm_step_down_stops_after_first_failure(self) -> None:
        result = holm_step_down({"H1": 0.01, "H2": 0.04, "H3": 0.03})

        self.assertTrue(result["H1"]["reject"])
        self.assertFalse(result["H3"]["reject"])
        self.assertFalse(result["H2"]["reject"])
        self.assertAlmostEqual(result["H1"]["holm_adjusted_p"], 0.03)
        self.assertAlmostEqual(result["H2"]["holm_adjusted_p"], 0.06)
        self.assertAlmostEqual(result["H3"]["holm_adjusted_p"], 0.06)

    def test_primary_inference_reports_target_and_cluster_units(self) -> None:
        result = primary_inference(
            "H3",
            [0.1, 0.2, -0.1],
            ["A", "A", "B"],
            bootstrap_replicates=100,
            permutation_replicates=999,
            seed=5,
        )

        self.assertEqual(result.target_n, 3)
        self.assertEqual(result.cluster_n, 2)
        self.assertAlmostEqual(result.mean_delta, 0.2 / 3.0)
        self.assertEqual(result.improved, 2)
        self.assertEqual(result.regressed, 1)

    def test_primary_rejects_missing_failure_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            primary_inference("H1", [0.1, np.nan], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
