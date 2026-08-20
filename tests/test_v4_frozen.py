"""Data-free tests for the frozen V4 native-blind candidate implementation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rna3d.pipeline.v4_frozen import (
    build_j_controlled_bank,
    build_thesis_tbm_bank,
    length_allowed,
    normalized_exclusions,
    stable_seed,
)


class FrozenV4PipelineTests(unittest.TestCase):
    def test_helpers_are_deterministic(self) -> None:
        self.assertEqual(stable_seed("x", 1), stable_seed("x", 1))
        self.assertNotEqual(stable_seed("x", 1), stable_seed("x", 2))
        self.assertEqual(normalized_exclusions(["8abc", "8ABC", " 9def "]), ("8ABC", "9DEF"))

    def test_length_filter_boundaries(self) -> None:
        self.assertTrue(length_allowed(40, 100))
        self.assertFalse(length_allowed(40, 101))
        self.assertTrue(length_allowed(100, 60))
        self.assertFalse(length_allowed(100, 50))
        self.assertTrue(length_allowed(1200, 960))
        self.assertFalse(length_allowed(1200, 959))

    def test_empty_database_uses_repeatable_fallbacks(self) -> None:
        meta = pd.DataFrame(columns=["chain_key", "pdb_id", "release_date", "length", "seq"])
        kwargs = dict(
            target_id="TEST_A",
            sequence="AUGCAUGC",
            cutoff="2025-01-01",
            excluded_pdb_ids=("TEST",),
            meta=meta,
            coordinates={},
            adjacent_distance=5.9,
            n=5,
        )
        first = build_thesis_tbm_bank(**kwargs)
        second = build_thesis_tbm_bank(**kwargs)
        self.assertEqual(first.fallback_slots, 5)
        np.testing.assert_array_equal(first.coords, second.coords)
        self.assertEqual(first.coords.shape, (5, 8, 3))
        self.assertTrue(np.isfinite(first.coords).all())
        self.assertTrue(np.all(first.confidence == np.float32(0.1)))

        john = build_j_controlled_bank(
            target_id="TEST_A",
            sequence="AUGCAUGC",
            cutoff="2025-01-01",
            excluded_pdb_ids=("TEST",),
            meta=meta,
            coordinates={},
            n=5,
        )
        self.assertEqual(john.fallback_slots, 5)
        self.assertEqual(john.coords.shape, (5, 8, 3))
        self.assertTrue(np.isfinite(john.coords).all())


if __name__ == "__main__":
    unittest.main()
