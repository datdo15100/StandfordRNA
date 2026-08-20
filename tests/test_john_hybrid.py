from __future__ import annotations

import unittest

import numpy as np

from rna3d.baselines.john_hybrid import (
    public_drfold_index_range,
    run_public_hybrid_route,
)


class JohnHybridRouteTests(unittest.TestCase):
    def test_captured_small_cohort_range_uses_only_retained_index_zero(self) -> None:
        self.assertEqual(public_drfold_index_range(12), (0, 0))

    def test_model_output_is_padded_to_five(self) -> None:
        result = run_public_hybrid_route(
            retained_dataframe_index=0,
            target_count=12,
            elapsed_seconds=0,
            drfold_time_limit_seconds=100,
            boltz_conditioned_drfold_runner=lambda: [np.zeros((3, 3))],
            template_runner=lambda: [np.ones((3, 3))],
        )
        self.assertEqual(result.executed_route, "boltz_conditioned_drfold2")
        self.assertEqual(len(result.structures), 5)

    def test_missing_hybrid_artifacts_use_template_fallback(self) -> None:
        result = run_public_hybrid_route(
            retained_dataframe_index=0,
            target_count=12,
            elapsed_seconds=0,
            drfold_time_limit_seconds=100,
            boltz_conditioned_drfold_runner=None,
            template_runner=lambda: [np.ones((3, 3)) for _ in range(5)],
        )
        self.assertEqual(result.executed_route, "template")
        self.assertEqual(result.fallback_reason, "boltz_conditioned_drfold2_artifacts_unavailable")


if __name__ == "__main__":
    unittest.main()
