"""Data-free tests for selective fusion and abstention."""
from __future__ import annotations

import unittest

import numpy as np

from rna3d.geofuse.candidate import StructureCandidate
from rna3d.geofuse.selective import contiguous_run_mask, selective_quality_fusion


def candidate(kind: str, coords: np.ndarray) -> StructureCandidate:
    length = len(coords)
    return StructureCandidate(
        target_id="T",
        sequence=("ACGU" * 10)[:length],
        candidate_id=kind,
        kind=kind,
        source=kind,
        model="test",
        coords=coords,
        confidence=np.full(length, 0.7),
        support_mask=np.ones(length, dtype=bool),
        global_confidence=0.7,
    )


class SelectiveFusionTests(unittest.TestCase):
    def test_short_runs_are_removed(self) -> None:
        mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(
            contiguous_run_mask(mask, minimum=4),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=bool),
        )

    def test_uncertain_router_abstains(self) -> None:
        template_coords = np.column_stack(
            [np.arange(24) * 5.0, np.sin(np.arange(24)), np.zeros(24)]
        )
        pretrained_coords = template_coords.copy()
        pretrained_coords[8:16, 1] += 3.0
        template = candidate("template", template_coords)
        pretrained = candidate("pretrained", pretrained_coords)
        self.assertIsNone(
            selective_quality_fusion(
                template,
                pretrained,
                np.full(24, 0.5),
                decision_threshold=0.5,
            )
        )

    def test_confident_long_segment_creates_candidate(self) -> None:
        template_coords = np.column_stack(
            [np.arange(30) * 5.0, np.sin(np.arange(30)), np.zeros(30)]
        )
        pretrained_coords = template_coords.copy()
        pretrained_coords[9:21, 1] += 4.0
        probability = np.full(30, 0.2)
        probability[9:21] = 0.9
        fused = selective_quality_fusion(
            candidate("template", template_coords),
            candidate("pretrained", pretrained_coords),
            probability,
            decision_threshold=0.5,
            probability_margin=0.15,
            minimum_disagreement=0.1,
        )
        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertGreater(fused.metadata["switched_fraction"], 0.0)
        self.assertEqual(fused.kind, "fused")


if __name__ == "__main__":
    unittest.main()
