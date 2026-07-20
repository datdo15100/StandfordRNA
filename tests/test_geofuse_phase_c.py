"""Data-free tests for GeoFuse fold clustering, fusion, and selection."""
from __future__ import annotations

import unittest

import numpy as np

from rna3d.geofuse.candidate import StructureCandidate
from rna3d.geofuse.phase_c import (
    cluster_fold_families,
    fuse_template_pretrained,
    native_blind_quality_scores,
    robust_superpose,
    select_quality_diversity,
)
from rna3d.geometry.transforms import apply_rigid, random_rotation


def candidate(
    candidate_id: str,
    source: str,
    kind: str,
    coords: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    global_confidence: float,
) -> StructureCandidate:
    return StructureCandidate(
        target_id="T1",
        sequence="ACGUAC"[: len(coords)],
        candidate_id=candidate_id,
        kind=kind,
        source=source,
        model="test",
        coords=coords,
        confidence=confidence,
        support_mask=support,
        global_confidence=global_confidence,
    )


class FoldClusteringTests(unittest.TestCase):
    def test_complete_link_separates_incompatible_fold(self) -> None:
        similarity = np.array(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.4], [0.3, 0.4, 1.0]]
        )
        labels = cluster_fold_families(similarity, threshold=0.5)
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_robust_alignment_trims_one_outlier(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(12, 3))
        rotation = random_rotation(rng)
        moving = apply_rigid(reference, rotation, np.array([5.0, -2.0, 3.0]))
        moving[0] += 50.0
        aligned, _, inliers = robust_superpose(moving, reference, trim_fraction=0.8)
        self.assertFalse(inliers[0])
        np.testing.assert_allclose(aligned[1:], reference[1:], atol=1e-5)


class SegmentFusionTests(unittest.TestCase):
    def test_gap_segment_prefers_pretrained_and_keeps_supported_template(self) -> None:
        template_coords = np.array(
            [[0, 0, 0], [5, 1, 0], [10, 0, 1], [15, 1, 1], [20, 0, 2], [25, 1, 2]],
            dtype=float,
        )
        pretrained_coords = template_coords.copy()
        pretrained_coords[2:4, 1] += 4.0
        support = np.array([True, True, False, False, True, True])
        template = candidate(
            "tbm", "tbm", "template", template_coords, np.ones(6), support, 0.8
        )
        pretrained = candidate(
            "pre", "drfold2", "pretrained", pretrained_coords,
            np.linspace(0.2, 0.8, 6), np.ones(6, dtype=bool), 0.3
        )
        fused = fuse_template_pretrained(template, pretrained)
        self.assertEqual(fused.kind, "fused")
        self.assertLess(
            np.linalg.norm(fused.coords[2] - pretrained_coords[2]),
            np.linalg.norm(fused.coords[2] - template_coords[2]),
        )
        np.testing.assert_allclose(
            fused.coords[[0, 1, 4, 5]], template_coords[[0, 1, 4, 5]], atol=1e-6
        )
        self.assertGreater(fused.metadata["pretrained_dominant_fraction"], 0.0)

        pretrained_heavy = fuse_template_pretrained(
            template, pretrained, mode="pretrained_heavy"
        )
        self.assertGreater(
            pretrained_heavy.metadata["mean_pretrained_weight"],
            fused.metadata["mean_pretrained_weight"],
        )


class SelectionTests(unittest.TestCase):
    def test_quality_uses_source_local_confidence_ranks(self) -> None:
        coords = np.arange(18, dtype=float).reshape(6, 3)
        support = np.ones(6, dtype=bool)
        candidates = [
            candidate("a-low", "a", "pretrained", coords, np.ones(6), support, 0.1),
            candidate("a-high", "a", "pretrained", coords, np.ones(6), support, 0.2),
            candidate("b-low", "b", "pretrained", coords, np.ones(6), support, 0.8),
            candidate("b-high", "b", "pretrained", coords, np.ones(6), support, 0.9),
        ]
        base = {
            "support_fraction": 1.0,
            "pair_like_fraction": 0.5,
            "angle_nll": 0.5,
            "torsion_nll": 0.5,
            "clash_per_res": 0.0,
            "bb_dev": 0.5,
            "sharp_kinks": 0.0,
        }
        quality = native_blind_quality_scores(candidates, [base] * 4)
        self.assertGreater(quality[1], quality[0])
        self.assertGreater(quality[3], quality[2])
        self.assertAlmostEqual(float(quality[0]), float(quality[2]))

    def test_selector_covers_a_second_fold_when_quality_is_close(self) -> None:
        coords = np.arange(18, dtype=float).reshape(6, 3)
        support = np.ones(6, dtype=bool)
        candidates = [
            candidate(str(i), "tbm", "template", coords, np.ones(6), support, 0.5)
            for i in range(3)
        ]
        similarity = np.array(
            [[1.0, 0.95, 0.2], [0.95, 1.0, 0.2], [0.2, 0.2, 1.0]]
        )
        selected = select_quality_diversity(
            candidates,
            similarity,
            np.array([0, 0, 1]),
            np.array([0.9, 0.85, 0.8]),
            limit=2,
        )
        self.assertIn(2, selected)


if __name__ == "__main__":
    unittest.main()
