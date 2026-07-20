"""Data-free tests for GeoFuse context geometry and v2 projection."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rna3d.geofuse.geometry_v2 import (
    histogram_nll,
    pair_like_mask,
    pseudo_angles,
    signed_pseudo_torsions,
)
from rna3d.geofuse.refine_v2 import (
    GeometryV2Config,
    backbone_huber_loss,
    histogram_nll_loss,
    kink_regression_loss,
    refine_structure_v2,
    torch_pseudo_angles,
    torch_signed_pseudo_torsions,
)


def flat_prior(lo: float, hi: float, bins: int = 12, periodic: bool = False) -> dict:
    return {
        "lo": lo,
        "hi": hi,
        "bins": bins,
        "periodic": periodic,
        "nll": [0.0] * bins,
    }


class GeometryPrimitiveTests(unittest.TestCase):
    def test_pseudo_angle_is_rigid_invariant(self) -> None:
        coords = np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0]])
        transformed = coords @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 7.0
        self.assertAlmostEqual(float(pseudo_angles(coords)[0]), np.pi / 2)
        np.testing.assert_allclose(pseudo_angles(coords), pseudo_angles(transformed))

    def test_signed_torsion_changes_sign_under_reflection(self) -> None:
        coords = np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [2.0, 1, 1]], dtype=float
        )
        mirrored = coords.copy()
        mirrored[:, 0] *= -1
        original = signed_pseudo_torsions(coords)[0]
        reflected = signed_pseudo_torsions(mirrored)[0]
        self.assertGreater(abs(original), 1e-3)
        self.assertAlmostEqual(float(original), -float(reflected), places=6)

    def test_pair_like_mask_uses_complement_distance_and_one_partner(self) -> None:
        sequence = "ACCCU"
        coords = np.array(
            [[0.0, 0, 0], [50.0, 0, 0], [60.0, 0, 0], [70.0, 0, 0], [10.5, 0, 0]]
        )
        np.testing.assert_array_equal(
            pair_like_mask(sequence, coords), [True, False, False, False, True]
        )

    def test_histogram_lookup_preserves_nan_without_invalid_index(self) -> None:
        prior = flat_prior(-np.pi, np.pi, periodic=True)
        result = histogram_nll(np.array([0.0, np.nan]), prior)
        self.assertTrue(np.isfinite(result[0]))
        self.assertTrue(np.isnan(result[1]))


class GeometryV2RefinementTests(unittest.TestCase):
    def test_robust_backbone_loss_caps_outlier_gradient(self) -> None:
        coords = torch.tensor(
            [[0.0, 0, 0], [6.0, 0, 0], [106.0, 0, 0]], requires_grad=True
        )
        loss = backbone_huber_loss(coords, mean=6.0, std=1.0, delta=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(float(coords.grad.abs().max()), 1.01)

    def test_kink_barrier_prevents_regression_without_forcing_raw_kink(self) -> None:
        source = torch.deg2rad(torch.tensor([90.0, 50.0]))
        current = torch.deg2rad(torch.tensor([60.0, 55.0]))
        floor = torch.deg2rad(torch.tensor(70.0))
        loss = kink_regression_loss(current, source, floor)
        expected = torch.deg2rad(torch.tensor(10.0)).square()
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_histogram_loss_has_finite_gradient(self) -> None:
        prior = flat_prior(-np.pi, np.pi, periodic=True)
        prior["nll"][4] = 2.0
        values = torch.tensor([0.1, -2.0], requires_grad=True)
        loss = histogram_nll_loss(values, prior)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_periodic_histogram_wraps_float32_pi_boundaries(self) -> None:
        prior = flat_prior(-np.pi, np.pi, periodic=True)
        values = torch.tensor([np.pi, -np.pi], dtype=torch.float32, requires_grad=True)
        loss = histogram_nll_loss(values, prior)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_torch_and_numpy_torsions_agree(self) -> None:
        coords = np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [2.0, 1, 1]], dtype=np.float32
        )
        expected = signed_pseudo_torsions(coords)
        actual = torch_signed_pseudo_torsions(torch.tensor(coords)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_degenerate_geometry_is_ignored_without_nan_gradient(self) -> None:
        coords = torch.tensor(
            [[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]],
            requires_grad=True,
        )
        values = torch_signed_pseudo_torsions(coords)
        self.assertTrue(torch.isnan(values).all())
        loss = histogram_nll_loss(values, flat_prior(-np.pi, np.pi, periodic=True))
        loss.backward()
        self.assertTrue(torch.isfinite(coords.grad).all())
        self.assertTrue(torch.isnan(torch_pseudo_angles(coords)).any())

    def test_short_projection_returns_finite_shape(self) -> None:
        sequence = "AAAAA"
        coords = np.array(
            [[0.0, 0, 0], [6.0, 0, 0], [11.0, 2, 0], [16.0, 4, 1], [21.0, 5, 3]],
            dtype=np.float32,
        )
        v1 = {
            "adjacent_c1": {"mean": 6.0, "std": 1.0},
            "clash": {"r_min": 4.0},
            "rg_powerlaw": {"a": 2.0, "b": 0.5},
        }
        contexts = {
            name: {
                "angle": flat_prior(0.0, np.pi),
                "torsion": flat_prior(-np.pi, np.pi, periodic=True),
            }
            for name in ("pair_like", "unpaired")
        }
        result, info = refine_structure_v2(
            coords,
            sequence,
            v1,
            {"contexts": contexts},
            cfg=GeometryV2Config(steps=3),
        )
        self.assertEqual(result.shape, coords.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(len(info["history"]), 2)


if __name__ == "__main__":
    unittest.main()
