"""Data-free tests for synthetic confidence-gate data and model shapes."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from rna3d.geofuse.phase_d import (
    FEATURE_NAMES,
    ConfidenceGate1D,
    fuse_with_learned_gate,
    pair_gate_features,
)
from scripts.train_geofuse_phase_d import (
    collate_examples,
    make_example,
    synthetic_pair,
    temporal_three_way_split,
    temporal_split,
)


def flat_prior(lo: float, hi: float, periodic: bool = False) -> dict:
    return {
        "lo": lo,
        "hi": hi,
        "bins": 12,
        "periodic": periodic,
        "nll": [0.0] * 12,
    }


def priors() -> tuple[dict, dict]:
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
    return v1, {"contexts": contexts}


def native_chain(length: int = 40) -> np.ndarray:
    index = np.arange(length, dtype=float)
    return np.column_stack(
        [5.5 * index, 2.0 * np.sin(index / 3.0), 2.0 * np.cos(index / 4.0)]
    )


class SyntheticGateTests(unittest.TestCase):
    def test_synthetic_pair_and_features_are_finite(self) -> None:
        sequence = ("ACGU" * 10)[:40]
        template, pretrained = synthetic_pair(
            native_chain(), sequence, "T", np.random.default_rng(4)
        )
        v1, v2 = priors()
        features, aligned, metadata = pair_gate_features(
            template, pretrained, v1, v2
        )
        self.assertEqual(features.shape, (40, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.isfinite(aligned).all())
        self.assertGreater(metadata["alignment_inlier_fraction"], 0.0)

    def test_example_and_collation_preserve_variable_length_mask(self) -> None:
        v1, v2 = priors()
        examples = [
            make_example(
                native_chain(length),
                ("ACGU" * 20)[:length],
                f"T{length}",
                np.random.default_rng(length),
                v1,
                v2,
            )
            for length in (32, 40)
        ]
        batch = collate_examples(examples)
        self.assertEqual(tuple(batch["features"].shape), (2, 40, len(FEATURE_NAMES)))
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), [32, 40])

    def test_gate_returns_one_logit_per_residue(self) -> None:
        model = ConfidenceGate1D()
        features = torch.zeros(3, 41, len(FEATURE_NAMES))
        self.assertEqual(tuple(model(features).shape), (3, 41))

    def test_frozen_gate_builds_finite_fused_candidate(self) -> None:
        sequence = ("ACGU" * 10)[:40]
        template, pretrained = synthetic_pair(
            native_chain(), sequence, "T", np.random.default_rng(9)
        )
        model = ConfidenceGate1D()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        checkpoint = {
            "model": model,
            "feature_mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            "feature_std": np.ones(len(FEATURE_NAMES), dtype=np.float32),
            "training": {"decision_threshold": 0.5},
        }
        v1, v2 = priors()
        fused = fuse_with_learned_gate(
            template, pretrained, v1, v2, checkpoint
        )
        self.assertEqual(fused.source, "geofuse_learned")
        self.assertTrue(np.isfinite(fused.coords).all())
        self.assertAlmostEqual(fused.metadata["mean_pretrained_probability"], 0.5)

    def test_temporal_split_has_no_target_or_date_overlap(self) -> None:
        frame = pd.DataFrame(
            {
                "target_id": [f"T{i}" for i in range(10)],
                "date": pd.date_range("2000-01-01", periods=10),
            }
        )
        train, validation = temporal_split(frame, 0.2)
        self.assertTrue(set(train.target_id).isdisjoint(set(validation.target_id)))
        self.assertLess(train.date.max(), validation.date.min())

        train, calibration, validation = temporal_three_way_split(
            pd.DataFrame(
                {
                    "target_id": [f"X{i}" for i in range(20)],
                    "date": pd.date_range("2000-01-01", periods=20),
                }
            ),
            calibration_fraction=0.1,
            validation_fraction=0.2,
        )
        self.assertLess(train.date.max(), calibration.date.min())
        self.assertLess(calibration.date.max(), validation.date.min())

        tied = pd.DataFrame(
            {
                "target_id": [f"D{i}" for i in range(20)],
                "date": pd.to_datetime([f"2000-01-{i // 2 + 1:02d}" for i in range(20)]),
            }
        )
        train, calibration, validation = temporal_three_way_split(tied, 0.1, 0.2)
        self.assertTrue(set(train.date).isdisjoint(set(calibration.date)))
        self.assertTrue(set(calibration.date).isdisjoint(set(validation.date)))


if __name__ == "__main__":
    unittest.main()
