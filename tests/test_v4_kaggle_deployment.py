import json
from pathlib import Path
import unittest

import numpy as np

from kaggle.v4_frozen_inference import GEOMETRY_CONFIG, _assemble_3t2d


REPO = Path(__file__).resolve().parents[1]


def fake_tbm(length: int = 4) -> dict[str, np.ndarray]:
    return {
        "coords": np.stack([np.full((length, 3), index) for index in range(5)]),
        "confidence": np.stack([np.full(length, index / 10) for index in range(5)]),
        "global_confidence": np.arange(5) / 10,
    }


def fake_deep(index: int, length: int = 4) -> dict:
    return {
        "coords": np.full((length, 3), 10 + index),
        "confidence": np.full(length, 0.8),
        "global_confidence": 0.8,
        "candidate_id": f"deep_{index}",
    }


class V4KaggleDeploymentTests(unittest.TestCase):
    def test_frozen_geometry_config_matches_preopening_freeze(self) -> None:
        freeze = json.loads(
            (REPO / "reports/thesis_v4/final_freeze/final_method_freeze.json").read_text()
        )
        self.assertEqual(GEOMETRY_CONFIG, freeze["final_pipeline"]["geometry_config"])

    def test_assemble_3t2d_with_two_deep_candidates(self) -> None:
        selected, status = _assemble_3t2d(fake_tbm(), [fake_deep(1), fake_deep(2)])
        self.assertEqual(
            status["candidate_sources"],
            ["template", "template", "template", "drfold2_e2e", "drfold2_e2e"],
        )
        self.assertEqual(status["fallback_template_indices_zero_based"], [])
        self.assertEqual(
            [float(item["coords"][0, 0]) for item in selected], [0, 1, 2, 11, 12]
        )

    def test_one_missing_deep_slot_uses_fifth_template_as_frozen(self) -> None:
        selected, status = _assemble_3t2d(fake_tbm(), [fake_deep(1)])
        self.assertEqual(status["fallback_template_indices_zero_based"], [4])
        self.assertEqual(
            [float(item["coords"][0, 0]) for item in selected], [0, 1, 2, 11, 4]
        )

    def test_two_missing_deep_slots_restore_five_template_bank(self) -> None:
        selected, status = _assemble_3t2d(fake_tbm(), [])
        self.assertEqual(status["fallback_template_indices_zero_based"], [3, 4])
        self.assertEqual(
            [float(item["coords"][0, 0]) for item in selected], [0, 1, 2, 3, 4]
        )


if __name__ == "__main__":
    unittest.main()
