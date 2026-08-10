from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from kaggle.hybrid_inference import read_c1_pdb, residue_confidence


class HybridInferenceHelpersTests(unittest.TestCase):
    def test_pair_confidence_becomes_symmetric_residue_confidence(self) -> None:
        pair = np.asarray([[0.2, 0.6], [0.4, 0.8]], dtype=np.float32)
        actual = residue_confidence(pair)
        expected = 0.5 * (pair.mean(axis=0) + pair.mean(axis=1))
        np.testing.assert_allclose(actual, expected)

    def test_read_c1_pdb_ignores_other_atoms_and_duplicate_altloc(self) -> None:
        rows = [
            "ATOM      1  P     A A   1       0.000   0.000   0.000  1.00  1.00           P\n",
            "ATOM      2  C1'   A A   1       1.000   2.000   3.000  1.00  1.00           C\n",
            "ATOM      3  C1'   A A   1       9.000   9.000   9.000  1.00  1.00           C\n",
            "ATOM      4  C1'   U A   2       4.000   5.000   6.000  1.00  1.00           C\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pdb"
            path.write_text("".join(rows))
            coords = read_c1_pdb(path, expected_length=2)
        np.testing.assert_allclose(coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_read_c1_pdb_rejects_wrong_length(self) -> None:
        row = "ATOM      1  C1'   A A   1       1.000   2.000   3.000  1.00  1.00           C\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pdb"
            path.write_text(row)
            with self.assertRaises(ValueError):
                read_c1_pdb(path, expected_length=2)


if __name__ == "__main__":
    unittest.main()
