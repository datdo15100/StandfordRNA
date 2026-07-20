"""Data-free tests for the GeoFuse Phase-A candidate gate."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rna3d.geofuse.candidate import CandidateCache, StructureCandidate
from rna3d.geofuse.overlap import audit_exact_sequence_overlap, read_fasta_sequences
from rna3d.geofuse.phase_a import evaluate_candidate_pool, select_source_balanced
from rna3d.geofuse.structure_io import import_structure


def make_candidate(
    candidate_id: str,
    *,
    kind: str,
    source: str,
    confidence: float,
    offset: float = 0.0,
) -> StructureCandidate:
    sequence = "ACG"
    coords = np.arange(9, dtype=np.float32).reshape(3, 3) + offset
    return StructureCandidate(
        target_id="T1",
        sequence=sequence,
        candidate_id=candidate_id,
        kind=kind,
        source=source,
        model="test",
        coords=coords,
        confidence=np.full(3, confidence),
        support_mask=np.ones(3, dtype=bool),
        global_confidence=confidence,
        metadata={"offset": offset},
        priors={"distance": np.eye(3, dtype=np.float32)},
    )


class CandidateCacheTests(unittest.TestCase):
    def test_round_trip_is_pickle_free_and_preserves_priors(self) -> None:
        candidate = make_candidate(
            "drfold2__cfg97__01", kind="pretrained", source="drfold2", confidence=0.8
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateCache(tmp, "validation")
            path = store.save(candidate)
            loaded = store.load_file(path)

        self.assertEqual(loaded.candidate_id, candidate.candidate_id)
        self.assertEqual(loaded.metadata, candidate.metadata)
        np.testing.assert_array_equal(loaded.coords, candidate.coords)
        np.testing.assert_array_equal(loaded.priors["distance"], candidate.priors["distance"])

    def test_sequence_mismatch_rejects_stale_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateCache(tmp, "validation")
            store.save(make_candidate("tbm__one", kind="template", source="tbm", confidence=0.7))
            with self.assertRaisesRegex(ValueError, "stale candidates"):
                store.load_target("T1", "AAA")

    def test_pdb_and_safe_sidecars_are_imported(self) -> None:
        pdb_lines = []
        for index in range(3):
            pdb_lines.append(
                f"ATOM  {index + 1:5d}  C1'   A A{index + 1:4d}    "
                f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_1.pdb"
            path.write_text("\n".join(pdb_lines) + "\nEND\n")
            np.savez_compressed(Path(tmp) / "plddt_model_1.npz", plddt=np.array([0.2, 0.5, 0.8]))
            np.savez_compressed(
                Path(tmp) / "priors_model_1.npz", dist_p=np.zeros((3, 3, 4), dtype=np.float16)
            )
            candidate = import_structure(
                path,
                target_id="T1",
                sequence="AAA",
                candidate_id="drfold2__01",
                source="drfold2",
                model="cfg97",
            )

        np.testing.assert_allclose(candidate.confidence, [0.2, 0.5, 0.8])
        self.assertIn("dist_p", candidate.priors)


class PhaseAGateTests(unittest.TestCase):
    def test_balanced_selection_keeps_both_sources(self) -> None:
        candidates = [
            make_candidate(f"tbm__{i}", kind="template", source="tbm", confidence=0.9 - i / 10)
            for i in range(4)
        ] + [
            make_candidate("drfold2__1", kind="pretrained", source="drfold2", confidence=0.1)
        ]
        selected = select_source_balanced(candidates, limit=3)
        self.assertEqual({candidate.source for candidate in selected}, {"tbm", "drfold2"})

    def test_oracle_gain_is_separate_from_confidence_selection(self) -> None:
        sequences = pd.DataFrame({"target_id": ["T1"], "sequence": ["ACG"]})
        labels = pd.DataFrame(
            {
                "ID": ["T1_1", "T1_2", "T1_3"],
                "resname": list("ACG"),
                "resid": [1, 2, 3],
                "x_1": [0.0, 1.0, 2.0],
                "y_1": [0.0, 0.0, 0.0],
                "z_1": [0.0, 0.0, 0.0],
            }
        )
        score_by_offset = {0.0: 0.4, 10.0: 0.9}

        def scorer(predictions, references, resnames):
            del references, resnames
            return score_by_offset[float(predictions[0][0, 0])]

        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateCache(tmp, "validation")
            store.save(
                make_candidate("tbm__one", kind="template", source="tbm", confidence=0.9)
            )
            store.save(
                make_candidate(
                    "drfold2__one",
                    kind="pretrained",
                    source="drfold2",
                    confidence=0.1,
                    offset=10.0,
                )
            )
            _, targets, summary = evaluate_candidate_pool(
                sequences, labels, store, scorer=scorer
            )

        self.assertAlmostEqual(targets.iloc[0]["oracle_gain_over_tbm"], 0.5)
        self.assertAlmostEqual(summary["mean_oracle_gain_over_tbm"], 0.5)


class PretrainedOverlapTests(unittest.TestCase):
    def test_exact_overlap_audit_normalizes_dna_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "train.fasta"
            fasta.write_text(">same\nAGTC\n>different\nCCCC\n")
            self.assertEqual(read_fasta_sequences(fasta)[0], ("same", "AGUC"))
            targets = pd.DataFrame(
                [
                    {"target_id": "T1", "sequence": "AGUC"},
                    {"target_id": "T2", "sequence": "AAAA"},
                ]
            )
            result = audit_exact_sequence_overlap(targets, [("model", fasta)])
            matched = result.set_index("target_id")
            self.assertTrue(bool(matched.loc["T1", "exact_overlap"]))
            self.assertEqual(matched.loc["T1", "matching_training_ids"], "same")
            self.assertFalse(bool(matched.loc["T2", "exact_overlap"]))


if __name__ == "__main__":
    unittest.main()
