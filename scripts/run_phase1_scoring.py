"""Phase 1 — scoring sanity + dummy baseline.

Runs three checks that validate the local US-align scoring harness, then scores
a trivial extended-chain baseline (B0) on the CASP15 validation targets.

    native vs native   -> TM ~= 1.0
    native vs rotated  -> TM ~= 1.0  (US-align is rotation invariant)
    native vs mirrored -> TM  < 1.0  (chirality matters; mirror must be penalised)

Outputs:
    reports/tables/phase1_sanity.csv
    reports/tables/phase1_dummy_baseline.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.data import io
from rna3d.eval.usalign import score_target, tm_score, write_c1_pdb
from rna3d.geometry import transforms as T
from rna3d.paths import tables


def first_resolved_reference(labels, target_id):
    refs = io.get_reference_coords(labels, target_id)
    seq = io.get_sequence_from_labels(labels, target_id)
    return refs[0], list(seq)


def run_sanity(labels) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # pick a target with a fully/mostly resolved single structure
    target_ids = labels["ID"].map(io.target_id_of).unique()
    chosen = None
    for tid in target_ids:
        ref, resn = first_resolved_reference(labels, tid)
        if np.isfinite(ref).all(axis=1).sum() >= 20:
            chosen = (tid, ref, resn)
            break
    tid, ref, resn = chosen
    res_mask = np.isfinite(ref).all(axis=1)
    ref_r = ref[res_mask]
    resn_r = [resn[i] for i, m in enumerate(res_mask) if m]

    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        native = td / "native.pdb"
        write_c1_pdb(ref_r, resn_r, native)

        # native vs native
        write_c1_pdb(ref_r, resn_r, td / "a.pdb")
        rows.append(("native_vs_native", tm_score(td / "a.pdb", native)))

        # native vs rotated+translated
        R = T.random_rotation(rng)
        rot = T.apply_rigid(ref_r, R, t=np.array([10.0, -5.0, 3.0]))
        write_c1_pdb(rot, resn_r, td / "b.pdb")
        rows.append(("native_vs_rotated", tm_score(td / "b.pdb", native)))

        # native vs mirrored
        mir = T.mirror(ref_r)
        write_c1_pdb(mir, resn_r, td / "c.pdb")
        rows.append(("native_vs_mirrored", tm_score(td / "c.pdb", native)))

    df = pd.DataFrame(rows, columns=["check", "tm_score"])
    df["target_used"] = tid
    return df


def run_dummy_baseline(sequences, labels) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    target_ids = labels["ID"].map(io.target_id_of).unique()
    for tid in target_ids:
        if tid not in set(sequences["target_id"]):
            continue
        refs = io.get_reference_coords(labels, tid)
        seq = list(io.get_sequence_from_labels(labels, tid))
        L = len(seq)
        preds = [T.extended_chain(L, rng=rng) for _ in range(5)]
        tm = score_target(preds, refs, seq)
        rows.append({"target_id": tid, "seq_len": L, "n_refs": len(refs), "tm_best_of_5": tm})
    df = pd.DataFrame(rows)
    return df


def main():
    labels = io.load_labels("validation")
    sequences = io.load_sequences("validation")

    print("== Sanity checks ==")
    sanity = run_sanity(labels)
    print(sanity.to_string(index=False))
    sanity.to_csv(tables() / "phase1_sanity.csv", index=False)

    print("\n== Dummy extended-chain baseline (B0) on CASP15 validation ==")
    dummy = run_dummy_baseline(sequences, labels)
    print(dummy.to_string(index=False))
    print(f"\nMean best-of-5 TM (B0 dummy): {dummy['tm_best_of_5'].mean():.4f}")
    dummy.to_csv(tables() / "phase1_dummy_baseline.csv", index=False)


if __name__ == "__main__":
    main()
