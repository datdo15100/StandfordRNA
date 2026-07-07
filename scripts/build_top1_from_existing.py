"""Build the top-1-style template set from our already-parsed template DB (instant).

Re-parsing 8.6k CIFs with gemmi is I/O-bound (~35 min on the /mnt mount) and redundant
— we already parsed them into data/cache/template_coords.pkl. We reconstruct a
"top-1 style" template library from it:
  - keep only canonical A/U/G/C positions that also have a resolved C1' (so seq and
    coords stay index-aligned — cleaner than the original's occasional seq/coord
    length mismatch);
  - de-duplicate by sequence, keeping the EARLIEST release date (correct for the
    temporal-safe check: a sequence is "available" from its first deposition) and one
    coordinate representative — this keeps the brute-force search tractable.

Note vs a byte-exact reproduction: our library maps modified bases to canonical (so it
keeps a few residues the original dropped) and is a superset — if anything this
*helps* the baseline. Documented in reports/thesis_notes/reproduce_top1.md.

Outputs the same files the runner expects:
    data/processed/top1_template_meta.parquet
    data/cache/top1_template_coords.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.paths import cache, processed
from rna3d.template import db

_CANON = set("ACGU")


def main():
    meta = db.load_meta()
    coords_db = db.load_coords()
    rel = dict(zip(meta["chain_key"], meta["release_date"]))
    pdb_of = dict(zip(meta["chain_key"], meta["pdb_id"]))

    best_by_seq: dict[str, dict] = {}
    for key, d in coords_db.items():
        seq = d["seq"]
        coords = np.asarray(d["coords"], float)
        mask = np.array([c in _CANON for c in seq]) & np.isfinite(coords).all(axis=1)
        if mask.sum() < 8:
            continue
        seq2 = "".join(c for c, m in zip(seq, mask) if m)
        coords2 = coords[mask]
        rd = str(rel.get(key, "9999-12-31"))
        prev = best_by_seq.get(seq2)
        if prev is None or rd < prev["release_date"]:
            best_by_seq[seq2] = {
                "target_id": key, "pdb_id": str(pdb_of.get(key, key.split("_")[0])),
                "sequence": seq2, "release_date": rd,
                "resid": np.arange(1, len(seq2) + 1, dtype=np.int32),
                "coords": coords2.astype(np.float32),
            }

    meta_rows, coords_out = [], {}
    for rec in best_by_seq.values():
        tid = rec["target_id"]
        meta_rows.append({"target_id": tid, "pdb_id": rec["pdb_id"],
                          "sequence": rec["sequence"], "length": len(rec["sequence"]),
                          "release_date": rec["release_date"]})
        coords_out[tid] = {"seq": rec["sequence"], "resid": rec["resid"],
                           "coords": rec["coords"]}

    mdf = pd.DataFrame(meta_rows)
    mdf.to_parquet(processed() / "top1_template_meta.parquet", index=False)
    with open(cache() / "top1_template_coords.pkl", "wb") as fh:
        pickle.dump(coords_out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"top-1 template library: {len(mdf)} unique sequences "
          f"(from {len(coords_db)} chains; 1st-place notebook had 18,881)")
    print(f"length: min {mdf.length.min()} / median {int(mdf.length.median())} / max {mdf.length.max()}")


if __name__ == "__main__":
    main()
