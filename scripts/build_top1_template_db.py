"""Reproduce the 1st-place `rna-cif-to-csv` template set — faithfully but fast.

The 1st-place notebook (`utilities/top1_rna_cif_to_csv.py`) parses every PDB_RNA CIF
with Biopython's MMCIFParser (took ~8 h) and extracts, per chain:
  - only residues whose resname is literally one of A/U/G/C (modified bases DROPPED);
  - the C1' coordinate for those residues that have one (resid = 1-based index over the
    kept residues; residues without C1' leave a gap);
  - de-duplicating identical sequences *within the same CIF file*.

We replicate that exact filter with gemmi (minutes, not hours). We ALSO attach the
release date per PDB id so the reproduction can be scored both leaked (all templates)
and temporal-safe (release < cutoff) — the notebook itself never filters by date.

Outputs:
    data/processed/top1_template_meta.parquet   (target_id, seq, length, release_date)
    data/cache/top1_template_coords.pkl          {target_id: {seq, resid, coords}}
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.paths import cache, comp_dir, processed

_CANON = {"A", "U", "G", "C"}


def _worker(path: str):
    pdb_id = Path(path).stem.upper()
    try:
        st = gemmi.read_structure(path)
    except Exception as e:  # noqa: BLE001
        return [], {}, {"path": path, "err": str(e)[:150]}
    if len(st) == 0:
        return [], {}, None
    meta_rows, coords = [], {}
    seen = set()  # de-dup identical sequences within this file (their behaviour)
    for chain in st[0]:
        rna_res = [r for r in chain if r.name.strip() in _CANON]
        if not rna_res:
            continue
        seq = "".join(r.name.strip() for r in rna_res)
        if seq in seen:
            continue
        seen.add(seq)
        target_id = f"{pdb_id}_{chain.name}"
        resids, xyz = [], []
        for i, r in enumerate(rna_res, 1):
            atom = None
            for a in r:
                if a.name == "C1'":
                    atom = a
                    break
            if atom is not None:
                resids.append(i)
                xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
        if not xyz:
            continue
        coords_arr = np.asarray(xyz, dtype=np.float32)
        meta_rows.append({"target_id": target_id, "pdb_id": pdb_id,
                          "sequence": seq, "length": len(seq),
                          "n_coords": len(xyz)})
        coords[target_id] = {"seq": seq, "resid": np.asarray(resids, np.int32),
                             "coords": coords_arr}
    return meta_rows, coords, None


def load_release_dates() -> dict:
    import csv
    import re
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    out = {}
    with open(comp_dir() / "PDB_RNA" / "pdb_release_dates_NA.csv", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and date_re.match(row[1].strip()):
                out[row[0].strip().upper()] = row[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(str(comp_dir() / "PDB_RNA" / "*.cif")))
    if args.limit:
        files = files[: args.limit]
    print(f"parsing {len(files)} CIFs (top-1 AUGC-only filter), {args.workers} workers")

    release = load_release_dates()
    all_meta, all_coords, errors = [], {}, []
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed(ex.submit(_worker, f) for f in files):
            meta_rows, coords, err = fut.result()
            if err:
                errors.append(err)
            all_meta.extend(meta_rows)
            all_coords.update(coords)
            done += 1
            if done % 1000 == 0 or done == len(files):
                print(f"  {done}/{len(files)}  chains={len(all_meta)}  "
                      f"{done/(time.time()-t0):.1f} files/s")

    meta = pd.DataFrame(all_meta)
    meta["release_date"] = meta["pdb_id"].map(release).fillna("9999-12-31")
    meta.to_parquet(processed() / "top1_template_meta.parquet", index=False)
    with open(cache() / "top1_template_coords.pkl", "wb") as fh:
        pickle.dump(all_coords, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nunique RNA chain-sequences: {len(meta)}   errors: {len(errors)}")
    print(f"(1st-place notebook reported 18,881 unique RNA sequences)")
    print(f"wrote data/processed/top1_template_meta.parquet + data/cache/top1_template_coords.pkl")


if __name__ == "__main__":
    main()
