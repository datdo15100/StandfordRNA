"""Phase 3 — parse all PDB_RNA CIF files into a template database.

Outputs:
    data/processed/template_meta.parquet   one row per RNA chain (metadata + seq)
    data/cache/template_coords.pkl         {chain_key: {seq, resid, coords(float32)}}
    data/processed/pdb_parse_report.json    coverage stats + unmapped residue codes

Temporal filtering and self-leakage guards are applied later at *search* time
(Phase 4); here we capture every chain plus its release date so the search can
filter correctly for any cutoff.

Usage:
    python scripts/build_template_db.py [--limit N] [--workers W]
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.cif.parser import parse_cif
from rna3d.paths import cache, comp_dir, comp_file, processed


def load_release_dates() -> dict[str, str]:
    """Robust parse: the file concatenates several download blocks, each with its
    own header row and trailing commas. Take the first two CSV fields per data row
    that looks like ``ID, yyyy-mm-dd``."""
    import csv
    import re

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    out: dict[str, str] = {}
    with open(comp_file("pdb_release_dates"), newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            eid, date = row[0].strip().upper(), row[1].strip()
            if date_re.match(date):
                out[eid] = date
    return out


def _worker(path: str) -> tuple[list[dict], dict, dict]:
    """Parse one CIF; return (meta rows, coords payload, unmapped counts)."""
    unmapped: dict[str, int] = {}
    try:
        recs = parse_cif(path, unmapped_out=unmapped)
    except Exception as e:  # noqa: BLE001 — keep going, report the failure
        return [], {"__error__": {"path": path, "err": str(e)[:200]}}, {}
    meta_rows, coords = [], {}
    for r in recs:
        meta_rows.append({
            "chain_key": r.key, "pdb_id": r.pdb_id, "chain_id": r.chain_id,
            "length": len(r.seq), "n_resolved": r.n_resolved,
            "n_unknown": r.n_unknown, "polymer_type": r.polymer_type, "seq": r.seq,
        })
        coords[r.key] = {"seq": r.seq, "resid": r.resids, "coords": r.coords}
    return meta_rows, coords, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = sorted(glob.glob(str(comp_dir() / "PDB_RNA" / "*.cif")))
    if args.limit:
        files = files[: args.limit]
    print(f"parsing {len(files)} CIF files with {args.workers} workers")

    release = load_release_dates()
    all_meta: list[dict] = []
    all_coords: dict = {}
    unmapped = Counter()
    errors = []

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, f): f for f in files}
        for fut in as_completed(futs):
            meta_rows, coords, unm = fut.result()
            if "__error__" in coords:
                errors.append(coords["__error__"])
            else:
                all_meta.extend(meta_rows)
                all_coords.update(coords)
            unmapped.update(unm)
            done += 1
            if done % 500 == 0 or done == len(files):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(files)}  chains={len(all_meta)}  "
                      f"{rate:.1f} files/s  unmapped_codes={len(unmapped)}")

    meta = pd.DataFrame(all_meta)
    meta["release_date"] = meta["pdb_id"].map(release).fillna("9999-12-31")
    meta = meta.sort_values("chain_key").reset_index(drop=True)

    meta_path = processed() / "template_meta.parquet"
    meta.to_parquet(meta_path, index=False)

    coords_path = cache() / "template_coords.pkl"
    with open(coords_path, "wb") as fh:
        pickle.dump(all_coords, fh, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "n_files": len(files),
        "n_chains": int(len(meta)),
        "n_with_release_date": int((meta["release_date"] != "9999-12-31").sum()),
        "total_residues": int(meta["length"].sum()),
        "total_resolved": int(meta["n_resolved"].sum()),
        "rna_chains": int((meta["polymer_type"] == "Rna").sum()),
        "hybrid_chains": int((meta["polymer_type"] == "DnaRnaHybrid").sum()),
        "n_errors": len(errors),
        "errors_sample": errors[:20],
        "top_unmapped_codes": unmapped.most_common(40),
        "n_unmapped_codes": len(unmapped),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(processed() / "pdb_parse_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nwrote {meta_path}  ({len(meta)} chains)")
    print(f"wrote {coords_path}")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("errors_sample", "top_unmapped_codes")}, indent=2))
    print("top unmapped codes:", unmapped.most_common(20))


if __name__ == "__main__":
    main()
