"""Reproduce the 1st-place TBM-only pipeline and score it on the 12 CASP15 targets.

Two modes on the SAME method (their notebook only ever ran the first):
  full_pdb      : all templates, NO temporal filter (their actual setup — natives
                  are in the PDB dump, so this is the *leaked* CASP15 number).
  temporal_safe : templates with release_date < target cutoff (honest number,
                  directly comparable to our pipeline's 0.16).

This is the faithful baseline the thesis compares against. It CANNOT reproduce their
0.593 private-leaderboard score — that is measured on ~40 hidden private targets, not
the 12 public CASP15 targets we can score locally.

Output: reports/tables/reproduce_top1.csv + reports/thesis_notes/reproduce_top1.md
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.baselines.top1 import predict_structures
from rna3d.data import io
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"


def native_pdb_ids(all_seq) -> set:
    if not isinstance(all_seq, str):
        return set()
    return {s.upper() for s in re.findall(r">([0-9][A-Za-z0-9]{3})_", all_seq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=900)
    args = ap.parse_args()

    meta = pd.read_parquet(processed() / "top1_template_meta.parquet")
    with open(cache() / "top1_template_coords.pkl", "rb") as fh:
        coords_db = pickle.load(fh)
    seqs = io.load_sequences("validation")
    labels = io.load_labels("validation")
    targets = list(seqs["target_id"])[: args.targets] if args.targets else list(seqs["target_id"])

    # pre-index template metadata
    meta = meta.reset_index(drop=True)
    rel = dict(zip(meta["target_id"], meta["release_date"]))
    pdb_of = dict(zip(meta["target_id"], meta["pdb_id"]))

    rows = []
    for tid in targets:
        sr = seqs[seqs["target_id"] == tid].iloc[0]
        seq, L = sr["sequence"], len(sr["sequence"])
        if L > args.max_len:
            continue
        cutoff = sr["temporal_cutoff"] or casp15_safe_cutoff()
        natives = native_pdb_ids(sr.get("all_sequences"))
        refs = io.get_reference_coords(labels, tid)
        resn = list(seq)

        def templates(mode):
            out = []
            for t in meta["target_id"]:
                if mode == "temporal_safe":
                    if not (str(rel[t]) < str(cutoff)):
                        continue
                    if str(pdb_of[t]).upper() in natives:
                        continue
                d = coords_db[t]
                out.append((t, d["seq"], np.asarray(d["coords"], float)))
            return out

        row = {"target_id": tid, "seq_len": L}
        for mode in ("full_pdb", "temporal_safe"):
            t0 = time.time()
            preds = predict_structures(seq, tid, templates(mode), n=5)
            tm = score_target(preds, refs, resn)
            row[mode] = round(tm, 4)
            row[f"{mode}_sec"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{tid}] L={L} full_pdb={row['full_pdb']:.3f} "
              f"temporal_safe={row['temporal_safe']:.3f} "
              f"({row['full_pdb_sec']:.0f}s/{row['temporal_safe_sec']:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(tables() / "reproduce_top1.csv", index=False)
    md = [
        "# 1st-place TBM-only — faithful reproduction on the 12 CASP15 targets\n",
        "Best-of-5 TM (US-align). Same method (composite similarity + KMeans diversity + "
        "transfer + rule-based refine + de novo), scored under two template regimes.\n",
        f"- **full_pdb (their setup — no temporal filter, LEAKED): {df['full_pdb'].mean():.4f}**",
        f"- **temporal_safe (honest): {df['temporal_safe'].mean():.4f}**\n",
        f"Leakage on CASP15 = **{df['full_pdb'].mean()-df['temporal_safe'].mean():+.4f}** TM. "
        "Their public 0.593 is a *private-set* score (≈40 hidden targets), NOT reproducible "
        "on these 12 public targets; full_pdb here is the local leaked proxy.\n",
        df.round(4).to_markdown(index=False),
    ]
    THESIS.mkdir(parents=True, exist_ok=True)
    (THESIS / "reproduce_top1.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[:8]))


if __name__ == "__main__":
    main()
