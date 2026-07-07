"""Phase 4/5/7/8/9 — end-to-end evaluation + ablation on CASP15 validation.

Steps:
  1. MMseqs nucleotide search of all validation targets vs the full template DB.
  2. Per target: build temporal-safe, leakage-guarded TBM candidates.
  3. Produce five structures for each method / ablation.
  4. Score best-of-5 TM with US-align; record template stats and self-TM diversity.

Leakage controls:
  - templates filtered to release_date < target temporal_cutoff (hard gate);
  - the target's own PDB id excluded;
  - geometry priors were estimated only from pre-2022-05-27 structures.

Outputs:
  reports/tables/eval_methods.csv        per-target x per-method best-of-5 TM
  reports/tables/eval_summary.csv        mean TM per method
  reports/tables/eval_template_stats.csv per-target template availability
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.data import io
from rna3d.eval.self_tm import mean_pairwise_self_tm
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables
from rna3d.pipeline import methods as M
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.refine.optimizer import RefineConfig
from rna3d.template import db, mmseqs_search


def guess_pdb_ids(all_sequences: str | float) -> tuple[str, ...]:
    """Pull PDB ids out of the all_sequences FASTA headers (e.g. '>7QR4_1|...')."""
    if not isinstance(all_sequences, str):
        return ()
    import re
    ids = re.findall(r">([0-9][A-Za-z0-9]{3})_", all_sequences)
    return tuple(s.upper() for s in set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=None, help="limit number of targets")
    ap.add_argument("--steps", type=int, default=300, help="refinement steps")
    ap.add_argument("--max-len", type=int, default=600, help="skip targets longer than this")
    args = ap.parse_args()

    cutoff_default = casp15_safe_cutoff()
    seqs = io.load_sequences("validation")
    labels = io.load_labels("validation")
    meta = db.load_meta()
    db.load_coords()  # warm cache
    priors = json.load(open(processed() / "geometry_priors.json"))

    targets = list(seqs["target_id"])
    if args.targets:
        targets = targets[: args.targets]

    # 1. one batched mmseqs search
    qf = cache() / "validation_query.fasta"
    with open(qf, "w") as fh:
        for _, r in seqs.iterrows():
            if r["target_id"] in targets:
                fh.write(f">{r['target_id']}\n{r['sequence']}\n")
    print("[search] running mmseqs ...")
    t0 = time.time()
    hits = mmseqs_search.search(qf, cache() / "validation_hits.m8")
    print(f"[search] {len(hits)} hits in {time.time()-t0:.1f}s")

    refine_cfg = RefineConfig(steps=args.steps)
    rows, tmpl_rows = [], []
    for tid in targets:
        srow = seqs[seqs["target_id"] == tid].iloc[0]
        seq = srow["sequence"]
        L = len(seq)
        cutoff = srow["temporal_cutoff"] or cutoff_default
        resnames = list(seq)
        refs = io.get_reference_coords(labels, tid)
        rng = np.random.default_rng(0)

        if L > args.max_len:
            print(f"[{tid}] L={L} > max-len, dummy only")
            cands = []
        else:
            thits = hits[hits["query"] == tid]
            cands = build_tbm_candidates(
                tid, seq, cutoff, thits, meta, rng=rng,
                adj_dist=priors["adjacent_c1"]["mean"],
                exclude_pdb_ids=guess_pdb_ids(srow.get("all_sequences")),
            )
        best_conf = cands[0].confidence if cands else 0.0
        tmpl_rows.append({"target_id": tid, "seq_len": L, "n_candidates": len(cands),
                          "best_conf": round(best_conf, 4),
                          "best_template": cands[0].chain_key if cands else "",
                          "best_identity": round(cands[0].identity, 3) if cands else 0.0,
                          "best_coverage": round(cands[0].coverage, 3) if cands else 0.0})

        method_preds = {
            "B0_dummy": M.m_dummy(L, np.random.default_rng(0)),
            "B1_tbm_top1": M.m_tbm_top1(cands, L, np.random.default_rng(1)),
            "B2_tbm_top5": M.m_tbm_top5(cands, L, np.random.default_rng(2)),
            "B4_tbm_refined": M.m_tbm_refined(cands, L, priors, np.random.default_rng(3), cfg=refine_cfg),
            "A_no_clash": M.m_tbm_refined(cands, L, priors, np.random.default_rng(4), cfg=refine_cfg, use_clash=False),
            "A_no_rg": M.m_tbm_refined(cands, L, priors, np.random.default_rng(5), cfg=refine_cfg, use_rg=False),
            "A_no_gapweights": M.m_tbm_refined(cands, L, priors, np.random.default_rng(6), cfg=refine_cfg, gap_aware=False),
        }

        row = {"target_id": tid, "seq_len": L, "n_refs": len(refs), "best_conf": round(best_conf, 4)}
        for name, preds in method_preds.items():
            tm = score_target([preds[k] for k in range(5)], refs, resnames)
            row[name] = round(tm, 4)
        # diversity of the headline method
        row["selftm_B4"] = round(mean_pairwise_self_tm(
            [method_preds["B4_tbm_refined"][k] for k in range(5)], resnames), 4)
        rows.append(row)
        print(f"[{tid}] L={L} cand={len(cands)} conf={best_conf:.3f} "
              f"top1={row['B1_tbm_top1']:.3f} top5={row['B2_tbm_top5']:.3f} "
              f"refined={row['B4_tbm_refined']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(tables() / "eval_methods.csv", index=False)
    pd.DataFrame(tmpl_rows).to_csv(tables() / "eval_template_stats.csv", index=False)

    method_cols = [c for c in res.columns if c.startswith(("B", "A_"))]
    summary = res[method_cols].mean().sort_values(ascending=False)
    summary.to_csv(tables() / "eval_summary.csv", header=["mean_tm"])
    print("\n==== mean best-of-5 TM by method (CASP15 validation) ====")
    print(summary.to_string())


if __name__ == "__main__":
    main()
