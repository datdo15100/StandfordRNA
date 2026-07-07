"""Ablation: does the composite-similarity search fallback close the gap to top-1?

For each CASP15 target, score best-of-5 TM (temporal-safe) with our gradient pipeline
under two settings:
  comp_off : MMseqs only (de novo fallback when 0 hits) — our previous pipeline
  comp_on  : MMseqs + exhaustive composite similarity, always merged and re-ranked by
             confidence (keeps the best template from either source)
and line them up against the faithfully-reproduced 1st-place temporal-safe score.

Outputs:
  reports/tables/composite_ablation.csv
  reports/thesis_notes/composite_ablation.md
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.data import io
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables
from rna3d.pipeline import methods as M
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.refine.optimizer import RefineConfig
from rna3d.template import db, mmseqs_search

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"


def native_ids(a):
    return tuple(x.upper() for x in set(re.findall(r">([0-9][A-Za-z0-9]{3})_", a))) \
        if isinstance(a, str) else ()


def main():
    seqs = io.load_sequences("validation")
    labels = io.load_labels("validation")
    meta = db.load_meta(); db.load_coords()
    priors = json.load(open(processed() / "geometry_priors.json"))
    cfg = RefineConfig(steps=300)
    qf = cache() / "validation_query.fasta"
    if not qf.exists():
        with open(qf, "w") as fh:
            for _, r in seqs.iterrows():
                fh.write(f">{r['target_id']}\n{r['sequence']}\n")
    hits = mmseqs_search.search(qf, cache() / "validation_hits.m8")

    try:
        top1 = pd.read_csv(tables() / "reproduce_top1.csv")[["target_id", "temporal_safe"]]
        top1 = dict(zip(top1["target_id"], top1["temporal_safe"]))
    except Exception:
        top1 = {}

    rows = []
    for _, sr in seqs.iterrows():
        tid, seq, L = sr["target_id"], sr["sequence"], len(sr["sequence"])
        if L > 900:
            continue
        cutoff = sr["temporal_cutoff"] or casp15_safe_cutoff()
        excl = native_ids(sr.get("all_sequences"))
        th = hits[hits["query"] == tid]
        refs = io.get_reference_coords(labels, tid)
        resn = list(seq)

        def run(comp):
            t0 = time.time()
            cands = build_tbm_candidates(tid, seq, cutoff, th, meta,
                                         rng=np.random.default_rng(0),
                                         adj_dist=priors["adjacent_c1"]["mean"],
                                         exclude_pdb_ids=excl, composite_fallback=comp)
            preds = M.m_tbm_grad(cands, seq, L, priors, np.random.default_rng(3), cfg=cfg)
            tm = score_target([preds[k] for k in range(5)], refs, resn)
            srcs = [c.meta.get("source") for c in cands]
            return round(tm, 4), len(cands), srcs, round(time.time() - t0, 1)

        off_tm, off_n, off_src, _ = run(False)
        on_tm, on_n, on_src, on_sec = run(True)
        n_comp = sum(1 for s in on_src if s == "composite")
        rows.append({"target_id": tid, "seq_len": L,
                     "comp_off": off_tm, "comp_on": on_tm,
                     "delta": round(on_tm - off_tm, 4),
                     "n_cand_off": off_n, "n_cand_on": on_n, "n_composite": n_comp,
                     "top1_tsafe": top1.get(tid, np.nan), "sec": on_sec})
        print(f"[{tid}] off={off_tm:.3f} on={on_tm:.3f} (Δ{on_tm-off_tm:+.3f}) "
              f"n_comp={n_comp} top1={top1.get(tid, float('nan')):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(tables() / "composite_ablation.csv", index=False)
    md = [
        "# Composite-search fallback ablation — CASP15 (temporal-safe, best-of-5 TM)\n",
        f"- **comp_off (MMseqs only, previous): {df['comp_off'].mean():.4f}**",
        f"- **comp_on (MMseqs + composite fallback): {df['comp_on'].mean():.4f}**  "
        f"(Δ **{df['comp_on'].mean()-df['comp_off'].mean():+.4f}**)",
        f"- top-1 reproduced (temporal-safe): {df['top1_tsafe'].mean():.4f}\n",
        f"Targets improved: {(df['delta'] > 1e-3).sum()}, unchanged "
        f"{(df['delta'].abs() <= 1e-3).sum()}, worse {(df['delta'] < -1e-3).sum()}.\n",
        df.round(4).to_markdown(index=False),
    ]
    THESIS.mkdir(parents=True, exist_ok=True)
    (THESIS / "composite_ablation.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[:6]))


if __name__ == "__main__":
    main()
