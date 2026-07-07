"""Refinement ablation on the NEW pipeline: is the geometry refinement truthful?

Same candidate pool (MMseqs + composite search + de novo hedge, temporal-safe) under
three refinement settings, scored on the 12 CASP15 targets:
  none     : raw gap-filled candidates, NO refinement
  gradient : our gradient geometry-energy refinement
  rule     : the 1st-place rule-based nudging (top-1 logic)

Metrics per setting (aux metrics averaged over all 5 predicted structures per target):
  TM (best-of-5)        -- accuracy, INDEPENDENT of the refinement objective
  clash_per_res, bb_dev,
  rg_err                -- exactly what the gradient refiner minimises (OPTIMIZED)
  sharp_kinks           -- pseudo-bond-angle kink rate, NOT in any objective (INDEPENDENT)

Truthfulness test: a refiner is truthful if it improves the optimized metrics while
PRESERVING the independent ones (TM, kinks) — i.e. it fixes physical validity without
distorting the fold or gaming the score.

Outputs: reports/tables/refine_ablation.csv + reports/thesis_notes/refine_ablation.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.data import io
from rna3d.eval.metrics import geom_metrics
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables
from rna3d.pipeline import methods as M
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.refine.optimizer import RefineConfig
from rna3d.template import db, mmseqs_search

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"
AUX = ["clash_per_res", "bb_dev", "rg_err", "sharp_kinks"]


def native_ids(a):
    return tuple(x.upper() for x in set(re.findall(r">([0-9][A-Za-z0-9]{3})_", a))) \
        if isinstance(a, str) else ()


def aux_mean(preds, priors):
    ms = [geom_metrics(preds[k], priors) for k in range(5)]
    return {k: float(np.mean([m[k] for m in ms])) for k in AUX}


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
        cands = build_tbm_candidates(tid, seq, cutoff, th, meta,
                                     rng=np.random.default_rng(0),
                                     adj_dist=priors["adjacent_c1"]["mean"],
                                     exclude_pdb_ids=excl)
        settings = {
            "none": M.m_tbm_none(cands, seq, L, np.random.default_rng(3)),
            "gradient": M.m_tbm_grad(cands, seq, L, priors, np.random.default_rng(3), cfg=cfg),
            "rule": M.m_tbm_rule(cands, seq, L, priors, np.random.default_rng(3)),
        }
        for name, preds in settings.items():
            tm = score_target([preds[k] for k in range(5)], refs, resn)
            row = {"target_id": tid, "seq_len": L, "setting": name, "tm": round(tm, 4)}
            row.update({k: round(v, 4) for k, v in aux_mean(preds, priors).items()})
            rows.append(row)
        g = {r["setting"]: r for r in rows[-3:]}
        print(f"[{tid}] TM none={g['none']['tm']:.3f} grad={g['gradient']['tm']:.3f} "
              f"rule={g['rule']['tm']:.3f} | clash none={g['none']['clash_per_res']:.2f}"
              f"->grad {g['gradient']['clash_per_res']:.2f} | kink none={g['none']['sharp_kinks']:.2f}"
              f"->grad {g['gradient']['sharp_kinks']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(tables() / "refine_ablation.csv", index=False)
    agg = df.groupby("setting").agg(
        tm=("tm", "mean"), clash=("clash_per_res", "mean"),
        bb_dev=("bb_dev", "mean"), rg_err=("rg_err", "mean"),
        kinks=("sharp_kinks", "mean")).reindex(["none", "rule", "gradient"])

    md = [
        "# Refinement ablation — is the geometry refinement truthful? (CASP15, temporal-safe)\n",
        "New pipeline (MMseqs + composite search + de novo hedge). Aux metrics averaged over "
        "all 5 predicted structures per target, then over targets.\n",
        "**OPTIMIZED by gradient** = clash / bb_dev / rg_err (drops are expected by construction). "
        "**INDEPENDENT** = TM and sharp_kinks (not in any objective) — the real truthfulness test.\n",
        agg.round(4).to_markdown(),
        "",
        f"- **TM (independent accuracy)**: none {agg.loc['none','tm']:.3f} -> "
        f"gradient {agg.loc['gradient','tm']:.3f} ({agg.loc['gradient','tm']-agg.loc['none','tm']:+.3f}), "
        f"rule {agg.loc['rule','tm']:.3f}.",
        f"- **clash/res (optimized)**: none {agg.loc['none','clash']:.3f} -> "
        f"gradient {agg.loc['gradient','clash']:.3f} / rule {agg.loc['rule','clash']:.3f}.",
        f"- **backbone dev Å (optimized)**: none {agg.loc['none','bb_dev']:.3f} -> "
        f"gradient {agg.loc['gradient','bb_dev']:.3f} / rule {agg.loc['rule','bb_dev']:.3f}.",
        f"- **sharp kinks (INDEPENDENT)**: none {agg.loc['none','kinks']:.3f} -> "
        f"gradient {agg.loc['gradient','kinks']:.3f} / rule {agg.loc['rule','kinks']:.3f}.",
    ]
    THESIS.mkdir(parents=True, exist_ok=True)
    (THESIS / "refine_ablation.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[3:]))


if __name__ == "__main__":
    main()
