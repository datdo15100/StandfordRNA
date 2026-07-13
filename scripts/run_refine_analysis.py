"""Isolate the refinement module: best-of-1 TM and physical-validity, before/after.

For each CASP15 target we take the single highest-confidence temporal-safe template
(transferred + gap-filled), then refine it. We report, before vs after refinement:
  - TM-score of that one structure (best over references)  [accuracy]
  - clashes per residue (non-adjacent C1' pairs < r_min)    [validity]
  - backbone deviation |d - mu| over consecutive C1'        [validity]
  - radius-of-gyration error vs the length-appropriate target

This separates the refinement effect from the best-of-5 max, and shows the module's
value on physical plausibility even where TM (robust to local error) moves little.

Output: reports/tables/refine_geometry.csv + reports/thesis_notes/refine_geometry.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rna3d.data import io
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.refine.optimizer import RefineConfig, refine_structure
from rna3d.template import db, mmseqs_search

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"


def geom_metrics(X: np.ndarray, priors: dict) -> dict:
    mu = priors["adjacent_c1"]["mean"]
    r_min = priors["clash"]["r_min"]
    a, b = priors["rg_powerlaw"]["a"], priors["rg_powerlaw"]["b"]
    d = np.linalg.norm(X[1:] - X[:-1], axis=1)
    bb_dev = float(np.abs(d - mu).mean())
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    L = len(X)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    nonadj = sep >= 2
    clashes = int(((D < r_min) & nonadj).sum() // 2)
    c = X.mean(0)
    rg = float(np.sqrt(((X - c) ** 2).sum(1).mean()))
    rg_err = abs(rg - a * L ** b)
    return {"clash_per_res": clashes / L, "bb_dev": bb_dev, "rg_err": rg_err}


def main():
    seqs = io.load_sequences("validation")
    labels = io.load_labels("validation")
    meta = db.load_meta()
    db.load_coords()
    priors = json.load(open(processed() / "geometry_priors.json"))

    qf = cache() / "validation_query.fasta"
    with open(qf, "w") as fh:
        for _, r in seqs.iterrows():
            fh.write(f">{r['target_id']}\n{r['sequence']}\n")
    hits = mmseqs_search.search(qf, cache() / "validation_hits.m8")

    import re
    rows = []
    for _, sr in seqs.iterrows():
        tid, seq = sr["target_id"], sr["sequence"]
        L = len(seq)
        if L > 900:
            continue
        cutoff = sr["temporal_cutoff"] or casp15_safe_cutoff()
        excl = tuple(s.upper() for s in set(re.findall(
            r">([0-9][A-Za-z0-9]{3})_", sr.get("all_sequences") if isinstance(sr.get("all_sequences"), str) else "")))
        thits = hits[hits["query"] == tid]
        cands = build_tbm_candidates(tid, seq, cutoff, thits, meta,
                                     rng=np.random.default_rng(0),
                                     adj_dist=priors["adjacent_c1"]["mean"],
                                     exclude_pdb_ids=excl, max_candidates=1)
        if not cands:
            continue
        c = cands[0]
        X0 = c.coords
        X1, _ = refine_structure(X0, priors, template_coords=X0,
                                 conf_residue=c.conf_residue,
                                 template_confidence=c.confidence,
                                 cfg=RefineConfig(steps=300))
        refs = io.get_reference_coords(labels, tid)
        resn = list(seq)
        g0, g1 = geom_metrics(X0, priors), geom_metrics(X1, priors)
        rows.append({
            "target_id": tid, "seq_len": L, "best_conf": round(c.confidence, 3),
            "template": c.chain_key, "coverage": round(c.coverage, 3),
            "tm_before": round(score_target([X0], refs, resn), 4),
            "tm_after": round(score_target([X1], refs, resn), 4),
            "clash_before": round(g0["clash_per_res"], 3),
            "clash_after": round(g1["clash_per_res"], 3),
            "bbdev_before": round(g0["bb_dev"], 3),
            "bbdev_after": round(g1["bb_dev"], 3),
            "rgerr_before": round(g0["rg_err"], 2),
            "rgerr_after": round(g1["rg_err"], 2),
        })
        print(f"[{tid}] conf={c.confidence:.2f} TM {rows[-1]['tm_before']:.3f}->{rows[-1]['tm_after']:.3f} "
              f"clash/res {g0['clash_per_res']:.2f}->{g1['clash_per_res']:.2f} "
              f"bbdev {g0['bb_dev']:.2f}->{g1['bb_dev']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(tables() / "refine_geometry.csv", index=False)
    THESIS.mkdir(parents=True, exist_ok=True)
    summary = (
        f"# Refinement isolated (best-of-1)\n\n"
        f"Targets: {len(df)}\n\n"
        f"- TM: {df['tm_before'].mean():.4f} -> {df['tm_after'].mean():.4f} "
        f"(mean delta {df['tm_after'].mean()-df['tm_before'].mean():+.4f})\n"
        f"- Clashes/residue: {df['clash_before'].mean():.3f} -> {df['clash_after'].mean():.3f} "
        f"({100*(1-df['clash_after'].mean()/max(df['clash_before'].mean(),1e-9)):.0f}% reduction)\n"
        f"- Backbone deviation (A): {df['bbdev_before'].mean():.3f} -> {df['bbdev_after'].mean():.3f}\n"
        f"- Rg error (A): {df['rgerr_before'].mean():.2f} -> {df['rgerr_after'].mean():.2f}\n\n"
        + df.to_markdown(index=False)
    )
    (THESIS / "refine_geometry.md").write_text(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
