"""Leakage demonstration (PLAN Phase 9): how temporal discipline bounds TBM on CASP15.

Runs the SAME TBM+refinement pipeline under three template-availability regimes:

  temporal_safe : release_date < target cutoff, and the target's own PDB excluded
                  -> the honest, leakage-free setting used for the thesis numbers.
  no_temporal   : ignore release dates, but still exclude the literal native PDB ids
                  -> "if we ignored the cutoff" (post-CASP15 homologs leak in).
  oracle_leak   : no filtering at all (the native structure itself is allowed)
                  -> pure-leakage upper bound / sanity ceiling.

The gap between temporal_safe and the others quantifies how much CASP15 leaderboard
performance can rely on structures unavailable at prediction time — and shows the
pipeline itself is strong when templates exist.

Output: reports/tables/leakage_demo.csv + reports/thesis_notes/leakage_demo.md
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
from rna3d.eval.usalign import score_target
from rna3d.paths import cache, casp15_safe_cutoff, processed, tables
from rna3d.pipeline import methods as M
from rna3d.pipeline.tbm import build_tbm_candidates
from rna3d.refine.optimizer import RefineConfig
from rna3d.template import db, mmseqs_search

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"


def native_pdb_ids(all_seq) -> tuple[str, ...]:
    if not isinstance(all_seq, str):
        return ()
    return tuple(s.upper() for s in set(re.findall(r">([0-9][A-Za-z0-9]{3})_", all_seq)))


REGIMES = {
    "temporal_safe": dict(apply_temporal=True, exclude_native=True),
    "no_temporal": dict(apply_temporal=False, exclude_native=True),
    "oracle_leak": dict(apply_temporal=False, exclude_native=False),
}


def main():
    seqs = io.load_sequences("validation")
    labels = io.load_labels("validation")
    meta = db.load_meta()
    db.load_coords()
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
        tid, seq = sr["target_id"], sr["sequence"]
        L = len(seq)
        if L > 900:
            rows.append({"target_id": tid, "seq_len": L, **{r: np.nan for r in REGIMES}})
            continue
        cutoff = sr["temporal_cutoff"] or casp15_safe_cutoff()
        natives = native_pdb_ids(sr.get("all_sequences"))
        thits = hits[hits["query"] == tid]
        refs = io.get_reference_coords(labels, tid)
        resn = list(seq)
        row = {"target_id": tid, "seq_len": L}
        for rname, opt in REGIMES.items():
            excl = natives if opt["exclude_native"] else ()
            cands = build_tbm_candidates(
                tid, seq, cutoff, thits, meta, rng=np.random.default_rng(0),
                adj_dist=priors["adjacent_c1"]["mean"], exclude_pdb_ids=excl,
                apply_temporal=opt["apply_temporal"],
            )
            preds = M.m_tbm_refined(cands, L, priors, np.random.default_rng(3), cfg=cfg)
            row[rname] = round(score_target([preds[k] for k in range(5)], refs, resn), 4)
            row[f"{rname}_conf"] = round(cands[0].confidence, 3) if cands else 0.0
        rows.append(row)
        print(f"[{tid}] safe={row['temporal_safe']:.3f} "
              f"no_temporal={row['no_temporal']:.3f} oracle={row['oracle_leak']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(tables() / "leakage_demo.csv", index=False)
    means = {r: df[r].mean() for r in REGIMES}
    md = [
        "# Leakage demonstration — CASP15 validation (TBM + refinement)\n",
        "Best-of-5 TM under three template-availability regimes.\n",
        f"- **temporal_safe (honest): {means['temporal_safe']:.4f}**",
        f"- no_temporal (ignore cutoff, exclude native pdb): {means['no_temporal']:.4f}",
        f"- oracle_leak (native allowed): {means['oracle_leak']:.4f}\n",
        f"Temporal discipline costs **{means['no_temporal']-means['temporal_safe']:+.4f}** TM vs "
        f"ignoring the cutoff, and **{means['oracle_leak']-means['temporal_safe']:+.4f}** vs full leakage. "
        "The oracle column confirms the TBM+refinement machinery reaches high TM when a true "
        "template is available — the honest score is bounded by template availability, not the pipeline.\n",
        df[["target_id", "seq_len", "temporal_safe", "no_temporal", "oracle_leak"]].round(4).to_markdown(index=False),
    ]
    (THESIS / "leakage_demo.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
