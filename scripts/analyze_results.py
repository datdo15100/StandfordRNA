"""Analyse eval results: stratify by template confidence, quantify refinement gain.

Reads reports/tables/eval_methods.csv and writes:
    reports/tables/eval_by_confidence.csv
    reports/thesis_notes/results_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rna3d.paths import tables  # noqa: E402

THESIS = Path(__file__).resolve().parents[1] / "reports" / "thesis_notes"
THESIS.mkdir(parents=True, exist_ok=True)

METHODS = ["B0_dummy", "B1_tbm_top1", "B2_tbm_top5", "B4_tbm_refined",
           "A_no_clash", "A_no_rg", "A_no_gapweights"]


def conf_bin(c: float) -> str:
    if c <= 0:
        return "none"
    if c < 0.4:
        return "low"
    if c < 0.6:
        return "medium"
    return "high"


def main():
    df = pd.read_csv(tables() / "eval_methods.csv")
    df["conf_bin"] = df["best_conf"].map(conf_bin)

    present = [m for m in METHODS if m in df.columns]
    overall = df[present].mean().sort_values(ascending=False)

    # refinement gain isolating the geometry module
    df["refine_gain"] = df["B4_tbm_refined"] - df["B2_tbm_top5"]

    by_conf = df.groupby("conf_bin").agg(
        n=("target_id", "count"),
        mean_best_conf=("best_conf", "mean"),
        B2_tbm_top5=("B2_tbm_top5", "mean"),
        B4_tbm_refined=("B4_tbm_refined", "mean"),
        refine_gain=("refine_gain", "mean"),
    ).reindex(["high", "medium", "low", "none"]).dropna(how="all")
    by_conf.to_csv(tables() / "eval_by_confidence.csv")

    lines = []
    lines.append("# CASP15 validation results\n")
    lines.append("Best-of-5 TM-score (US-align), 12 CASP15 targets, temporal-safe templates.\n")
    lines.append("## Mean TM by method\n")
    lines.append(overall.round(4).to_frame("mean_TM").to_markdown())
    lines.append("\n## Refinement (B4 = TBM+refine) vs TBM-only (B2)\n")
    lines.append(f"- Overall mean gain: **{df['refine_gain'].mean():+.4f}** "
                 f"(B2 {df['B2_tbm_top5'].mean():.4f} -> B4 {df['B4_tbm_refined'].mean():.4f})")
    wins = int((df["refine_gain"] > 1e-4).sum())
    ties = int((df["refine_gain"].abs() <= 1e-4).sum())
    losses = int((df["refine_gain"] < -1e-4).sum())
    lines.append(f"- Per-target: {wins} improved, {ties} unchanged, {losses} worse\n")
    lines.append("## Stratified by template confidence\n")
    lines.append(by_conf.round(4).to_markdown())
    lines.append("\n## Ablations (mean TM)\n")
    for m in ["B4_tbm_refined", "A_no_clash", "A_no_rg", "A_no_gapweights"]:
        if m in df.columns:
            lines.append(f"- {m}: {df[m].mean():.4f}")
    if "selftm_B4" in df.columns:
        lines.append(f"\n## Best-of-5 diversity\n- mean pairwise self-TM (B4): "
                     f"{df['selftm_B4'].mean():.4f} (lower = more diverse)")
    lines.append("\n## Per-target detail\n")
    cols = ["target_id", "seq_len", "best_conf", "B1_tbm_top1", "B2_tbm_top5",
            "B4_tbm_refined", "refine_gain"]
    cols = [c for c in cols if c in df.columns]
    lines.append(df[cols].round(4).to_markdown(index=False))

    out = THESIS / "results_summary.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")
    print(f"wrote {tables() / 'eval_by_confidence.csv'}")


if __name__ == "__main__":
    main()
