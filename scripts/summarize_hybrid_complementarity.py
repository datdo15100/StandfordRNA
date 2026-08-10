#!/usr/bin/env python
"""Summarize TBM/DRfold2 complementarity from frozen held-out artifacts only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.eval.statistics import paired_target_summary


DEFAULT_INPUT = REPO_ROOT / "reports" / "tables" / "geometry_v2_confirmatory" / "validation"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "tables" / "hybrid_complementarity"
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_notes" / "hybrid_complementarity.md"


def _effect(baseline: pd.Series, method: pd.Series, name: str) -> dict:
    shared = baseline.index.intersection(method.index)
    result = paired_target_summary(
        baseline.loc[shared].to_numpy(),
        method.loc[shared].to_numpy(),
        higher_is_better=True,
    )
    return {"effect": name, **result}


def run(args: argparse.Namespace) -> None:
    source = Path(args.input_dir)
    candidates = pd.read_csv(source / "candidate_metrics.csv")
    allocation = pd.read_csv(source / "candidate_allocation.csv")
    raw = candidates[candidates["setting"] == "raw"].copy()
    raw["generator"] = np.where(raw["kind"] == "template", "TBM", "DRfold2")

    oracle = raw.pivot_table(
        index="target_id", columns="generator", values="raw_candidate_tm", aggfunc="max"
    )
    oracle["Union"] = oracle[["TBM", "DRfold2"]].max(axis=1)
    oracle["union_gain_over_tbm"] = oracle["Union"] - oracle["TBM"]
    oracle["winner"] = np.where(
        oracle["DRfold2"] > oracle["TBM"],
        "DRfold2",
        np.where(oracle["TBM"] > oracle["DRfold2"], "TBM", "tie"),
    )

    raw_alloc = allocation[allocation["geometry"] == "off"].copy()
    allocation_means = raw_alloc.groupby(
        ["total_candidates", "n_tbm", "n_drfold2"]
    )["best_tm"].agg(["mean", "std", "count"]).reset_index()
    pivot = raw_alloc.pivot(
        index="target_id", columns=["n_tbm", "n_drfold2"], values="best_tm"
    )
    effects = pd.DataFrame(
        [
            _effect(
                pivot[(3, 0)],
                pivot[(2, 1)],
                "fixed-N=3: replace one TBM with one DRfold2",
            ),
            _effect(
                pivot[(3, 0)],
                pivot[(3, 2)],
                "production bank: add two DRfold2 to three TBM",
            ),
            _effect(
                pivot[(2, 0)],
                pivot[(1, 1)],
                "fixed-N=2: replace one TBM with one DRfold2",
            ),
            _effect(
                pivot[(2, 0)],
                pivot[(0, 2)],
                "two DRfold2 versus two TBM",
            ),
            _effect(
                oracle["TBM"],
                oracle["Union"],
                "oracle ceiling: union versus TBM",
            ),
        ]
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oracle.reset_index().to_csv(output / "source_oracle.csv", index=False)
    allocation_means.to_csv(output / "allocation_means.csv", index=False)
    effects.to_csv(output / "paired_effects.csv", index=False)

    winner_counts = oracle["winner"].value_counts().to_dict()
    report = [
        "# TBM–DRfold2 complementarity on 20 newest held-out RNA",
        "",
        "This report reads the already frozen validation artifacts; it does not rerun a "
        "model or access native labels again. Candidate selection is confidence-based and "
        "native blind. Oracle rows use native scores only to measure the available ceiling.",
        "",
        f"- Mean best individual TBM candidate: **{oracle['TBM'].mean():.6f}**",
        f"- Mean best individual DRfold2 candidate: **{oracle['DRfold2'].mean():.6f}**",
        f"- Mean union oracle: **{oracle['Union'].mean():.6f}**",
        f"- DRfold2/TBM/tie oracle winner counts: "
        f"**{winner_counts.get('DRfold2', 0)}/{winner_counts.get('TBM', 0)}/"
        f"{winner_counts.get('tie', 0)}**",
        "",
        "## Candidate allocation (Geometry off)",
        "",
        allocation_means.round(6).to_markdown(index=False),
        "",
        "## Paired target effects",
        "",
        "Positive means the method named after the colon is better. The fixed-N rows "
        "separate source composition from candidate-count advantage; the production-bank "
        "row is pragmatic but receives two additional prediction slots.",
        "",
        effects.round(6).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "DRfold2 is not uniformly stronger than TBM. Its value is complementary coverage: "
        "on a minority of targets it supplies a fold absent from the TBM bank. A fixed-size "
        "hybrid allocation has a positive mean effect but a wide interval, while the union "
        "oracle quantifies how much a perfect native-blind selector could in principle gain.",
        "",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report))
    print(oracle[["TBM", "DRfold2", "Union", "union_gain_over_tbm"]].mean().round(6))
    print(effects.round(6).to_string(index=False))
    print(f"[report] {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
