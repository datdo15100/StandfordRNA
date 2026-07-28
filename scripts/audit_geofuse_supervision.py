#!/usr/bin/env python
"""Audit whether GeoFuse source-quality labels agree on real OOF pairs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.paths import cache, processed
from train_geofuse_phase_d import load_priors
from train_geofuse_real_gate import build_examples


ADVANTAGES = {
    "aligned_point": ("aligned_template_error", "aligned_pretrained_error", "lower"),
    "c1_lddt": ("template_lddt", "pretrained_lddt", "higher"),
    "window15_rmsd": ("template_window_rmsd", "pretrained_window_rmsd", "lower"),
}


def _residue_frame(examples: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for split, split_examples in examples.items():
        for example in split_examples:
            length = len(example["features"])
            for index in range(length):
                row = {
                    "split": split,
                    "target_id": example["target_id"],
                    "pair_id": example["pair_id"],
                    "residue": index + 1,
                }
                valid = bool(
                    example["resolved_mask"][index]
                    and example["lddt_resolved_mask"][index]
                    and example["window_resolved_mask"][index]
                )
                row["valid"] = valid
                for name, (template_key, pretrained_key, direction) in ADVANTAGES.items():
                    template = float(example[template_key][index])
                    pretrained = float(example[pretrained_key][index])
                    row[f"{name}_template"] = template
                    row[f"{name}_pretrained"] = pretrained
                    row[f"{name}_advantage"] = (
                        pretrained - template
                        if direction == "higher"
                        else template - pretrained
                    )
                    row[f"{name}_pretrained_better"] = row[f"{name}_advantage"] > 0
                rows.append(row)
    return pd.DataFrame(rows)


def _agreement(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"]].copy()
    rows = []
    names = list(ADVANTAGES)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_value = valid[f"{left}_advantage"]
            right_value = valid[f"{right}_advantage"]
            centered_left = left_value - valid.groupby("target_id")[
                f"{left}_advantage"
            ].transform("mean")
            centered_right = right_value - valid.groupby("target_id")[
                f"{right}_advantage"
            ].transform("mean")
            rows.append(
                {
                    "label_a": left,
                    "label_b": right,
                    "decision_agreement": float(
                        (
                            valid[f"{left}_pretrained_better"]
                            == valid[f"{right}_pretrained_better"]
                        ).mean()
                    ),
                    "pooled_spearman": float(spearmanr(left_value, right_value).statistic),
                    "target_centered_spearman": float(
                        spearmanr(centered_left, centered_right).statistic
                    ),
                    "n_residues": len(valid),
                }
            )
    return pd.DataFrame(rows)


def _source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"]]
    rows = []
    for split, group in valid.groupby("split"):
        for name, (_, _, direction) in ADVANTAGES.items():
            template = group[f"{name}_template"]
            pretrained = group[f"{name}_pretrained"]
            rows.append(
                {
                    "split": split,
                    "label": name,
                    "direction": direction,
                    "template_mean": float(template.mean()),
                    "pretrained_mean": float(pretrained.mean()),
                    "pretrained_better_fraction": float(
                        group[f"{name}_pretrained_better"].mean()
                    ),
                    "targets": int(group["target_id"].nunique()),
                    "pairs": int(group["pair_id"].nunique()),
                    "residues": len(group),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, dtype={"target_id": str})
    priors_v1, priors_v2 = load_priors()
    examples, failures = build_examples(args, manifest, priors_v1, priors_v2)
    frame = _residue_frame(examples)
    agreement = _agreement(frame)
    source = _source_summary(frame)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "supervision_residues.csv", index=False)
    agreement.to_csv(output / "supervision_agreement.csv", index=False)
    source.to_csv(output / "source_quality_summary.csv", index=False)

    old_lddt = agreement[
        (agreement["label_a"] == "aligned_point")
        & (agreement["label_b"] == "c1_lddt")
    ].iloc[0]
    old_window = agreement[
        (agreement["label_a"] == "aligned_point")
        & (agreement["label_b"] == "window15_rmsd")
    ].iloc[0]
    lines = [
        "# GeoFuse real-OOF supervision audit",
        "",
        "This is E11 under the frozen confirmatory protocol. Native coordinates are used "
        "only here to compare training labels; no native-derived column is available "
        "to the inference-time gate.",
        "",
        f"- Manifest targets: {len(manifest)}",
        f"- Ready targets: {frame['target_id'].nunique()}",
        f"- Pair examples: {frame['pair_id'].nunique()}",
        f"- Rejected pair attempts: {len(failures)}",
        f"- Jointly valid residue rows: {int(frame['valid'].sum())}",
        "",
        "Positive advantage always means the pretrained source is locally better.",
        "",
        "## Pairwise label agreement",
        "",
        agreement.round(5).to_markdown(index=False),
        "",
        "## Source summaries",
        "",
        source.round(5).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- The old globally aligned point-error choice agrees with C1′-lDDT on "
        f"{old_lddt['decision_agreement']:.1%} of jointly valid residue rows "
        f"(target-centred rho={old_lddt['target_centered_spearman']:.3f}).",
        f"- It agrees with 15-residue window RMSD on "
        f"{old_window['decision_agreement']:.1%} "
        f"(target-centred rho={old_window['target_centered_spearman']:.3f}).",
        "- Disagreement is expected because global point error assigns displacement "
        "after one whole-fold fit, lDDT measures local distance preservation without "
        "superposition, and window RMSD refits each local segment.",
        "- C1′-lDDT is the primary gate supervision in subsequent confirmatory models. "
        "Aligned point error and window RMSD remain explicit ablations.",
        "",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(agreement.round(5).to_string(index=False))
    print(f"[report] {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=processed() / "geofuse_real_oof_v2" / "medium_manifest.csv",
    )
    parser.add_argument(
        "--cache-root", default=cache() / "geofuse_candidates"
    )
    parser.add_argument("--max-templates", type=int, default=2)
    parser.add_argument("--max-pretrained", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_supervision_audit",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT
        / "reports"
        / "thesis_notes"
        / "geofuse_supervision_audit.md",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
