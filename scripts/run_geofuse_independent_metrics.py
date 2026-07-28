#!/usr/bin/env python
"""Re-evaluate raw/geometry-v1/v2 with independent native C1' metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.data import io
from rna3d.eval.local_metrics import local_accuracy_metrics
from rna3d.eval.statistics import paired_target_summary
from rna3d.eval.usalign import score_target
from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.phase_a import select_source_balanced
from rna3d.paths import cache
from run_geofuse_phase_b import _cached_refinement, load_priors


METRICS = ("c1_rmsd", "c1_lddt", "sw_rmsd_9", "sw_rmsd_15", "sw_rmsd_31")


def _selected_sequences(target_ids: str | None) -> pd.DataFrame:
    frame = io.load_sequences("validation")
    if target_ids:
        requested = {item.strip() for item in target_ids.split(",") if item.strip()}
        missing = requested - set(frame["target_id"])
        if missing:
            raise KeyError(f"unknown validation targets: {sorted(missing)}")
        frame = frame[frame["target_id"].isin(requested)]
    return frame.reset_index(drop=True)


def _best_reference(
    coordinates: np.ndarray, references: list[np.ndarray], sequence: str
) -> tuple[int, float]:
    values = [
        float(score_target([coordinates], [reference], list(sequence)))
        for reference in references
    ]
    index = int(np.nanargmax(values))
    return index, values[index]


def _bootstrap_rows(targets: pd.DataFrame, setting: str) -> list[dict]:
    output = []
    raw = targets[targets["setting"] == "raw"].set_index("target_id")
    method = targets[targets["setting"] == setting].set_index("target_id")
    shared = raw.index.intersection(method.index)
    for metric in METRICS:
        higher = metric == "c1_lddt"
        result = paired_target_summary(
            raw.loc[shared, metric].to_numpy(),
            method.loc[shared, metric].to_numpy(),
            higher_is_better=higher,
        )
        output.append({"setting": setting, "metric": metric, **result})
    result = paired_target_summary(
        raw.loc[shared, "best5_tm"].to_numpy(),
        method.loc[shared, "best5_tm"].to_numpy(),
        higher_is_better=True,
    )
    output.append({"setting": setting, "metric": "best5_tm", **result})
    return output


def _write_report(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    report: Path,
    tm_tolerance: float,
) -> None:
    aggregate = (
        targets.groupby("setting")[
            ["best5_tm", *METRICS]
        ]
        .mean()
        .reindex(["raw", "v1", "v2"])
    )
    lookup = bootstrap.set_index(["setting", "metric"])
    v2_tm_loss_ok = (
        aggregate.loc["v2", "best5_tm"]
        >= aggregate.loc["raw", "best5_tm"] - tm_tolerance
    )
    v2_lddt_ok = lookup.loc[("v2", "c1_lddt"), "mean_delta"] > 0
    local_ok = sum(
        lookup.loc[("v2", f"sw_rmsd_{window}"), "mean_delta"] > 0
        for window in (9, 15, 31)
    )
    passed = bool(v2_tm_loss_ok and v2_lddt_ok and local_ok >= 2)
    candidate_delta = candidates.pivot_table(
        index=["target_id", "candidate_id"],
        columns="setting",
        values="c1_lddt",
    )
    v2_candidate_improved = int((candidate_delta["v2"] > candidate_delta["raw"]).sum())
    v2_candidate_regressed = int((candidate_delta["v2"] < candidate_delta["raw"]).sum())
    lines = [
        "# GeoFuse independent local-metric ablation",
        "",
        "This is E10 under `geofuse_confirmatory_protocol.md`. Raw, geometry v1 and "
        "geometry v2 use identical source-balanced candidates. For each evaluated "
        "structure, the native conformation with highest TM-score is selected first; "
        "all local metrics then use that same conformation.",
        "",
        f"- Targets: {targets['target_id'].nunique()}",
        f"- Paired candidates: {candidate_delta.shape[0]}",
        f"- Confirmatory geometry-v2 gate: **{'pass' if passed else 'fail'}**",
        f"- TM preservation (loss <= {tm_tolerance:.3f}): "
        f"**{'pass' if v2_tm_loss_ok else 'fail'}**",
        f"- Positive C1′-lDDT delta: **{'pass' if v2_lddt_ok else 'fail'}**",
        f"- Improved sliding-window scales: {local_ok}/3",
        f"- Candidate C1′-lDDT: {v2_candidate_improved} improved, "
        f"{v2_candidate_regressed} regressed",
        "",
        "## Equal-weight target means",
        "",
        aggregate.round(6).to_markdown(),
        "",
        "## Paired target bootstrap",
        "",
        "Every delta is oriented so positive means the method is better. Confidence "
        "intervals resample targets (10,000 samples, seed 2025), not residues.",
        "",
        bootstrap.round(6).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "- Geometry v2 passes the pre-registered independent local-accuracy gate."
            if passed
            else "- Geometry v2 does not pass the pre-registered independent "
            "local-accuracy gate. Improvements in its clash/kink/NLL objectives must "
            "therefore remain a physical-plausibility claim, not a native-accuracy claim."
        ),
        "- C1′-lDDT and sliding-window RMSD are independent native metrics; sharp-kink, "
        "clash proxy, angle NLL and torsion NLL are project diagnostics and are not used "
        "to pass this gate.",
        "- With only the existing validation targets, target-bootstrap intervals may be "
        "wide. The counts and intervals are reported rather than converting residues "
        "into falsely independent samples.",
        "",
        "## Per-target",
        "",
        targets.round(6).to_markdown(index=False),
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    sequences = _selected_sequences(args.target_ids)
    labels = io.load_labels("validation")
    priors_v1, priors_v2 = load_priors()
    store = CandidateCache(cache() / "geofuse_candidates", "validation")
    candidate_rows: list[dict] = []
    target_rows: list[dict] = []

    for target in sequences.itertuples(index=False):
        candidates = select_source_balanced(
            store.load_target(target.target_id, target.sequence), args.candidates
        )
        if not candidates:
            continue
        coordinates = {"raw": [candidate.coords for candidate in candidates]}
        for setting in ("v1", "v2"):
            coordinates[setting] = [
                _cached_refinement(
                    candidate,
                    target.sequence,
                    setting,
                    priors_v1,
                    priors_v2,
                    args.steps,
                    args.device,
                    False,
                )[0]
                for candidate in candidates
            ]
        references = io.get_reference_coords(labels, target.target_id)
        for setting, structures in coordinates.items():
            rows = []
            for candidate, structure in zip(candidates, structures):
                reference_index, candidate_tm = _best_reference(
                    structure, references, target.sequence
                )
                metrics = local_accuracy_metrics(
                    structure, references[reference_index], windows=(9, 15, 31)
                )
                row = {
                    "target_id": target.target_id,
                    "candidate_id": candidate.candidate_id,
                    "source": candidate.source,
                    "setting": setting,
                    "reference_index": reference_index,
                    "candidate_tm": candidate_tm,
                    **metrics,
                }
                rows.append(row)
                candidate_rows.append(row)
            target_rows.append(
                {
                    "target_id": target.target_id,
                    "seq_len": len(target.sequence),
                    "setting": setting,
                    "n_candidates": len(candidates),
                    "best5_tm": float(
                        score_target(structures, references, list(target.sequence))
                    ),
                    **{
                        metric: float(np.nanmean([row[metric] for row in rows]))
                        for metric in METRICS
                    },
                }
            )
        recent = {row["setting"]: row for row in target_rows[-3:]}
        print(
            f"[{target.target_id}] lDDT "
            f"{recent['raw']['c1_lddt']:.4f}->{recent['v2']['c1_lddt']:.4f}; "
            f"SW15 {recent['raw']['sw_rmsd_15']:.3f}->"
            f"{recent['v2']['sw_rmsd_15']:.3f}",
            flush=True,
        )

    targets = pd.DataFrame(target_rows)
    candidates = pd.DataFrame(candidate_rows)
    bootstrap = pd.DataFrame(
        _bootstrap_rows(targets, "v1") + _bootstrap_rows(targets, "v2")
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "target_metrics.csv", index=False)
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    bootstrap.to_csv(output / "target_bootstrap.csv", index=False)
    _write_report(targets, candidates, bootstrap, Path(args.report), args.tm_tolerance)
    print(targets.groupby("setting")[["best5_tm", *METRICS]].mean().round(6))
    print(f"[report] {args.report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ids")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tm-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_independent_metrics",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT
        / "reports"
        / "thesis_notes"
        / "geofuse_independent_metrics.md",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
