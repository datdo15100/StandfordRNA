"""Phase-A candidate-pool evaluation for GeoFuse-RNA."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..data import io
from ..eval.usalign import score_target
from .candidate import CandidateCache, StructureCandidate


ScoreFunction = Callable[[list[np.ndarray], list[np.ndarray], list[str]], float]


def rank_without_native(candidates: list[StructureCandidate]) -> list[StructureCandidate]:
    """Deterministic source-local ranking using only model-side confidence."""
    return sorted(candidates, key=lambda c: (-c.global_confidence, c.candidate_id))


def select_source_balanced(
    candidates: list[StructureCandidate], limit: int = 5
) -> list[StructureCandidate]:
    """Round-robin sources so an uncalibrated confidence scale cannot erase a source."""
    groups: dict[str, list[StructureCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.source].append(candidate)
    for source in groups:
        groups[source] = rank_without_native(groups[source])

    source_order = sorted(groups, key=lambda source: (source != "tbm", source))
    selected: list[StructureCandidate] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for source in source_order:
            if depth < len(groups[source]):
                selected.append(groups[source][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _best(rows: list[dict], candidate_ids: set[str]) -> float:
    values = [row["tm_score"] for row in rows if row["candidate_id"] in candidate_ids]
    return max(values) if values else float("nan")


def _max_or_nan(values: list[float]) -> float:
    return max(values) if values else float("nan")


def evaluate_candidate_pool(
    sequences: pd.DataFrame,
    labels: pd.DataFrame,
    candidate_cache: CandidateCache,
    *,
    max_selected: int = 5,
    scorer: ScoreFunction = score_target,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score candidates individually, then compute selection and oracle ceilings.

    Native labels are used only in the individual score/oracle analysis.  The
    candidates entering each selected set are chosen before those scores are read.
    """
    candidate_rows: list[dict] = []
    target_rows: list[dict] = []

    for _, sequence_row in sequences.iterrows():
        target_id = str(sequence_row["target_id"])
        sequence = str(sequence_row["sequence"])
        candidates = candidate_cache.load_target(target_id, sequence)
        if not candidates:
            continue
        references = io.get_reference_coords(labels, target_id)
        resnames = list(sequence)
        rows: list[dict] = []
        for candidate in candidates:
            tm = float(scorer([candidate.coords], references, resnames))
            row = {
                "target_id": target_id,
                "seq_len": len(sequence),
                "candidate_id": candidate.candidate_id,
                "kind": candidate.kind,
                "source": candidate.source,
                "model": candidate.model,
                "global_confidence": candidate.global_confidence,
                "resolved_fraction": float(candidate.valid_mask.mean()),
                "supported_fraction": float(candidate.support_mask.mean()),
                "tm_score": tm,
            }
            rows.append(row)
            candidate_rows.append(row)

        templates = [candidate for candidate in candidates if candidate.kind == "template"]
        pretrained = [candidate for candidate in candidates if candidate.kind == "pretrained"]
        tbm_selected = rank_without_native(templates)[:max_selected]
        pretrained_selected = rank_without_native(pretrained)[:max_selected]
        union_selected = select_source_balanced(templates + pretrained, max_selected)

        tbm_scores = [row["tm_score"] for row in rows if row["kind"] == "template"]
        pretrained_scores = [row["tm_score"] for row in rows if row["kind"] == "pretrained"]
        union_scores = tbm_scores + pretrained_scores
        tbm_oracle = _max_or_nan(tbm_scores)
        pretrained_oracle = _max_or_nan(pretrained_scores)
        union_oracle = _max_or_nan(union_scores)
        union_selected_tm = _best(rows, {candidate.candidate_id for candidate in union_selected})

        target_rows.append(
            {
                "target_id": target_id,
                "seq_len": len(sequence),
                "n_tbm": len(templates),
                "n_pretrained": len(pretrained),
                "n_sources": len({candidate.source for candidate in templates + pretrained}),
                "tbm_selected_tm": _best(rows, {candidate.candidate_id for candidate in tbm_selected}),
                "pretrained_selected_tm": _best(
                    rows, {candidate.candidate_id for candidate in pretrained_selected}
                ),
                "union_selected_tm": union_selected_tm,
                "tbm_oracle_tm": tbm_oracle,
                "pretrained_oracle_tm": pretrained_oracle,
                "union_oracle_tm": union_oracle,
                "oracle_gain_over_tbm": union_oracle - tbm_oracle
                if np.isfinite(tbm_oracle) and np.isfinite(union_oracle)
                else float("nan"),
                "selection_regret": union_oracle - union_selected_tm
                if np.isfinite(union_oracle) and np.isfinite(union_selected_tm)
                else float("nan"),
                "union_selected_ids": ";".join(candidate.candidate_id for candidate in union_selected),
            }
        )

    candidates_df = pd.DataFrame(candidate_rows)
    targets_df = pd.DataFrame(target_rows)
    paired = targets_df[
        (targets_df.get("n_tbm", pd.Series(dtype=int)) > 0)
        & (targets_df.get("n_pretrained", pd.Series(dtype=int)) > 0)
    ] if not targets_df.empty else targets_df
    gains = paired["oracle_gain_over_tbm"].dropna() if not paired.empty else pd.Series(dtype=float)
    summary = {
        "schema_version": 1,
        "n_targets_in_sequences": int(len(sequences)),
        "n_targets_with_candidates": int(len(targets_df)),
        "n_targets_with_tbm": int((targets_df.get("n_tbm", pd.Series(dtype=int)) > 0).sum()),
        "n_targets_with_pretrained": int(
            (targets_df.get("n_pretrained", pd.Series(dtype=int)) > 0).sum()
        ),
        "n_paired_targets": int(len(paired)),
        "mean_tbm_selected_tm": _mean_column(targets_df, "tbm_selected_tm"),
        "mean_pretrained_selected_tm": _mean_column(targets_df, "pretrained_selected_tm"),
        "mean_union_selected_tm": _mean_column(targets_df, "union_selected_tm"),
        "mean_tbm_oracle_tm_paired": _mean_column(paired, "tbm_oracle_tm"),
        "mean_pretrained_oracle_tm_paired": _mean_column(paired, "pretrained_oracle_tm"),
        "mean_union_oracle_tm_paired": _mean_column(paired, "union_oracle_tm"),
        "mean_oracle_gain_over_tbm": float(gains.mean()) if len(gains) else None,
        "median_oracle_gain_over_tbm": float(gains.median()) if len(gains) else None,
        "n_targets_oracle_improved": int((gains > 1e-8).sum()),
        "n_targets_oracle_tied": int((gains.abs() <= 1e-8).sum()),
        "gate_status": "not_evaluable" if not len(gains) else "pending_threshold",
    }
    return candidates_df, targets_df, summary


def _mean_column(frame: pd.DataFrame, name: str) -> float | None:
    if frame.empty or name not in frame:
        return None
    values = frame[name].dropna()
    return float(values.mean()) if len(values) else None


def write_phase_a_report(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    summary: dict,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    min_mean_gain: float = 0.0,
) -> dict:
    """Persist machine-readable tables and a compact thesis-facing Markdown report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "candidate_scores.csv", index=False)
    targets.to_csv(output_dir / "target_pool_metrics.csv", index=False)

    result = dict(summary)
    result["min_mean_gain"] = float(min_mean_gain)
    gain = result.get("mean_oracle_gain_over_tbm")
    if gain is None:
        result["gate_status"] = "not_evaluable"
    else:
        result["gate_status"] = "pass" if gain > min_mean_gain else "fail"
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GeoFuse-RNA Phase A — pretrained candidate gate\n",
        "This gate asks whether pretrained candidates add a useful fold hypothesis before "
        "confidence fusion or geometry v2 is implemented. Candidate selection never sees native "
        "labels; the oracle columns use labels only to measure the candidate-generation ceiling.\n",
        "## Result\n",
        f"- Gate status: **{result['gate_status']}**",
        f"- Targets with TBM candidates: {result['n_targets_with_tbm']}",
        f"- Targets with pretrained candidates: {result['n_targets_with_pretrained']}",
        f"- Paired targets used by the gate: {result['n_paired_targets']}",
        f"- Mean paired TBM oracle TM: {_fmt(result['mean_tbm_oracle_tm_paired'])}",
        f"- Mean paired pretrained oracle TM: {_fmt(result['mean_pretrained_oracle_tm_paired'])}",
        f"- Mean paired union oracle TM: {_fmt(result['mean_union_oracle_tm_paired'])}",
        f"- Mean oracle gain over TBM: {_fmt(result['mean_oracle_gain_over_tbm'], signed=True)}",
        f"- Improved/tied paired targets: {result['n_targets_oracle_improved']}/"
        f"{result['n_targets_oracle_tied']}",
        "\n## Per-target metrics\n",
    ]
    if targets.empty:
        lines.append("No cached candidates were available.\n")
    else:
        columns = [
            "target_id", "seq_len", "n_tbm", "n_pretrained", "tbm_selected_tm",
            "pretrained_selected_tm", "union_selected_tm", "tbm_oracle_tm",
            "pretrained_oracle_tm", "union_oracle_tm", "oracle_gain_over_tbm",
            "selection_regret",
        ]
        lines.append(targets[columns].round(4).to_markdown(index=False) + "\n")
    lines.extend(
        [
            "## Interpretation\n",
            "- **Pass**: the pretrained branch raises mean oracle-pool TM above the configured "
            "threshold; proceed to geometry v2 and fold-aware fusion.",
            "- **Fail**: candidate generation is still the bottleneck; improve model coverage, "
            "sampling, or checkpoints before building a more complex refiner.",
            "- **Not evaluable**: at least one target needs both a TBM and a pretrained candidate.\n",
        ]
    )
    report_path.write_text("\n".join(lines))
    return result


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"
