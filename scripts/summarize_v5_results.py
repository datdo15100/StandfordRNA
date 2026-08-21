#!/usr/bin/env python
"""Create the verified V5 CASP15 evidence ledger from frozen experiment tables."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import run_v5_casp15 as core


ROOT = REPO / "reports" / "thesis_v5" / "experiments"
RAW = ROOT / "raw_results"
SECONDARY = ROOT / "secondary_tbm"
REFINEMENT = ROOT / "refinement_results"
OUTPUT_JSON = ROOT / "V5_VERIFIED_EVIDENCE.json"
OUTPUT_MD = ROOT / "V5_EXPERIMENT_RESULTS.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effect(delta: pd.Series | np.ndarray, clusters: pd.Series | np.ndarray) -> dict:
    values = np.asarray(delta, dtype=float)
    groups = np.asarray(clusters)
    return {
        **core._bootstrap_effect(values, groups),
        "exact_cluster_sign_flip": core._exact_sign_flip(values, groups),
    }


def pivot_effect(
    frame: pd.DataFrame,
    index: list[str],
    column: str,
    value: str,
    first: str,
    second: str,
    *,
    lower_is_better: bool = False,
) -> dict:
    wide = frame.pivot(index=index, columns=column, values=value).reset_index()
    delta = (
        wide[second] - wide[first]
        if lower_is_better
        else wide[first] - wide[second]
    )
    result = effect(delta, wide["sequence_cluster"])
    result.update(
        {
            "first": first,
            "second": second,
            "metric": value,
            "positive_favours": first,
            "first_mean": float(wide[first].mean()),
            "second_mean": float(wide[second].mean()),
        }
    )
    return result


def main() -> None:
    required = [
        RAW / "rq1_target_best5_tm.csv",
        RAW / "rq2_allocation_target_tm.csv",
        RAW / "rq2_candidate_diversity.csv",
        SECONDARY / "target_best5_tm.csv",
        SECONDARY / "candidate_allocation_target_tm.csv",
        REFINEMENT / "factorial_candidate_metrics.csv",
        REFINEMENT / "factorial_bank_metrics.csv",
        REFINEMENT / "complete_pipeline_target_tm.csv",
    ]
    missing = [str(path.relative_to(REPO)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V5 experiment tables are incomplete: {missing}")

    targets = core.read_targets()
    target_clusters = targets.set_index("target_id")["sequence_cluster"]
    rq1 = pd.read_csv(required[0])
    clusters = rq1["sequence_cluster"]
    h1 = effect(rq1["V3_EXACT_RAW"] - rq1["J_SS_RAW"], clusters)
    h1.update(
        {
            "comparison": "exact reconstructed V3 TBM Raw minus J-SS-RAW",
            "j_mean_tm": float(rq1["J_SS_RAW"].mean()),
            "v3_mean_tm": float(rq1["V3_EXACT_RAW"].mean()),
            "relative_change_percent": float(
                100.0 * h1["mean_delta"] / rq1["J_SS_RAW"].mean()
            ),
        }
    )

    allocations = pd.read_csv(required[1])
    v3_alloc = allocations[allocations["tbm_base"] == "V3_EXACT_RAW"].pivot(
        index=["target_id", "sequence_cluster"], columns="allocation", values="best_tm"
    ).reset_index()
    h2 = effect(v3_alloc["3T+2D"] - v3_alloc["5T"], v3_alloc["sequence_cluster"])
    h2.update(
        {
            "comparison": "V3 3T+2D Raw minus V3 5T Raw",
            "five_t_mean_tm": float(v3_alloc["5T"].mean()),
            "three_t_two_d_mean_tm": float(v3_alloc["3T+2D"].mean()),
            "relative_change_percent": float(
                100.0 * h2["mean_delta"] / v3_alloc["5T"].mean()
            ),
        }
    )
    h2_4t1d = effect(v3_alloc["4T+1D"] - v3_alloc["5T"], v3_alloc["sequence_cluster"])
    h2_2d_vs_5t = effect(v3_alloc["2D"] - v3_alloc["5T"], v3_alloc["sequence_cluster"])
    h2_template_complement = effect(
        v3_alloc["3T+2D"] - v3_alloc["2D"], v3_alloc["sequence_cluster"]
    )

    candidate_metrics = pd.read_csv(required[5])
    target_metrics = candidate_metrics.groupby(
        ["target_id", "sequence_cluster", "bank", "setting"], as_index=False
    ).mean(numeric_only=True)
    v3_hybrid = target_metrics[target_metrics["bank"] == "V3_3T2D"]
    h3 = {
        metric: pivot_effect(
            v3_hybrid,
            ["target_id", "sequence_cluster"],
            "setting",
            metric,
            "Geometry_historical",
            "Simple",
            lower_is_better=metric in {"sw_rmsd_9", "sw_rmsd_15", "c1_rmsd"},
        )
        for metric in (
            "sw_rmsd_9",
            "sw_rmsd_15",
            "c1_rmsd",
            "c1_lddt",
            "candidate_tm_same_reference",
        )
    }
    geometry_vs_raw = {
        metric: pivot_effect(
            v3_hybrid,
            ["target_id", "sequence_cluster"],
            "setting",
            metric,
            "Geometry_historical",
            "Raw",
            lower_is_better=metric in {"sw_rmsd_9", "sw_rmsd_15", "c1_rmsd"},
        )
        for metric in (
            "sw_rmsd_9",
            "sw_rmsd_15",
            "c1_rmsd",
            "c1_lddt",
            "candidate_tm_same_reference",
        )
    }
    bank_metrics = pd.read_csv(required[6])
    v3_hybrid_bank = bank_metrics[bank_metrics["bank"] == "V3_3T2D"]
    h3["best5_tm"] = pivot_effect(
        v3_hybrid_bank,
        ["target_id", "sequence_cluster"],
        "setting",
        "best5_tm",
        "Geometry_historical",
        "Simple",
    )
    geometry_vs_raw["best5_tm"] = pivot_effect(
        v3_hybrid_bank,
        ["target_id", "sequence_cluster"],
        "setting",
        "best5_tm",
        "Geometry_historical",
        "Raw",
    )

    complete = pd.read_csv(required[7])
    complete_wide = complete.pivot(
        index=["target_id", "sequence_cluster"], columns="pipeline", values="best5_tm"
    ).reset_index()
    raw_vs_j_complete = effect(
        complete_wide["V3_hybrid_raw"]
        - complete_wide["J_same_sandbox_complete"],
        complete_wide["sequence_cluster"],
    )
    raw_vs_j_complete.update(
        {
            "j_complete_mean_tm": float(
                complete_wide["J_same_sandbox_complete"].mean()
            ),
            "v3_hybrid_raw_mean_tm": float(complete_wide["V3_hybrid_raw"].mean()),
        }
    )
    complete_means = {
        column: float(complete_wide[column].mean())
        for column in complete_wide.columns
        if column not in {"target_id", "sequence_cluster"}
    }

    staged = pd.read_csv(RAW / "rq1_staged_effects.csv")
    secondary = pd.read_csv(SECONDARY / "summary.csv")
    secondary_alloc = pd.read_csv(SECONDARY / "candidate_allocation_target_tm.csv")
    sec_wide = secondary_alloc.pivot(
        index=["target_id", "sequence_cluster"],
        columns=["tbm_base", "allocation"],
        values="best_tm",
    )
    v3_3t2d = v3_alloc.set_index("target_id")["3T+2D"]
    secondary_hybrid_comparisons = {}
    for base in secondary_alloc["tbm_base"].unique():
        values = sec_wide[(base, "3T+2D")]
        aligned_v3 = v3_3t2d.reindex(values.index)
        secondary_hybrid_comparisons[base] = effect(
            values.to_numpy() - aligned_v3.to_numpy(),
            target_clusters.reindex(values.index).to_numpy(),
        )

    diversity = pd.read_csv(required[2])
    diversity_v3 = diversity[diversity["tbm_base"] == "V3_EXACT_RAW"].pivot(
        index=["target_id", "sequence_cluster"],
        columns="allocation",
        values="mean_pairwise_self_tm",
    ).reset_index()
    diversity_effect = effect(
        diversity_v3["5T"] - diversity_v3["3T+2D"],
        diversity_v3["sequence_cluster"],
    )
    diversity_effect.update(
        {
            "positive_definition": "positive means 3T+2D has lower self-TM and therefore greater structural diversity",
            "five_t_mean_self_tm": float(diversity_v3["5T"].mean()),
            "three_t_two_d_mean_self_tm": float(diversity_v3["3T+2D"].mean()),
        }
    )

    validation_sequences = REPO / "data" / "stanford-rna-3d-folding" / "validation_sequences.csv"
    kaggle_test_sequences = REPO / "data" / "stanford-rna-3d-folding" / "test_sequences.csv"
    same_sequence_file = sha256(validation_sequences) == sha256(kaggle_test_sequences)

    evidence = {
        "status": "V5_CASP15_EVIDENCE_VERIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "CASP15 local validation and model development",
        "target_n": 12,
        "sequence_cluster_n": int(targets["sequence_cluster"].nunique()),
        "primary_questions": {"RQ1": h1, "RQ2": h2, "RQ3_geometry_vs_simple": h3},
        "supporting": {
            "RQ2_4T1D_vs_5T": h2_4t1d,
            "RQ2_2D_vs_5T": h2_2d_vs_5t,
            "RQ2_3T2D_vs_2D_template_complement": h2_template_complement,
            "RQ2_diversity": diversity_effect,
            "Geometry_vs_Raw": geometry_vs_raw,
            "complete_V3_hybrid_raw_vs_J_complete": raw_vs_j_complete,
            "complete_pipeline_mean_tm": complete_means,
            "secondary_hybrid_vs_exact_V3_hybrid": secondary_hybrid_comparisons,
            "staged_tbm_effects": staged.to_dict("records"),
            "secondary_tbm_summary": secondary.to_dict("records"),
        },
        "selection": {
            "frozen_candidate_pipeline": "V3 exact reconstructed 3T+2D Raw",
            "reason": "It has the highest CASP15 complete-pipeline mean best-of-five TM among the evaluated complete contenders. Geometry improves SW-RMSD but does not improve the primary TM objective and conflicts with lDDT relative to Simple.",
            "casp15_mean_best5_tm": complete_means["V3_hybrid_raw"],
            "refinement": "none",
            "geometry_role": "supported metric-specific local-analysis component; not selected in the TM-optimized final deployment",
        },
        "kaggle_external_check_limitation": {
            "validation_sequences_path": str(validation_sequences.relative_to(REPO)),
            "test_sequences_path": str(kaggle_test_sequences.relative_to(REPO)),
            "validation_sha256": sha256(validation_sequences),
            "test_sha256": sha256(kaggle_test_sequences),
            "byte_identical": same_sequence_file,
            "conclusion": "The downloaded late-submission Kaggle test sequence file is byte-identical to the CASP15 validation sequence file. A new late leaderboard score after CASP15 tuning is therefore a scorer/deployment compatibility check, not an independent hidden-target evaluation.",
        },
        "input_table_hashes": {
            str(path.relative_to(REPO)): sha256(path) for path in required
        },
        "code_sha256": sha256(Path(__file__)),
    }
    OUTPUT_JSON.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    def fmt(value: float) -> str:
        return f"{value:.6f}"

    markdown = f"""# V5 CASP15 experiment results

Status: complete local development/validation evidence. This is not a thesis chapter.

## Main results

| Question | Comparison | Mean scores | Paired delta | 95% cluster-bootstrap CI | Wins/ties/losses |
|---|---|---:|---:|---:|---:|
| RQ1 | exact V3 TBM Raw vs J-SS-RAW | {fmt(h1['v3_mean_tm'])} vs {fmt(h1['j_mean_tm'])} | {fmt(h1['mean_delta'])} ({h1['relative_change_percent']:.2f}%) | [{fmt(h1['ci_lower'])}, {fmt(h1['ci_upper'])}] | {h1['wins']}/{h1['ties']}/{h1['losses']} |
| RQ2 | V3 3T+2D Raw vs V3 5T Raw | {fmt(h2['three_t_two_d_mean_tm'])} vs {fmt(h2['five_t_mean_tm'])} | {fmt(h2['mean_delta'])} ({h2['relative_change_percent']:.2f}%) | [{fmt(h2['ci_lower'])}, {fmt(h2['ci_upper'])}] | {h2['wins']}/{h2['ties']}/{h2['losses']} |
| RQ3 local | Geometry vs Simple, SW-RMSD9 | {fmt(h3['sw_rmsd_9']['first_mean'])} vs {fmt(h3['sw_rmsd_9']['second_mean'])} A | {fmt(h3['sw_rmsd_9']['mean_delta'])} A reduction | [{fmt(h3['sw_rmsd_9']['ci_lower'])}, {fmt(h3['sw_rmsd_9']['ci_upper'])}] | {h3['sw_rmsd_9']['wins']}/{h3['sw_rmsd_9']['ties']}/{h3['sw_rmsd_9']['losses']} |
| RQ3 global | Geometry vs Simple, best-of-five TM | {fmt(h3['best5_tm']['first_mean'])} vs {fmt(h3['best5_tm']['second_mean'])} | {fmt(h3['best5_tm']['mean_delta'])} | [{fmt(h3['best5_tm']['ci_lower'])}, {fmt(h3['best5_tm']['ci_upper'])}] | {h3['best5_tm']['wins']}/{h3['best5_tm']['ties']}/{h3['best5_tm']['losses']} |

## Interpretation locked to the evidence

- RQ1: the reconstructed thesis TBM improves local CASP15 best-of-five TM over the inherited John algorithm in the common sandbox.
- The clearest internal TBM Lego is identity x query coverage: +{fmt(float(staged.loc[staged['stage'] == 'identity_to_identity_coverage', 'mean_delta'].iloc[0]))} TM. MMseqs retrieval, completeness, and distinct-PDB selection produce no TM change on this 12-target cohort; they are safeguards/availability mechanisms here.
- RQ2: direct DRfold2 candidates are much stronger than template-only candidates on CASP15. The 2D bank already reaches {fmt(float(v3_alloc['2D'].mean()))}; adding three templates raises this to {fmt(float(v3_alloc['3T+2D'].mean()))}, a further {fmt(h2_template_complement['mean_delta'])}.
- The structural-diversity mechanism is weak: mean self-TM changes from {fmt(diversity_effect['five_t_mean_self_tm'])} for 5T to {fmt(diversity_effect['three_t_two_d_mean_self_tm'])} for 3T+2D. The large TM gain is therefore better described as pretrained-source accuracy/coverage plus a smaller template complement, not as proven generic diversity.
- RQ3: Geometry improves SW-RMSD9 beyond Simple by {fmt(h3['sw_rmsd_9']['mean_delta'])} A on all 12 targets, but C1'-lDDT changes by {fmt(h3['c1_lddt']['mean_delta'])} in the opposite direction. Its bank TM delta versus Simple is small and uncertain. This is a metric-specific local trade-off, not universal refinement superiority.
- The selected TM-optimized deployment is `V3 exact reconstructed 3T+2D Raw` at {fmt(complete_means['V3_hybrid_raw'])} CASP15 TM. Geometry remains an analyzed local-refinement contribution but is not in the selected deployment.

## Kaggle limitation discovered before submission

`validation_sequences.csv` and the downloaded `test_sequences.csv` are byte-identical ({evidence['kaggle_external_check_limitation']['validation_sha256']}). Therefore a late Kaggle score obtained after these CASP15 experiments cannot honestly be called an independent hidden-target test. It may still be reported as an official-scorer/deployment compatibility check.
"""
    OUTPUT_MD.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
