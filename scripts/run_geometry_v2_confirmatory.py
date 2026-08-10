#!/usr/bin/env python
"""Confirmatory 60/20/20 evaluation for conservative Geometry v2.

The protocol deliberately separates three roles:

* 60 temporally earlier RNAs estimate empirical geometry priors;
* 20 calibration RNAs choose between pre-declared Geometry-v2 variants;
* 20 newest RNAs are evaluated only from the frozen calibration document.

The validation command never changes the frozen configuration.  Exact reruns are
allowed for reproducibility, but an existing result directory is not overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.eval.local_metrics import local_accuracy_metrics
from rna3d.eval.statistics import paired_target_summary
from rna3d.eval.usalign import score_target
from rna3d.geofuse.candidate import CandidateCache, StructureCandidate, safe_name
from rna3d.geofuse.geometry_v2 import (
    estimate_geometry_v2_priors,
    geometry_v2_metrics,
)
from rna3d.geofuse.phase_a import rank_without_native, select_source_balanced
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2
from rna3d.geometry.priors import compute_priors
from rna3d.paths import cache, processed


PROTOCOL_VERSION = 1
MANIFEST = processed() / "geofuse_real_oof_v2" / "medium_manifest.csv"
PROTOCOL_DIR = processed() / "geometry_v2_confirmatory"
PRIORS_V1 = PROTOCOL_DIR / "priors_v1_train60.json"
PRIORS_V2 = PROTOCOL_DIR / "priors_v2_train60.json"
PRIOR_PROVENANCE = PROTOCOL_DIR / "prior_provenance.json"
FROZEN_CONFIG = PROTOCOL_DIR / "frozen_geometry_v2_config.json"
REFINEMENT_CACHE = cache() / "geometry_v2_confirmatory"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "tables" / "geometry_v2_confirmatory"
DEFAULT_REPORT = REPO_ROOT / "reports" / "thesis_notes" / "geometry_v2_confirmatory.md"

LOCAL_METRICS = ("c1_rmsd", "c1_lddt", "sw_rmsd_9", "sw_rmsd_15", "sw_rmsd_31")
PHYSICAL_METRICS = (
    "bb_dev",
    "clash_per_res",
    "rg_err",
    "sharp_kinks",
    "angle_nll",
    "torsion_nll",
    "pair_like_fraction",
)


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    config: GeometryV2Config
    selection_eligible: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_priors(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "_raw"}


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST, dtype={"target_id": str, "sequence": str, "split": str})
    required = {
        "target_id",
        "sequence",
        "seq_len",
        "date",
        "split",
        "sequence_group",
        "family_group",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")
    expected = {"train": 60, "calibration": 20, "validation": 20}
    counts = frame.groupby("split")["target_id"].nunique().to_dict()
    if counts != expected:
        raise ValueError(f"expected split counts {expected}, found {counts}")
    dates = frame.groupby("split")["date"].agg(["min", "max"])
    if not (dates.loc["train", "max"] < dates.loc["calibration", "min"]):
        raise ValueError("train and calibration dates overlap or are out of order")
    if not (dates.loc["calibration", "max"] < dates.loc["validation", "min"]):
        raise ValueError("calibration and validation dates overlap or are out of order")
    if len(frame) != frame["target_id"].nunique():
        raise ValueError("confirmatory manifest must contain exactly one row per target")
    for group_column in ("sequence_group", "family_group"):
        if frame[group_column].isna().any():
            raise ValueError(f"{group_column} contains missing values")
        split_counts = frame.groupby(group_column)["split"].nunique()
        overlapping = split_counts[split_counts > 1]
        if not overlapping.empty:
            raise ValueError(
                f"{group_column} crosses train/calibration/validation splits: "
                f"{overlapping.index.tolist()[:10]}"
            )
    return frame


def _labels_for(target_ids: Iterable[str]) -> pd.DataFrame:
    labels = io.load_labels("train_v2")
    wanted = set(target_ids)
    tids = labels["ID"].map(io.target_id_of)
    selected = labels[tids.isin(wanted)].copy()
    found = set(selected["ID"].map(io.target_id_of))
    if found != wanted:
        raise KeyError(f"labels missing targets: {sorted(wanted - found)}")
    return selected


def prepare(_: argparse.Namespace) -> None:
    manifest = _manifest()
    train = manifest[manifest["split"] == "train"].copy()
    labels = _labels_for(train["target_id"])
    priors_v1 = _jsonable_priors(compute_priors(labels))
    priors_v2 = estimate_geometry_v2_priors(labels)
    _write_json(PRIORS_V1, priors_v1)
    _write_json(PRIORS_V2, priors_v2)
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": _now(),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "split_counts": manifest.groupby("split")["target_id"].nunique().to_dict(),
        "split_date_ranges": manifest.groupby("split")["date"].agg(["min", "max"]).to_dict("index"),
        "training_targets": train["target_id"].tolist(),
        "training_target_count": int(train["target_id"].nunique()),
        "separation_audit": {
            "one_row_per_target": bool(len(manifest) == manifest["target_id"].nunique()),
            "sequence_groups_crossing_splits": int(
                (manifest.groupby("sequence_group")["split"].nunique() > 1).sum()
            ),
            "family_groups_crossing_splits": int(
                (manifest.groupby("family_group")["split"].nunique() > 1).sum()
            ),
        },
        "rule": "Only the 60 earliest training targets estimate geometry priors.",
        "priors_v1_sha256": _sha256(PRIORS_V1),
        "priors_v2_sha256": _sha256(PRIORS_V2),
    }
    _write_json(PRIOR_PROVENANCE, provenance)
    print(f"[prepare] wrote {PRIORS_V1}")
    print(f"[prepare] wrote {PRIORS_V2}")
    print(json.dumps(provenance["split_date_ranges"], indent=2))


def _load_priors() -> tuple[dict, dict]:
    if not PRIORS_V1.exists() or not PRIORS_V2.exists():
        raise FileNotFoundError("run the prepare command before calibration or validation")
    return json.loads(PRIORS_V1.read_text()), json.loads(PRIORS_V2.read_text())


def _cfg(**updates: object) -> GeometryV2Config:
    document = asdict(GeometryV2Config())
    document.update(updates)
    return GeometryV2Config(**document)


def calibration_variants(suite: str) -> list[Variant]:
    """Pre-declared cumulative, leave-one-out, and robustness variants."""
    variants = [
        Variant(
            "simple_source_backbone",
            "dumb_baseline",
            _cfg(w_clash=0.0, w_rg=0.0, w_angle=0.0, w_torsion=0.0, w_kink=0.0),
        ),
        Variant(
            "cumulative_plus_angle",
            "cumulative",
            _cfg(steps=100, w_clash=0.0, w_rg=0.0, w_torsion=0.0, w_kink=0.0),
        ),
        Variant(
            "cumulative_plus_torsion",
            "cumulative",
            _cfg(steps=100, w_clash=0.0, w_rg=0.0, w_kink=0.0),
        ),
        Variant(
            "cumulative_plus_kink",
            "cumulative",
            _cfg(steps=100, w_clash=0.0, w_rg=0.0),
        ),
        Variant(
            "full_fixed",
            "adaptive_ablation",
            _cfg(adaptive_strength=False, fixed_strength=0.6),
            True,
        ),
        Variant("full_adaptive", "full", _cfg(), True),
        Variant("mechanism_full_100", "mechanism_reference", _cfg(steps=100)),
        Variant("minus_backbone", "leave_one_out", _cfg(steps=100, w_backbone=0.0)),
        Variant("minus_angle", "leave_one_out", _cfg(steps=100, w_angle=0.0)),
        Variant("minus_torsion", "leave_one_out", _cfg(steps=100, w_torsion=0.0)),
        Variant("minus_kink", "leave_one_out", _cfg(steps=100, w_kink=0.0)),
        Variant(
            "minus_clash_rg",
            "leave_one_out",
            _cfg(steps=100, w_clash=0.0, w_rg=0.0),
        ),
    ]
    if suite == "full":
        variants.extend(
            [
                Variant("source_1p5", "source_sensitivity", _cfg(w_source=1.5), True),
                Variant("source_6p0", "source_sensitivity", _cfg(w_source=6.0), True),
                Variant("lr_0p02", "optimizer_sensitivity", _cfg(lr=0.02), True),
                Variant("lr_0p08", "optimizer_sensitivity", _cfg(lr=0.08), True),
                Variant("steps_25", "convergence", _cfg(steps=25), True),
                Variant("steps_50", "convergence", _cfg(steps=50), True),
                Variant("steps_100", "convergence", _cfg(steps=100), True),
                Variant("steps_200", "convergence", _cfg(steps=200), True),
            ]
        )
    return variants


def _variant_digest(variant: Variant) -> str:
    document = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation": "refine_structure_v2_adaptive_ablation_v1",
        "name": variant.name,
        "config": asdict(variant.config),
        "priors_v1": _sha256(PRIORS_V1),
        "priors_v2": _sha256(PRIORS_V2),
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]


def _refine_cached(
    candidate: StructureCandidate,
    sequence: str,
    variant: Variant,
    priors_v1: dict,
    priors_v2: dict,
    device: str,
) -> tuple[np.ndarray, float]:
    digest = _variant_digest(variant)
    path = (
        REFINEMENT_CACHE
        / safe_name(candidate.target_id)
        / f"{safe_name(candidate.candidate_id)}__{safe_name(variant.name)}__{digest}.npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            return payload["coords"].copy(), float(payload["seconds"])
    started = time.time()
    coords, info = refine_structure_v2(
        candidate.coords,
        sequence,
        priors_v1,
        priors_v2,
        source_confidence=candidate.confidence,
        global_confidence=candidate.global_confidence,
        cfg=variant.config,
        device=device,
    )
    elapsed = time.time() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": variant.name,
        "family": variant.family,
        "config": asdict(variant.config),
        "info": info,
    }
    np.savez_compressed(
        path,
        coords=np.asarray(coords, dtype=np.float32),
        seconds=np.asarray(elapsed),
        document=np.asarray(json.dumps(document, sort_keys=True)),
    )
    return coords, elapsed


def _best_reference_index(
    coords: np.ndarray, references: list[np.ndarray], sequence: str
) -> tuple[int, float]:
    scores = [
        float(score_target([coords], [reference], list(sequence)))
        for reference in references
    ]
    index = int(np.nanargmax(scores))
    return index, scores[index]


def _candidate_metadata(candidate: StructureCandidate, seq_len: int) -> dict:
    return {
        "target_id": candidate.target_id,
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "kind": candidate.kind,
        "model": candidate.model,
        "seq_len": seq_len,
        "global_confidence": candidate.global_confidence,
        "mean_residue_confidence": float(candidate.confidence.mean()),
        "support_fraction": float(candidate.support_mask.mean()),
        "gap_fraction": float(1.0 - candidate.support_mask.mean()),
    }


def _worker_init() -> None:
    # Tiny C1' traces are faster and more reproducible with one BLAS/Torch
    # thread per worker than with nested thread pools.
    import torch

    torch.set_num_threads(1)


def _refine_task(payload: tuple) -> tuple[np.ndarray, float]:
    return _refine_cached(*payload)


def _evaluate_split(
    split: str,
    variants: list[Variant],
    *,
    device: str,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = _manifest()
    sequences = manifest[manifest["split"] == split].reset_index(drop=True)
    labels = _labels_for(sequences["target_id"])
    priors_v1, priors_v2 = _load_priors()
    store = CandidateCache(cache() / "geofuse_candidates", "train_v2")
    candidate_rows: list[dict] = []
    target_rows: list[dict] = []
    runtime_rows: list[dict] = []

    executor = (
        ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
        if workers > 1
        else None
    )
    try:
        for target_number, target in enumerate(sequences.itertuples(index=False), 1):
            candidates = select_source_balanced(
                store.load_target(target.target_id, target.sequence), 5
            )
            if len(candidates) != 5:
                raise ValueError(f"{target.target_id}: expected five candidates, found {len(candidates)}")
            references = io.get_reference_coords(labels, target.target_id)
            raw_reference: dict[str, int] = {}
            raw_candidate_tm: dict[str, float] = {}
            coordinates: dict[str, list[np.ndarray]] = {
                "raw": [candidate.coords for candidate in candidates]
            }
            runtimes = {"raw": 0.0}

            for candidate in candidates:
                reference_index, tm = _best_reference_index(
                    candidate.coords, references, target.sequence
                )
                raw_reference[candidate.candidate_id] = reference_index
                raw_candidate_tm[candidate.candidate_id] = tm

            for variant in variants:
                payloads = [
                    (
                        candidate,
                        target.sequence,
                        variant,
                        priors_v1,
                        priors_v2,
                        device,
                    )
                    for candidate in candidates
                ]
                outputs = (
                    list(executor.map(_refine_task, payloads))
                    if executor is not None
                    else [_refine_task(payload) for payload in payloads]
                )
                coordinates[variant.name] = [item[0] for item in outputs]
                runtimes[variant.name] = float(sum(item[1] for item in outputs))

            for setting, structures in coordinates.items():
                per_candidate = []
                for candidate, coords in zip(candidates, structures):
                    reference_index = raw_reference[candidate.candidate_id]
                    local = local_accuracy_metrics(
                        coords, references[reference_index], windows=(9, 15, 31)
                    )
                    physical = geometry_v2_metrics(
                        coords, target.sequence, priors_v1, priors_v2
                    )
                    displacement = np.linalg.norm(coords - candidate.coords, axis=1)
                    row = {
                        **_candidate_metadata(candidate, len(target.sequence)),
                        "split": split,
                        "setting": setting,
                        "reference_index_fixed_from_raw": reference_index,
                        "raw_candidate_tm": raw_candidate_tm[candidate.candidate_id],
                        "mean_drift": float(displacement.mean()),
                        "max_drift": float(displacement.max()),
                        **local,
                        **physical,
                    }
                    per_candidate.append(row)
                    candidate_rows.append(row)
                target_rows.append(
                    {
                        "target_id": target.target_id,
                        "split": split,
                        "seq_len": len(target.sequence),
                        "setting": setting,
                        "n_candidates": len(candidates),
                        "best5_tm": float(
                            score_target(structures, references, list(target.sequence))
                        ),
                        **{
                            metric: float(np.nanmean([row[metric] for row in per_candidate]))
                            for metric in (*LOCAL_METRICS, *PHYSICAL_METRICS, "mean_drift")
                        },
                    },
                )
                runtime_rows.append(
                    {
                        "target_id": target.target_id,
                        "split": split,
                        "setting": setting,
                        "seconds": runtimes[setting],
                    }
                )
            latest = {
                row["setting"]: row
                for row in target_rows
                if row["target_id"] == target.target_id
            }
            full_name = "full_adaptive" if "full_adaptive" in latest else variants[0].name
            print(
                f"[{split} {target_number:02d}/{len(sequences)} {target.target_id}] "
                f"TM {latest['raw']['best5_tm']:.4f}->{latest[full_name]['best5_tm']:.4f}; "
                f"lDDT {latest['raw']['c1_lddt']:.4f}->{latest[full_name]['c1_lddt']:.4f}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return pd.DataFrame(target_rows), pd.DataFrame(candidate_rows), pd.DataFrame(runtime_rows)


def _aggregate(targets: pd.DataFrame) -> pd.DataFrame:
    columns = ["best5_tm", *LOCAL_METRICS, *PHYSICAL_METRICS, "mean_drift"]
    return targets.groupby("setting")[columns].mean()


def _delta(aggregate: pd.DataFrame, setting: str, metric: str) -> float:
    return float(aggregate.loc[setting, metric] - aggregate.loc["raw", metric])


def _passes_gate(aggregate: pd.DataFrame, setting: str, tm_tolerance: float) -> bool:
    return bool(
        _delta(aggregate, setting, "best5_tm") >= -tm_tolerance
        and _delta(aggregate, setting, "c1_lddt") > 0
        and sum(
            _delta(aggregate, setting, f"sw_rmsd_{window}") < 0
            for window in (9, 15, 31)
        )
        >= 2
    )


def _bootstrap(targets: pd.DataFrame, settings: Iterable[str]) -> pd.DataFrame:
    rows = []
    raw = targets[targets["setting"] == "raw"].set_index("target_id")
    for setting in settings:
        method = targets[targets["setting"] == setting].set_index("target_id")
        shared = raw.index.intersection(method.index)
        for metric in ("best5_tm", *LOCAL_METRICS):
            result = paired_target_summary(
                raw.loc[shared, metric].to_numpy(),
                method.loc[shared, metric].to_numpy(),
                higher_is_better=metric in {"best5_tm", "c1_lddt"},
            )
            rows.append(
                {"baseline": "raw", "setting": setting, "metric": metric, **result}
            )
    return pd.DataFrame(rows)


def _compare_settings(
    targets: pd.DataFrame, baseline_name: str, method_name: str
) -> pd.DataFrame:
    rows = []
    baseline = targets[targets["setting"] == baseline_name].set_index("target_id")
    method = targets[targets["setting"] == method_name].set_index("target_id")
    shared = baseline.index.intersection(method.index)
    for metric in ("best5_tm", *LOCAL_METRICS):
        result = paired_target_summary(
            baseline.loc[shared, metric].to_numpy(),
            method.loc[shared, metric].to_numpy(),
            higher_is_better=metric in {"best5_tm", "c1_lddt"},
        )
        rows.append(
            {
                "baseline": baseline_name,
                "setting": method_name,
                "metric": metric,
                **result,
            }
        )
    return pd.DataFrame(rows)


def calibrate(args: argparse.Namespace) -> None:
    if FROZEN_CONFIG.exists() and not args.replace_frozen:
        raise FileExistsError(
            f"{FROZEN_CONFIG} already exists; pass --replace-frozen only before validation"
        )
    variants = calibration_variants(args.suite)
    targets, candidates, runtimes = _evaluate_split(
        "calibration", variants, device=args.device, workers=args.workers
    )
    aggregate = _aggregate(targets)
    eligible = [variant.name for variant in variants if variant.selection_eligible]
    passed = [name for name in eligible if _passes_gate(aggregate, name, args.tm_tolerance)]
    ranking_pool = passed or eligible
    selected_name = max(
        ranking_pool,
        key=lambda name: (
            _delta(aggregate, name, "c1_lddt"),
            -_delta(aggregate, name, "sw_rmsd_15"),
            _delta(aggregate, name, "best5_tm"),
        ),
    )
    selected = next(variant for variant in variants if variant.name == selected_name)
    output = Path(args.output_dir) / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output / "target_metrics.csv", index=False)
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    runtimes.to_csv(output / "runtimes.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv")
    boot = _bootstrap(targets, [variant.name for variant in variants])
    boot.to_csv(output / "target_bootstrap.csv", index=False)
    frozen = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at_utc": _now(),
        "validation_labels_seen": False,
        "manifest_sha256": _sha256(MANIFEST),
        "priors_v1_sha256": _sha256(PRIORS_V1),
        "priors_v2_sha256": _sha256(PRIORS_V2),
        "calibration_suite": args.suite,
        "tm_tolerance": args.tm_tolerance,
        "eligible_variants": eligible,
        "gate_passing_variants": passed,
        "selection_status": "passed_gate" if passed else "no_variant_passed_gate",
        "selection_rule": (
            "Among pre-declared full variants passing TM preservation, positive C1 lDDT, "
            "and >=2/3 sliding-window improvements: maximize lDDT delta, then SW15 and TM."
        ),
        "selected_variant": selected.name,
        "selected_family": selected.family,
        "selected_config": asdict(selected.config),
        "calibration_aggregate": aggregate.reset_index().to_dict("records"),
    }
    _write_json(FROZEN_CONFIG, frozen)
    report = [
        "# Geometry v2 calibration (20 RNA)",
        "",
        "The 20 calibration RNAs select among the pre-declared complete-method "
        "hyperparameter and convergence variants. Cumulative, leave-one-out and dumb-baseline "
        "rows explain mechanism but are not eligible to win the configuration search.",
        "",
        f"- Frozen variant: **{selected.name}**",
        f"- Calibration gate: **{frozen['selection_status']}**",
        f"- Untouched validation labels read at this point: **no**",
        "",
        "## Equal-weight target means",
        "",
        aggregate.round(6).to_markdown(),
        "",
        "## Deltas from raw",
        "",
        (aggregate - aggregate.loc["raw"]).round(6).to_markdown(),
        "",
        "## Paired target bootstrap",
        "",
        boot.round(6).to_markdown(index=False),
        "",
    ]
    (output / "calibration_report.md").write_text("\n".join(report))
    print(aggregate.round(6).to_string())
    print(f"[freeze] {selected.name} -> {FROZEN_CONFIG}")


def _allocation_rows(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    frozen_name: str,
    manifest_split: pd.DataFrame,
) -> pd.DataFrame:
    """Score all source allocations supported by the cached 3T/2D bank."""
    labels = _labels_for(manifest_split["target_id"])
    priors_v1, priors_v2 = _load_priors()
    frozen = json.loads(FROZEN_CONFIG.read_text())
    variant = Variant(frozen_name, "frozen", GeometryV2Config(**frozen["selected_config"]))
    store = CandidateCache(cache() / "geofuse_candidates", "train_v2")
    rows = []
    for target in manifest_split.itertuples(index=False):
        all_candidates = store.load_target(target.target_id, target.sequence)
        tbm = rank_without_native([candidate for candidate in all_candidates if candidate.kind == "template"])
        deep = rank_without_native([candidate for candidate in all_candidates if candidate.kind == "pretrained"])
        raw_map = {candidate.candidate_id: candidate.coords for candidate in all_candidates}
        refined_map = {
            candidate.candidate_id: _refine_cached(
                candidate, target.sequence, variant, priors_v1, priors_v2, "cuda"
            )[0]
            for candidate in all_candidates
        }
        references = io.get_reference_coords(labels, target.target_id)
        allocations = []
        for total in (1, 2, 3):
            for n_tbm in range(total, -1, -1):
                n_deep = total - n_tbm
                if n_tbm <= len(tbm) and n_deep <= len(deep):
                    allocations.append((n_tbm, n_deep))
        allocations.append((3, 2))
        for n_tbm, n_deep in allocations:
            chosen = tbm[:n_tbm] + deep[:n_deep]
            for geometry, coord_map in (("off", raw_map), ("on", refined_map)):
                rows.append(
                    {
                        "target_id": target.target_id,
                        "n_tbm": n_tbm,
                        "n_drfold2": n_deep,
                        "total_candidates": n_tbm + n_deep,
                        "geometry": geometry,
                        "best_tm": float(
                            score_target(
                                [coord_map[item.candidate_id] for item in chosen],
                                references,
                                list(target.sequence),
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _factorial(allocation: pd.DataFrame) -> pd.DataFrame:
    # Use a fixed candidate count so the bank factor measures composition rather
    # than getting two extra chances.  The full production 3T+2D row remains in
    # candidate_allocation.csv as a separate pragmatic comparison.
    chosen = allocation[
        ((allocation["n_tbm"] == 3) & (allocation["n_drfold2"] == 0))
        | ((allocation["n_tbm"] == 2) & (allocation["n_drfold2"] == 1))
    ].copy()
    chosen["bank"] = np.where(
        chosen["n_drfold2"] == 0, "TBM (3T+0D)", "Hybrid (2T+1D)"
    )
    return chosen.groupby(["bank", "geometry"])["best_tm"].agg(["mean", "std", "count"]).reset_index()


def _factorial_effects(allocation: pd.DataFrame) -> pd.DataFrame:
    selected = allocation[
        ((allocation["n_tbm"] == 3) & (allocation["n_drfold2"] == 0))
        | ((allocation["n_tbm"] == 2) & (allocation["n_drfold2"] == 1))
    ].copy()
    selected["bank"] = np.where(selected["n_drfold2"] == 0, "tbm", "hybrid")
    pivot = selected.pivot(index="target_id", columns=["bank", "geometry"], values="best_tm")
    effects = {
        "hybrid_gain_geometry_off": pivot[("hybrid", "off")] - pivot[("tbm", "off")],
        "hybrid_gain_geometry_on": pivot[("hybrid", "on")] - pivot[("tbm", "on")],
        "geometry_effect_tbm": pivot[("tbm", "on")] - pivot[("tbm", "off")],
        "geometry_effect_hybrid": pivot[("hybrid", "on")] - pivot[("hybrid", "off")],
    }
    effects["interaction"] = effects["geometry_effect_hybrid"] - effects["geometry_effect_tbm"]
    rows = []
    for name, values in effects.items():
        result = paired_target_summary(
            np.zeros(len(values)), values.to_numpy(), higher_is_better=True
        )
        rows.append({"effect": name, **result})
    return pd.DataFrame(rows)


def _subgroups(candidates: pd.DataFrame, frozen_name: str) -> pd.DataFrame:
    raw = candidates[candidates["setting"] == "raw"].copy()
    refined = candidates[candidates["setting"] == frozen_name].copy()
    keys = ["target_id", "candidate_id"]
    metric_columns = [*LOCAL_METRICS, *PHYSICAL_METRICS, "mean_drift"]
    merged = raw[
        keys
        + [
            "source",
            "kind",
            "seq_len",
            "global_confidence",
            "support_fraction",
            "gap_fraction",
            "raw_candidate_tm",
            *metric_columns,
        ]
    ].merge(
        refined[keys + metric_columns], on=keys, suffixes=("_raw", "_refined")
    )
    merged["length_group"] = pd.cut(
        merged["seq_len"], bins=[0, 50, 75, np.inf], labels=["short", "medium", "long"]
    )
    for column, group_name in (
        ("raw_candidate_tm", "initial_quality_group"),
        ("global_confidence", "confidence_group"),
        ("support_fraction", "support_group"),
    ):
        merged[group_name] = pd.qcut(
            merged[column].rank(method="first"), 3, labels=["low", "mid", "high"]
        )
    rows = []
    for group_type in (
        "source",
        "length_group",
        "initial_quality_group",
        "confidence_group",
        "support_group",
    ):
        for group_value, group in merged.groupby(group_type, observed=True):
            for metric in ("c1_lddt", "sw_rmsd_15", "sharp_kinks", "angle_nll", "torsion_nll"):
                higher = metric == "c1_lddt"
                # Candidates from one RNA are correlated.  First average the
                # paired candidate values within each target/group cell, then
                # bootstrap targets; never pretend 100 candidates are 100 RNAs.
                target_pairs = group.groupby("target_id")[[
                    f"{metric}_raw", f"{metric}_refined"
                ]].mean()
                result = paired_target_summary(
                    target_pairs[f"{metric}_raw"].to_numpy(),
                    target_pairs[f"{metric}_refined"].to_numpy(),
                    higher_is_better=higher,
                    bootstrap_samples=5000,
                )
                rows.append(
                    {
                        "group_type": group_type,
                        "group": str(group_value),
                        "metric": metric,
                        "n_candidate_pairs": int(len(group)),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _write_validation_summary(
    output: Path,
    report_path: Path,
    frozen: dict,
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    allocation: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    selected_name = str(frozen["selected_variant"])
    aggregate = _aggregate(targets)
    aggregate.to_csv(output / "aggregate.csv")
    bootstrap = pd.concat(
        [
            _bootstrap(targets, ["simple_source_backbone", selected_name]),
            _compare_settings(targets, "simple_source_backbone", selected_name),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output / "target_bootstrap.csv", index=False)
    factorial = _factorial(allocation)
    factorial.to_csv(output / "factorial_2x2.csv", index=False)
    factorial_effects = _factorial_effects(allocation)
    factorial_effects.to_csv(output / "factorial_effects.csv", index=False)
    subgroups = _subgroups(candidates, selected_name)
    subgroups.to_csv(output / "subgroup_analysis.csv", index=False)
    gate = _passes_gate(aggregate, selected_name, float(frozen["tm_tolerance"]))
    report = [
        "# Geometry v2 confirmatory validation (20 newest RNA)",
        "",
        f"- Frozen method: **{selected_name}**",
        f"- Confirmatory independent-metric gate: **{'pass' if gate else 'fail'}**",
        "- Split protocol: 60 train-prior → 20 calibration → 20 newest validation, "
        "with no sequence/family group crossing a split.",
        "- Native reference for each refined candidate is fixed from its raw candidate, "
        "so refinement cannot benefit by switching to an easier native conformation.",
        "- All headline means and confidence intervals give equal weight to each RNA target.",
        "",
        "## Equal-weight target means",
        "",
        aggregate.round(6).to_markdown(),
        "",
        "## Deltas from raw",
        "",
        (aggregate - aggregate.loc["raw"]).round(6).to_markdown(),
        "",
        "## Paired target bootstrap",
        "",
        "Positive deltas always mean the method in `setting` is better than `baseline`. "
        "C1-lDDT, sliding-window RMSD and TM are independent of the optimized Geometry-v2 "
        "loss. Angle/torsion NLL, clash and kink values are objective diagnostics only.",
        "",
        bootstrap.round(6).to_markdown(index=False),
        "",
        "## Fixed-N 2×2 source-bank × Geometry factorial",
        "",
        "Both banks contain exactly three candidates: 3 TBM versus 2 TBM + 1 DRfold2. "
        "This removes candidate-count advantage from the source-composition factor.",
        "",
        factorial.round(6).to_markdown(index=False),
        "",
        factorial_effects.round(6).to_markdown(index=False),
        "",
        "The cache contains 3 TBM and 2 DRfold2 candidates per RNA. Therefore the "
        "requested full fixed-five 5T→0T allocation sweep is not identifiable from current "
        "artifacts. `candidate_allocation.csv` reports all honest fixed-size allocations "
        "supported at N=1,2,3, plus the production 3T+2D bank.",
        "",
        "## Stratified paired effects",
        "",
        "Candidates are first averaged inside each target/group cell; intervals then "
        "bootstrap RNA targets, not correlated candidate structures.",
        "",
        subgroups.round(6).to_markdown(index=False),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))
    return gate, aggregate


def validate(args: argparse.Namespace) -> None:
    frozen = json.loads(FROZEN_CONFIG.read_text())
    if frozen.get("validation_labels_seen"):
        raise RuntimeError("frozen config claims validation labels were seen before freezing")
    selected = Variant(
        frozen["selected_variant"],
        "frozen_full",
        GeometryV2Config(**frozen["selected_config"]),
    )
    simple = Variant(
        "simple_source_backbone",
        "dumb_baseline",
        _cfg(w_clash=0.0, w_rg=0.0, w_angle=0.0, w_torsion=0.0, w_kink=0.0),
    )
    output = Path(args.output_dir) / "validation"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} already contains held-out results; refusing to overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "validation_run_provenance.json",
        {
            "started_at_utc": _now(),
            "frozen_config_sha256": _sha256(FROZEN_CONFIG),
            "selected_variant": selected.name,
            "rule": "The selected configuration was frozen before validation labels were loaded.",
        },
    )
    targets, candidates, runtimes = _evaluate_split(
        "validation", [simple, selected], device=args.device, workers=args.workers
    )
    targets.to_csv(output / "target_metrics.csv", index=False)
    candidates.to_csv(output / "candidate_metrics.csv", index=False)
    runtimes.to_csv(output / "runtimes.csv", index=False)
    manifest_split = _manifest().query("split == 'validation'").reset_index(drop=True)
    allocation = _allocation_rows(targets, candidates, selected.name, manifest_split)
    allocation.to_csv(output / "candidate_allocation.csv", index=False)
    gate, aggregate = _write_validation_summary(
        output, Path(args.report), frozen, targets, candidates, allocation
    )
    provenance = json.loads((output / "validation_run_provenance.json").read_text())
    provenance["completed_at_utc"] = _now()
    provenance["gate"] = "pass" if gate else "fail"
    _write_json(output / "validation_run_provenance.json", provenance)
    print(aggregate.round(6).to_string())
    print(f"[validation gate] {'pass' if gate else 'fail'}")
    print(f"[report] {args.report}")


def summarize(args: argparse.Namespace) -> None:
    """Rebuild derived tables without re-running or re-reading native labels."""
    frozen = json.loads(FROZEN_CONFIG.read_text())
    output = Path(args.output_dir) / "validation"
    required = [
        output / "target_metrics.csv",
        output / "candidate_metrics.csv",
        output / "candidate_allocation.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"validation artifacts missing: {missing}")
    targets = pd.read_csv(required[0])
    candidates = pd.read_csv(required[1])
    allocation = pd.read_csv(required[2])
    gate, aggregate = _write_validation_summary(
        output, Path(args.report), frozen, targets, candidates, allocation
    )
    print(aggregate.round(6).to_string())
    print(f"[validation gate] {'pass' if gate else 'fail'}")
    print(f"[report rebuilt without native-label access] {args.report}")


def inventory(_: argparse.Namespace) -> None:
    manifest = _manifest()
    store = CandidateCache(cache() / "geofuse_candidates", "train_v2")
    rows = []
    for target in manifest.itertuples(index=False):
        candidates = store.load_target(target.target_id, target.sequence)
        rows.append(
            {
                "target_id": target.target_id,
                "split": target.split,
                "seq_len": target.seq_len,
                "n_candidates": len(candidates),
                "n_tbm": sum(candidate.kind == "template" for candidate in candidates),
                "n_drfold2": sum(candidate.kind == "pretrained" for candidate in candidates),
                "all_finite": all(candidate.valid_mask.all() for candidate in candidates),
            }
        )
    frame = pd.DataFrame(rows)
    print(frame.groupby(["split", "n_candidates", "n_tbm", "n_drfold2", "all_finite"]).size())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory").set_defaults(func=inventory)
    commands.add_parser("prepare").set_defaults(func=prepare)

    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--suite", choices=("core", "full"), default="full")
    calibration.add_argument("--device", default="cuda")
    calibration.add_argument("--workers", type=int, default=1)
    calibration.add_argument("--tm-tolerance", type=float, default=0.005)
    calibration.add_argument("--replace-frozen", action="store_true")
    calibration.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    calibration.set_defaults(func=calibrate)

    validation = commands.add_parser("validate")
    validation.add_argument("--device", default="cuda")
    validation.add_argument("--workers", type=int, default=1)
    validation.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    validation.add_argument("--report", default=DEFAULT_REPORT)
    validation.set_defaults(func=validate)

    summary = commands.add_parser("summarize")
    summary.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    summary.add_argument("--report", default=DEFAULT_REPORT)
    summary.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
