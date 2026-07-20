"""Leakage-audited supervision from real TBM and pretrained candidates.

The synthetic Phase-D gate is useful only as initialization.  This module builds
the same residue-wise target from actual predictor errors while keeping native
coordinates out of candidate selection and feature construction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import io
from .candidate import StructureCandidate
from .phase_c import robust_superpose
from .phase_d import pair_gate_features


SPLITS = ("train", "calibration", "validation")


def sequence_group(sequence: str) -> str:
    """Return an exact-sequence group used as a minimum duplicate guard."""
    normalized = sequence.upper().replace("T", "U")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()[:16]


def grouped_temporal_split(
    frame: pd.DataFrame,
    calibration_fraction: float = 0.15,
    validation_fraction: float = 0.20,
    *,
    group_column: str = "sequence_group",
) -> pd.DataFrame:
    """Assign whole sequence groups to oldest/train, calibration, newest/validation.

    Dates first define chronological bands.  If a family crosses a boundary, its
    later band wins and older members are excluded.  This is intentionally
    conservative: it preserves both strict chronology and family disjointness.
    """
    if frame.empty:
        raise ValueError("cannot split an empty manifest")
    if calibration_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("calibration and validation fractions must be positive")
    if calibration_fraction + validation_fraction >= 0.8:
        raise ValueError("at least 20% of targets must remain for training")
    required = {"target_id", "date", group_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing split columns: {sorted(missing)}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    dates = (
        result.groupby("date", as_index=False)
        .agg(size=("target_id", "size"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    total = int(dates["size"].sum())
    train_limit = total * (1.0 - calibration_fraction - validation_fraction)
    calibration_limit = total * (1.0 - validation_fraction)
    cumulative = dates["size"].cumsum()
    dates["provisional_split"] = np.where(
        cumulative <= train_limit,
        "train",
        np.where(cumulative <= calibration_limit, "calibration", "validation"),
    )
    rank = {name: index for index, name in enumerate(SPLITS)}
    result["provisional_split"] = result["date"].map(
        dates.set_index("date")["provisional_split"]
    )
    group_split = result.groupby(group_column)["provisional_split"].agg(
        lambda values: max(values, key=rank.__getitem__)
    )
    result["split"] = result[group_column].map(group_split)
    result = result[result["provisional_split"] == result["split"]].drop(
        columns="provisional_split"
    )
    return result.sort_values(["date", "target_id"]).reset_index(drop=True)


def audit_pretrained_oof(
    candidate: StructureCandidate,
    target_date: str | pd.Timestamp,
) -> dict:
    """Validate structural-training provenance for one frozen-model prediction.

    Date-based OOF means the target structure was released strictly after the
    model's structural training cutoff.  A declared exclusion manifest is also
    accepted, but is reported separately because the caller must retain that file.
    """
    if candidate.kind != "pretrained":
        raise ValueError(f"{candidate.candidate_id}: expected pretrained candidate")
    metadata = candidate.metadata
    cutoff_raw = metadata.get("model_training_cutoff")
    exclusion = metadata.get("oof_exclusion_manifest")
    target = pd.Timestamp(target_date)
    cutoff = pd.to_datetime(cutoff_raw, errors="coerce") if cutoff_raw else pd.NaT
    date_safe = bool(pd.notna(cutoff) and pd.Timestamp(cutoff) < target)
    exclusion_safe = False
    if exclusion:
        exclusion_path = Path(str(exclusion))
        if exclusion_path.is_file():
            tokens = {
                token.strip()
                for line in exclusion_path.read_text().splitlines()
                for token in line.replace(",", " ").split()
            }
            exclusion_safe = candidate.target_id in tokens
    if not date_safe and not exclusion_safe:
        raise ValueError(
            f"{candidate.candidate_id}: not auditable as OOF; provide a structural "
            "model_training_cutoff earlier than the target date or an exclusion manifest"
        )
    return {
        "oof_mode": "date" if date_safe else "explicit_exclusion",
        "target_date": str(target.date()),
        "model_training_cutoff": str(pd.Timestamp(cutoff).date()) if date_safe else None,
        "model_training_data": metadata.get("model_training_data"),
        "oof_exclusion_manifest": exclusion,
    }


def audit_template_oof(
    candidate: StructureCandidate,
    target_date: str | pd.Timestamp,
    excluded_pdb_ids: set[str] | None = None,
) -> dict:
    """Validate temporal and direct-self exclusion metadata for a TBM candidate."""
    if candidate.kind != "template":
        raise ValueError(f"{candidate.candidate_id}: expected template candidate")
    metadata = candidate.metadata
    release_raw = metadata.get("release_date")
    release = pd.to_datetime(release_raw, errors="coerce") if release_raw else pd.NaT
    target = pd.Timestamp(target_date)
    if pd.isna(release) or pd.Timestamp(release) >= target:
        raise ValueError(
            f"{candidate.candidate_id}: missing or non-temporal template release date"
        )
    pdb_id = str(metadata.get("pdb_id", "")).upper()
    excluded = {value.upper() for value in (excluded_pdb_ids or set())}
    if pdb_id and pdb_id in excluded:
        raise ValueError(f"{candidate.candidate_id}: direct target PDB {pdb_id} was not excluded")
    return {
        "template_release_date": str(pd.Timestamp(release).date()),
        "template_pdb_id": pdb_id,
    }


def _aligned_error(coords: np.ndarray, native: np.ndarray) -> np.ndarray:
    resolved = np.isfinite(native).all(axis=1)
    aligned, _, _ = robust_superpose(coords, native, resolved)
    error = np.linalg.norm(aligned - native, axis=1)
    error[~resolved] = np.nan
    return error.astype(np.float32)


def make_real_example(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    native_references: list[np.ndarray],
    priors_v1: dict,
    priors_v2: dict,
) -> dict:
    """Build one native-supervised example from inference-available pair features."""
    features, _, alignment = pair_gate_features(template, pretrained, priors_v1, priors_v2)
    choices = []
    for native in native_references:
        native = np.asarray(native, dtype=float)
        if native.shape != template.coords.shape or np.isfinite(native).all(axis=1).sum() < 3:
            continue
        template_error = _aligned_error(template.coords, native)
        pretrained_error = _aligned_error(pretrained.coords, native)
        joint = float(np.nanmean(np.minimum(template_error, pretrained_error)))
        choices.append((joint, template_error, pretrained_error))
    if not choices:
        raise ValueError(f"{template.target_id}: no usable native conformation")
    _, template_error, pretrained_error = min(choices, key=lambda item: item[0])
    resolved = np.isfinite(template_error) & np.isfinite(pretrained_error)
    difference = np.nan_to_num(template_error - pretrained_error)
    target = 1.0 / (1.0 + np.exp(-np.clip(difference / 1.5, -12.0, 12.0)))
    weight = (0.25 + np.clip(np.abs(difference) / 5.0, 0.0, 1.0)) * resolved
    return {
        "target_id": template.target_id,
        "pair_id": f"{template.candidate_id}::{pretrained.candidate_id}",
        "features": features.astype(np.float32),
        "target": target.astype(np.float32),
        "weight": weight.astype(np.float32),
        "template_error": np.nan_to_num(template_error).astype(np.float32),
        "pretrained_error": np.nan_to_num(pretrained_error).astype(np.float32),
        "gap_rule": (~template.support_mask).astype(np.float32),
        "confidence_rule": (pretrained.confidence > template.confidence).astype(np.float32),
        "resolved_mask": resolved,
        "alignment_rmsd": alignment["alignment_rmsd"],
    }


def native_reference_map(labels: pd.DataFrame, target_ids: set[str]) -> dict[str, list[np.ndarray]]:
    """Extract native conformations only for requested targets."""
    return {target_id: io.get_reference_coords(labels, target_id) for target_id in target_ids}
