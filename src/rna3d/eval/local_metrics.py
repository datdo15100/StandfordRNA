"""Independent C1'-trace accuracy metrics for RNA structures.

These metrics compare predictions with native coordinates.  They are deliberately
separate from the geometry diagnostics optimized by the refiners.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..geometry.transforms import apply_rigid, kabsch


DEFAULT_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def _coordinate_pair(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        prediction.ndim != 2
        or prediction.shape[1:] != (3,)
        or prediction.shape != reference.shape
    ):
        raise ValueError("prediction and reference must have the same (L, 3) shape")
    return prediction, reference


def shared_finite_mask(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return residues with finite coordinates in both structures."""
    prediction, reference = _coordinate_pair(prediction, reference)
    return np.isfinite(prediction).all(axis=1) & np.isfinite(reference).all(axis=1)


def c1_rmsd(
    prediction: np.ndarray, reference: np.ndarray, *, min_residues: int = 3
) -> float:
    """Kabsch-aligned C1' RMSD over shared finite residues."""
    prediction, reference = _coordinate_pair(prediction, reference)
    mask = shared_finite_mask(prediction, reference)
    if mask.sum() < min_residues:
        return float("nan")
    rotation, translation = kabsch(prediction[mask], reference[mask])
    aligned = apply_rigid(prediction[mask], rotation, translation)
    return float(np.sqrt(np.mean(np.sum(np.square(aligned - reference[mask]), axis=1))))


def c1_lddt(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    inclusion_radius: float = 15.0,
    thresholds: Iterable[float] = DEFAULT_LDDT_THRESHOLDS,
) -> dict:
    """C1'-trace lDDT and per-residue scores without superposition.

    Native C1' pairs within ``inclusion_radius`` are evaluated at the supplied
    distance-error thresholds.  Missing predicted residues count as non-preserved
    pairs; unresolved native residues are excluded.  The global score is the mean
    of finite per-residue scores, matching the atom-local nature of lDDT.
    """
    prediction, reference = _coordinate_pair(prediction, reference)
    thresholds = np.asarray(tuple(thresholds), dtype=float)
    if (
        inclusion_radius <= 0
        or thresholds.ndim != 1
        or len(thresholds) == 0
        or not np.isfinite(thresholds).all()
        or (thresholds <= 0).any()
    ):
        raise ValueError("radius and all lDDT thresholds must be positive and finite")

    native_valid = np.isfinite(reference).all(axis=1)
    predicted_valid = np.isfinite(prediction).all(axis=1)
    native_indices = np.flatnonzero(native_valid)
    length = len(reference)
    per_sum = np.zeros(length, dtype=float)
    per_count = np.zeros(length, dtype=np.int64)
    pair_scores: list[float] = []

    if len(native_indices) >= 2:
        native = reference[native_indices]
        native_distances = np.linalg.norm(native[:, None] - native[None, :], axis=2)
        upper_i, upper_j = np.triu_indices(len(native_indices), k=1)
        included = native_distances[upper_i, upper_j] <= inclusion_radius
        for local_i, local_j in zip(upper_i[included], upper_j[included]):
            i = int(native_indices[local_i])
            j = int(native_indices[local_j])
            if predicted_valid[i] and predicted_valid[j]:
                predicted_distance = float(np.linalg.norm(prediction[i] - prediction[j]))
                error = abs(predicted_distance - float(native_distances[local_i, local_j]))
                score = float(np.mean(error < thresholds))
            else:
                score = 0.0
            pair_scores.append(score)
            per_sum[i] += score
            per_sum[j] += score
            per_count[i] += 1
            per_count[j] += 1

    per_residue = np.full(length, np.nan, dtype=float)
    has_neighbours = per_count > 0
    per_residue[has_neighbours] = per_sum[has_neighbours] / per_count[has_neighbours]
    return {
        "score": float(np.nanmean(per_residue)) if has_neighbours.any() else float("nan"),
        "pair_score": float(np.mean(pair_scores)) if pair_scores else float("nan"),
        "per_residue": per_residue,
        "n_pairs": len(pair_scores),
        "n_scored_residues": int(has_neighbours.sum()),
        "inclusion_radius": float(inclusion_radius),
        "thresholds": thresholds.copy(),
    }


def sliding_window_c1_rmsd(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    window: int = 15,
    min_residues: int = 3,
) -> dict:
    """Local C1' RMSD after an independent Kabsch fit in every full window.

    Windows use consecutive sequence positions.  Only full windows are used; when
    ``L < window``, the complete sequence is evaluated once.  ``per_residue`` is the
    mean RMSD of all valid windows containing that residue.
    """
    prediction, reference = _coordinate_pair(prediction, reference)
    if window <= 0 or min_residues < 3:
        raise ValueError("window must be positive and min_residues must be at least 3")
    length = len(reference)
    if length == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "per_residue": np.empty(0, dtype=float),
            "window_rmsd": np.empty(0, dtype=float),
            "window_starts": np.empty(0, dtype=int),
            "effective_window": 0,
            "n_windows": 0,
        }

    effective = min(window, length)
    starts = np.arange(length - effective + 1, dtype=int)
    values = np.full(len(starts), np.nan, dtype=float)
    per_sum = np.zeros(length, dtype=float)
    per_count = np.zeros(length, dtype=np.int64)
    for index, start in enumerate(starts):
        stop = int(start + effective)
        value = c1_rmsd(
            prediction[start:stop],
            reference[start:stop],
            min_residues=min_residues,
        )
        values[index] = value
        if np.isfinite(value):
            per_sum[start:stop] += value
            per_count[start:stop] += 1

    per_residue = np.full(length, np.nan, dtype=float)
    valid = per_count > 0
    per_residue[valid] = per_sum[valid] / per_count[valid]
    finite_values = values[np.isfinite(values)]
    return {
        "mean": float(finite_values.mean()) if len(finite_values) else float("nan"),
        "median": float(np.median(finite_values)) if len(finite_values) else float("nan"),
        "per_residue": per_residue,
        "window_rmsd": values,
        "window_starts": starts,
        "effective_window": effective,
        "n_windows": int(len(finite_values)),
    }


def local_accuracy_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    windows: Iterable[int] = (9, 15, 31),
) -> dict[str, float]:
    """Return scalar independent C1' accuracy metrics for one native reference."""
    lddt = c1_lddt(prediction, reference)
    result = {
        "c1_rmsd": c1_rmsd(prediction, reference),
        "c1_lddt": float(lddt["score"]),
        "c1_lddt_pair": float(lddt["pair_score"]),
        "c1_lddt_pairs": int(lddt["n_pairs"]),
    }
    for window in windows:
        local = sliding_window_c1_rmsd(prediction, reference, window=int(window))
        result[f"sw_rmsd_{int(window)}"] = float(local["mean"])
    return result
