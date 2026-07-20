"""Context-aware C1' geometry statistics for GeoFuse Phase B.

The context is intentionally available at inference time.  A residue is marked
``pair_like`` when the *candidate* contains a plausible non-local canonical or
wobble pair at a C1'-C1' separation near the RNA base-pairing range.  This is a
coarse structural proxy, not an annotation of true secondary structure; using
that name keeps the scientific claim honest while still testing the plan's
paired/unpaired conditioning idea without consulting native labels.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..data import io
from ..eval.metrics import geom_metrics


PAIR_TYPES = {"AU", "UA", "GC", "CG", "GU", "UG"}


def pseudo_angles(coords: np.ndarray) -> np.ndarray:
    """Return C1'(i-1)-C1'(i)-C1'(i+1) angles in radians."""
    xyz = np.asarray(coords, dtype=float)
    if len(xyz) < 3:
        return np.empty(0, dtype=float)
    left = xyz[:-2] - xyz[1:-1]
    right = xyz[2:] - xyz[1:-1]
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.divide(
        np.einsum("ij,ij->i", left, right),
        denom,
        out=np.full(len(left), np.nan),
        where=denom > 1e-8,
    )
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def signed_pseudo_torsions(coords: np.ndarray) -> np.ndarray:
    """Return signed four-C1' pseudo-dihedrals in ``[-pi, pi]``."""
    xyz = np.asarray(coords, dtype=float)
    if len(xyz) < 4:
        return np.empty(0, dtype=float)
    b0 = xyz[1:-2] - xyz[:-3]
    b1 = xyz[2:-1] - xyz[1:-2]
    b2 = xyz[3:] - xyz[2:-1]
    norm = np.linalg.norm(b1, axis=1, keepdims=True)
    unit = np.divide(b1, norm, out=np.zeros_like(b1), where=norm > 1e-8)
    v = b0 - np.einsum("ij,ij->i", b0, unit)[:, None] * unit
    w = b2 - np.einsum("ij,ij->i", b2, unit)[:, None] * unit
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(unit, v), w)
    values = np.arctan2(y, x)
    degenerate = (np.linalg.norm(v, axis=1) < 1e-8) | (np.linalg.norm(w, axis=1) < 1e-8)
    values[degenerate] = np.nan
    return values


def pair_like_mask(
    sequence: str,
    coords: np.ndarray,
    *,
    min_separation: int = 4,
    target_distance: float = 10.5,
    tolerance: float = 2.5,
) -> np.ndarray:
    """Greedily mark residues in plausible one-to-one base-pair-like contacts.

    Only sequence complementarity and candidate coordinates are used.  Crossing
    pairs are allowed because RNA pseudoknots exist; each residue can be assigned
    to at most one partner.
    """
    seq = str(sequence).upper().replace("T", "U")
    xyz = np.asarray(coords, dtype=float)
    if xyz.shape != (len(seq), 3):
        raise ValueError(f"coordinate shape {xyz.shape} does not match sequence length {len(seq)}")
    finite = np.isfinite(xyz).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    candidates: list[tuple[float, int, int]] = []
    if len(finite_indices) >= 2:
        points = xyz[finite_indices]
        pairs = cKDTree(points).query_pairs(target_distance + tolerance, output_type="ndarray")
        for local_i, local_j in pairs:
            i, j = int(finite_indices[local_i]), int(finite_indices[local_j])
            if abs(i - j) < min_separation or seq[i] + seq[j] not in PAIR_TYPES:
                continue
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            deviation = abs(distance - target_distance)
            if deviation <= tolerance:
                candidates.append((deviation, i, j))
    paired = np.zeros(len(seq), dtype=bool)
    for _, i, j in sorted(candidates):
        if not paired[i] and not paired[j]:
            paired[i] = paired[j] = True
    return paired


def _smooth_counts(counts: np.ndarray, periodic: bool) -> np.ndarray:
    kernel = np.asarray([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    if periodic:
        return sum(weight * np.roll(counts, offset - 3) for offset, weight in enumerate(kernel))
    padded = np.pad(counts, (3, 3), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _histogram_prior(
    values: Iterable[float], *, lo: float, hi: float, bins: int, periodic: bool
) -> dict:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("cannot estimate a geometry distribution from no finite values")
    counts, edges = np.histogram(data, bins=bins, range=(lo, hi))
    smooth = _smooth_counts(counts.astype(float) + 1.0, periodic=periodic)
    probability = smooth / smooth.sum()
    nll = -np.log(np.maximum(probability, 1e-12))
    nll -= nll.min()
    return {
        "lo": float(lo),
        "hi": float(hi),
        "bins": int(bins),
        "periodic": bool(periodic),
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p05": float(np.percentile(data, 5)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "probability": probability.tolist(),
        "nll": nll.tolist(),
    }


def _valid_local_geometry(
    resids: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(coords).all(axis=1)
    consecutive = np.diff(resids) == 1
    angle_valid = finite[:-2] & finite[1:-1] & finite[2:] & consecutive[:-1] & consecutive[1:]
    torsion_valid = (
        finite[:-3]
        & finite[1:-2]
        & finite[2:-1]
        & finite[3:]
        & consecutive[:-2]
        & consecutive[1:-1]
        & consecutive[2:]
    )
    return (
        pseudo_angles(coords)[angle_valid],
        signed_pseudo_torsions(coords)[torsion_valid],
        angle_valid,
        torsion_valid,
    )


def estimate_geometry_v2_priors(labels: pd.DataFrame, *, bins: int = 72) -> dict:
    """Estimate global and pair-like/unpaired angle/torsion distributions."""
    required = {"ID", "resid", "resname", "x_1", "y_1", "z_1"}
    missing = required - set(labels.columns)
    if missing:
        raise KeyError(f"labels are missing columns: {sorted(missing)}")

    values: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"angle": [], "torsion": []}
    )
    pair_fractions = []
    tid = labels["ID"].map(io.target_id_of)
    n_chains = 0
    for _, sub in labels.assign(_tid=tid).groupby("_tid", sort=False):
        sub = sub.sort_values("resid")
        coords = sub[["x_1", "y_1", "z_1"]].to_numpy(float).copy()
        coords[coords[:, 0] <= io.RESOLVED_THRESHOLD] = np.nan
        sequence = "".join(sub["resname"].astype(str)).upper().replace("T", "U")
        resids = sub["resid"].to_numpy(int)
        angles, torsions, angle_valid, torsion_valid = _valid_local_geometry(resids, coords)
        if not len(angles):
            continue
        pair_mask = pair_like_mask(sequence, coords)
        angle_context = pair_mask[1:-1][angle_valid]
        torsion_context = (pair_mask[1:-2] | pair_mask[2:-1])[torsion_valid]
        finite_pairs = pair_mask[np.isfinite(coords).all(axis=1)]
        if len(finite_pairs):
            pair_fractions.append(float(finite_pairs.mean()))

        values["global"]["angle"].append(angles)
        values["global"]["torsion"].append(torsions)
        for name, mask in (("pair_like", angle_context), ("unpaired", ~angle_context)):
            if mask.any():
                values[name]["angle"].append(angles[mask])
        for name, mask in (("pair_like", torsion_context), ("unpaired", ~torsion_context)):
            if mask.any():
                values[name]["torsion"].append(torsions[mask])
        n_chains += 1

    contexts = {}
    for context in ("global", "pair_like", "unpaired"):
        angles = np.concatenate(values[context]["angle"])
        torsions = np.concatenate(values[context]["torsion"])
        contexts[context] = {
            "angle": _histogram_prior(angles, lo=0.0, hi=np.pi, bins=bins, periodic=False),
            "torsion": _histogram_prior(
                torsions, lo=-np.pi, hi=np.pi, bins=bins, periodic=True
            ),
        }
    return {
        "schema_version": 1,
        "context_method": {
            "name": "candidate_pair_like_c1",
            "pair_types": sorted(PAIR_TYPES),
            "min_sequence_separation": 4,
            "target_c1_distance": 10.5,
            "distance_tolerance": 2.5,
            "note": "inference-time structural proxy, not native secondary-structure annotation",
        },
        "n_chains": n_chains,
        "mean_pair_like_fraction": float(np.mean(pair_fractions)),
        "contexts": contexts,
    }


def histogram_nll(values: np.ndarray, prior: dict) -> np.ndarray:
    """Piecewise-linear lookup of empirical negative log-density."""
    x = np.asarray(values, dtype=float)
    result = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return result
    valid = x[finite]
    nll = np.asarray(prior["nll"], dtype=float)
    bins = int(prior["bins"])
    lo, hi = float(prior["lo"]), float(prior["hi"])
    position = (valid - lo) / (hi - lo) * bins - 0.5
    if prior.get("periodic", False):
        position = np.mod(position, bins)
        left = np.floor(position).astype(int)
        right = (left + 1) % bins
    else:
        position = np.clip(position, 0.0, bins - 1.0)
        left = np.floor(position).astype(int)
        right = np.minimum(left + 1, bins - 1)
    fraction = position - np.floor(position)
    result[finite] = (1.0 - fraction) * nll[left] + fraction * nll[right]
    return result


def geometry_v2_metrics(
    coords: np.ndarray, sequence: str, priors_v1: dict, priors_v2: dict
) -> dict:
    """Native-blind physical diagnostics used by Phase-B routing/refinement."""
    xyz = np.asarray(coords, dtype=float)
    base = geom_metrics(xyz, priors_v1)
    angles = pseudo_angles(xyz)
    torsions = signed_pseudo_torsions(xyz)
    pair_mask = pair_like_mask(sequence, xyz)
    angle_context = pair_mask[1:-1]
    torsion_context = pair_mask[1:-2] | pair_mask[2:-1]

    angle_scores = np.full(len(angles), np.nan)
    torsion_scores = np.full(len(torsions), np.nan)
    for name, mask in (("pair_like", angle_context), ("unpaired", ~angle_context)):
        if mask.any():
            angle_scores[mask] = histogram_nll(
                angles[mask], priors_v2["contexts"][name]["angle"]
            )
    for name, mask in (("pair_like", torsion_context), ("unpaired", ~torsion_context)):
        if mask.any():
            torsion_scores[mask] = histogram_nll(
                torsions[mask], priors_v2["contexts"][name]["torsion"]
            )
    return {
        **base,
        "angle_nll": float(np.nanmean(angle_scores)) if len(angle_scores) else 0.0,
        "torsion_nll": float(np.nanmean(torsion_scores)) if len(torsion_scores) else 0.0,
        "pair_like_fraction": float(pair_mask.mean()) if len(pair_mask) else 0.0,
    }
