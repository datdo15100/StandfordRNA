"""Exact cluster-aware inference frozen in the V4 preregistration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_SEED = 20260819


@dataclass(frozen=True)
class PrimaryInference:
    hypothesis: str
    target_n: int
    cluster_n: int
    mean_delta: float
    median_delta: float
    ci_lower: float
    ci_upper: float
    raw_one_sided_p: float
    improved: int
    tied: int
    regressed: int
    bootstrap_replicates: int
    permutation_replicates: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validated(
    deltas: Sequence[float], cluster_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(deltas, dtype=float)
    clusters = np.asarray(cluster_ids, dtype=str)
    if values.ndim != 1 or clusters.ndim != 1 or len(values) != len(clusters):
        raise ValueError("deltas and cluster_ids must be aligned one-dimensional arrays")
    if not len(values):
        raise ValueError("at least one paired target delta is required")
    if not np.isfinite(values).all():
        raise ValueError("primary paired target deltas must be finite after frozen fallback")
    if np.any(clusters == ""):
        raise ValueError("every target must have a regenerated MMseqs cluster ID")
    unique = sorted(set(clusters.tolist()))
    return values, clusters, unique


def cluster_bootstrap_means(
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Resample clusters and carry every member target, preserving target weighting."""
    values, clusters, unique = _validated(deltas, cluster_ids)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    members = [values[clusters == cluster] for cluster in unique]
    cluster_sums = np.asarray([group.sum() for group in members], dtype=float)
    cluster_sizes = np.asarray([len(group) for group in members], dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty(replicates, dtype=float)
    cluster_n = len(unique)
    for index in range(replicates):
        sampled = rng.integers(0, cluster_n, size=cluster_n)
        result[index] = cluster_sums[sampled].sum() / cluster_sizes[sampled].sum()
    return result


def cluster_sign_flip_pvalue(
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    permutations: int = 100_000,
    seed: int = DEFAULT_SEED,
    batch_size: int = 10_000,
) -> float:
    """One-sided p-value using one shared random sign per dependence cluster."""
    values, clusters, unique = _validated(deltas, cluster_ids)
    if permutations <= 0 or batch_size <= 0:
        raise ValueError("permutations and batch_size must be positive")
    cluster_sums = np.asarray(
        [values[clusters == cluster].sum() for cluster in unique], dtype=float
    )
    observed = float(values.mean())
    denominator = float(len(values))
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = 0
    generated = 0
    while generated < permutations:
        current = min(batch_size, permutations - generated)
        signs = rng.integers(0, 2, size=(current, len(unique)), dtype=np.int8)
        signs = signs.astype(float) * 2.0 - 1.0
        statistics = signs @ cluster_sums / denominator
        exceed += int(np.count_nonzero(statistics >= observed))
        generated += current
    return (1.0 + exceed) / (permutations + 1.0)


def primary_inference(
    hypothesis: str,
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    bootstrap_replicates: int = 10_000,
    permutation_replicates: int = 100_000,
    seed: int = DEFAULT_SEED,
    tie_threshold: float = 1e-6,
) -> PrimaryInference:
    values, _, unique = _validated(deltas, cluster_ids)
    bootstrap = cluster_bootstrap_means(
        values,
        cluster_ids,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    raw_p = cluster_sign_flip_pvalue(
        values,
        cluster_ids,
        permutations=permutation_replicates,
        seed=seed,
    )
    tied = np.abs(values) < tie_threshold
    return PrimaryInference(
        hypothesis=hypothesis,
        target_n=len(values),
        cluster_n=len(unique),
        mean_delta=float(values.mean()),
        median_delta=float(np.median(values)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        raw_one_sided_p=float(raw_p),
        improved=int(np.count_nonzero(values > tie_threshold)),
        tied=int(np.count_nonzero(tied)),
        regressed=int(np.count_nonzero(values < -tie_threshold)),
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        seed=seed,
    )


def holm_step_down(
    raw_pvalues: Mapping[str, float], alpha: float = 0.05
) -> dict[str, dict[str, float | bool | int]]:
    """Holm adjustment and step-down rejection in original hypothesis names."""
    if not raw_pvalues:
        raise ValueError("at least one p-value is required")
    for name, value in raw_pvalues.items():
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
    ordered = sorted(raw_pvalues.items(), key=lambda item: (item[1], item[0]))
    family_n = len(ordered)
    adjusted_ordered: list[float] = []
    running = 0.0
    still_rejecting = True
    results: dict[str, dict[str, float | bool | int]] = {}
    for rank, (name, pvalue) in enumerate(ordered, start=1):
        multiplier = family_n - rank + 1
        running = max(running, multiplier * pvalue)
        adjusted_ordered.append(min(1.0, running))
        threshold = alpha / multiplier
        reject = bool(still_rejecting and pvalue <= threshold)
        if not reject:
            still_rejecting = False
        results[name] = {
            "rank": rank,
            "raw_p": float(pvalue),
            "holm_adjusted_p": adjusted_ordered[-1],
            "step_down_threshold": float(threshold),
            "reject": reject,
        }
    return results
