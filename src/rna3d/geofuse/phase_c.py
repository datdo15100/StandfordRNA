"""Fold clustering, heuristic segment fusion, and diverse selection for GeoFuse."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from ..geometry.transforms import apply_rigid, kabsch
from .candidate import StructureCandidate, safe_name


@dataclass
class FusionConfig:
    """Native-blind controls for conservative template/pretrained fusion."""

    reliable_template_confidence: float = 0.5
    reliable_template_partner_cap: float = 0.15
    unsupported_partner_weight: float = 0.9
    pretrained_heavy_floor: float = 0.7
    max_supported_disagreement: float = 12.0
    smoothing_radius: int = 2
    alignment_trim_fraction: float = 0.8
    alignment_iterations: int = 3


@dataclass
class SelectionConfig:
    """Weights for deterministic quality-diversity selection."""

    diversity_weight: float = 0.25
    new_cluster_bonus: float = 0.20
    cluster_support_weight: float = 0.10


QUALITY_WEIGHTS = {
    "source_confidence": 0.25,
    "support_fraction": 0.20,
    "pair_like_fraction": 0.10,
    "angle_nll": 0.15,
    "torsion_nll": 0.10,
    "clash_per_res": 0.07,
    "bb_dev": 0.08,
    "sharp_kinks": 0.05,
}


def cluster_fold_families(similarity: np.ndarray, threshold: float = 0.45) -> np.ndarray:
    """Complete-link clustering of a symmetric same-length self-TM matrix."""
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if not len(matrix):
        return np.empty(0, dtype=int)
    if len(matrix) == 1:
        return np.zeros(1, dtype=int)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError("similarity must be finite and symmetric")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")

    distance = np.clip(1.0 - matrix, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    raw = fcluster(tree, t=1.0 - threshold, criterion="distance")
    # Stable zero-based labels ordered by first candidate, not scipy tree order.
    remap: dict[int, int] = {}
    return np.asarray([remap.setdefault(int(value), len(remap)) for value in raw], dtype=int)


def robust_superpose(
    moving: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    trim_fraction: float = 0.8,
    iterations: int = 3,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Align coordinates with iterative residual trimming."""
    source = np.asarray(moving, dtype=float)
    target = np.asarray(reference, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("moving and reference must have the same (L, 3) shape")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    base = finite if mask is None else finite & np.asarray(mask, dtype=bool)
    if base.sum() < 3:
        base = finite
    if base.sum() < 3:
        raise ValueError("at least three shared finite residues are required for alignment")

    active = base.copy()
    aligned = source.copy()
    for _ in range(max(iterations, 1)):
        rotation, translation = kabsch(source[active], target[active])
        aligned = apply_rigid(source, rotation, translation)
        residual = np.linalg.norm(aligned - target, axis=1)
        if active.sum() <= 3 or not 0.0 < trim_fraction < 1.0:
            break
        cutoff = float(np.quantile(residual[active], trim_fraction))
        updated = base & (residual <= cutoff)
        if updated.sum() < 3 or np.array_equal(updated, active):
            break
        active = updated

    rmsd = float(
        np.sqrt(np.mean(np.square((aligned - target)[active]).sum(axis=1)))
    )
    return aligned.astype(np.float32), rmsd, active


def _relative_confidence(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) <= 1 or np.ptp(data) <= 1e-8:
        return np.full(len(data), 0.625, dtype=float)
    percentile = (rankdata(data, method="average") - 1.0) / (len(data) - 1.0)
    return 0.25 + 0.75 * percentile


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) <= 1:
        return np.asarray(values, dtype=float).copy()
    offsets = np.arange(-radius, radius + 1)
    scale = max(radius / 1.5, 1.0)
    kernel = np.exp(-0.5 * np.square(offsets / scale))
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fuse_template_pretrained(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    cfg: FusionConfig | None = None,
    *,
    mode: str = "template_conservative",
) -> StructureCandidate:
    """Conservatively patch a template with an aligned pretrained candidate."""
    cfg = cfg or FusionConfig()
    if template.target_id != pretrained.target_id or template.sequence != pretrained.sequence:
        raise ValueError("fusion parents must describe the same target and sequence")
    if template.kind != "template" or pretrained.kind != "pretrained":
        raise ValueError("fusion requires one template and one pretrained parent")
    if mode not in {"template_conservative", "pretrained_heavy"}:
        raise ValueError(f"unknown fusion mode: {mode}")

    alignment_mask = (
        template.valid_mask
        & pretrained.valid_mask
        & template.support_mask
        & (template.confidence >= 0.25)
    )
    aligned, alignment_rmsd, inliers = robust_superpose(
        pretrained.coords,
        template.coords,
        alignment_mask,
        trim_fraction=cfg.alignment_trim_fraction,
        iterations=cfg.alignment_iterations,
    )
    disagreement = np.linalg.norm(aligned - template.coords, axis=1)
    template_quality = np.clip(template.confidence.astype(float), 0.01, 1.0)
    pretrained_quality = _relative_confidence(pretrained.confidence)
    alpha = pretrained_quality / (template_quality + pretrained_quality)

    reliable = template.support_mask & (
        template.confidence >= cfg.reliable_template_confidence
    )
    strong_disagreement = disagreement > cfg.max_supported_disagreement
    if mode == "template_conservative":
        alpha[reliable] = np.minimum(
            alpha[reliable], cfg.reliable_template_partner_cap
        )
        alpha[~template.support_mask] = cfg.unsupported_partner_weight
        alpha[strong_disagreement & template.support_mask] = 0.0
        alpha[strong_disagreement & ~template.support_mask] = 1.0
    else:
        # A distinct source hypothesis, not a small lambda perturbation: keep the
        # aligned pretrained fold and blend template evidence only where sources
        # agree.  At large disagreement choose one source instead of averaging.
        alpha = np.maximum(alpha, cfg.pretrained_heavy_floor)
        alpha[~template.support_mask] = 1.0
        alpha[strong_disagreement] = 1.0

    alpha = _smooth(alpha, cfg.smoothing_radius)
    # Reapply hard safety caps after boundary smoothing.
    if mode == "template_conservative":
        alpha[reliable] = np.minimum(
            alpha[reliable], cfg.reliable_template_partner_cap
        )
        alpha[strong_disagreement & template.support_mask] = 0.0
        alpha[strong_disagreement & ~template.support_mask] = 1.0
    else:
        alpha = np.maximum(alpha, cfg.pretrained_heavy_floor)
        alpha[~template.support_mask] = 1.0
        alpha[strong_disagreement] = 1.0
    alpha = np.clip(alpha, 0.0, 1.0)

    coordinates = (1.0 - alpha[:, None]) * template.coords + alpha[:, None] * aligned
    confidence = (
        (1.0 - alpha) * template.confidence + alpha * pretrained.confidence
    )
    candidate_id = (
        f"fused__{safe_name(mode)}__{safe_name(template.candidate_id)}__"
        f"{safe_name(pretrained.candidate_id)}"
    )
    return StructureCandidate(
        target_id=template.target_id,
        sequence=template.sequence,
        candidate_id=candidate_id,
        kind="fused",
        source="geofuse",
        model="heuristic_segment_v1",
        coords=coordinates,
        confidence=np.clip(confidence, 0.01, 1.0),
        support_mask=template.support_mask | pretrained.support_mask,
        global_confidence=(template.global_confidence + pretrained.global_confidence) / 2.0,
        metadata={
            "template_parent": template.candidate_id,
            "pretrained_parent": pretrained.candidate_id,
            "fusion_mode": mode,
            "alignment_rmsd": alignment_rmsd,
            "alignment_inlier_fraction": float(inliers.mean()),
            "mean_pretrained_weight": float(alpha.mean()),
            "pretrained_dominant_fraction": float((alpha > 0.5).mean()),
            "boundary_count": int(np.count_nonzero(np.diff(alpha > 0.5))),
        },
    )


def _percentile(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) <= 1 or np.ptp(data) <= 1e-12:
        result = np.full(len(data), 0.5, dtype=float)
    else:
        result = (rankdata(data, method="average") - 1.0) / (len(data) - 1.0)
    return result if higher_is_better else 1.0 - result


def native_blind_quality_scores(
    candidates: list[StructureCandidate], features: list[dict]
) -> np.ndarray:
    """Rank-aggregate heterogeneous confidence and geometry without native labels."""
    if len(candidates) != len(features):
        raise ValueError("candidates and features must have the same length")
    count = len(candidates)
    if not count:
        return np.empty(0, dtype=float)

    source_confidence = np.zeros(count, dtype=float)
    for source in sorted({candidate.source for candidate in candidates}):
        indices = [i for i, candidate in enumerate(candidates) if candidate.source == source]
        source_confidence[indices] = _percentile(
            np.asarray([candidates[i].global_confidence for i in indices]),
            higher_is_better=True,
        )

    values = {"source_confidence": source_confidence}
    positive = {"support_fraction", "pair_like_fraction"}
    for name in QUALITY_WEIGHTS:
        if name == "source_confidence":
            continue
        values[name] = _percentile(
            np.asarray([float(row[name]) for row in features]),
            higher_is_better=name in positive,
        )
    return sum(QUALITY_WEIGHTS[name] * values[name] for name in QUALITY_WEIGHTS)


def select_quality_diversity(
    candidates: list[StructureCandidate],
    similarity: np.ndarray,
    cluster_labels: np.ndarray,
    quality: np.ndarray,
    *,
    limit: int = 5,
    cfg: SelectionConfig | None = None,
) -> list[int]:
    """Greedy MMR selection with fold-family support and coverage bonuses."""
    cfg = cfg or SelectionConfig()
    matrix = np.asarray(similarity, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    score = np.asarray(quality, dtype=float)
    n = len(candidates)
    if matrix.shape != (n, n) or labels.shape != (n,) or score.shape != (n,):
        raise ValueError("candidate, similarity, cluster, and quality shapes disagree")
    if limit <= 0 or not n:
        return []

    sizes = {label: int((labels == label).sum()) for label in np.unique(labels)}
    max_size = max(sizes.values())
    selected: list[int] = []
    used_clusters: set[int] = set()
    remaining = set(range(n))
    while remaining and len(selected) < limit:
        ranked = []
        for index in remaining:
            cluster = int(labels[index])
            support = sizes[cluster] / max_size
            redundancy = max((matrix[index, other] for other in selected), default=0.0)
            value = score[index] + cfg.cluster_support_weight * support
            if cluster not in used_clusters:
                value += cfg.new_cluster_bonus
            value -= cfg.diversity_weight * redundancy
            ranked.append((value, score[index], candidates[index].candidate_id, index))
        _, _, _, chosen = sorted(
            ranked, key=lambda item: (-item[0], -item[1], item[2], item[3])
        )[0]
        selected.append(chosen)
        remaining.remove(chosen)
        used_clusters.add(int(labels[chosen]))
    return selected


def mean_selected_similarity(similarity: np.ndarray, indices: list[int]) -> float:
    """Mean off-diagonal similarity for a selected set."""
    if len(indices) < 2:
        return 1.0
    submatrix = np.asarray(similarity)[np.ix_(indices, indices)]
    upper = np.triu_indices(len(indices), k=1)
    return float(submatrix[upper].mean())
