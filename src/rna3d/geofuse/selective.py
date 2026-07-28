"""Conservative local fusion primitives with explicit abstention."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from .candidate import StructureCandidate, safe_name
from .phase_c import robust_superpose


def contiguous_run_mask(mask: np.ndarray, minimum: int = 7) -> np.ndarray:
    """Keep only true runs with at least ``minimum`` consecutive residues."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or minimum <= 0:
        raise ValueError("mask must be one-dimensional and minimum must be positive")
    output = np.zeros_like(values)
    padded = np.pad(values.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start >= minimum:
            output[start:stop] = True
    return output


def local_source_disagreement(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    *,
    window: int = 15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return aligned pretrained coordinates and window-smoothed disagreement."""
    if template.target_id != pretrained.target_id or template.sequence != pretrained.sequence:
        raise ValueError("fusion parents must describe the same target")
    alignment_mask = (
        template.valid_mask
        & pretrained.valid_mask
        & template.support_mask
        & (template.confidence >= 0.25)
    )
    aligned, rmsd, inliers = robust_superpose(
        pretrained.coords, template.coords, alignment_mask
    )
    point = np.linalg.norm(aligned - template.coords, axis=1)
    effective = min(max(int(window), 1), len(point))
    smooth = uniform_filter1d(point, size=effective, mode="nearest")
    return aligned, smooth, {
        "alignment_rmsd": rmsd,
        "alignment_inlier_fraction": float(inliers.mean()),
        "mean_point_disagreement": float(point.mean()),
        "mean_window_disagreement": float(smooth.mean()),
    }


def selective_quality_fusion(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    pretrained_probability: np.ndarray,
    *,
    decision_threshold: float,
    probability_margin: float = 0.15,
    minimum_segment: int = 7,
    disagreement_window: int = 15,
    minimum_disagreement: float = 0.5,
    maximum_disagreement: float = 12.0,
    smoothing_sigma: float = 1.0,
) -> StructureCandidate | None:
    """Patch only confident, contiguous pretrained-better regions.

    Ambiguous residues and implausibly large source disagreements abstain to the
    template anchor. Returning ``None`` means no segment passed the frozen gate.
    """
    probability = np.asarray(pretrained_probability, dtype=float)
    if probability.shape != (len(template.sequence),):
        raise ValueError("pretrained_probability must have one value per residue")
    if (
        not np.isfinite(probability).all()
        or not 0.0 < decision_threshold < 1.0
        or probability_margin < 0
        or decision_threshold + probability_margin > 1.0
    ):
        raise ValueError("invalid probability or threshold/margin")
    aligned, disagreement, alignment = local_source_disagreement(
        template, pretrained, window=disagreement_window
    )
    confident_pretrained = probability >= decision_threshold + probability_margin
    safe_disagreement = (
        (disagreement >= minimum_disagreement)
        & (disagreement <= maximum_disagreement)
    )
    switch = contiguous_run_mask(
        confident_pretrained & safe_disagreement, minimum=minimum_segment
    )
    if not switch.any():
        return None
    alpha = switch.astype(float)
    if smoothing_sigma > 0:
        alpha = gaussian_filter1d(alpha, sigma=smoothing_sigma, mode="nearest")
    alpha = np.clip(alpha, 0.0, 1.0)
    coordinates = (1.0 - alpha[:, None]) * template.coords + alpha[:, None] * aligned
    confidence = (
        (1.0 - alpha) * template.confidence + alpha * pretrained.confidence
    )
    return StructureCandidate(
        target_id=template.target_id,
        sequence=template.sequence,
        candidate_id=(
            f"fused__selective_quality__{safe_name(template.candidate_id)}__"
            f"{safe_name(pretrained.candidate_id)}"
        ),
        kind="fused",
        source="geofuse_selective",
        model="quality_gate_abstain_v1",
        coords=coordinates,
        confidence=np.clip(confidence, 0.01, 1.0),
        support_mask=template.support_mask | pretrained.support_mask,
        global_confidence=float(
            (template.global_confidence + pretrained.global_confidence) / 2.0
        ),
        metadata={
            "template_parent": template.candidate_id,
            "pretrained_parent": pretrained.candidate_id,
            "decision_threshold": float(decision_threshold),
            "probability_margin": float(probability_margin),
            "minimum_segment": int(minimum_segment),
            "switched_fraction": float(switch.mean()),
            "boundary_count": int(np.count_nonzero(np.diff(switch))),
            **alignment,
        },
    )

def oracle_source_fusion(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    pretrained_better: np.ndarray,
) -> StructureCandidate:
    """Non-deployable per-residue source oracle used only as an upper bound."""
    choice = np.asarray(pretrained_better, dtype=bool)
    if choice.shape != (len(template.sequence),):
        raise ValueError("oracle choice must have one value per residue")
    aligned, _, alignment = local_source_disagreement(template, pretrained)
    coordinates = np.where(choice[:, None], aligned, template.coords)
    confidence = np.where(choice, pretrained.confidence, template.confidence)
    return StructureCandidate(
        target_id=template.target_id,
        sequence=template.sequence,
        candidate_id=(
            f"fused__native_oracle__{safe_name(template.candidate_id)}__"
            f"{safe_name(pretrained.candidate_id)}"
        ),
        kind="fused",
        source="geofuse_native_oracle",
        model="non_deployable_upper_bound",
        coords=coordinates,
        confidence=confidence,
        support_mask=template.support_mask | pretrained.support_mask,
        global_confidence=float(confidence.mean()),
        metadata={
            "non_deployable": True,
            "pretrained_fraction": float(choice.mean()),
            **alignment,
        },
    )
