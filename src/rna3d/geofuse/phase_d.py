"""Tiny residue-wise confidence gate and inference-available pair features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.stats import rankdata
import torch
from torch import nn

from .candidate import StructureCandidate
from .geometry_v2 import (
    histogram_nll,
    pair_like_mask,
    pseudo_angles,
    signed_pseudo_torsions,
)
from .phase_c import robust_superpose


LOCAL_GEOMETRY_NAMES = [
    "bb_dev",
    "angle_nll",
    "torsion_nll",
    "sharp_kink",
    "pair_like",
]
FEATURE_NAMES = [
    "template_confidence",
    "pretrained_confidence",
    "template_confidence_rank",
    "pretrained_confidence_rank",
    "template_support",
    "template_support_distance",
    "source_disagreement",
    *[f"template_{name}" for name in LOCAL_GEOMETRY_NAMES],
    *[f"pretrained_{name}" for name in LOCAL_GEOMETRY_NAMES],
    "base_A",
    "base_C",
    "base_G",
    "base_U",
    "log_length",
]


@dataclass
class GateConfig:
    hidden_channels: int = 32
    kernel_size: int = 5
    dropout: float = 0.05


class ConfidenceGate1D(nn.Module):
    """Small contextual network that predicts probability of trusting pretrained."""

    def __init__(self, n_features: int = len(FEATURE_NAMES), cfg: GateConfig | None = None):
        super().__init__()
        cfg = cfg or GateConfig()
        padding = cfg.kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(n_features, cfg.hidden_channels, cfg.kernel_size, padding=padding),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(
                cfg.hidden_channels,
                cfg.hidden_channels,
                cfg.kernel_size,
                padding=padding,
            ),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(batch, length)`` from ``(batch, length, feature)``."""
        return self.network(features.transpose(1, 2)).squeeze(1)


def _rank01(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) <= 1 or np.ptp(data) <= 1e-8:
        return np.full(len(data), 0.5, dtype=float)
    return (rankdata(data, method="average") - 1.0) / (len(data) - 1.0)


def _scatter_average(length: int, values: np.ndarray, indices: list[np.ndarray]) -> np.ndarray:
    total = np.zeros(length, dtype=float)
    count = np.zeros(length, dtype=float)
    for index in indices:
        np.add.at(total, index, values)
        np.add.at(count, index, 1.0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def local_geometry_features(
    coords: np.ndarray, sequence: str, priors_v1: dict, priors_v2: dict
) -> np.ndarray:
    """Return five local, native-blind geometry channels for each residue."""
    xyz = np.asarray(coords, dtype=float)
    length = len(sequence)
    if xyz.shape != (length, 3) or not np.isfinite(xyz).all():
        raise ValueError("local geometry requires complete finite coordinates")
    mean_distance = float(priors_v1["adjacent_c1"]["mean"])
    distance = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
    bb = _scatter_average(
        length,
        np.abs(distance - mean_distance),
        [np.arange(length - 1), np.arange(1, length)],
    )

    pair_mask = pair_like_mask(sequence, xyz)
    angles = pseudo_angles(xyz)
    angle_context = pair_mask[1:-1]
    angle_score = np.zeros(len(angles), dtype=float)
    for name, mask in (("pair_like", angle_context), ("unpaired", ~angle_context)):
        if mask.any():
            angle_score[mask] = histogram_nll(
                angles[mask], priors_v2["contexts"][name]["angle"]
            )
    angle_local = np.zeros(length, dtype=float)
    angle_local[1:-1] = np.nan_to_num(angle_score)

    torsions = signed_pseudo_torsions(xyz)
    torsion_context = pair_mask[1:-2] | pair_mask[2:-1]
    torsion_score = np.zeros(len(torsions), dtype=float)
    for name, mask in (("pair_like", torsion_context), ("unpaired", ~torsion_context)):
        if mask.any():
            torsion_score[mask] = histogram_nll(
                torsions[mask], priors_v2["contexts"][name]["torsion"]
            )
    torsion_local = _scatter_average(
        length,
        np.nan_to_num(torsion_score),
        [np.arange(1, length - 2), np.arange(2, length - 1)],
    )
    kink = np.zeros(length, dtype=float)
    kink[1:-1] = np.nan_to_num(angles < np.deg2rad(70.0)).astype(float)
    return np.column_stack([bb, angle_local, torsion_local, kink, pair_mask.astype(float)])


def pair_gate_features(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    priors_v1: dict,
    priors_v2: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build pair features and return pretrained coordinates aligned to template."""
    if template.target_id != pretrained.target_id or template.sequence != pretrained.sequence:
        raise ValueError("gate candidates must describe the same target")
    if template.kind != "template" or pretrained.kind != "pretrained":
        raise ValueError("gate expects template and pretrained candidates")
    sequence = template.sequence
    alignment_mask = (
        template.valid_mask
        & pretrained.valid_mask
        & template.support_mask
        & (template.confidence >= 0.25)
    )
    aligned, alignment_rmsd, inliers = robust_superpose(
        pretrained.coords, template.coords, alignment_mask
    )
    disagreement = np.linalg.norm(aligned - template.coords, axis=1)
    support_distance = distance_transform_edt(template.support_mask).astype(float)
    support_distance = np.clip(support_distance / 20.0, 0.0, 1.0)
    template_geometry = local_geometry_features(
        template.coords, sequence, priors_v1, priors_v2
    )
    pretrained_geometry = local_geometry_features(
        aligned, sequence, priors_v1, priors_v2
    )
    one_hot = np.zeros((len(sequence), 4), dtype=float)
    alphabet = {base: index for index, base in enumerate("ACGU")}
    for index, base in enumerate(sequence):
        if base in alphabet:
            one_hot[index, alphabet[base]] = 1.0
    features = np.column_stack(
        [
            template.confidence,
            pretrained.confidence,
            _rank01(template.confidence),
            _rank01(pretrained.confidence),
            template.support_mask.astype(float),
            support_distance,
            np.clip(disagreement / 20.0, 0.0, 2.0),
            template_geometry,
            pretrained_geometry,
            one_hot,
            np.full(len(sequence), np.log1p(len(sequence)) / 7.0),
        ]
    ).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError(f"feature schema mismatch: {features.shape[1]}")
    return features, aligned, {
        "alignment_rmsd": alignment_rmsd,
        "alignment_inlier_fraction": float(inliers.mean()),
    }


def predict_pretrained_probability(
    model: ConfidenceGate1D,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Apply a trained gate to one variable-length pair."""
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    normalized = (np.asarray(features, dtype=np.float32) - mean) / std
    tensor = torch.as_tensor(normalized[None], dtype=torch.float32, device=dev)
    model = model.to(dev).eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0]
    return probability.cpu().numpy()


def load_gate_checkpoint(path: str, *, map_location: str = "cpu") -> dict:
    """Load and validate a locally trained confidence-gate checkpoint."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"unsupported gate checkpoint schema: {checkpoint.get('schema_version')}")
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError("gate checkpoint feature schema does not match this code")
    config = GateConfig(**checkpoint["gate_config"])
    model = ConfidenceGate1D(cfg=config)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint = dict(checkpoint)
    checkpoint["model"] = model.eval()
    return checkpoint


def fuse_with_learned_gate(
    template: StructureCandidate,
    pretrained: StructureCandidate,
    priors_v1: dict,
    priors_v2: dict,
    checkpoint: dict,
    *,
    device: str = "cpu",
    disagreement_threshold: float = 12.0,
) -> StructureCandidate:
    """Fuse an aligned pair using a frozen, calibrated residue gate."""
    features, aligned, alignment = pair_gate_features(
        template, pretrained, priors_v1, priors_v2
    )
    probability = predict_pretrained_probability(
        checkpoint["model"],
        features,
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_std"], dtype=np.float32),
        device=device,
    )
    threshold = float(checkpoint["training"]["decision_threshold"])
    epsilon = 1e-5
    logit = np.log(np.clip(probability, epsilon, 1.0 - epsilon)) - np.log1p(
        -np.clip(probability, epsilon, 1.0 - epsilon)
    )
    threshold_logit = np.log(threshold) - np.log1p(-threshold)
    alpha = 1.0 / (1.0 + np.exp(-np.clip(logit - threshold_logit, -12.0, 12.0)))
    alpha = gaussian_filter1d(alpha, sigma=1.5, mode="nearest")
    disagreement = np.linalg.norm(aligned - template.coords, axis=1)
    strong_disagreement = disagreement > disagreement_threshold
    alpha[strong_disagreement] = (probability[strong_disagreement] >= threshold).astype(float)
    alpha = np.clip(alpha, 0.0, 1.0)

    coordinates = (1.0 - alpha[:, None]) * template.coords + alpha[:, None] * aligned
    confidence = (
        (1.0 - alpha) * template.confidence + alpha * pretrained.confidence
    )
    return StructureCandidate(
        target_id=template.target_id,
        sequence=template.sequence,
        candidate_id=(
            f"fused__learned_gate__{template.candidate_id}__{pretrained.candidate_id}"
        ),
        kind="fused",
        source="geofuse_learned",
        model="confidence_gate_1d_v1",
        coords=coordinates,
        confidence=np.clip(confidence, 0.01, 1.0),
        support_mask=template.support_mask | pretrained.support_mask,
        global_confidence=(template.global_confidence + pretrained.global_confidence) / 2.0,
        metadata={
            "template_parent": template.candidate_id,
            "pretrained_parent": pretrained.candidate_id,
            "fusion_mode": "learned_gate",
            "decision_threshold": threshold,
            "alignment_rmsd": alignment["alignment_rmsd"],
            "alignment_inlier_fraction": alignment["alignment_inlier_fraction"],
            "mean_pretrained_probability": float(probability.mean()),
            "mean_pretrained_weight": float(alpha.mean()),
            "pretrained_dominant_fraction": float((alpha > 0.5).mean()),
            "boundary_count": int(np.count_nonzero(np.diff(alpha > 0.5))),
        },
    )
