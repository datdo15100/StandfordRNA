"""Motif-proxy-conditioned geometry projection for GeoFuse Phase B."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..refine import losses as base_losses
from ..refine.optimizer import _rg_target
from .geometry_v2 import pair_like_mask


@dataclass
class GeometryV2Config:
    """Fixed Phase-B pilot weights; tune only on temporal-safe training data."""

    steps: int = 300
    lr: float = 0.04
    # Source anchoring is deliberately stronger than v1: Phase A already found
    # useful global folds, so Phase B should project local geometry without
    # rewriting them. Angle/torsion terms dominate the local repair to prevent
    # the distance-only sharp-kink failure observed in v1.
    w_source: float = 3.0
    w_backbone: float = 1.0
    w_clash: float = 0.3
    w_rg: float = 0.02
    w_angle: float = 0.30
    w_torsion: float = 0.15
    w_kink: float = 20.0
    kink_floor_deg: float = 70.0
    kink_margin_deg: float = 5.0
    backbone_huber_delta: float = 2.0
    rg_huber_delta: float = 5.0
    sep_clash: int = 2
    huber_delta: float = 2.0
    grad_clip: float = 10.0


def torch_pseudo_angles(coords: torch.Tensor) -> torch.Tensor:
    if len(coords) < 3:
        return coords.new_empty((0,))
    left = coords[:-2] - coords[1:-1]
    right = coords[2:] - coords[1:-1]
    left_norm = torch.linalg.norm(left, dim=-1)
    right_norm = torch.linalg.norm(right, dim=-1)
    valid = (left_norm > 1e-6) & (right_norm > 1e-6)
    denominator = torch.where(valid, left_norm * right_norm, torch.ones_like(left_norm))
    cosine = (left * right).sum(-1) / denominator
    angle = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
    return torch.where(valid, angle, torch.full_like(angle, float("nan")))


def torch_signed_pseudo_torsions(coords: torch.Tensor) -> torch.Tensor:
    if len(coords) < 4:
        return coords.new_empty((0,))
    b0 = coords[1:-2] - coords[:-3]
    b1 = coords[2:-1] - coords[1:-2]
    b2 = coords[3:] - coords[2:-1]
    b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
    unit = b1 / torch.clamp(b1_norm, min=1e-6)
    v = b0 - (b0 * unit).sum(-1, keepdim=True) * unit
    w = b2 - (b2 * unit).sum(-1, keepdim=True) * unit
    v_norm = torch.linalg.norm(v, dim=-1)
    w_norm = torch.linalg.norm(w, dim=-1)
    valid = (b1_norm.squeeze(-1) > 1e-6) & (v_norm > 1e-6) & (w_norm > 1e-6)
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(unit, v, dim=-1) * w).sum(-1)
    # atan2(0, 0) has an undefined gradient.  Substitute a constant safe point
    # before atan2, then mark it missing so the histogram loss can ignore it.
    safe_x = torch.where(valid, x, torch.ones_like(x))
    safe_y = torch.where(valid, y, torch.zeros_like(y))
    torsion = torch.atan2(safe_y, safe_x)
    return torch.where(valid, torsion, torch.full_like(torsion, float("nan")))


def histogram_nll_loss(values: torch.Tensor, prior: dict) -> torch.Tensor:
    """Differentiable piecewise-linear empirical negative log-density."""
    values = values[torch.isfinite(values)]
    if not len(values):
        return values.sum()
    table = torch.as_tensor(prior["nll"], dtype=values.dtype, device=values.device)
    bins = int(prior["bins"])
    lo, hi = float(prior["lo"]), float(prior["hi"])
    position = (values - lo) / (hi - lo) * bins - 0.5
    if prior.get("periodic", False):
        base = torch.floor(position)
        # Integer modulo is robust at the +pi float32 boundary, where floating
        # remainder can round to exactly ``bins`` on CUDA.
        left = torch.remainder(base.long(), bins)
        right = torch.remainder(left + 1, bins)
        fraction = position - base
    else:
        position = torch.clamp(position, 0.0, bins - 1.0)
        base = torch.floor(position)
        left = torch.clamp(base.long(), min=0, max=bins - 1)
        right = torch.clamp(left + 1, max=bins - 1)
        fraction = position - base
    return ((1.0 - fraction) * table[left] + fraction * table[right]).mean()


def source_huber_loss(
    coords: torch.Tensor,
    source: torch.Tensor,
    confidence: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    distance = torch.linalg.norm(coords - source, dim=-1)
    error = torch.nn.functional.huber_loss(
        distance, torch.zeros_like(distance), reduction="none", delta=delta
    )
    return (confidence * error).sum() / (confidence.sum() + 1e-8)


def backbone_huber_loss(
    coords: torch.Tensor, mean: float, std: float, delta: float
) -> torch.Tensor:
    """Robust adjacent-distance loss in prior-standard-deviation units.

    Gap filling can leave a handful of very long TBM bonds.  A squared loss lets
    those outliers dominate all angle terms and creates the exact kink failure
    observed in geometry v1.  Huber retains a repair gradient without allowing
    one seam to dictate the entire projection.
    """
    distance = torch.linalg.norm(coords[1:] - coords[:-1], dim=-1)
    standardized = (distance - mean) / std
    return torch.nn.functional.huber_loss(
        standardized,
        torch.zeros_like(standardized),
        reduction="mean",
        delta=delta,
    )


def rg_huber_loss(coords: torch.Tensor, target: float, delta: float) -> torch.Tensor:
    """Robust global-size regularizer in Angstroms."""
    centered = coords - coords.mean(dim=0, keepdim=True)
    rg = torch.sqrt((centered.square().sum(dim=-1)).mean() + 1e-8)
    return torch.nn.functional.huber_loss(
        rg, rg.new_tensor(target), reduction="mean", delta=delta
    )


def kink_regression_loss(
    angles: torch.Tensor, source_angles: torch.Tensor, floor: torch.Tensor
) -> torch.Tensor:
    """Penalize new/worsened kinks while not forcing a dubious raw kink upright.

    Raw angles above the reporting threshold should remain above it.  Existing
    raw kinks use their own angle as the lower bound, so the projection may
    improve them but cannot cheaply make them sharper.
    """
    valid = torch.isfinite(angles) & torch.isfinite(source_angles)
    lower_bound = torch.minimum(source_angles, floor)
    violation = torch.where(
        valid, torch.clamp(lower_bound - angles, min=0.0), torch.zeros_like(angles)
    )
    # This is a constraint-style term: averaging over all residues would make a
    # single new kink almost free on a long RNA.  Summing keeps the cost of the
    # same local angular regression independent of sequence length.
    return violation.square().sum()


def _context_geometry_loss(
    values: torch.Tensor,
    pair_context: torch.Tensor,
    pair_count: int,
    priors_v2: dict,
    geometry_name: str,
) -> torch.Tensor:
    loss = values.new_zeros(())
    weight = 0
    for context, mask, count in (
        ("pair_like", pair_context, pair_count),
        ("unpaired", ~pair_context, len(values) - pair_count),
    ):
        if count:
            loss = loss + count * histogram_nll_loss(
                values[mask], priors_v2["contexts"][context][geometry_name]
            )
            weight += count
    return loss / max(weight, 1)


def refine_structure_v2(
    coords: np.ndarray,
    sequence: str,
    priors_v1: dict,
    priors_v2: dict,
    *,
    source_confidence: np.ndarray | None = None,
    global_confidence: float = 0.5,
    cfg: GeometryV2Config | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Project a raw candidate onto context-conditioned empirical geometry."""
    cfg = cfg or GeometryV2Config()
    x0 = np.asarray(coords, dtype=np.float32)
    if x0.shape != (len(sequence), 3) or not np.isfinite(x0).all():
        raise ValueError("Geometry v2 requires a complete finite C1' candidate")
    confidence = (
        np.asarray(source_confidence, dtype=np.float32)
        if source_confidence is not None
        else np.ones(len(sequence), dtype=np.float32)
    )
    if confidence.shape != (len(sequence),):
        raise ValueError("source confidence must have one value per residue")

    torch.manual_seed(seed)
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    source = torch.as_tensor(x0, dtype=torch.float32, device=dev)
    source_angles = torch_pseudo_angles(source).detach()
    conf = torch.as_tensor(np.clip(confidence, 0.01, 1.0), dtype=torch.float32, device=dev)
    current = source.clone().requires_grad_(True)

    pair_mask = pair_like_mask(sequence, x0)
    angle_context = torch.as_tensor(pair_mask[1:-1], dtype=torch.bool, device=dev)
    angle_pair_count = int(pair_mask[1:-1].sum())
    torsion_context = torch.as_tensor(
        pair_mask[1:-2] | pair_mask[2:-1], dtype=torch.bool, device=dev
    )
    torsion_pair_count = int((pair_mask[1:-2] | pair_mask[2:-1]).sum())
    mu = float(priors_v1["adjacent_c1"]["mean"])
    sigma = max(float(priors_v1["adjacent_c1"]["std"]), 1e-3)
    r_min = float(priors_v1["clash"]["r_min"])
    rg_target = _rg_target(len(sequence), priors_v1)
    strength = 0.2 + 0.8 * (1.0 - float(np.clip(global_confidence, 0.0, 1.0)))

    optimizer = torch.optim.Adam([current], lr=cfg.lr)
    history = []
    for step in range(cfg.steps):
        optimizer.zero_grad()
        terms = {
            "source": source_huber_loss(current, source, conf, cfg.huber_delta),
            "backbone": backbone_huber_loss(
                current, mu, sigma, cfg.backbone_huber_delta
            )
            if len(sequence) >= 2
            else current.new_zeros(()),
            "clash": base_losses.loss_clash(current, r_min, sep=cfg.sep_clash)
            if len(sequence) >= 3
            else current.new_zeros(()),
            "rg": rg_huber_loss(current, rg_target, cfg.rg_huber_delta)
            if len(sequence) >= 3
            else current.new_zeros(()),
            "angle": _context_geometry_loss(
                torch_pseudo_angles(current),
                angle_context,
                angle_pair_count,
                priors_v2,
                "angle",
            ),
            "torsion": _context_geometry_loss(
                torch_signed_pseudo_torsions(current),
                torsion_context,
                torsion_pair_count,
                priors_v2,
                "torsion",
            ),
        }
        angles = torch_pseudo_angles(current)
        kink_guard = torch.deg2rad(
            current.new_tensor(cfg.kink_floor_deg + cfg.kink_margin_deg)
        )
        terms["kink"] = kink_regression_loss(angles, source_angles, kink_guard)
        loss = (
            cfg.w_source * terms["source"]
            + strength
            * (
                cfg.w_backbone * terms["backbone"]
                + cfg.w_clash * terms["clash"]
                + cfg.w_rg * terms["rg"]
                + cfg.w_angle * terms["angle"]
                + cfg.w_torsion * terms["torsion"]
                + cfg.w_kink * terms["kink"]
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([current], cfg.grad_clip)
        optimizer.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    **{name: float(value.detach()) for name, value in terms.items()},
                }
            )

    result = current.detach().cpu().numpy()
    info = {
        "config": asdict(cfg),
        "device": str(dev),
        "strength": strength,
        "pair_like_fraction": float(pair_mask.mean()),
        "rg_target": rg_target,
        "history": history,
    }
    return result, info
