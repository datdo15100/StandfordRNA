"""Geometry-informed refinement losses (PyTorch, autograd).

Energy minimised:
    E(X) = w_tpl * L_tpl + w_bb * L_bb + w_clash * L_clash + w_rg * L_rg [+ w_dist * L_dist]

L_tpl   confidence-weighted pull toward transferred template coordinates. Per-residue
        weights are high where a coordinate was reliably transferred and low/zero in
        gaps — this is the core "trust the template where reliable, let geometry repair
        the rest" mechanism.
L_bb    consecutive C1'-C1' distance kept near the data-estimated mean/std (backbone
        continuity).
L_clash steric term: penalise non-adjacent C1' pairs closer than r_min.
L_rg    keep the radius of gyration near the length-appropriate target (anti collapse/
        explosion). Applied softly.
L_dist  optional pairwise prior (e.g. from a pretrained distogram); off in v1.
"""
from __future__ import annotations

import torch


def loss_template(X: torch.Tensor, X_tpl: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    diff = ((X - X_tpl) ** 2).sum(dim=-1)  # (L,)
    return (w * diff).sum() / (w.sum() + 1e-8)


def loss_backbone(X: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    d = torch.linalg.norm(X[1:] - X[:-1], dim=-1)
    return (((d - mu) / sigma) ** 2).mean()


def loss_clash(X: torch.Tensor, r_min: float, sep: int = 2,
               max_pairs: int = 200_000) -> torch.Tensor:
    L = X.shape[0]
    D = torch.cdist(X, X)  # (L, L)
    idx = torch.arange(L, device=X.device)
    nonadj = (idx[None, :] - idx[:, None]).abs() >= sep
    tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=X.device), diagonal=1)
    pair = nonadj & tri
    viol = torch.clamp(r_min - D[pair], min=0.0)
    return (viol ** 2).sum() / (L + 1e-8)


def radius_of_gyration(X: torch.Tensor) -> torch.Tensor:
    c = X.mean(dim=0, keepdim=True)
    return torch.sqrt(((X - c) ** 2).sum(dim=-1).mean() + 1e-8)


def loss_rg(X: torch.Tensor, rg_target: float) -> torch.Tensor:
    return (radius_of_gyration(X) - rg_target) ** 2


def loss_distance(X: torch.Tensor, D_pred: torch.Tensor, W: torch.Tensor,
                  sep: int = 3) -> torch.Tensor:
    """Smooth-L1 between predicted and realised pairwise distances (non-local only)."""
    L = X.shape[0]
    D = torch.cdist(X, X)
    idx = torch.arange(L, device=X.device)
    mask = (idx[None, :] - idx[:, None]).abs() >= sep
    err = torch.nn.functional.smooth_l1_loss(D[mask], D_pred[mask], reduction="none")
    return (W[mask] * err).sum() / (W[mask].sum() + 1e-8)
