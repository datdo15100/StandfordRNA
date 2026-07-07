"""Confidence-weighted geometry-informed refinement of a C1' structure.

Given an initial structure (from TBM transfer + gap fill, or a DL candidate),
minimise the geometry energy with Adam. Two confidence signals shape the refinement:

  - per-residue confidence -> template-pull weights (gaps are free to move).
  - scalar template confidence -> overall refinement strength: high-confidence
    templates are barely perturbed (preserve the good fold); low-confidence
    candidates get stronger backbone/clash/Rg regularisation (repair geometry).

This is the thesis's core module. It is deliberately light (a few hundred Adam
steps on C1' only) so it runs inside the Kaggle time budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import losses as L


@dataclass
class RefineConfig:
    steps: int = 300
    lr: float = 0.05
    # base weights (further scaled by adaptive strength)
    w_tpl: float = 1.0
    w_bb: float = 1.0
    w_clash: float = 0.5
    w_rg: float = 0.05
    w_dist: float = 0.0
    sep_clash: int = 2
    sep_dist: int = 3


def _rg_target(L_len: int, priors: dict) -> float:
    a = priors["rg_powerlaw"]["a"]
    b = priors["rg_powerlaw"]["b"]
    return float(a * (L_len ** b))


def refine_structure(
    x0: np.ndarray,
    priors: dict,
    template_coords: np.ndarray | None = None,
    conf_residue: np.ndarray | None = None,
    template_confidence: float = 0.5,
    cfg: RefineConfig | None = None,
    D_pred: np.ndarray | None = None,
    W_dist: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Return (refined_coords (L,3), info)."""
    cfg = cfg or RefineConfig()
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    L_len = len(x0)
    X = torch.tensor(np.asarray(x0, float), dtype=torch.float32, device=dev,
                     requires_grad=True)

    mu = priors["adjacent_c1"]["mean"]
    sigma = max(priors["adjacent_c1"]["std"], 1e-3)
    r_min = priors["clash"]["r_min"]
    rg_t = _rg_target(L_len, priors)

    have_tpl = template_coords is not None and conf_residue is not None
    if have_tpl:
        Xt = torch.tensor(np.asarray(template_coords, float), dtype=torch.float32, device=dev)
        w = torch.tensor(np.asarray(conf_residue, float), dtype=torch.float32, device=dev)
    have_dist = D_pred is not None and cfg.w_dist > 0
    if have_dist:
        Dp = torch.tensor(np.asarray(D_pred, float), dtype=torch.float32, device=dev)
        Wd = (torch.tensor(np.asarray(W_dist, float), dtype=torch.float32, device=dev)
              if W_dist is not None else torch.ones(L_len, L_len, device=dev))

    # adaptive strength: confident templates -> gentle; weak -> assertive geometry
    s = 0.2 + 0.8 * (1.0 - min(max(template_confidence, 0.0), 1.0))

    opt = torch.optim.Adam([X], lr=cfg.lr)
    history = []
    for step in range(cfg.steps):
        opt.zero_grad()
        loss = X.new_zeros(())
        if have_tpl:
            loss = loss + cfg.w_tpl * L.loss_template(X, Xt, w)
        if L_len >= 2:
            loss = loss + s * cfg.w_bb * L.loss_backbone(X, mu, sigma)
        if L_len >= 3:
            loss = loss + s * cfg.w_clash * L.loss_clash(X, r_min, sep=cfg.sep_clash)
            loss = loss + s * cfg.w_rg * L.loss_rg(X, rg_t)
        if have_dist:
            loss = loss + cfg.w_dist * L.loss_distance(X, Dp, Wd, sep=cfg.sep_dist)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            history.append((step, float(loss.detach())))

    Xf = X.detach().cpu().numpy()
    info = {"rg_target": rg_t, "rg_final": float(L.radius_of_gyration(X.detach())),
            "strength": s, "loss_history": history}
    return Xf, info
