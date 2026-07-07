"""Physical-validity metrics for a C1' structure.

Two kinds, to keep the refinement evaluation honest:
  - **optimized** metrics (clash, backbone deviation, Rg error): these are exactly the
    energies the gradient refinement minimises, so improvements are expected *by
    construction* — they show the optimiser works, not that it is truthful.
  - **independent** metrics (TM-score, handled by the scorer; and `sharp_kinks` here):
    NOT part of the refinement objective. The refinement has no bond-angle term, so the
    pseudo-bond-angle kink rate is a genuinely independent check that fixing clashes /
    backbone does not secretly distort the fold.
"""
from __future__ import annotations

import numpy as np


def sharp_kink_fraction(X: np.ndarray, thresh_deg: float = 70.0) -> float:
    """Fraction of interior residues whose C1'(i-1)-C1'(i)-C1'(i+1) pseudo-bond-angle is
    sharper than `thresh_deg` — an unphysical backbone kink. NOT optimized by refinement."""
    if len(X) < 3:
        return 0.0
    v1 = X[:-2] - X[1:-1]
    v2 = X[2:] - X[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    cos = np.clip((v1 * v2).sum(1) / (n1 * n2), -1, 1)
    ang = np.degrees(np.arccos(cos))
    return float((ang < thresh_deg).mean())


def geom_metrics(X: np.ndarray, priors: dict) -> dict:
    mu = priors["adjacent_c1"]["mean"]
    r_min = priors["clash"]["r_min"]
    a, b = priors["rg_powerlaw"]["a"], priors["rg_powerlaw"]["b"]
    L = len(X)
    d = np.linalg.norm(X[1:] - X[:-1], axis=1)
    bb_dev = float(np.abs(d - mu).mean())
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    nonadj = sep >= 2
    clashes = int(((D < r_min) & nonadj).sum() // 2)
    c = X.mean(0)
    rg = float(np.sqrt(((X - c) ** 2).sum(1).mean()))
    return {"clash_per_res": clashes / L, "bb_dev": bb_dev,
            "rg_err": abs(rg - a * L ** b),
            "sharp_kinks": sharp_kink_fraction(X)}
