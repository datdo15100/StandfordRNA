"""Physical-validity metrics for a C1' structure (clashes, backbone, Rg)."""
from __future__ import annotations

import numpy as np


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
    return {"clash_per_res": clashes / L, "bb_dev": bb_dev, "rg_err": abs(rg - a * L ** b)}
