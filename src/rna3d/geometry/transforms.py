"""Rigid-body and geometric helpers used across the pipeline."""
from __future__ import annotations

import numpy as np


def centroid(x: np.ndarray) -> np.ndarray:
    return np.nanmean(x, axis=0)


def radius_of_gyration(x: np.ndarray) -> float:
    x = x[np.isfinite(x).all(axis=1)]
    c = x.mean(axis=0)
    return float(np.sqrt(((x - c) ** 2).sum(axis=1).mean()))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3x3 rotation matrix (QR of a Gaussian)."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def apply_rigid(x: np.ndarray, R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    y = x @ R.T
    if t is not None:
        y = y + t
    return y


def mirror(x: np.ndarray) -> np.ndarray:
    """Reflect through the x-plane (flips chirality)."""
    y = x.copy()
    y[:, 0] *= -1
    return y


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rotation+translation mapping P onto Q (both (N,3), paired).

    Returns (R, t) such that ``P @ R.T + t`` best matches Q in least squares.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R.T
    return R, t


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))


def extended_chain(n: int, spacing: float = 5.9, rng: np.random.Generator | None = None) -> np.ndarray:
    """A trivial baseline structure: a (slightly wiggly) extended chain.

    Used as a sanity / floor baseline (B0). With an rng, adds small lateral
    jitter so five such chains can be made distinct.
    """
    z = np.arange(n) * spacing
    x = np.zeros(n)
    y = np.zeros(n)
    if rng is not None:
        x = rng.standard_normal(n) * spacing * 0.15
        y = rng.standard_normal(n) * spacing * 0.15
    return np.stack([x, y, z], axis=1)
