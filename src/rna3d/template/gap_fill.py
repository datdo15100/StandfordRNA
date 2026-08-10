"""Geometry-aware reconstruction of missing C1' positions after coordinate transfer.

Follows the backbone-geometry recipe used by strong TBM solutions:
  - internal gap, short  : linear interpolation between flanking C1' atoms
  - internal gap, long   : interpolation + sinusoidal perturbation perpendicular to
                           the backbone, preserving ~adjacent C1'-C1' spacing and
                           giving the segment realistic curvature instead of a rod
  - terminal gap         : extend along the established backbone direction at the
                           target adjacent spacing
  - fully unresolved     : fall back to an extended chain

Returns fully-populated coordinates plus a per-residue confidence in [0, 1]:
transferred residues = 1.0, filled residues decay with distance into the gap.
"""
from __future__ import annotations

import numpy as np


def _perp_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to `direction`."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - d * (ref @ d)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(d, u)
    return u, v


def fill_gaps(coords: np.ndarray, mask: np.ndarray, adj_dist: float = 6.0,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN positions in `coords`. `mask[i]` is True where coords[i] is a real
    transferred coordinate. Returns (filled_coords, confidence)."""
    if rng is None:
        rng = np.random.default_rng(0)
    coords = np.array(coords, dtype=float)
    L = len(coords)
    conf = mask.astype(float)

    resolved_idx = np.where(mask)[0]
    if len(resolved_idx) == 0:
        # nothing transferred: extended chain fallback
        z = np.arange(L) * adj_dist
        out = np.stack([np.zeros(L), np.zeros(L), z], axis=1)
        return out, conf

    # ---- internal gaps ----
    for a, b in zip(resolved_idx[:-1], resolved_idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        pa, pb = coords[a], coords[b]
        seg = pb - pa
        seg_len = np.linalg.norm(seg)
        for k in range(1, gap + 1):
            t = k / (gap + 1)
            p = pa + t * seg
            if gap >= 3 and seg_len > 1e-6:
                # add curvature so a long span isn't a straight rod
                u, _ = _perp_basis(seg)
                amp = adj_dist * 0.5 * min(1.0, gap / 6.0)
                p = p + u * amp * np.sin(np.pi * t)
            coords[a + k] = p
            conf[a + k] = 0.3 * (1.0 - abs(2 * t - 1))  # lowest mid-gap

    # ---- terminal gaps ----
    first, last = resolved_idx[0], resolved_idx[-1]
    if first > 0:
        # direction from the first resolved pair (or arbitrary if only one point)
        if len(resolved_idx) >= 2:
            d = coords[first] - coords[resolved_idx[1]]
        else:
            d = np.array([0.0, 0.0, -1.0])
        d = d / (np.linalg.norm(d) + 1e-8)
        for k in range(1, first + 1):
            coords[first - k] = coords[first] + d * adj_dist * k
            conf[first - k] = 0.1
    if last < L - 1:
        if len(resolved_idx) >= 2:
            d = coords[last] - coords[resolved_idx[-2]]
        else:
            d = np.array([0.0, 0.0, 1.0])
        d = d / (np.linalg.norm(d) + 1e-8)
        for k in range(1, L - last):
            coords[last + k] = coords[last] + d * adj_dist * k
            conf[last + k] = 0.1

    return coords, conf


def fill_gaps_linear(
    coords: np.ndarray,
    mask: np.ndarray,
    adj_dist: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple fair baseline: linear internal interpolation + terminal extension.

    It shares the same transferred coordinates and confidence convention as
    :func:`fill_gaps`; the only removed heuristic is long-gap curvature.
    """
    coords = np.array(coords, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    length = len(coords)
    confidence = mask.astype(float)
    resolved = np.flatnonzero(mask)
    if not len(resolved):
        z = np.arange(length) * adj_dist
        return np.stack([np.zeros(length), np.zeros(length), z], axis=1), confidence

    for left, right in zip(resolved[:-1], resolved[1:]):
        gap = right - left - 1
        for offset in range(1, gap + 1):
            fraction = offset / (gap + 1)
            coords[left + offset] = (
                (1.0 - fraction) * coords[left] + fraction * coords[right]
            )
            confidence[left + offset] = 0.3 * (1.0 - abs(2 * fraction - 1))

    first, last = int(resolved[0]), int(resolved[-1])
    if first:
        direction = (
            coords[first] - coords[resolved[1]]
            if len(resolved) >= 2
            else np.array([0.0, 0.0, -1.0])
        )
        direction /= np.linalg.norm(direction) + 1e-8
        for offset in range(1, first + 1):
            coords[first - offset] = coords[first] + direction * adj_dist * offset
            confidence[first - offset] = 0.1
    if last < length - 1:
        direction = (
            coords[last] - coords[resolved[-2]]
            if len(resolved) >= 2
            else np.array([0.0, 0.0, 1.0])
        )
        direction /= np.linalg.norm(direction) + 1e-8
        for offset in range(1, length - last):
            coords[last + offset] = coords[last] + direction * adj_dist * offset
            confidence[last + offset] = 0.1
    return coords, confidence
