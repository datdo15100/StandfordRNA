"""Rule-based C1' refinement — a faithful baseline from the 1st-place TBM notebook.

Adapted from ``adaptive_rna_constraints`` in the 1st-place Stanford RNA 3D Folding
notebook ("RNA 3D Folds: TBM-only approach"). It refines coordinates by direct,
single-pass geometric *nudging* (no gradient / no autodiff):

  1. sequential distance: pull consecutive C1' back into [5.5, 6.5] A when outside;
  2. steric clash: push non-consecutive C1' apart when closer than 3.8 A;
  3. base pairing (only for low-confidence templates): gently pull complementary
     bases within a window toward a 10.5 A C1'-C1' separation.

All adjustments are scaled by ``constraint_strength = 0.8 * (1 - min(conf, 0.8))``,
so confident templates are barely touched. This exists so the thesis can compare it
head-to-head against our gradient-based geometry-energy refinement (``refine.optimizer``).
The two differ in kind: this makes local, greedy corrections in one pass; ours
minimises a global differentiable energy (template + backbone + clash + Rg) with Adam.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import distance_matrix

_PAIRS = {"A": "U", "U": "A", "G": "C", "C": "G"}


def refine_rule_based(
    coordinates: np.ndarray,
    sequence: str,
    confidence: float = 1.0,
    seq_min_dist: float = 5.5,
    seq_max_dist: float = 6.5,
    min_allowed_distance: float = 3.8,
) -> np.ndarray:
    coords = np.array(coordinates, dtype=float)
    n = len(sequence)
    strength = 0.8 * (1.0 - min(confidence, 0.8))

    # 1. sequential distance constraints (adjust only the following residue)
    for i in range(n - 1):
        cur, nxt = coords[i], coords[i + 1]
        d = np.linalg.norm(nxt - cur)
        if d < seq_min_dist or d > seq_max_dist:
            target = (seq_min_dist + seq_max_dist) / 2
            direction = (nxt - cur) / (np.linalg.norm(nxt - cur) + 1e-10)
            adjustment = (target - d) * strength
            coords[i + 1] = cur + direction * (d + adjustment)

    # 2. steric-clash prevention (non-consecutive C1' too close)
    dmat = distance_matrix(coords, coords)
    clash_i, clash_j = np.where((dmat < min_allowed_distance) & (dmat > 0))
    for idx in range(len(clash_i)):
        i, j = int(clash_i[idx]), int(clash_j[idx])
        if abs(i - j) <= 1 or i >= j:
            continue
        pos_i, pos_j = coords[i], coords[j]
        d = dmat[i, j]
        direction = (pos_j - pos_i) / (np.linalg.norm(pos_j - pos_i) + 1e-10)
        adjustment = (min_allowed_distance - d) * strength
        coords[i] = pos_i - direction * (adjustment / 2)
        coords[j] = pos_j + direction * (adjustment / 2)

    # 3. light base-pair constraint (only when template confidence is low)
    if strength > 0.3:
        for i in range(n):
            comp = _PAIRS.get(sequence[i])
            if not comp:
                continue
            for j in range(i + 3, min(i + 20, n)):
                if sequence[j] == comp:
                    d = np.linalg.norm(coords[i] - coords[j])
                    if 8.0 < d < 14.0:
                        target = 10.5
                        direction = (coords[j] - coords[i]) / (np.linalg.norm(coords[j] - coords[i]) + 1e-10)
                        adjustment = (target - d) * (strength * 0.3)
                        coords[i] = coords[i] - direction * (adjustment / 2)
                        coords[j] = coords[j] + direction * (adjustment / 2)
                        break
    return coords
