"""De novo C1' structure generation for targets with no usable template.

Adapted from the 1st-place Stanford RNA 3D Folding TBM notebook
("RNA 3D Folds: TBM-only approach", functions ``generate_rna_structure`` /
``identify_potential_stems`` / ``generate_improved_rna_structure``). It is a cheap,
sequence-only heuristic — NOT a learned model — that builds a plausible fold by
detecting complementary stem candidates and laying residues along helix/loop/single
paths with stochastic backbone steps. Different seeds give diverse structures, which
is exactly what best-of-5 needs when no template exists.

We use this instead of a bare extended chain for the ~7 CASP15 targets that have no
temporal-safe homolog. It is stochastic-only; refinement (gradient or rule-based)
runs on top of it.
"""
from __future__ import annotations

import numpy as np

_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}


def identify_potential_stems(sequence: str, min_stem_length: int = 3) -> list[tuple[int, int, int, int]]:
    """Find short self-complementary segments that could form stems.

    Returns (start1, end1, start2, end2) index tuples (inclusive).
    """
    seq = sequence.upper()
    n = len(seq)
    stems = []
    for i in range(n - min_stem_length):
        for j in range(i + min_stem_length + 3, n - min_stem_length + 1):
            stem_len = min(min_stem_length, n - j)
            ok = True
            for k in range(stem_len):
                a = seq[i + k]
                b = seq[j + stem_len - k - 1]
                if a not in _COMPLEMENT or _COMPLEMENT[a] != b:
                    ok = False
                    break
            if ok:
                stems.append((i, i + stem_len - 1, j, j + stem_len - 1))
    return stems


def de_novo_structure(sequence: str, seed: int = 0) -> np.ndarray:
    """Heuristic C1' fold driven by local base-pairing (top-1's generate_rna_structure).

    Deterministic given ``seed``; vary the seed to produce diverse candidates.
    """
    rng = np.random.default_rng(seed)
    import random as _random
    pyrng = _random.Random(seed)

    n = len(sequence)
    coords = np.zeros((n, 3))
    seq = sequence.upper()

    # seed the first few residues as a short helix
    for i in range(min(3, n)):
        angle = i * 0.6
        coords[i] = [10.0 * np.cos(angle), 10.0 * np.sin(angle), i * 2.5]

    direction = np.array([0.0, 0.0, 1.0])
    for i in range(3, n):
        base = seq[i]
        comp = _COMPLEMENT.get(base, "X")
        # look back for a nearby complementary partner
        pair_idx = -1
        window = min(i, 15)
        for j in range(i - window, i):
            if j >= 0 and seq[j] == comp:
                pair_idx = j
                break

        if pair_idx != -1 and (i - pair_idx) <= 10 and pyrng.random() < 0.7:
            pair_pos = coords[pair_idx]
            offset = rng.normal(0, 1, 3) * 2.0
            bp_dist = 10.0 + pyrng.uniform(-1.0, 1.0)
            center = coords[:i].mean(axis=0)
            d = center - pair_pos
            d = d / (np.linalg.norm(d) + 1e-10)
            coords[i] = pair_pos + d * bp_dist + offset
            direction = rng.normal(0, 0.3, 3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
        else:
            if pyrng.random() < 0.3:
                # larger reorientation via a random-axis rotation
                angle = pyrng.uniform(0.2, 0.6)
                axis = rng.normal(0, 1, 3)
                axis = axis / (np.linalg.norm(axis) + 1e-10)
                direction = _rotate(direction, axis, angle)
            else:
                direction = direction + rng.normal(0, 0.15, 3)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            step = pyrng.uniform(3.5, 4.5)
            coords[i] = coords[i - 1] + step * direction

    return coords


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of `vec` about unit `axis` by `angle` radians."""
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    c, s = np.cos(angle), np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * (axis @ vec) * (1 - c)


def de_novo_ensemble(sequence: str, n: int = 5, base_seed: int = 0) -> list[np.ndarray]:
    """`n` diverse de novo candidates (distinct seeds)."""
    return [de_novo_structure(sequence, seed=base_seed + k * 1000 + len(sequence))
            for k in range(n)]
