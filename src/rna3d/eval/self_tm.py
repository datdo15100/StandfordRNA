"""Pairwise self-TM diversity among a target's five predictions.

If the five predictions are near-identical (mean pairwise self-TM ~1.0) then
best-of-5 is wasted. This measures that.
"""
from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np

from .usalign import tm_score, write_c1_pdb


def self_tm_matrix(structs: list[np.ndarray], resnames: list[str]) -> np.ndarray:
    n = len(structs)
    M = np.eye(n)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pdbs = []
        for i, s in enumerate(structs):
            p = td / f"s{i}.pdb"
            write_c1_pdb(s, resnames, p)
            pdbs.append(p)
        for i, j in itertools.combinations(range(n), 2):
            v = tm_score(pdbs[i], pdbs[j])
            M[i, j] = M[j, i] = v
    return M


def mean_pairwise_self_tm(structs: list[np.ndarray], resnames: list[str]) -> float:
    M = self_tm_matrix(structs, resnames)
    n = len(structs)
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(M[iu].mean())
