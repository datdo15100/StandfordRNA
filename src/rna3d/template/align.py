"""Pairwise target<->template alignment and C1' coordinate transfer.

Given a target RNA sequence and a parsed template chain (sequence + per-residue
C1' coordinates, some NaN), we globally align the two sequences and copy the
template's resolved C1' coordinates onto the matched target positions. The result
carries an explicit per-target-residue mask distinguishing:
  - matched & resolved   (coordinate transferred, high confidence)
  - matched & unresolved (template residue had no C1' -> treated as gap)
  - unmatched            (insertion in target -> gap to be filled)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio.Align import PairwiseAligner


def make_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = 2.0
    a.mismatch_score = -1.0
    a.open_gap_score = -6.0
    a.extend_gap_score = -0.5
    # do not penalise gaps at the ends (allow partial templates / local coverage)
    a.end_gap_score = 0.0
    return a


_ALIGNER = make_aligner()


@dataclass
class TransferResult:
    chain_key: str
    coords: np.ndarray      # (L_target, 3) transferred C1', NaN where no coord
    mask: np.ndarray        # bool (L_target,), True where a resolved coord was transferred
    identity: float         # matched identical / aligned columns
    coverage: float         # fraction of target residues with a transferred coord
    n_aligned: int          # aligned (non-gap-gap) columns
    template_len: int
    template_resolved: int


def align_and_transfer(target_seq: str, tmpl: dict, chain_key: str) -> TransferResult:
    t_seq = tmpl["seq"]
    t_coords = np.asarray(tmpl["coords"], dtype=float)
    L = len(target_seq)

    coords = np.full((L, 3), np.nan)
    mask = np.zeros(L, dtype=bool)

    aln = _ALIGNER.align(target_seq, t_seq)[0]
    # aln.aligned: array of [[t_start,t_end], ...], [[q_start,q_end], ...] block pairs
    tgt_blocks, tmpl_blocks = aln.aligned
    n_ident = 0
    n_cols = 0
    for (qs, qe), (ts, te) in zip(tgt_blocks, tmpl_blocks):
        length = qe - qs
        n_cols += length
        for k in range(length):
            qi, ti = qs + k, ts + k
            if target_seq[qi] == t_seq[ti]:
                n_ident += 1
            if np.all(np.isfinite(t_coords[ti])):
                coords[qi] = t_coords[ti]
                mask[qi] = True

    identity = n_ident / n_cols if n_cols else 0.0
    coverage = float(mask.sum()) / L if L else 0.0
    return TransferResult(
        chain_key=chain_key,
        coords=coords,
        mask=mask,
        identity=identity,
        coverage=coverage,
        n_aligned=n_cols,
        template_len=len(t_seq),
        template_resolved=int(np.isfinite(t_coords).all(axis=1).sum()),
    )
