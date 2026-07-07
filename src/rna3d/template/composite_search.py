"""Composite-similarity template search — a high-recall fallback for MMseqs.

Our MMseqs k=13 nucleotide prefilter is fast but misses weak / remote / partial RNA
matches: on the CASP15 no-homolog targets it returns 0 candidates, so we fall back to
de novo. The reproduced 1st-place method showed that an exhaustive composite-similarity
scan (global + local Smith-Waterman + RNA k-mer/feature similarity) finds *some* real
RNA template for essentially every target, and copying a plausible real fold beats a de
novo heuristic (temporal-safe CASP15: 0.30 vs our 0.21). This module adds that scan as a
recall fallback, still temporal-safe and leakage-guarded.

Library = the deduped top-1 template set (`top1_template_meta.parquet` /
`top1_template_coords.pkl`, 7,155 unique sequences). Loaded once and cached.
"""
from __future__ import annotations

import functools
import pickle

import numpy as np
import pandas as pd

from ..baselines.top1 import find_similar_sequences
from ..paths import cache, processed


@functools.lru_cache(maxsize=1)
def _library() -> tuple[pd.DataFrame, dict]:
    meta = pd.read_parquet(processed() / "top1_template_meta.parquet")
    with open(cache() / "top1_template_coords.pkl", "rb") as fh:
        coords = pickle.load(fh)
    return meta, coords


def search(seq: str, cutoff: str, exclude_pdb_ids: tuple[str, ...] = (),
           top_n: int = 5) -> list[dict]:
    """Temporal-safe composite search. Returns up to top_n dicts with keys
    chain_key, seq, coords, score, pdb_id, release_date."""
    meta, coords = _library()
    excl = {p.upper() for p in exclude_pdb_ids}

    templates = []  # (chain_key, seq, coords) for find_similar_sequences
    info = {}
    for tid, tseq, pdb_id, rd in zip(meta["target_id"], meta["sequence"],
                                     meta["pdb_id"], meta["release_date"]):
        if not (str(rd) < str(cutoff)):
            continue
        if str(pdb_id).upper() in excl:
            continue
        d = coords.get(tid)
        if d is None:
            continue
        templates.append((tid, tseq, np.asarray(d["coords"], float)))
        info[tid] = (str(pdb_id).upper(), str(rd))

    if not templates:
        return []
    hits = find_similar_sequences(seq, templates, top_n=top_n)
    out = []
    for tid, tseq, score, tcoords in hits:
        pdb_id, rd = info[tid]
        out.append({"chain_key": tid, "seq": tseq, "coords": tcoords,
                    "score": float(score), "pdb_id": pdb_id, "release_date": rd})
    return out
