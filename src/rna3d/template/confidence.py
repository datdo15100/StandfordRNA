"""Template confidence scoring and temporal-safe ranking.

Template confidence combines four factors, each in [0, 1]:
    conf = identity * coverage * completeness * temporal_validity
where
    identity         = sequence identity over aligned columns
    coverage         = fraction of target residues that received a coordinate
    completeness     = fraction of the template chain that is resolved (has C1')
    temporal_validity= 1 if release_date < target cutoff else 0 (hard leakage gate)

The temporal gate is a hard filter (drops the candidate), but is also exposed as a
multiplicative factor for transparency in ablation tables.
"""
from __future__ import annotations

import pandas as pd


def temporal_valid(release_date: str, cutoff: str) -> bool:
    """Strictly-before gate. Unknown dates ('9999-12-31') are treated as invalid."""
    return str(release_date) < str(cutoff)


def template_confidence(identity: float, coverage: float, completeness: float) -> float:
    return float(identity * coverage * completeness)


def rank_candidates(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with identity, coverage, completeness, confidence, chain_key, ...
    Returns a DataFrame sorted by confidence descending."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("confidence", ascending=False).reset_index(drop=True)
