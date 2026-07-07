"""Template-Based Modeling candidate generation for one target.

Pipeline per target:
    mmseqs hits  ->  temporal/leakage filter  ->  re-align top-K with Biopython
                 ->  coordinate transfer       ->  gap fill
                 ->  confidence-ranked candidates

Each candidate is a fully-populated C1' structure (no NaN) plus the metadata the
refinement stage needs (per-residue confidence, transfer mask, template key).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..template import db
from ..template.align import align_and_transfer
from ..template.confidence import temporal_valid, template_confidence
from ..template.gap_fill import fill_gaps


@dataclass
class Candidate:
    target_id: str
    source: str                 # 'tbm'
    chain_key: str
    coords: np.ndarray          # (L, 3) fully populated
    conf_residue: np.ndarray    # (L,) per-residue confidence in [0,1]
    mask: np.ndarray            # (L,) bool, True where coordinate came from template
    confidence: float           # scalar template confidence
    identity: float
    coverage: float
    meta: dict = field(default_factory=dict)


def build_tbm_candidates(
    target_id: str,
    target_seq: str,
    cutoff: str,
    hits: pd.DataFrame,
    meta: pd.DataFrame,
    adj_dist: float = 6.0,
    realign_topk: int = 40,
    max_candidates: int = 5,
    exclude_pdb_ids: tuple[str, ...] = (),
    apply_temporal: bool = True,
    rng: np.random.Generator | None = None,
) -> list[Candidate]:
    if rng is None:
        rng = np.random.default_rng(0)

    meta_idx = meta.set_index("chain_key")
    # candidate chain_keys from mmseqs, best first
    if hits.empty:
        return []
    cand_keys = (hits.sort_values("bits", ascending=False)["target"]
                 .drop_duplicates().tolist())

    scored: list[dict] = []
    seen_templates = 0
    for key in cand_keys:
        if key not in meta_idx.index:
            continue
        row = meta_idx.loc[key]
        # ---- temporal-safety + self-leakage gate ----
        if apply_temporal and not temporal_valid(row["release_date"], cutoff):
            continue
        if str(row["pdb_id"]).upper() in exclude_pdb_ids:
            continue

        tmpl = db.get_chain(key)
        tr = align_and_transfer(target_seq, tmpl, key)
        if tr.coverage <= 0:
            continue
        completeness = tr.template_resolved / max(tr.template_len, 1)
        conf = template_confidence(tr.identity, tr.coverage, completeness)
        scored.append({
            "chain_key": key, "pdb_id": str(row["pdb_id"]).upper(),
            "transfer": tr, "confidence": conf, "completeness": completeness,
        })
        seen_templates += 1
        if seen_templates >= realign_topk:
            break

    if not scored:
        return []
    scored.sort(key=lambda d: d["confidence"], reverse=True)

    # diverse top candidates: distinct PDB entries first
    chosen, used_pdb = [], set()
    for s in scored:
        if s["pdb_id"] in used_pdb:
            continue
        used_pdb.add(s["pdb_id"])
        chosen.append(s)
        if len(chosen) >= max_candidates:
            break
    # backfill from remaining if we still have < max
    if len(chosen) < max_candidates:
        for s in scored:
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= max_candidates:
                break

    candidates: list[Candidate] = []
    for s in chosen:
        tr = s["transfer"]
        filled, conf_res = fill_gaps(tr.coords, tr.mask, adj_dist=adj_dist, rng=rng)
        candidates.append(Candidate(
            target_id=target_id, source="tbm", chain_key=s["chain_key"],
            coords=filled, conf_residue=conf_res, mask=tr.mask,
            confidence=s["confidence"], identity=tr.identity, coverage=tr.coverage,
            meta={"pdb_id": s["pdb_id"], "completeness": s["completeness"],
                  "template_len": tr.template_len},
        ))
    return candidates
