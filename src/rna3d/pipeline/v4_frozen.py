"""Frozen native-blind candidate generation used by the V4 confirmatory study.

This module contains no label loading or structural scoring.  It materializes the
development-selected TBM method and the common J-controlled raw comparator from a
sequence, cutoff date, and the frozen template database.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from ..baselines.top1 import build_raw_candidates, composite_similarity_components, rna_features
from ..geometry.denovo import de_novo_structure
from ..template.align import align_and_transfer
from ..template.gap_fill import fill_gaps_linear


@dataclass(frozen=True)
class FrozenBank:
    coords: np.ndarray
    confidence: np.ndarray
    global_confidence: np.ndarray
    candidate_ids: tuple[str, ...]
    pdb_ids: tuple[str, ...]
    fallback_slots: int


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def normalized_exclusions(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip().upper() for value in values if str(value).strip()}))


def length_allowed(query_len: int, template_len: int) -> bool:
    ratio = abs(template_len - query_len) / max(template_len, query_len)
    if query_len < 50 or template_len < 50:
        return ratio <= 0.6
    if query_len > 1000 or template_len > 1000:
        return ratio <= 0.2
    return ratio <= 0.4


def _candidate_pool(
    sequence: str,
    cutoff: str,
    exclusions: tuple[str, ...],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    query_features = rna_features(sequence)
    excluded = set(exclusions)
    rows = []
    for row in meta.itertuples(index=False):
        pdb_id = str(row.pdb_id).upper()
        if pdb_id in excluded or str(row.release_date) >= str(cutoff):
            continue
        if not length_allowed(len(sequence), int(row.length)):
            continue
        components = composite_similarity_components(sequence, row.seq, query_features)
        rows.append(
            {
                "chain_key": str(row.chain_key),
                "pdb_id": pdb_id,
                "global": float(components["global"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["chain_key", "pdb_id", "global"])
    return pd.DataFrame(rows).sort_values(
        ["global", "chain_key"], ascending=[False, True]
    ).head(50)


def build_thesis_tbm_bank(
    *,
    target_id: str,
    sequence: str,
    cutoff: str,
    excluded_pdb_ids: Iterable[str],
    meta: pd.DataFrame,
    coordinates: dict,
    adjacent_distance: float,
    n: int = 5,
    fallback_version: str = "v4-final-raw-1",
) -> FrozenBank:
    """Build global-only, identity-by-coverage, distinct-PDB, linear-gap TBM."""
    exclusions = normalized_exclusions(excluded_pdb_ids)
    pool = _candidate_pool(sequence, cutoff, exclusions, meta)
    materialized = []
    for row in pool.itertuples(index=False):
        transfer = align_and_transfer(sequence, coordinates[row.chain_key], row.chain_key)
        filled, local_confidence = fill_gaps_linear(
            transfer.coords, transfer.mask, adj_dist=adjacent_distance
        )
        materialized.append(
            {
                "chain_key": row.chain_key,
                "pdb_id": row.pdb_id,
                "rank": float(transfer.identity * transfer.coverage),
                "coords": filled,
                "confidence": local_confidence,
            }
        )
    ranked = sorted(materialized, key=lambda item: (-item["rank"], item["chain_key"]))
    selected, used_pdb = [], set()
    for item in ranked:
        if item["pdb_id"] in used_pdb:
            continue
        selected.append(item)
        used_pdb.add(item["pdb_id"])
        if len(selected) == n:
            break
    if len(selected) < n:
        selected.extend(item for item in ranked if item not in selected)
    selected = selected[:n]

    coords = [np.asarray(item["coords"], dtype=float) for item in selected]
    confidence = [np.asarray(item["confidence"], dtype=float) for item in selected]
    global_confidence = [float(item["rank"]) for item in selected]
    candidate_ids = [str(item["chain_key"]) for item in selected]
    pdb_ids = [str(item["pdb_id"]) for item in selected]
    available = len(coords)
    while len(coords) < n:
        index = len(coords)
        coords.append(
            de_novo_structure(
                sequence,
                stable_seed(fallback_version, "retained_tbm", target_id, index),
            )
        )
        confidence.append(np.full(len(sequence), 0.1))
        global_confidence.append(0.1)
        candidate_ids.append(f"de_novo_{index}")

    return FrozenBank(
        coords=np.asarray(coords, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        global_confidence=np.asarray(global_confidence, dtype=np.float32),
        candidate_ids=tuple(candidate_ids),
        pdb_ids=tuple(pdb_ids),
        fallback_slots=max(0, n - available),
    )


def build_j_controlled_bank(
    *,
    target_id: str,
    sequence: str,
    cutoff: str,
    excluded_pdb_ids: Iterable[str],
    meta: pd.DataFrame,
    coordinates: dict,
    n: int = 5,
) -> FrozenBank:
    """Build the raw publicly released John comparator under common data rules."""
    excluded = set(normalized_exclusions(excluded_pdb_ids))
    safe = meta[
        (meta["release_date"].astype(str) < str(cutoff))
        & ~meta["pdb_id"].str.upper().isin(excluded)
    ]
    templates = [
        (row.chain_key, row.seq, coordinates[row.chain_key]["coords"])
        for row in safe.itertuples(index=False)
    ]
    raw = build_raw_candidates(sequence, target_id, templates, n=n, base_seed=42)
    return FrozenBank(
        coords=np.asarray([item.coords for item in raw], dtype=np.float32),
        confidence=np.asarray(
            [np.full(len(sequence), item.confidence, dtype=np.float32) for item in raw],
            dtype=np.float32,
        ),
        global_confidence=np.asarray([item.confidence for item in raw], dtype=np.float32),
        candidate_ids=tuple(item.template_id or f"john_de_novo_{index}" for index, item in enumerate(raw)),
        pdb_ids=tuple(),
        fallback_slots=sum(item.source != "template" for item in raw),
    )
