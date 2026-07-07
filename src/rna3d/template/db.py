"""Access to the parsed template database (metadata + C1' coordinates)."""
from __future__ import annotations

import functools
import pickle

import numpy as np
import pandas as pd

from ..paths import cache, processed


@functools.lru_cache(maxsize=1)
def load_meta() -> pd.DataFrame:
    return pd.read_parquet(processed() / "template_meta.parquet")


@functools.lru_cache(maxsize=1)
def load_coords() -> dict:
    with open(cache() / "template_coords.pkl", "rb") as fh:
        return pickle.load(fh)


def get_chain(chain_key: str) -> dict:
    """Return {'seq', 'resid', 'coords'} for a template chain."""
    return load_coords()[chain_key]


def write_fasta(meta: pd.DataFrame, path, min_len: int = 8) -> int:
    """Write template sequences to a FASTA for MMseqs. Returns count written.

    Skips chains shorter than ``min_len`` and all-unknown ('N') sequences.
    """
    n = 0
    with open(path, "w") as fh:
        for key, seq in zip(meta["chain_key"], meta["seq"]):
            if len(seq) < min_len or set(seq) <= {"N"}:
                continue
            fh.write(f">{key}\n{seq}\n")
            n += 1
    return n
