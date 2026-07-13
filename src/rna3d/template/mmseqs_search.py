"""MMseqs2 nucleotide search for candidate-template retrieval.

A persistent, split-on-disk target index is built once over all template chains;
searches then stream one split at a time, so peak RAM stays under the box's ~5 GB
(this also mirrors the Kaggle setup: precompute the index, search at inference).

Temporal-safety and self-leakage filtering are applied afterwards on the hit list
(by release date / PDB id), so one index serves any cutoff.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import pandas as pd

from ..paths import cache
from . import db

M8_COLS = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
           "qstart", "qend", "tstart", "tend", "evalue", "bits"]

_FMT = ("query,target,pident,alnlen,mismatch,gapopen,"
        "qstart,qend,tstart,tend,evalue,bits")


def mmseqs_bin() -> str:
    """Resolve the mmseqs binary robustly.

    Prefer PATH, but fall back to the sibling of the running interpreter (mmseqs is
    installed in the same conda env's bin/) so background shells without the env's
    PATH still find it.
    """
    configured = os.environ.get("RNA3D_MMSEQS")
    if configured:
        return configured
    found = shutil.which("mmseqs")
    if found:
        return found
    sibling = Path(sys.executable).parent / "mmseqs"
    if sibling.exists():
        return str(sibling)
    return "mmseqs"


def _run(cmd: list[str]):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mmseqs {cmd[1]} failed:\n{res.stderr[-2500:]}")
    return res


def mmseqs_dir() -> Path:
    d = cache() / "mmseqs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_template_fasta(min_len: int = 8) -> Path:
    path = cache() / "template_db.fasta"
    if not path.exists():
        n = db.write_fasta(db.load_meta(), path, min_len=min_len)
        print(f"[mmseqs] wrote template FASTA: {n} chains -> {path}")
    return path


def ensure_target_db() -> Path:
    """createdb over the full template FASTA (once; cached on disk).

    We do NOT precompute a k-mer index: for nucleotides mmseqs uses k=15, whose
    offset table cannot fit this box's ~5 GB RAM. We instead search on the fly with
    a smaller k (the table is built per search and re-aligned downstream anyway).
    """
    d = mmseqs_dir()
    target_db = d / "targetDB"
    if (d / "targetDB.dbtype").exists():
        return target_db
    fasta = ensure_template_fasta()
    print("[mmseqs] building target DB (one-time) ...")
    _run([mmseqs_bin(), "createdb", fasta, target_db, "--dbtype", "2"])  # 2 = nucleotide
    return target_db


def search(query_fasta: str | Path, out_m8: str | Path,
           sensitivity: float = 7.5, max_seqs: int = 300, kmer: int = 13,
           evalue: float = 100.0, split_memory_limit: str = "3G") -> pd.DataFrame:
    """Nucleotide search of query sequences against the template DB.

    ``kmer=13`` keeps the k-mer offset table within RAM (k=15 default does not on a
    ~5 GB box); ``split_memory_limit`` caps the prefilter. Hits are re-aligned and
    re-scored downstream, so the lower k only widens the candidate net.
    """
    target_db = ensure_target_db()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qdb = tmp / "queryDB"
        rdb = tmp / "resDB"
        _run([mmseqs_bin(), "createdb", query_fasta, qdb, "--dbtype", "2"])
        _run([mmseqs_bin(), "search", qdb, target_db, rdb, tmp / "s",
              "--search-type", "3", "-s", sensitivity, "--max-seqs", max_seqs,
              "-k", kmer, "-e", evalue, "--split-memory-limit", split_memory_limit])
        _run([mmseqs_bin(), "convertalis", qdb, target_db, rdb, out_m8,
              "--format-output", _FMT])
    return read_m8(out_m8)


def read_m8(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=M8_COLS)
    return pd.read_csv(p, sep="\t", names=M8_COLS)
