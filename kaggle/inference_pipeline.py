"""Kaggle inference entry point (production, offline, <8h).

This module is the single function the final Kaggle notebook calls. It assumes a
*precomputed* artifact bundle is mounted (built locally by the WSL pipeline and
uploaded as a Kaggle Dataset) so the notebook never parses CIFs, builds DBs, or
trains anything — it only searches, transfers, refines and writes submission.csv.

Expected precomputed bundle (mounted read-only):
    template_meta.parquet
    template_coords.pkl
    template_db.fasta            (+ optional prebuilt mmseqs index)
    geometry_priors.json
    USalign                      (not needed for inference; scoring is Kaggle-side)

The notebook wiring (paths, reading test_sequences.csv, MMseqs availability) is
finalised in Phase 8 packaging; this file documents and implements the core loop
so it can be unit-tested locally against the same code as the thesis experiments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def run_inference(test_sequences: pd.DataFrame, artifacts: Path, *,
                  work_dir: Path | None = None,
                  sample_submission: pd.DataFrame | None = None,
                  steps: int = 300, max_len: int = 1000) -> pd.DataFrame:
    """Produce a submission DataFrame for the given test sequences.

    Kept import-light at module load; heavy imports happen inside so this file is
    safe to import in environments without the full stack.
    """
    import sys
    artifacts = Path(artifacts).resolve()
    work_dir = Path(work_dir or Path.cwd() / "rna3d_work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle input datasets are read-only. Load all reusable files from the
    # mounted artifact bundle while keeping query/search outputs in working.
    os.environ["RNA3D_PROCESSED"] = str(artifacts)
    os.environ["RNA3D_CACHE"] = str(artifacts)
    bundled_mmseqs = artifacts / "bin" / "mmseqs"
    if bundled_mmseqs.is_file():
        os.environ["RNA3D_MMSEQS"] = str(bundled_mmseqs)
        bundled_lib = artifacts / "lib"
        if bundled_lib.is_dir():
            old = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{bundled_lib}:{old}" if old else str(bundled_lib)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from rna3d.data import io
    from rna3d.pipeline import methods as M
    from rna3d.pipeline.tbm import build_tbm_candidates
    from rna3d.refine.optimizer import RefineConfig
    from rna3d.template import db, mmseqs_search

    priors = json.load(open(artifacts / "geometry_priors.json"))
    meta = db.load_meta()
    cfg = RefineConfig(steps=steps)

    # one batched search
    qf = work_dir / "query.fasta"
    with open(qf, "w") as fh:
        for _, r in test_sequences.iterrows():
            fh.write(f">{r['target_id']}\n{r['sequence']}\n")
    hits = mmseqs_search.search(qf, work_dir / "hits.m8")

    predictions: dict[str, np.ndarray] = {}
    for _, r in test_sequences.iterrows():
        tid, seq = r["target_id"], r["sequence"]
        L = len(seq)
        cutoff = r.get("temporal_cutoff", "9999-12-31")
        rng = np.random.default_rng(0)
        if L > max_len:
            predictions[tid] = M.m_dummy(L, rng)
            continue
        thits = hits[hits["query"] == tid]
        cands = build_tbm_candidates(tid, seq, cutoff, thits, meta, rng=rng,
                                     adj_dist=priors["adjacent_c1"]["mean"])
        # m_tbm_grad = TBM + gradient refinement, with de novo (not extended-chain)
        # fallback for targets lacking a template — the better no-template branch.
        predictions[tid] = M.m_tbm_grad(cands, seq, L, priors, rng, cfg=cfg)

    sub = io.build_submission(predictions, test_sequences)
    if sample_submission is not None:
        sub = io.order_submission_like(sub, sample_submission)
    io.validate_submission(sub, test_sequences)
    return sub
