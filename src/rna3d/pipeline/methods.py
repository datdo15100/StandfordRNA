"""Assemble the five C1' structures per target for each evaluated method.

Methods (used as ablation baselines on the CASP15 validation set):
    B0  dummy            extended chain (floor)
    B1  tbm_top1         best template, gap-filled, 5 copies
    B2  tbm_top5         five distinct templates, gap-filled
    B4  tbm_refined      five distinct templates + geometry refinement
    ablations of B4: no-clash, no-rg, no-gap-weights (uniform template weights)

Every method returns an array of shape (5, L, 3) with no NaN.
"""
from __future__ import annotations

import numpy as np

from ..geometry.denovo import de_novo_ensemble
from ..geometry.transforms import extended_chain
from ..refine.optimizer import RefineConfig, refine_structure
from ..refine.rule_based import refine_rule_based
from .tbm import Candidate


def _pad_to_five(structs: list[np.ndarray], L: int, rng: np.random.Generator) -> np.ndarray:
    while len(structs) < 5:
        if structs:
            base = structs[len(structs) % max(len(structs), 1)]
            structs.append(base + rng.normal(scale=1.5, size=base.shape))
        else:
            structs.append(extended_chain(L, rng=rng))
    return np.stack(structs[:5]).astype(float)


def m_dummy(L: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack([extended_chain(L, rng=rng) for _ in range(5)])


def m_tbm_top1(cands: list[Candidate], L: int, rng: np.random.Generator) -> np.ndarray:
    if not cands:
        return m_dummy(L, rng)
    x = cands[0].coords
    return np.stack([x for _ in range(5)]).astype(float)


def m_tbm_top5(cands: list[Candidate], L: int, rng: np.random.Generator) -> np.ndarray:
    structs = [c.coords for c in cands[:5]]
    return _pad_to_five(structs, L, rng)


def m_tbm_refined(cands: list[Candidate], L: int, priors: dict,
                  rng: np.random.Generator, cfg: RefineConfig | None = None,
                  gap_aware: bool = True, use_clash: bool = True,
                  use_rg: bool = True) -> np.ndarray:
    if not cands:
        # refine a dummy extended chain (still benefits from backbone/clash terms)
        cfg2 = cfg or RefineConfig()
        x, _ = refine_structure(extended_chain(L, rng=rng), priors,
                                template_confidence=0.0, cfg=cfg2)
        return _pad_to_five([x], L, rng)

    cfg = cfg or RefineConfig()
    cfg = RefineConfig(steps=cfg.steps, lr=cfg.lr, w_tpl=cfg.w_tpl, w_bb=cfg.w_bb,
                       w_clash=cfg.w_clash if use_clash else 0.0,
                       w_rg=cfg.w_rg if use_rg else 0.0,
                       w_dist=cfg.w_dist, sep_clash=cfg.sep_clash, sep_dist=cfg.sep_dist)
    structs = []
    for i, c in enumerate(cands[:5]):
        conf = c.conf_residue if gap_aware else np.ones_like(c.conf_residue)
        x, _ = refine_structure(
            c.coords, priors, template_coords=c.coords, conf_residue=conf,
            template_confidence=c.confidence, cfg=cfg, seed=i,
        )
        structs.append(x)
    return _pad_to_five(structs, L, rng)


# --------------------------------------------------------------------------- #
# De novo fallback + refiner comparison (adapted from the 1st-place TBM notebook)
# --------------------------------------------------------------------------- #
def m_denovo(seq: str, rng: np.random.Generator, base_seed: int = 0) -> np.ndarray:
    """Five diverse sequence-only de novo folds (no template, no refinement)."""
    return np.stack(de_novo_ensemble(seq, n=5, base_seed=base_seed)).astype(float)


def _candidate_pool(cands: list[Candidate], seq: str, base_seed: int, n: int = 5):
    """Up to `n` (coords, template_coords, conf_residue, confidence) tuples.

    Uses TBM / composite candidates first, then fills any remaining slots with de
    novo folds (template_coords=None) as a best-of-5 hedge — so weak-template targets
    still carry both a real-fold copy and a de novo alternative.
    """
    pool = []
    for c in cands[:n]:
        pool.append((c.coords, c.coords, c.conf_residue, c.confidence))
    if len(pool) < n:
        for x in de_novo_ensemble(seq, n=n - len(pool), base_seed=base_seed):
            pool.append((x, None, None, 0.1))  # no template to anchor to
    return pool


def m_tbm_none(cands: list[Candidate], seq: str, L: int,
               rng: np.random.Generator) -> np.ndarray:
    """TBM (or de novo fallback) candidate pool with NO refinement (raw gap-filled)."""
    pool = _candidate_pool(cands, seq, base_seed=0)
    return _pad_to_five([x0 for x0, _t, _c, _tc in pool], L, rng)


def m_tbm_grad(cands: list[Candidate], seq: str, L: int, priors: dict,
               rng: np.random.Generator, cfg: RefineConfig | None = None) -> np.ndarray:
    """TBM (or de novo fallback) + our gradient geometry-energy refinement."""
    cfg = cfg or RefineConfig()
    pool = _candidate_pool(cands, seq, base_seed=0)
    structs = []
    for i, (x0, tpl, conf, tconf) in enumerate(pool):
        x, _ = refine_structure(x0, priors, template_coords=tpl, conf_residue=conf,
                                template_confidence=tconf, cfg=cfg, seed=i)
        structs.append(x)
    return _pad_to_five(structs, L, rng)


def m_tbm_rule(cands: list[Candidate], seq: str, L: int, priors: dict,
               rng: np.random.Generator) -> np.ndarray:
    """TBM (or de novo fallback) + the 1st-place rule-based nudging refinement."""
    pool = _candidate_pool(cands, seq, base_seed=0)
    structs = []
    for x0, _tpl, _conf, tconf in pool:
        structs.append(refine_rule_based(x0, seq, confidence=tconf))
    return _pad_to_five(structs, L, rng)
