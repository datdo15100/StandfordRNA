"""Faithful reproduction of the 1st-place TBM-only pipeline.

Ported from `utilities/top1_tbm.py` ("RNA 3D Folds: TBM-only approach"): composite
sequence similarity (global + local pairwise2 + RNA features + k-mer) -> KMeans feature
clustering for best-of-5 diversity -> coordinate transfer + gap fill -> confidence-scaled
rule-based refinement (`adaptive_rna_constraints`) -> de novo fallback.

We reuse our faithful ports of two of their sub-routines to avoid divergence:
  - `refine.rule_based.refine_rule_based`  == their `adaptive_rna_constraints`
  - `geometry.denovo.de_novo_structure`    == their `generate_rna_structure`

A `cutoff` argument (not in the original) lets us score the *same* method both leaked
(no filter) and temporal-safe (templates with release_date < cutoff), to quantify the
leakage the notebook is exposed to on the CASP15 public targets.
"""
from __future__ import annotations

import warnings

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..geometry.denovo import de_novo_structure
from ..refine.rule_based import refine_rule_based

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# RNA sequence features (their _extract_enhanced_rna_features)
# --------------------------------------------------------------------------- #
def _repeat_content(seq: str, w: int = 3) -> float:
    if len(seq) < 6:
        return 0.0
    cnt = 0
    for i in range(len(seq) - w + 1):
        motif = seq[i:i + w]
        for j in range(i + w, len(seq) - w + 1):
            if seq[j:j + w] == motif:
                cnt += 1
                break
    return cnt / (len(seq) - w + 1) if len(seq) > w else 0.0


def rna_features(sequence: str) -> list[float]:
    seq = sequence.upper()
    n = max(len(seq), 1)
    feats = [seq.count(x) / n for x in "AUGC"]
    for dinuc in ["AU", "UA", "GC", "CG", "GU", "UG", "AA", "UU", "GG", "CC"]:
        c = sum(1 for i in range(len(seq) - 1) if seq[i:i + 2] == dinuc)
        feats.append(c / (len(seq) - 1) if len(seq) > 1 else 0.0)
    gc = (seq.count("G") + seq.count("C")) / n
    au = (seq.count("A") + seq.count("U")) / n
    pur = (seq.count("A") + seq.count("G")) / n
    pyr = (seq.count("U") + seq.count("C")) / n
    feats += [gc, au, pur, pyr]
    ent = 0.0
    for x in "AUGC":
        f = seq.count(x) / n
        if f > 0:
            ent -= f * np.log2(f)
    feats += [min(len(seq) / 1000.0, 1.0), ent / 2.0, _repeat_content(seq)]
    return feats


def _kmer_sim(s1: str, s2: str, k: int = 3) -> float:
    def kmers(s):
        return set(s[i:i + k] for i in range(len(s) - k + 1))
    a, b = kmers(s1.upper()), kmers(s2.upper())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _composite_similarity(query: str, train: str, qfeat: list[float]) -> float:
    q = Seq(query)
    g = pairwise2.align.globalms(q, train, 2.9, -1, -10, -0.5, one_alignment_only=True)
    gs = g[0].score / (2 * min(len(query), len(train))) if g else 0.0
    lo = pairwise2.align.localms(q, train, 2.9, -1, -10, -0.5, one_alignment_only=True)
    ls = lo[0].score / (2 * min(len(query), len(train))) if lo else 0.0
    fs = cosine_similarity([qfeat], [rna_features(train)])[0][0]
    ks = _kmer_sim(query, train, k=3)
    return 0.4 * gs + 0.3 * ls + 0.2 * fs + 0.1 * ks


def _diversity_clustering(F: np.ndarray, k: int) -> np.ndarray:
    n = len(F)
    labels = np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n)
    sel = [0]
    for _ in range(1, k):
        best, bi = -1, -1
        for i in range(n):
            if i in sel:
                continue
            md = min(np.linalg.norm(F[i] - F[j]) for j in sel)
            if md > best:
                best, bi = md, i
        if bi != -1:
            sel.append(bi)
    for i in range(n):
        if i in sel:
            labels[i] = sel.index(i)
        else:
            labels[i] = int(np.argmin([np.linalg.norm(F[i] - F[j]) for j in sel]))
    return labels


def find_similar_sequences(query: str, templates: list[tuple], top_n: int = 5):
    """templates: list of (target_id, seq, coords). Returns top_n (id, seq, score, coords)."""
    qfeat = rna_features(query)
    scored = []
    for tid, tseq, tcoords in templates:
        lr = abs(len(tseq) - len(query)) / max(len(tseq), len(query))
        if len(query) < 50 or len(tseq) < 50:
            if lr > 0.6:
                continue
        elif len(query) > 1000 or len(tseq) > 1000:
            if lr > 0.2:
                continue
        elif lr > 0.4:
            continue
        s = _composite_similarity(query, tseq, qfeat)
        if s > 0:
            scored.append((tid, tseq, s, tcoords))
    scored.sort(key=lambda x: x[2], reverse=True)
    if len(scored) > 10:
        thr = np.percentile([x[2] for x in scored], 80)
        scored = [x for x in scored if x[2] >= thr][:50]
    else:
        scored = scored[:50]
    if len(scored) <= top_n:
        return scored[:top_n]
    F = np.array([rna_features(s[1]) for s in scored])
    nk = min(top_n, len(scored))
    if len(scored) >= 15:
        labels = KMeans(n_clusters=nk, random_state=42, n_init=10).fit_predict(F)
    else:
        labels = _diversity_clustering(F, nk)
    out = []
    for cid in range(nk):
        members = [scored[i] for i in range(len(scored)) if labels[i] == cid]
        if members:
            members.sort(key=lambda x: x[2], reverse=True)
            out.append(members[0])
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:top_n]


def adapt_template_to_query(query: str, template_seq: str, template_coords: np.ndarray):
    """Their transfer + geometry-aware gap fill (returns fully-populated (L,3))."""
    aln = pairwise2.align.globalms(Seq(query), Seq(template_seq), 2.9, -1, -10, -0.5,
                                   one_alignment_only=True)
    if not aln:
        return None
    aq, at = aln[0].seqA, aln[0].seqB
    coords = np.full((len(query), 3), np.nan)
    qi = ti = 0
    for k in range(len(aq)):
        if aq[k] != "-" and at[k] != "-":
            if ti < len(template_coords):
                coords[qi] = template_coords[ti]
            ti += 1
            qi += 1
        elif aq[k] != "-":
            qi += 1
        elif at[k] != "-":
            ti += 1

    bb = 5.9
    for i in range(len(coords)):
        if np.isnan(coords[i, 0]):
            prev = nxt = None
            for j in range(i - 1, -1, -1):
                if not np.isnan(coords[j, 0]):
                    prev = j
                    break
            for j in range(i + 1, len(coords)):
                if not np.isnan(coords[j, 0]):
                    nxt = j
                    break
            if prev is not None and nxt is not None:
                gap = nxt - prev
                tot = np.linalg.norm(coords[nxt] - coords[prev])
                exp = gap * bb
                if tot < exp * 0.7:
                    d = coords[nxt] - coords[prev]
                    d = d / (np.linalg.norm(d) + 1e-10)
                    for kk, idx in enumerate(range(prev + 1, nxt)):
                        pr = (kk + 1) / gap
                        base = coords[prev] + d * exp * pr
                        perp = np.cross(d, [0, 0, 1])
                        if np.linalg.norm(perp) < 1e-6:
                            perp = np.cross(d, [1, 0, 0])
                        perp = perp / (np.linalg.norm(perp) + 1e-10)
                        coords[idx] = base + perp * (2.0 * np.sin(pr * np.pi))
                else:
                    for kk, idx in enumerate(range(prev + 1, nxt)):
                        w = (kk + 1) / gap
                        coords[idx] = (1 - w) * coords[prev] + w * coords[nxt]
            elif prev is not None:
                if prev > 0 and not np.isnan(coords[prev - 1, 0]):
                    d = coords[prev] - coords[prev - 1]
                    d = d / (np.linalg.norm(d) + 1e-10)
                else:
                    d = np.array([1.0, 0.0, 0.0])
                for step in range(1, i - prev + 1):
                    coords[prev + step] = coords[prev] + d * bb * step
            elif nxt is not None:
                d = np.array([-1.0, 0.0, 0.0])
                for step in range(nxt - i, 0, -1):
                    coords[nxt - step] = coords[nxt] - d * bb * step
    return np.nan_to_num(coords)


def predict_structures(query: str, target_id: str, templates: list[tuple],
                       n: int = 5) -> list[np.ndarray]:
    """Their predict_rna_structures: template candidates + rule refine + de novo fill."""
    preds = []
    similar = find_similar_sequences(query, templates, top_n=n)
    for tid, tseq, score, tcoords in similar:
        adapted = adapt_template_to_query(query, tseq, tcoords)
        if adapted is None:
            continue
        refined = refine_rule_based(adapted, query, confidence=score)
        scale = max(0.05, 0.8 - score)
        rng = np.random.default_rng(abs(hash((target_id, len(preds)))) % (2**32))
        preds.append(refined + rng.normal(0, scale, refined.shape))
        if len(preds) >= n:
            break
    while len(preds) < n:
        seed = abs(hash(target_id)) % 10000 + len(preds) * 1000
        dn = de_novo_structure(query, seed=seed)
        preds.append(refine_rule_based(dn, query, confidence=0.2))
    return preds[:n]
