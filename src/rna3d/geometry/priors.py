"""Estimate generic RNA C1' geometry priors from (temporal-safe) train labels.

Priors produced (all data-driven, no hard-coded constants):
  - adjacent C1'-C1' distance: mean / std / robust median+IQR   (backbone term)
  - clash radius r_min: low percentile of non-adjacent pair distances (clash term)
  - radius of gyration vs length: power-law fit Rg = a * L^b + per-bin medians (Rg term)

Temporal safety: callers pass only chains whose source structure was released
before the evaluation cutoff, so priors used on CASP15 validation never peek at
post-2022 structures. The priors are physical/statistical, not target-specific.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import io


def _chain_coords(labels: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """target_id -> (resid array, (L,3) coords with NaN for unresolved)."""
    tid = labels["ID"].map(io.target_id_of)
    out = {}
    for t, sub in labels.assign(_tid=tid).groupby("_tid"):
        sub = sub.sort_values("resid")
        xyz = sub[["x_1", "y_1", "z_1"]].to_numpy(float).copy()
        xyz[xyz[:, 0] <= io.RESOLVED_THRESHOLD] = np.nan
        out[t] = (sub["resid"].to_numpy(int), xyz)
    return out


def compute_priors(labels: pd.DataFrame, clash_pct: float = 0.5,
                   max_pairs_per_chain: int = 5000,
                   rng_seed: int = 0) -> dict:
    rng = np.random.default_rng(rng_seed)
    chains = _chain_coords(labels)

    adj_d: list[float] = []
    nn_d: list[float] = []  # nearest non-adjacent neighbour distance per residue
    rg_pairs: list[tuple[int, float]] = []

    for resids, xyz in chains.values():
        finite = np.isfinite(xyz).all(axis=1)
        n = len(xyz)

        # adjacent distances: consecutive residues that are sequence-adjacent and both resolved
        for i in range(n - 1):
            if finite[i] and finite[i + 1] and resids[i + 1] - resids[i] == 1:
                adj_d.append(float(np.linalg.norm(xyz[i + 1] - xyz[i])))

        idx = np.where(finite)[0]
        if len(idx) >= 3:
            pts = xyz[idx]
            rids = resids[idx]
            # radius of gyration for this chain
            c = pts.mean(axis=0)
            rg = float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean()))
            rg_pairs.append((len(idx), rg))

            # For clash radius: nearest *non-adjacent* (|resid diff| >= 2) neighbour
            # of each residue. This captures how close residues legitimately pack,
            # which is the meaningful floor for a steric-clash penalty.
            if len(idx) > 1200:  # bound the O(m^2) matrix for very long chains
                sel = rng.choice(len(idx), size=1200, replace=False)
                pts, rids = pts[sel], rids[sel]
            D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            sep = np.abs(rids[:, None] - rids[None, :])
            D[sep < 2] = np.inf
            nn = D.min(axis=1)
            nn = nn[np.isfinite(nn)]
            nn_d.extend(nn.tolist())

    adj = np.array(adj_d)
    nn = np.array(nn_d)
    # robust trim of adjacent distances (drop chain breaks mislabeled as adjacent)
    lo, hi = np.percentile(adj, [1, 99])
    adj_trim = adj[(adj >= lo) & (adj <= hi)]

    # Clash radius: a low percentile of the nearest non-adjacent neighbour distance,
    # floored at a physical C1'-C1' van der Waals minimum so rare data artifacts
    # (overlapping atoms at sub-Angstrom distance) cannot drive it to zero.
    PHYS_FLOOR = 4.0
    nn_phys = nn[nn >= PHYS_FLOOR]
    r_min = float(max(PHYS_FLOOR, np.percentile(nn_phys, 1)))

    rg_arr = np.array(rg_pairs, dtype=float)  # (N, 2): length, rg
    # power-law fit: log Rg = log a + b log L
    mask = (rg_arr[:, 0] >= 5) & (rg_arr[:, 1] > 0)
    logL = np.log(rg_arr[mask, 0])
    logRg = np.log(rg_arr[mask, 1])
    b, loga = np.polyfit(logL, logRg, 1)
    a = float(np.exp(loga))

    # per-length-bin medians for a robust lookup table
    bins = [0, 20, 40, 60, 80, 120, 160, 220, 300, 400, 100000]
    bin_med = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        sel = (rg_arr[:, 0] >= lo_b) & (rg_arr[:, 0] < hi_b)
        if sel.sum() >= 3:
            bin_med.append([int(lo_b), int(hi_b), float(np.median(rg_arr[sel, 1])), int(sel.sum())])

    return {
        "n_chains": len(chains),
        "adjacent_c1": {
            "mean": float(adj_trim.mean()),
            "std": float(adj_trim.std()),
            "median": float(np.median(adj_trim)),
            "p05": float(np.percentile(adj, 5)),
            "p95": float(np.percentile(adj, 95)),
            "n": int(len(adj_trim)),
        },
        "clash": {
            "r_min": r_min,
            "phys_floor": PHYS_FLOOR,
            "nn_p01": float(np.percentile(nn, 1)),
            "nn_p05": float(np.percentile(nn, 5)),
            "nn_median": float(np.median(nn)),
            "n_residues": int(len(nn)),
            "note": "r_min = max(phys_floor, 1st-pct of nearest non-adjacent C1' neighbour)",
        },
        "rg_powerlaw": {"a": a, "b": float(b), "form": "Rg = a * L**b (Angstrom)"},
        "rg_bins": bin_med,
        "_raw": {"adj": adj, "rg": rg_arr},  # for plotting; not serialised to JSON
    }
