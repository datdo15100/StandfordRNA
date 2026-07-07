"""Phase 2 — estimate temporal-safe RNA C1' geometry priors.

Uses only train chains whose temporal_cutoff < the CASP15 safe cutoff, so the
priors are valid to apply on the CASP15 validation targets.

Outputs:
    data/processed/geometry_priors.json
    reports/figures/adjacent_distance_distribution.png
    reports/figures/rg_by_length.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rna3d.data import io
from rna3d.geometry.priors import compute_priors
from rna3d.paths import casp15_safe_cutoff, figures, processed


def main(use_v2: bool = True):
    cutoff = casp15_safe_cutoff()

    seqs = io.load_sequences("train_v2" if use_v2 else "train")
    labels = io.load_labels("train_v2" if use_v2 else "train")

    # temporal-safe filter: keep only chains released strictly before the cutoff
    safe_ids = set(seqs.loc[seqs["temporal_cutoff"] < cutoff, "target_id"])
    tid = labels["ID"].map(io.target_id_of)
    labels_safe = labels[tid.isin(safe_ids)].copy()
    n_total = labels["ID"].map(io.target_id_of).nunique()
    n_safe = labels_safe["ID"].map(io.target_id_of).nunique()
    print(f"temporal-safe chains: {n_safe} / {n_total} (cutoff < {cutoff})")

    priors = compute_priors(labels_safe)
    raw = priors.pop("_raw")

    out = processed() / "geometry_priors.json"
    meta = {"source": "train_v2" if use_v2 else "train", "cutoff": cutoff,
            "n_safe_chains": n_safe}
    with open(out, "w") as fh:
        json.dump({"_meta": meta, **priors}, fh, indent=2)
    print(f"\nwrote {out}")
    print(json.dumps({k: v for k, v in priors.items() if k != "rg_bins"}, indent=2))
    print("rg_bins (lo, hi, median_rg, n):")
    for row in priors["rg_bins"]:
        print("  ", row)

    # --- plots ---
    adj = raw["adj"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adj[(adj > 2) & (adj < 12)], bins=120, color="steelblue")
    ax.axvline(priors["adjacent_c1"]["mean"], color="red",
               label=f"mean={priors['adjacent_c1']['mean']:.2f}A")
    ax.set(xlabel="Adjacent C1'-C1' distance (A)", ylabel="count",
           title="Adjacent C1'-C1' distance (temporal-safe train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures() / "adjacent_distance_distribution.png", dpi=130)

    rg = raw["rg"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rg[:, 0], rg[:, 1], s=4, alpha=0.3)
    L = np.linspace(5, rg[:, 0].max(), 200)
    a, b = priors["rg_powerlaw"]["a"], priors["rg_powerlaw"]["b"]
    ax.plot(L, a * L ** b, "r-", label=f"Rg={a:.2f}*L^{b:.2f}")
    ax.set(xlabel="chain length (resolved)", ylabel="radius of gyration (A)",
           title="Rg vs length (temporal-safe train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures() / "rg_by_length.png", dpi=130)
    print(f"wrote plots to {figures()}")


if __name__ == "__main__":
    main()
