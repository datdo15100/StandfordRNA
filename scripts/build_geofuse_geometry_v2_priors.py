#!/usr/bin/env python
"""Build temporal-safe empirical angle/torsion priors for GeoFuse Phase B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.geometry_v2 import estimate_geometry_v2_priors
from rna3d.paths import casp15_safe_cutoff, processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train_v2", choices=["train", "train_v2"])
    parser.add_argument("--bins", type=int, default=72)
    parser.add_argument(
        "--output",
        type=Path,
        default=processed() / "geofuse_geometry_v2_priors.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_geometry_v2_priors.md",
    )
    args = parser.parse_args()

    cutoff = casp15_safe_cutoff()
    sequences = io.load_sequences(args.split)
    labels = io.load_labels(args.split)
    safe_ids = set(sequences.loc[sequences["temporal_cutoff"] < cutoff, "target_id"])
    label_targets = labels["ID"].map(io.target_id_of)
    safe_labels = labels[label_targets.isin(safe_ids)].copy()
    started = time.time()
    priors = estimate_geometry_v2_priors(safe_labels, bins=args.bins)
    priors["_meta"] = {
        "source": args.split,
        "cutoff_rule": f"temporal_cutoff < {cutoff}",
        "n_safe_target_ids": len(safe_ids),
        "seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(priors, indent=2) + "\n")

    rows = []
    for context, distributions in priors["contexts"].items():
        rows.append(
            {
                "context": context,
                "angle_n": distributions["angle"]["n"],
                "angle_median_deg": distributions["angle"]["median"] * 180.0 / 3.141592653589793,
                "angle_p05_deg": distributions["angle"]["p05"] * 180.0 / 3.141592653589793,
                "angle_p95_deg": distributions["angle"]["p95"] * 180.0 / 3.141592653589793,
                "torsion_n": distributions["torsion"]["n"],
                "torsion_median_deg": distributions["torsion"]["median"]
                * 180.0
                / 3.141592653589793,
            }
        )
    import pandas as pd

    table = pd.DataFrame(rows)
    lines = [
        "# GeoFuse Geometry v2 temporal-safe priors",
        "",
        f"- Source: `{args.split}` with `temporal_cutoff < {cutoff}`",
        f"- Chains contributing local geometry: {priors['n_chains']}",
        f"- Mean pair-like residue fraction: {priors['mean_pair_like_fraction']:.4f}",
        f"- Histogram bins: {args.bins}",
        f"- Runtime: {priors['_meta']['seconds']:.1f} seconds",
        "",
        "`pair_like` is an inference-available structural proxy based on complementary "
        "bases and candidate C1' distance. It is not a native secondary-structure label.",
        "",
        table.round(3).to_markdown(index=False),
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(table.round(3).to_string(index=False))
    print(f"[priors] {args.output}")
    print(f"[report] {args.report}")


if __name__ == "__main__":
    main()
