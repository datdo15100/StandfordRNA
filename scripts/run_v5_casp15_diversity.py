#!/usr/bin/env python
"""Measure fixed-budget candidate diversity for V5 RQ2 without native labels."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rna3d.eval.usalign import tm_score, write_c1_pdb
import run_v5_casp15 as core


OUT = REPO / "reports" / "thesis_v5" / "experiments" / "raw_results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(coords: np.ndarray) -> str:
    return core.array_sha256(np.asarray(coords, dtype=np.float32))


def _target(record: dict) -> list[dict]:
    target_id = record["target_id"]
    sequence = record["sequence"]
    drfold = core._drfold_records(target_id, sequence)
    allocations = {"5T": (5, 0), "4T+1D": (4, 1), "3T+2D": (3, 2), "2T": (2, 0), "1T+1D": (1, 1), "2D": (0, 2)}
    banks = {}
    for tbm_base in ("J_SS_RAW", "V3_EXACT_RAW"):
        templates = core._template_records(target_id, tbm_base)
        for name, (template_n, drfold_n) in allocations.items():
            bank, realized = core.allocate_bank(
                templates, drfold, template_n, drfold_n
            )
            banks[(tbm_base, name)] = (bank, realized)

    unique = {}
    for bank, _ in banks.values():
        for candidate in bank:
            unique.setdefault(_digest(candidate["coords"]), candidate["coords"])
    pair_cache = {}
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        paths = {}
        for index, (digest, coords) in enumerate(unique.items()):
            path = directory / f"candidate_{index}.pdb"
            write_c1_pdb(coords, list(sequence), path)
            paths[digest] = path

        rows = []
        for (tbm_base, allocation), (bank, realized) in banks.items():
            digests = [_digest(candidate["coords"]) for candidate in bank]
            values = []
            for left in range(len(digests)):
                for right in range(left + 1, len(digests)):
                    key = (digests[left], digests[right])
                    if key not in pair_cache:
                        pair_cache[key] = float(
                            tm_score(paths[key[0]], paths[key[1]])
                        )
                    values.append(pair_cache[key])
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "target_id": target_id,
                    "sequence_cluster": record["sequence_cluster"],
                    "length": len(sequence),
                    "tbm_base": tbm_base,
                    "allocation": allocation,
                    "candidate_n": len(bank),
                    "mean_pairwise_self_tm": float(array.mean()) if len(array) else float("nan"),
                    "near_duplicate_pair_fraction_tm_ge_0_95": float(np.mean(array >= 0.95)) if len(array) else float("nan"),
                    "pair_n": len(array),
                    **realized,
                }
            )
    return rows


def main() -> None:
    core.verify_raw()
    targets = core.read_targets()
    rows = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_target, record): record["target_id"]
            for record in targets.to_dict("records")
        }
        for index, future in enumerate(as_completed(futures), start=1):
            value = future.result()
            rows.extend(value)
            print(f"[{index:02d}/12 {value[0]['target_id']}] diversity scored", flush=True)
    frame = pd.DataFrame(rows).sort_values(["tbm_base", "allocation", "target_id"])
    path = OUT / "rq2_candidate_diversity.csv"
    frame.to_csv(path, index=False)
    summary = frame.groupby(["tbm_base", "allocation"], as_index=False).agg(
        target_n=("target_id", "count"),
        mean_pairwise_self_tm=("mean_pairwise_self_tm", "mean"),
        near_duplicate_pair_fraction_tm_ge_0_95=("near_duplicate_pair_fraction_tm_ge_0_95", "mean"),
        mean_realized_T=("realized_T", "mean"),
        mean_realized_D=("realized_D", "mean"),
    )
    summary.to_csv(OUT / "rq2_candidate_diversity_summary.csv", index=False)
    receipt = {
        "status": "V5_CASP15_RQ2_DIVERSITY_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "native_labels_loaded": False,
        "target_n": 12,
        "sequence_cluster_n": targets["sequence_cluster"].nunique(),
        "target_metrics_sha256": sha256(path),
        "summary_sha256": sha256(OUT / "rq2_candidate_diversity_summary.csv"),
        "code_sha256": sha256(Path(__file__)),
    }
    (OUT / "rq2_candidate_diversity_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
