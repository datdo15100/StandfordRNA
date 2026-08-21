#!/usr/bin/env python
"""Run the remaining preregistered V5 Geometry-global CASP15 cell.

This is a separate post-freeze CASP15 development analysis.  It never mutates
the frozen Raw Kaggle deployment or the original V5 refinement factorial.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_v5_casp15 as v5
from rna3d.data import io
from rna3d.eval.local_metrics import local_accuracy_metrics
from rna3d.geofuse.geometry_v2 import geometry_v2_metrics
from rna3d.geofuse.refine_v2 import GeometryV2Config, refine_structure_v2


NAME = "Geometry_global"
CACHE = REPO / "data" / "cache" / "v5_casp15_geometry_global"
OUT = REPO / "reports" / "thesis_v5" / "experiments" / "geometry_global_secondary"
CONFIG = GeometryV2Config(
    steps=300,
    lr=0.04,
    w_source=3.0,
    w_backbone=1.0,
    w_clash=0.3,
    w_rg=0.0,
    w_angle=0.30,
    w_torsion=0.15,
    w_kink=20.0,
    adaptive_strength=True,
    fixed_strength=1.0,
    context_mode="global",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(device: str = "cuda") -> Path:
    v5.verify_raw()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    priors_v1 = json.loads(v5.P0.read_text())
    priors_v2 = json.loads(v5.P1.read_text())
    targets = v5.read_targets()

    for number, target in enumerate(targets.itertuples(index=False), start=1):
        banks, allocation = v5._factorial_raw_banks(target.target_id, target.sequence)
        arrays: dict[str, np.ndarray] = {}
        info: dict[str, list[dict]] = {}
        memo: dict[str, tuple[np.ndarray, dict]] = {}
        for bank_name, candidates in banks.items():
            values: list[np.ndarray] = []
            records: list[dict] = []
            for index, candidate in enumerate(candidates):
                key = v5.array_sha256(np.asarray(candidate["coords"], np.float32))
                if key in memo:
                    value, details = memo[key]
                    details = {**details, "memoized": True}
                else:
                    try:
                        value, details = refine_structure_v2(
                            candidate["coords"],
                            target.sequence,
                            priors_v1,
                            priors_v2,
                            source_confidence=candidate["confidence"],
                            global_confidence=float(candidate["global_confidence"]),
                            cfg=CONFIG,
                            device=device,
                            seed=v5._stable_uint32(
                                v5.SEED, target.target_id, NAME, index
                            ),
                        )
                        value = np.asarray(value, dtype=np.float32)
                        if value.shape != candidate["coords"].shape or not np.isfinite(value).all():
                            raise FloatingPointError("invalid Geometry-global output")
                    except Exception as error:
                        value = np.asarray(candidate["coords"], dtype=np.float32).copy()
                        details = {"fallback_to_raw": True}
                        failures.append(
                            {
                                "target_id": target.target_id,
                                "bank": bank_name,
                                "candidate_index": index,
                                "candidate_id": candidate["candidate_id"],
                                "reason": f"{type(error).__name__}:{error}",
                            }
                        )
                    memo[key] = (value, details)
                values.append(value)
                records.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "source": candidate["source"],
                        "global_confidence": float(candidate["global_confidence"]),
                        **details,
                    }
                )
            arrays[bank_name] = np.asarray(values, dtype=np.float32)
            info[bank_name] = records

        output = CACHE / f"{target.target_id}.npz"
        np.savez_compressed(output, **arrays)
        metadata = output.with_suffix(".json")
        write_json(
            metadata,
            {
                "status": "V5_CASP15_GEOMETRY_GLOBAL_GENERATED",
                "scientific_role": "CASP15 development analysis; not frozen Kaggle method",
                "target_id": target.target_id,
                "config": asdict(CONFIG),
                "allocation": allocation,
                "settings_info": info,
                "array_hashes": {
                    key: v5.array_sha256(value) for key, value in arrays.items()
                },
            },
        )
        rows.append(
            {
                "target_id": target.target_id,
                "path": str(output.relative_to(REPO)),
                "sha256": sha256(output),
                "metadata_sha256": sha256(metadata),
            }
        )
        print(f"[{number:02d}/12 {target.target_id}] Geometry-global generated", flush=True)

    manifest = OUT / "generation_manifest.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.DataFrame(
        failures,
        columns=["target_id", "bank", "candidate_index", "candidate_id", "reason"],
    ).to_csv(OUT / "generation_failures.csv", index=False)
    write_json(
        OUT / "generation_receipt.json",
        {
            "status": "V5_CASP15_GEOMETRY_GLOBAL_FROZEN",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_role": "secondary CASP15 development analysis",
            "does_not_modify_frozen_kaggle_method": True,
            "raw_bank_freeze_sha256": sha256(v5.RAW_RECEIPT),
            "config": asdict(CONFIG),
            "target_n": len(rows),
            "failure_n": len(failures),
            "manifest_sha256": sha256(manifest),
            "generation_code_sha256": sha256(Path(__file__)),
        },
    )
    return manifest


def evaluate() -> None:
    manifest = pd.read_csv(OUT / "generation_manifest.csv", dtype=str)
    if len(manifest) != 12:
        raise RuntimeError("Geometry-global manifest is incomplete")
    for row in manifest.itertuples(index=False):
        path = REPO / row.path
        if sha256(path) != row.sha256 or sha256(path.with_suffix(".json")) != row.metadata_sha256:
            raise RuntimeError(f"Geometry-global artifact changed: {row.target_id}")

    targets = v5.read_targets()
    labels = io.load_labels("validation")
    labels = labels[labels["ID"].map(io.target_id_of).isin(set(targets["target_id"]))]
    priors_v1 = json.loads(v5.P0.read_text())
    priors_v2 = json.loads(v5.P1.read_text())
    scorer = v5._MetricScorer()
    locked = pd.read_csv(
        v5.OUT / "refinement_results" / "factorial_candidate_metrics.csv"
    )
    locked = locked[locked["setting"] == "Raw"].set_index(
        ["target_id", "bank", "candidate_index"]
    )
    candidate_rows: list[dict] = []
    bank_rows: list[dict] = []

    for number, target in enumerate(targets.itertuples(index=False), start=1):
        references = io.get_reference_coords(labels, target.target_id)
        with np.load(CACHE / f"{target.target_id}.npz", allow_pickle=False) as payload:
            for bank_name in v5.FACTORIAL_BANKS:
                bank = np.asarray(payload[bank_name], dtype=np.float32)
                bank_rows.append(
                    {
                        "target_id": target.target_id,
                        "sequence_cluster": target.sequence_cluster,
                        "length": len(target.sequence),
                        "bank": bank_name,
                        "setting": NAME,
                        "best5_tm": scorer.bank(
                            target.target_id, f"{bank_name}_{NAME}", bank, references, target.sequence
                        ),
                    }
                )
                for index, coords in enumerate(bank):
                    raw_row = locked.loc[(target.target_id, bank_name, index)]
                    ref_index = int(raw_row["raw_reference_index"])
                    reference = references[ref_index]
                    candidate_rows.append(
                        {
                            "target_id": target.target_id,
                            "sequence_cluster": target.sequence_cluster,
                            "length": len(target.sequence),
                            "bank": bank_name,
                            "candidate_index": index,
                            "setting": NAME,
                            "raw_reference_index": ref_index,
                            "raw_candidate_tm": float(raw_row["raw_candidate_tm"]),
                            "candidate_tm_same_reference": scorer.candidate(
                                target.target_id,
                                f"{bank_name}_{NAME}_{index}",
                                coords,
                                reference,
                                target.sequence,
                            ),
                            **local_accuracy_metrics(coords, reference, windows=(9, 15)),
                            **geometry_v2_metrics(
                                coords, target.sequence, priors_v1, priors_v2
                            ),
                        }
                    )
        print(f"[{number:02d}/12 {target.target_id}] Geometry-global scored", flush=True)

    candidates = pd.DataFrame(candidate_rows)
    banks = pd.DataFrame(bank_rows)
    candidates.to_csv(OUT / "candidate_metrics.csv", index=False)
    banks.to_csv(OUT / "bank_metrics.csv", index=False)
    target_metrics = (
        candidates.groupby(
            ["target_id", "sequence_cluster", "length", "bank", "setting"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )
    target_metrics.to_csv(OUT / "target_metrics.csv", index=False)
    candidates.groupby(["bank", "setting"], as_index=False).mean(
        numeric_only=True
    ).to_csv(OUT / "summary.csv", index=False)

    existing_target = pd.read_csv(
        v5.OUT / "refinement_results" / "factorial_target_metrics.csv"
    )
    existing_bank = pd.read_csv(
        v5.OUT / "refinement_results" / "factorial_bank_metrics.csv"
    )
    combined_target = pd.concat([existing_target, target_metrics], ignore_index=True)
    combined_bank = pd.concat([existing_bank, banks], ignore_index=True)
    directions = {
        "sw_rmsd_9": False,
        "sw_rmsd_15": False,
        "c1_rmsd": False,
        "c1_lddt": True,
        "candidate_tm_same_reference": True,
    }
    effects: dict[str, dict] = {}
    for bank_name in v5.FACTORIAL_BANKS:
        frame = combined_target[combined_target["bank"] == bank_name]
        for comparator in ("Raw", "Simple", "Geometry_historical"):
            for metric, higher_better in directions.items():
                wide = frame.pivot(
                    index=["target_id", "sequence_cluster"],
                    columns="setting",
                    values=metric,
                ).reset_index()
                delta = (
                    wide[NAME].to_numpy() - wide[comparator].to_numpy()
                    if higher_better
                    else wide[comparator].to_numpy() - wide[NAME].to_numpy()
                )
                effects[f"{bank_name}:{NAME}_vs_{comparator}:{metric}"] = {
                    **v5._bootstrap_effect(delta, wide["sequence_cluster"].to_numpy()),
                    "exact_sign_flip": v5._exact_sign_flip(
                        delta, wide["sequence_cluster"].to_numpy()
                    ),
                    "positive_favours": NAME,
                }
        frame = combined_bank[combined_bank["bank"] == bank_name]
        for comparator in ("Raw", "Simple", "Geometry_historical"):
            wide = frame.pivot(
                index=["target_id", "sequence_cluster"],
                columns="setting",
                values="best5_tm",
            ).reset_index()
            delta = wide[NAME].to_numpy() - wide[comparator].to_numpy()
            effects[f"{bank_name}:{NAME}_vs_{comparator}:best5_tm"] = {
                **v5._bootstrap_effect(delta, wide["sequence_cluster"].to_numpy()),
                "exact_sign_flip": v5._exact_sign_flip(
                    delta, wide["sequence_cluster"].to_numpy()
                ),
                "positive_favours": NAME,
            }

    write_json(OUT / "paired_effects.json", effects)
    pd.DataFrame(
        scorer.failures,
        columns=["target_id", "scope", "setting", "reason"],
    ).to_csv(OUT / "evaluation_failures.csv", index=False)
    write_json(
        OUT / "evaluation_receipt.json",
        {
            "status": "V5_CASP15_GEOMETRY_GLOBAL_EVALUATION_COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_role": "secondary CASP15 development analysis",
            "does_not_modify_frozen_kaggle_method": True,
            "target_n": 12,
            "cluster_n": int(targets["sequence_cluster"].nunique()),
            "evaluation_failure_n": len(scorer.failures),
            "candidate_metrics_sha256": sha256(OUT / "candidate_metrics.csv"),
            "bank_metrics_sha256": sha256(OUT / "bank_metrics.csv"),
            "evaluation_code_sha256": sha256(Path(__file__)),
        },
    )

    v3 = banks[banks["bank"] == "V3_3T2D"]
    print("\nV3 3T+2D Geometry-global bank TM:", float(v3["best5_tm"].mean()))
    for metric in ("sw_rmsd_9", "c1_lddt", "candidate_tm_same_reference", "best5_tm"):
        key = f"V3_3T2D:{NAME}_vs_Simple:{metric}"
        print(key, json.dumps(effects[key], sort_keys=True))


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    build(device)
    evaluate()
