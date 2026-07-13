"""Create a compact, measured EDA snapshot for the supervisor handoff."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "stanford-rna-3d-folding"


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def dir_inventory(path: Path, pattern: str = "*") -> dict:
    files = [p for p in path.glob(pattern) if p.is_file()] if path.is_dir() else []
    return {
        "exists": path.is_dir(),
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }


def sequence_summary(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_csv(path)
    lengths = frame["sequence"].astype(str).str.len()
    return {
        "exists": True,
        "rows": int(len(frame)),
        "targets": int(frame["target_id"].nunique()),
        "residues": int(lengths.sum()),
        "min_len": int(lengths.min()),
        "median_len": float(lengths.median()),
        "mean_len": round(float(lengths.mean()), 2),
        "max_len": int(lengths.max()),
        "columns": list(frame.columns),
    }


def label_summary(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    frame = pd.read_csv(path)
    return {
        "exists": True,
        "rows": int(len(frame)),
        "coordinate_sets": sum(c.startswith("x_") for c in frame.columns),
        "columns": list(frame.columns),
    }


def main() -> None:
    sequence_names = [
        "train_sequences.csv",
        "train_sequences.v2.csv",
        "validation_sequences.csv",
        "test_sequences.csv",
    ]
    label_names = [
        "train_labels.csv",
        "train_labels.v2.csv",
        "validation_labels.csv",
    ]
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA),
        "sequences": {name: sequence_summary(DATA / name) for name in sequence_names},
        "labels": {name: label_summary(DATA / name) for name in label_names},
        "msa": dir_inventory(DATA / "MSA", "*.fasta"),
        "msa_v2": dir_inventory(DATA / "MSA_v2", "*.fasta"),
        "pdb_rna": dir_inventory(DATA / "PDB_RNA", "*.cif"),
    }

    validation = DATA / "validation_sequences.csv"
    test = DATA / "test_sequences.csv"
    if validation.is_file() and test.is_file():
        v = pd.read_csv(validation).sort_values("target_id").reset_index(drop=True)
        t = pd.read_csv(test).sort_values("target_id").reset_index(drop=True)
        common = [c for c in ("target_id", "sequence", "temporal_cutoff") if c in v and c in t]
        audit["validation_test_identical"] = bool(v[common].equals(t[common]))
    else:
        audit["validation_test_identical"] = None

    (HERE / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# Competition data audit",
        "",
        f"Generated: `{audit['generated_utc']}`",
        "",
        f"Data path: `{DATA}`",
        "",
        "## Sequence tables",
        "",
        "| file | targets | residues | min | median | mean | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in audit["sequences"].items():
        if item.get("exists"):
            lines.append(
                f"| {name} | {item['targets']:,} | {item['residues']:,} | "
                f"{item['min_len']} | {item['median_len']:g} | {item['mean_len']:g} | {item['max_len']} |"
            )
        else:
            lines.append(f"| {name} | _not extracted_ | | | | | |")

    lines.extend(
        [
            "",
            "## Label tables",
            "",
            "| file | residue rows | coordinate/reference sets |",
            "|---|---:|---:|",
        ]
    )
    for name, item in audit["labels"].items():
        if item.get("exists"):
            lines.append(f"| {name} | {item['rows']:,} | {item['coordinate_sets']} |")
        else:
            lines.append(f"| {name} | _not extracted_ | |")

    lines.extend(
        [
            "",
            "## Large-file inventory",
            "",
            "| component | files | size of matched files |",
            "|---|---:|---:|",
            f"| MSA/*.fasta | {audit['msa']['files']:,} | {human_bytes(audit['msa']['bytes'])} |",
            f"| MSA_v2/*.fasta | {audit['msa_v2']['files']:,} | {human_bytes(audit['msa_v2']['bytes'])} |",
            f"| PDB_RNA/*.cif | {audit['pdb_rna']['files']:,} | {human_bytes(audit['pdb_rna']['bytes'])} |",
            "",
            f"Validation and local test sequence tables identical: **{audit['validation_test_identical']}**.",
            "",
            "## Interpretation",
            "",
            "- The local test table is public CASP15 development data, not the hidden Kaggle private set.",
            "- Length variation makes memory/runtime strongly target-dependent; the 720-nt target is a stress case.",
            "- PDB_RNA is a search/template resource. Release dates must be filtered for temporal-safe local evaluation.",
            "- MSA depth varies substantially, so pretrained methods should retain a single-sequence/TBM fallback.",
            "",
            "Historical full-parse result recorded in the repository: 23,869 RNA/hybrid chains, "
            "10.86M residues, 99.9% modelled C1′, zero parser errors.",
        ]
    )
    (HERE / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(HERE / "data_audit.md")


if __name__ == "__main__":
    main()

