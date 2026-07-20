#!/usr/bin/env python
"""Resumable DRfold2 candidate generation for GeoFuse Phase A.

The upstream repository and weights remain outside Git.  ``cfg97`` reproduces
the pretrained branch chosen in the first-place hybrid notebook; ``official``
runs DRfold2's four-configuration ensemble.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io


CONFIGS = {
    "cfg97": ["cfg_97"],
    "official": ["cfg_95", "cfg_96", "cfg_97", "cfg_99"],
}


def apply_scipy_compatibility(repo: Path) -> list[Path]:
    """Patch removed SciPy ``iprint`` calls in the external checkout, if needed."""
    from scipy.optimize import fmin_l_bfgs_b

    if "iprint" in inspect.signature(fmin_l_bfgs_b).parameters:
        return []
    changed = []
    for relative in ("PotentialFold/Selection.py", "PotentialFold/Optimization.py"):
        path = repo / relative
        text = path.read_text()
        patched = re.sub(r",\s*iprint\s*=\s*10", "", text)
        if patched != text:
            path.write_text(patched)
            changed.append(path)
    return changed


def configured_driver(repo: Path, mode: str) -> Path:
    upstream = repo / "DRfold_infer.py"
    if not upstream.exists():
        raise FileNotFoundError(f"DRfold2 entry point not found: {upstream}")
    if mode == "official":
        return upstream
    text = upstream.read_text()
    replacement = f"dlexps = {CONFIGS[mode]!r}"
    patched, count = re.subn(r"^dlexps\s*=\s*\[[^\n]*\]\s*$", replacement, text, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError("could not locate upstream dlexps assignment; DRfold2 layout changed")
    # The upstream driver invokes child scripts via the bare word ``python``.
    # That can escape the active conda environment even when this runner itself
    # was launched with the right interpreter.
    patched, python_count = re.subn(
        r"cmd = f'python \{", "cmd = f'{sys.executable} {", patched
    )
    if python_count < 3:
        raise RuntimeError("could not make upstream child Python calls environment-safe")
    patched = patched.replace(
        "#print(output,error)",
        "print(output.decode(errors='replace') if isinstance(output, bytes) else output)",
    )
    target = repo / f"DRfold_infer_geofuse_{mode}.py"
    if not target.exists() or target.read_text() != patched:
        target.write_text(patched)
    return target


def configured_e2e_driver(repo: Path, mode: str) -> Path:
    """Create a tiny upstream-only driver that stops before CPU optimization."""
    target = repo / f"DRfold_e2e_geofuse_{mode}.py"
    source = f'''import subprocess
import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parent
fasta = Path(sys.argv[1]).resolve()
outdir = Path(sys.argv[2]).resolve()
ret_dir = outdir / "rets_dir"
ret_dir.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
for config in {CONFIGS[mode]!r}:
    command = [
        sys.executable,
        str(repo / config / "test_modeldir.py"),
        device,
        str(fasta),
        str(ret_dir / (config + "_")),
        str(repo / "model_hub" / config),
    ]
    print("[e2e]", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)
(ret_dir / "done").write_text("1")
'''
    if not target.exists() or target.read_text() != source:
        target.write_text(source)
    return target


def validate_install(repo: Path, mode: str) -> None:
    missing = []
    for config in CONFIGS[mode]:
        model_dir = repo / "model_hub" / config
        if not model_dir.exists() or not any(model_dir.iterdir()):
            missing.append(str(model_dir))
    arena = repo / "Arena" / "Arena"
    if not arena.exists():
        missing.append(str(arena))
    if missing:
        raise FileNotFoundError(
            "DRfold2 is not fully installed. Missing:\n- "
            + "\n- ".join(missing)
            + f"\nRun `cd {repo} && bash install.sh` first."
        )


def expected_ret_count(repo: Path, mode: str) -> int:
    return sum(
        1
        for config in CONFIGS[mode]
        for path in (repo / "model_hub" / config).iterdir()
        if path.is_file() and "model" in path.name and "opt" not in path.name
    )


def select_sequences(args: argparse.Namespace):
    sequences = io.load_sequences(args.split)
    if args.target_ids:
        requested = {item.strip() for item in args.target_ids.split(",") if item.strip()}
        missing = requested - set(sequences["target_id"])
        if missing:
            raise KeyError(f"unknown target IDs: {sorted(missing)}")
        sequences = sequences[sequences["target_id"].isin(requested)]
    sequences = sequences[
        (sequences["seq_len"] >= args.min_len) & (sequences["seq_len"] <= args.max_len)
    ].sort_values(["seq_len", "target_id"])
    if args.limit:
        sequences = sequences.head(args.limit)
    return sequences


def export_safe_sidecars(target_dir: Path) -> dict:
    """Convert trusted local DRfold ret pickles to safe confidence/prior NPZs."""
    ret_paths = sorted((target_dir / "rets_dir").glob("*.ret"))
    plddt_sum = None
    prior_sums: dict[str, np.ndarray] = {}
    count = 0
    for path in ret_paths:
        # Security boundary: only call this for outputs generated by the trusted
        # official DRfold2 checkout, never for downloaded/untrusted .ret files.
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        plddt = np.asarray(payload["plddt"], dtype=np.float32)
        plddt_sum = plddt if plddt_sum is None else plddt_sum + plddt
        for key in ("dist_p", "dist_c", "dist_n"):
            if key in payload:
                value = np.asarray(payload[key], dtype=np.float32)
                prior_sums[key] = value if key not in prior_sums else prior_sums[key] + value
        count += 1
    if count == 0 or plddt_sum is None:
        return {"ret_files": 0, "sidecars": 0}

    pair_confidence = plddt_sum / count
    if pair_confidence.ndim == 2:
        residue_confidence = 0.5 * (pair_confidence.mean(0) + pair_confidence.mean(1))
    elif pair_confidence.ndim == 1:
        residue_confidence = pair_confidence
    else:
        raise ValueError(f"unexpected DRfold pLDDT shape: {pair_confidence.shape}")
    residue_confidence = np.clip(residue_confidence, 0.0, 1.0).astype(np.float32)
    priors = {key: (value / count).astype(np.float16) for key, value in prior_sums.items()}

    models = sorted((target_dir / "relax").glob("model_*.pdb"))
    for model in models:
        np.savez_compressed(model.parent / f"plddt_{model.stem}.npz", plddt=residue_confidence)
        if priors:
            np.savez_compressed(model.parent / f"priors_{model.stem}.npz", **priors)
    return {
        "ret_files": count,
        "sidecars": len(models),
        "mean_confidence": float(residue_confidence.mean()),
        "prior_keys": sorted(priors),
    }


def _residue_confidence(pair_or_residue: np.ndarray) -> np.ndarray:
    confidence = np.asarray(pair_or_residue, dtype=np.float32)
    if confidence.ndim == 2:
        confidence = 0.5 * (confidence.mean(0) + confidence.mean(1))
    elif confidence.ndim != 1:
        raise ValueError(f"unexpected DRfold pLDDT shape: {confidence.shape}")
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def export_e2e_candidates(target_dir: Path, arena: Path, limit: int) -> dict:
    """Turn the best direct checkpoint outputs into all-atom candidates.

    This deliberately stops before DRfold's expensive potential optimization.
    The resulting source is labelled ``e2e`` downstream and must not be confused
    with the official optimized DRfold structure.
    """
    if limit <= 0:
        return {"e2e_models": 0}
    records = []
    for ret_path in sorted((target_dir / "rets_dir").glob("*.ret")):
        with open(ret_path, "rb") as handle:
            payload = pickle.load(handle)
        confidence = _residue_confidence(payload["plddt"])
        records.append((float(confidence.mean()), ret_path))
        del payload
    records.sort(key=lambda item: (-item[0], item[1].name))
    selected = records[:limit]
    output_dir = target_dir / "e2e_relax"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, (score, ret_path) in enumerate(selected, start=1):
        with open(ret_path, "rb") as handle:
            payload = pickle.load(handle)
        confidence = _residue_confidence(payload["plddt"])
        raw_pdb = ret_path.with_suffix(".pdb")
        output_pdb = output_dir / f"model_{rank}.pdb"
        subprocess.run(
            [str(arena), str(raw_pdb), str(output_pdb), "7"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        np.savez_compressed(output_dir / f"plddt_model_{rank}.npz", plddt=confidence)
        priors = {
            key: np.asarray(payload[key], dtype=np.float16)
            for key in ("dist_p", "dist_c", "dist_n")
            if key in payload
        }
        if priors:
            np.savez_compressed(output_dir / f"priors_model_{rank}.npz", **priors)
        manifest.append(
            {"rank": rank, "checkpoint_ret": ret_path.name, "global_confidence": score}
        )
        del payload
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "e2e_models": len(selected),
        "e2e_best_confidence": selected[0][0] if selected else None,
    }


def write_status(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_target(
    *,
    driver: Path,
    python: Path,
    repo: Path,
    output_root: Path,
    target_id: str,
    sequence: str,
    cluster: bool,
    timeout_seconds: float,
    expected_rets: int,
    e2e_only: bool,
    e2e_candidates: int,
    force: bool,
) -> dict:
    target_dir = output_root / target_id
    complete_models = sorted((target_dir / "relax").glob("model_*.pdb"))
    existing_rets = sorted((target_dir / "rets_dir").glob("*.ret"))
    if e2e_only and len(existing_rets) == expected_rets and not force:
        exported = export_e2e_candidates(target_dir, repo / "Arena" / "Arena", e2e_candidates)
        return {"status": "cached_e2e", "ret_files": len(existing_rets), **exported}
    if complete_models and not force:
        sidecars = export_safe_sidecars(target_dir)
        exported = export_e2e_candidates(target_dir, repo / "Arena" / "Arena", e2e_candidates)
        return {"status": "cached", "models": len(complete_models), **sidecars, **exported}
    # An incomplete upstream run may still leave rets_dir/done, which would make
    # DRfold silently skip neural inference on retry.  Only remove the one target
    # directory under the explicitly supplied derived-output root.
    if target_dir.exists():
        if target_dir.resolve().parent != output_root.resolve():
            raise RuntimeError(f"refusing to clean unsafe target path: {target_dir}")
        reusable_rets = list((target_dir / "rets_dir").glob("*.ret"))
        done_marker = target_dir / "rets_dir" / "done"
        if not force and done_marker.exists() and len(reusable_rets) == expected_rets:
            # Neural inference is the expensive part. A failure in downstream
            # SciPy optimization can resume from complete .ret files safely.
            for derived in (target_dir / "folds", target_dir / "relax"):
                if derived.exists():
                    shutil.rmtree(derived)
        else:
            shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    fasta_dir = output_root / "_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    fasta = fasta_dir / f"{target_id}.fasta"
    fasta.write_text(f">{target_id}\n{sequence}\n")
    log_path = target_dir / "drfold2.log"
    command = [str(python), str(driver), str(fasta), str(target_dir)]
    if cluster:
        command.append("1")
    started = time.time()
    print(
        f"[{target_id}] start L={len(sequence)} mode={driver.stem} "
        f"timeout={timeout_seconds / 60:.0f}m",
        flush=True,
    )
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {
                "status": "timeout",
                "seconds": round(time.time() - started, 1),
                "log": str(log_path),
            }
    models = sorted((target_dir / "relax").glob("model_*.pdb"))
    generated_rets = sorted((target_dir / "rets_dir").glob("*.ret"))
    success = len(generated_rets) == expected_rets if e2e_only else bool(models)
    if returncode != 0 or not success:
        tail = ""
        if log_path.exists():
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        return {
            "status": "failed",
            "returncode": returncode,
            "seconds": round(time.time() - started, 1),
            "log": str(log_path),
            "log_tail": tail,
        }
    if e2e_only:
        exported = export_e2e_candidates(target_dir, repo / "Arena" / "Arena", e2e_candidates)
        return {
            "status": "complete_e2e",
            "ret_files": len(generated_rets),
            "seconds": round(time.time() - started, 1),
            "log": str(log_path),
            **exported,
        }
    sidecars = export_safe_sidecars(target_dir)
    exported = export_e2e_candidates(target_dir, repo / "Arena" / "Arena", e2e_candidates)
    return {
        "status": "complete",
        "models": len(models),
        "seconds": round(time.time() - started, 1),
        "log": str(log_path),
        **sidecars,
        **exported,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="installed official DRfold2 repo")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "train_v2", "validation", "test"])
    parser.add_argument("--mode", choices=sorted(CONFIGS), default="cfg97")
    parser.add_argument("--target-ids", help="comma-separated target IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-len", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument("--cluster", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--e2e-only",
        action="store_true",
        help="export direct checkpoint candidates and skip PotentialFold CPU optimization",
    )
    parser.add_argument(
        "--e2e-candidates", type=int, default=5, help="top confidence direct candidates to export"
    )
    parser.add_argument("--timeout-minutes", type=float, default=120.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.repo = args.repo.resolve()
    args.output_root = args.output_root.resolve()
    validate_install(args.repo, args.mode)
    compatibility_changes = apply_scipy_compatibility(args.repo)
    if compatibility_changes:
        print("[compat] removed unsupported SciPy iprint from:")
        for path in compatibility_changes:
            print(f"  {path}")
    driver = (
        configured_e2e_driver(args.repo, args.mode)
        if args.e2e_only
        else configured_driver(args.repo, args.mode)
    )
    expected_rets = expected_ret_count(args.repo, args.mode)
    sequences = select_sequences(args)
    print(f"[plan] {len(sequences)} targets, shortest first, output={args.output_root}", flush=True)
    if args.dry_run:
        print(sequences[["target_id", "seq_len"]].to_string(index=False))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for _, row in sequences.iterrows():
        target_id = str(row["target_id"])
        result = run_target(
            driver=driver,
            python=args.python,
            repo=args.repo,
            output_root=args.output_root,
            target_id=target_id,
            sequence=str(row["sequence"]),
            cluster=args.cluster,
            timeout_seconds=args.timeout_minutes * 60,
            expected_rets=expected_rets,
            e2e_only=args.e2e_only,
            e2e_candidates=args.e2e_candidates,
            force=args.force,
        )
        result.update(
            {
                "length": int(row["seq_len"]),
                "mode": args.mode,
                "cluster": bool(args.cluster),
                "e2e_only": bool(args.e2e_only),
            }
        )
        status[target_id] = result
        write_status(status_path, status)
        print(f"[{target_id}] {result['status']}: {result}")
        if result["status"] in {"failed", "timeout"}:
            print("[stop] first failure retained for diagnosis; rerun resumes completed targets")
            break


if __name__ == "__main__":
    main()
