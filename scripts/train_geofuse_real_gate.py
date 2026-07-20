#!/usr/bin/env python
"""Train the GeoFuse residue gate on leakage-audited real predictor pairs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.geofuse.candidate import CandidateCache
from rna3d.geofuse.phase_d import FEATURE_NAMES, ConfidenceGate1D, GateConfig, load_gate_checkpoint
from rna3d.geofuse.real_oof import (
    audit_pretrained_oof,
    audit_template_oof,
    make_real_example,
    native_reference_map,
)
from rna3d.paths import cache, comp_file, processed
from train_geofuse_phase_d import (
    ExampleDataset,
    calibrate_threshold,
    collate_examples,
    evaluate,
    feature_normalization,
    load_priors,
    normalize_examples,
)


def _load_requested_labels(target_ids: set[str]) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(comp_file("train_labels_v2"), chunksize=250_000):
        target = chunk["ID"].str.rsplit("_", n=1).str[0]
        selected = chunk[target.isin(target_ids)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("no native labels found for ready OOF targets")
    return pd.concat(chunks, ignore_index=True)


def _candidate_pairs(row, store: CandidateCache, max_templates: int, max_pretrained: int):
    candidates = store.load_target(row.target_id, row.sequence)
    excluded = set(str(row.excluded_pdb_ids).split(";")) - {"", "nan"}
    templates = []
    pretrained = []
    for candidate in candidates:
        try:
            if candidate.kind == "template":
                audit_template_oof(candidate, row.date, excluded)
                templates.append(candidate)
            elif candidate.kind == "pretrained":
                audit_pretrained_oof(candidate, row.date)
                pretrained.append(candidate)
        except ValueError:
            continue
    # This choice is made without native coordinates and is identical in every split.
    templates.sort(key=lambda value: (-value.global_confidence, value.candidate_id))
    pretrained.sort(key=lambda value: (-value.global_confidence, value.candidate_id))
    return [
        (template, prediction)
        for template in templates[:max_templates]
        for prediction in pretrained[:max_pretrained]
    ]


def build_examples(args, manifest: pd.DataFrame, priors_v1: dict, priors_v2: dict):
    store = CandidateCache(Path(args.cache_root), "train_v2")
    available = []
    pairs_by_target = {}
    for row in manifest.itertuples(index=False):
        pairs = _candidate_pairs(row, store, args.max_templates, args.max_pretrained)
        if pairs:
            available.append(row.target_id)
            pairs_by_target[row.target_id] = pairs
    labels = _load_requested_labels(set(available))
    references = native_reference_map(labels, set(available))
    examples = {split: [] for split in ("train", "calibration", "validation")}
    failures = []
    for row in manifest.itertuples(index=False):
        for template, prediction in pairs_by_target.get(row.target_id, []):
            try:
                example = make_real_example(
                    template, prediction, references[row.target_id], priors_v1, priors_v2
                )
                example["split"] = row.split
                examples[row.split].append(example)
            except (ValueError, RuntimeError) as exc:
                failures.append((row.target_id, template.candidate_id, prediction.candidate_id, str(exc)))
    return examples, failures


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    manifest = pd.read_csv(args.manifest, dtype={"target_id": str})
    expected_splits = {"train", "calibration", "validation"}
    if set(manifest["split"]) != expected_splits:
        raise ValueError(f"manifest must contain all splits: {sorted(expected_splits)}")
    priors_v1, priors_v2 = load_priors()
    examples, failures = build_examples(args, manifest, priors_v1, priors_v2)
    counts = {name: len(values) for name, values in examples.items()}
    target_counts = {
        name: len({example["target_id"] for example in values}) for name, values in examples.items()
    }
    if min(counts.values()) == 0 or min(target_counts.values()) < args.min_targets_per_split:
        raise RuntimeError(
            f"real OOF data incomplete: pairs={counts}, targets={target_counts}; "
            "generate and import both audited source types before training"
        )
    mean, std = feature_normalization(examples["train"])
    for values in examples.values():
        normalize_examples(values, mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            ExampleDataset(values), batch_size=args.batch_size,
            shuffle=name == "train", generator=generator if name == "train" else None,
            collate_fn=collate_examples,
        )
        for name, values in examples.items()
    }
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.initialize_from:
        initial = load_gate_checkpoint(args.initialize_from)
        model = initial["model"].to(device)
        gate_config = GateConfig(**initial["gate_config"])
    else:
        gate_config = GateConfig(hidden_channels=args.hidden_channels)
        model = ConfidenceGate1D(cfg=gate_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features)
            loss_by_residue = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (loss_by_residue[mask] * weight[mask]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration = evaluate(model, loaders["calibration"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "calibration_loss": calibration["weighted_bce"],
            "calibration_auc": calibration["roc_auc"],
        }
        history.append(row)
        print(
            f"[epoch {epoch:02d}] train={row['train_loss']:.4f} "
            f"cal={row['calibration_loss']:.4f} auc={row['calibration_auc']:.4f}",
            flush=True,
        )
        if row["calibration_loss"] < best_loss:
            best_loss = row["calibration_loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    threshold, calibration_metrics = calibrate_threshold(model, loaders["calibration"], device)
    validation_metrics = evaluate(
        model, loaders["validation"], device, decision_threshold=threshold
    )
    validation_metrics["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    checkpoint = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_std": std,
        "gate_config": asdict(gate_config),
        "state_dict": best_state,
        "training": {
            "supervision": "real_oof_v1",
            "seed": args.seed,
            "decision_threshold": threshold,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "pair_counts": counts,
            "target_counts": target_counts,
            "history": history,
        },
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    write_report(
        Path(args.report), manifest, counts, target_counts, calibration_metrics,
        validation_metrics, history, failures, args, time.time() - started,
    )
    print(pd.Series(validation_metrics).round(5).to_string())
    print(f"[checkpoint] {checkpoint_path}")


def real_gate_passed(metrics: dict) -> bool:
    """Require improvement over both whole-source and residue-rule baselines."""
    return metrics["learned_gate_error"] < min(
        metrics["template_error"],
        metrics["pretrained_error"],
        metrics["gap_rule_error"],
        metrics["confidence_rule_error"],
    )


def write_report(path, manifest, counts, target_counts, calibration, validation, history, failures, args, seconds):
    passed = real_gate_passed(validation)
    lines = [
        "# GeoFuse real-OOF confidence gate",
        "",
        "This experiment uses actual temporal-safe TBM candidates and frozen pretrained-model "
        "predictions. Native coordinates are used only to create residue labels and metrics; "
        "candidate selection and all gate features are native-blind.",
        "",
        f"- Gate: **{'pass' if passed else 'fail'}**",
        f"- Runtime: {seconds:.1f} seconds",
        f"- Pair counts: `{counts}`",
        f"- Target counts: `{target_counts}`",
        f"- Rejected pair attempts: {len(failures)}",
        f"- Initialized from synthetic gate: `{bool(args.initialize_from)}`",
        "",
        "## Held-out newest-target metrics",
        "",
        pd.Series(validation, name="value").to_frame().round(5).to_markdown(),
        "",
        f"The decision threshold ({validation['decision_threshold']:.3f}) was selected only "
        f"on calibration data; its calibration gate error was {calibration['learned_gate_error']:.4f} Å.",
        "The pass criterion requires lower held-out error than always-template, "
        "always-pretrained, gap-rule, and confidence-rule baselines.",
        "",
        "## Training history",
        "",
        pd.DataFrame(history).round(5).to_markdown(index=False),
        "",
        "## Leakage boundary",
        "",
        "The manifest groups >=80% identity sequences when prepared with MMseqs. Every pretrained "
        "candidate must declare either a structural-training cutoff older than its target or an "
        "explicit exclusion manifest. Every TBM template must predate its target and direct target "
        "PDB IDs are rejected. Sequence-language-model pretraining overlap is not claimed absent.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(processed() / "geofuse_real_oof" / "manifest.csv"))
    parser.add_argument("--cache-root", default=str(cache() / "geofuse_candidates"))
    parser.add_argument("--initialize-from", default=str(processed() / "geofuse_confidence_gate_v1.pt"))
    parser.add_argument("--checkpoint", default=str(processed() / "geofuse_confidence_gate_real_oof_v1.pt"))
    parser.add_argument("--report", default=str(REPO_ROOT / "reports" / "thesis_notes" / "geofuse_real_oof_gate.md"))
    parser.add_argument("--max-templates", type=int, default=2)
    parser.add_argument("--max-pretrained", type=int, default=2)
    parser.add_argument("--min-targets-per-split", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
