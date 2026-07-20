#!/usr/bin/env python
"""Train the GeoFuse tiny 1D confidence gate on temporal-safe synthetic corruptions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rna3d.data import io
from rna3d.geofuse.candidate import StructureCandidate
from rna3d.geofuse.phase_d import (
    FEATURE_NAMES,
    ConfidenceGate1D,
    GateConfig,
    pair_gate_features,
)
from rna3d.geometry.transforms import apply_rigid, random_rotation
from rna3d.paths import casp15_safe_cutoff, processed


def load_priors() -> tuple[dict, dict]:
    return (
        json.loads((processed() / "geometry_priors.json").read_text()),
        json.loads((processed() / "geofuse_geometry_v2_priors.json").read_text()),
    )


def _blocks(length: int, rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    result = []
    for _ in range(count):
        block_length = int(rng.integers(max(3, length // 20), max(4, length // 6) + 1))
        block_length = min(block_length, max(length - 4, 1))
        start = int(rng.integers(1, max(2, length - block_length)))
        result.append((start, min(start + block_length, length - 1)))
    return result


def synthetic_pair(
    native: np.ndarray,
    sequence: str,
    target_id: str,
    rng: np.random.Generator,
) -> tuple[StructureCandidate, StructureCandidate]:
    """Create one template-like and one pretrained-like corruption."""
    length = len(sequence)
    template_coords = np.asarray(native, dtype=float).copy()
    template_coords += rng.normal(0.0, 0.20, size=template_coords.shape)
    support = np.ones(length, dtype=bool)
    template_confidence = np.clip(rng.normal(0.9, 0.05, size=length), 0.6, 1.0)
    for start, end in _blocks(length, rng, int(rng.integers(1, 3))):
        support[start:end] = False
        bridge = np.linspace(template_coords[start - 1], template_coords[end], end - start + 2)
        template_coords[start:end] = bridge[1:-1]
        template_coords[start:end] += rng.normal(0.0, 0.5, size=(end - start, 3))
        template_confidence[start:end] = rng.uniform(0.03, 0.18, size=end - start)

    # A supported but locally misregistered segment prevents the support mask from
    # being a perfect label and makes calibration necessary.
    if rng.random() < 0.8:
        start, end = _blocks(length, rng, 1)[0]
        shift = rng.normal(size=3)
        shift *= rng.uniform(2.0, 7.0) / (np.linalg.norm(shift) + 1e-8)
        taper = np.sin(np.linspace(0.0, np.pi, end - start))[:, None]
        template_coords[start:end] += taper * shift
        template_confidence[start:end] *= rng.uniform(0.45, 0.8)
    template_confidence *= rng.uniform(0.55, 1.05)
    template_confidence = np.clip(template_confidence, 0.01, 1.0)

    raw_noise = rng.normal(size=native.shape)
    smooth_noise = gaussian_filter1d(raw_noise, sigma=2.0, axis=0, mode="nearest")
    smooth_noise *= rng.uniform(1.0, 2.5) / (np.std(smooth_noise) + 1e-8)
    pretrained_coords = np.asarray(native, dtype=float) + smooth_noise
    if rng.random() < 0.9:
        start, end = _blocks(length, rng, 1)[0]
        shift = rng.normal(size=3)
        shift *= rng.uniform(2.0, 8.0) / (np.linalg.norm(shift) + 1e-8)
        taper = np.sin(np.linspace(0.0, np.pi, end - start))[:, None]
        pretrained_coords[start:end] += taper * shift
    pretrained_error = np.linalg.norm(pretrained_coords - native, axis=1)
    pretrained_confidence = 0.1 + 0.8 * np.exp(-np.square(pretrained_error) / 18.0)
    pretrained_confidence += rng.normal(0.0, 0.08, size=length)
    # Model-side confidence scales are not comparable (Phase-B audit).  Randomize
    # the absolute scale so the gate must use within-source rank, support, local
    # geometry, and disagreement instead of memorizing a synthetic score range.
    pretrained_confidence *= np.exp(rng.uniform(np.log(0.08), np.log(1.2)))
    pretrained_confidence = np.clip(pretrained_confidence, 0.02, 0.98)

    rotation = random_rotation(rng)
    translation = rng.normal(0.0, 20.0, size=3)
    pretrained_coords = apply_rigid(pretrained_coords, rotation, translation)
    template = StructureCandidate(
        target_id=target_id,
        sequence=sequence,
        candidate_id=f"synthetic_template__{target_id}",
        kind="template",
        source="synthetic_tbm",
        model="corruption_v1",
        coords=template_coords,
        confidence=template_confidence,
        support_mask=support,
        global_confidence=float(template_confidence.mean() * support.mean()),
    )
    pretrained = StructureCandidate(
        target_id=target_id,
        sequence=sequence,
        candidate_id=f"synthetic_pretrained__{target_id}",
        kind="pretrained",
        source="synthetic_pretrained",
        model="corruption_v1",
        coords=pretrained_coords,
        confidence=pretrained_confidence,
        support_mask=np.ones(length, dtype=bool),
        global_confidence=float(pretrained_confidence.mean()),
    )
    return template, pretrained


def make_example(
    native: np.ndarray,
    sequence: str,
    target_id: str,
    rng: np.random.Generator,
    priors_v1: dict,
    priors_v2: dict,
) -> dict:
    template, pretrained = synthetic_pair(native, sequence, target_id, rng)
    features, aligned, alignment = pair_gate_features(
        template, pretrained, priors_v1, priors_v2
    )
    template_error = np.linalg.norm(template.coords - native, axis=1)
    pretrained_error = np.linalg.norm(aligned - native, axis=1)
    logit = np.clip((template_error - pretrained_error) / 1.5, -12.0, 12.0)
    target = 1.0 / (1.0 + np.exp(-logit))
    weight = 0.25 + np.clip(np.abs(template_error - pretrained_error) / 5.0, 0.0, 1.0)
    return {
        "target_id": target_id,
        "features": features.astype(np.float32),
        "target": target.astype(np.float32),
        "weight": weight.astype(np.float32),
        "template_error": template_error.astype(np.float32),
        "pretrained_error": pretrained_error.astype(np.float32),
        "gap_rule": (~template.support_mask).astype(np.float32),
        "confidence_rule": (pretrained.confidence > template.confidence).astype(np.float32),
        "alignment_rmsd": alignment["alignment_rmsd"],
    }


def load_complete_chains(args: argparse.Namespace) -> pd.DataFrame:
    sequences = io.load_sequences("train_v2").copy()
    sequences["date"] = pd.to_datetime(sequences["temporal_cutoff"], errors="coerce")
    cutoff = pd.Timestamp(casp15_safe_cutoff())
    eligible = sequences[
        (sequences["date"] < cutoff)
        & sequences["date"].notna()
        & sequences["sequence"].str.len().between(args.min_len, args.max_len)
    ].copy()
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.max_targets:
        chosen = rng.choice(eligible.index.to_numpy(), size=args.max_targets, replace=False)
        eligible = eligible.loc[chosen]
    eligible = eligible.sort_values(["date", "target_id"]).reset_index(drop=True)
    requested = set(eligible["target_id"])
    labels = io.load_labels("train_v2")
    label_target = labels["ID"].map(io.target_id_of)
    labels = labels[label_target.isin(requested)].assign(_target_id=label_target)
    rows = []
    sequence_map = eligible.set_index("target_id")
    for target_id, group in labels.groupby("_target_id", sort=False):
        group = group.sort_values("resid")
        coordinates = group[["x_1", "y_1", "z_1"]].to_numpy(np.float32)
        if not np.isfinite(coordinates).all() or (coordinates[:, 0] <= io.RESOLVED_THRESHOLD).any():
            continue
        sequence = str(sequence_map.loc[target_id, "sequence"]).upper().replace("T", "U")
        if len(sequence) != len(coordinates):
            continue
        rows.append(
            {
                "target_id": target_id,
                "sequence": sequence,
                "date": sequence_map.loc[target_id, "date"],
                "coords": coordinates,
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "target_id"]).reset_index(drop=True)


def temporal_split(
    chains: pd.DataFrame, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(chains) < 10:
        raise ValueError("at least ten complete chains are required")
    boundary = min(max(int(round(len(chains) * (1.0 - validation_fraction))), 1), len(chains) - 1)
    return chains.iloc[:boundary].copy(), chains.iloc[boundary:].copy()


def temporal_three_way_split(
    chains: pd.DataFrame,
    calibration_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Oldest train, then calibration, with the newest targets untouched."""
    if len(chains) < 15:
        raise ValueError("at least fifteen complete chains are required")
    if calibration_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("calibration and validation fractions must be positive")
    if calibration_fraction + validation_fraction >= 0.8:
        raise ValueError("at least 20% of chains must remain for training")
    train_end = int(round(len(chains) * (1.0 - calibration_fraction - validation_fraction)))
    calibration_end = int(round(len(chains) * (1.0 - validation_fraction)))
    train_end = min(max(train_end, 1), len(chains) - 2)
    calibration_end = min(max(calibration_end, train_end + 1), len(chains) - 1)
    # Keep every deposition-date tie wholly in the later split.
    while train_end > 1 and chains.iloc[train_end - 1]["date"] == chains.iloc[train_end]["date"]:
        train_end -= 1
    while (
        calibration_end > train_end + 1
        and chains.iloc[calibration_end - 1]["date"] == chains.iloc[calibration_end]["date"]
    ):
        calibration_end -= 1
    return (
        chains.iloc[:train_end].copy(),
        chains.iloc[train_end:calibration_end].copy(),
        chains.iloc[calibration_end:].copy(),
    )


class ExampleDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate_examples(examples: list[dict]) -> dict[str, torch.Tensor]:
    batch = len(examples)
    length = max(len(example["features"]) for example in examples)
    feature_count = len(FEATURE_NAMES)
    output = {
        "features": torch.zeros(batch, length, feature_count),
        "target": torch.zeros(batch, length),
        "weight": torch.zeros(batch, length),
        "mask": torch.zeros(batch, length, dtype=torch.bool),
        "template_error": torch.zeros(batch, length),
        "pretrained_error": torch.zeros(batch, length),
        "gap_rule": torch.zeros(batch, length),
        "confidence_rule": torch.zeros(batch, length),
    }
    for row, example in enumerate(examples):
        size = len(example["features"])
        output["features"][row, :size] = torch.from_numpy(example["features"])
        for name in (
            "target",
            "weight",
            "template_error",
            "pretrained_error",
            "gap_rule",
            "confidence_rule",
        ):
            output[name][row, :size] = torch.from_numpy(example[name])
        if "resolved_mask" in example:
            output["mask"][row, :size] = torch.from_numpy(
                np.asarray(example["resolved_mask"], dtype=bool)
            )
        else:
            output["mask"][row, :size] = True
    return output


def feature_normalization(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([example["features"] for example in examples], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return mean, std


def normalize_examples(examples: list[dict], mean: np.ndarray, std: np.ndarray) -> None:
    for example in examples:
        example["features"] = ((example["features"] - mean) / std).astype(np.float32)


def evaluate(
    model: ConfidenceGate1D,
    loader: DataLoader,
    device: torch.device,
    *,
    decision_threshold: float = 0.5,
) -> dict:
    model.eval()
    probabilities = []
    targets = []
    template_errors = []
    pretrained_errors = []
    gap_rules = []
    confidence_rules = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            losses.extend((loss[mask] * weight[mask]).cpu().numpy())
            probabilities.extend(torch.sigmoid(logits)[mask].cpu().numpy())
            targets.extend(target[mask].cpu().numpy())
            template_errors.extend(batch["template_error"][batch["mask"]].numpy())
            pretrained_errors.extend(batch["pretrained_error"][batch["mask"]].numpy())
            gap_rules.extend(batch["gap_rule"][batch["mask"]].numpy())
            confidence_rules.extend(batch["confidence_rule"][batch["mask"]].numpy())
    probability = np.asarray(probabilities)
    soft_target = np.asarray(targets)
    hard_target = soft_target >= 0.5
    template_error = np.asarray(template_errors)
    pretrained_error = np.asarray(pretrained_errors)

    def selected_error(decision: np.ndarray) -> float:
        return float(np.where(decision, pretrained_error, template_error).mean())

    hard_prediction = probability >= decision_threshold
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(probability[mask].mean() - hard_target[mask].mean())
    return {
        "weighted_bce": float(np.mean(losses)),
        "accuracy": float((hard_prediction == hard_target).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(hard_target, hard_prediction)),
        "roc_auc": float(roc_auc_score(hard_target, probability)),
        "brier": float(brier_score_loss(hard_target, probability)),
        "ece_10bin": float(ece),
        "template_error": float(template_error.mean()),
        "pretrained_error": float(pretrained_error.mean()),
        "oracle_residue_error": float(np.minimum(template_error, pretrained_error).mean()),
        "learned_gate_error": selected_error(hard_prediction),
        "gap_rule_error": selected_error(np.asarray(gap_rules) >= 0.5),
        "confidence_rule_error": selected_error(np.asarray(confidence_rules) >= 0.5),
        "n_residues": int(len(probability)),
        "pretrained_better_fraction": float(hard_target.mean()),
        "decision_threshold": float(decision_threshold),
    }


def calibrate_threshold(
    model: ConfidenceGate1D, loader: DataLoader, device: torch.device
) -> tuple[float, dict]:
    candidates = np.linspace(0.05, 0.95, 37)
    results = [
        evaluate(model, loader, device, decision_threshold=float(threshold))
        for threshold in candidates
    ]
    best = min(
        zip(candidates, results),
        key=lambda item: (item[1]["learned_gate_error"], abs(float(item[0]) - 0.5)),
    )
    return float(best[0]), best[1]


def build_examples(
    chains: pd.DataFrame,
    variants: int,
    seed: int,
    priors_v1: dict,
    priors_v2: dict,
) -> list[dict]:
    examples = []
    for row_index, row in enumerate(chains.itertuples(index=False)):
        for variant in range(variants):
            derived_seed = seed + row_index * 1009 + variant * 9176
            examples.append(
                make_example(
                    row.coords,
                    row.sequence,
                    row.target_id,
                    np.random.default_rng(derived_seed),
                    priors_v1,
                    priors_v2,
                )
            )
    return examples


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    priors_v1, priors_v2 = load_priors()
    chains = load_complete_chains(args)
    train_chains, calibration_chains, validation_chains = temporal_three_way_split(
        chains, args.calibration_fraction, args.validation_fraction
    )
    print(
        f"[data] complete chains={len(chains)} train={len(train_chains)} "
        f"calibration={len(calibration_chains)} validation={len(validation_chains)}",
        flush=True,
    )
    train_examples = build_examples(
        train_chains, args.variants, args.seed, priors_v1, priors_v2
    )
    calibration_examples = build_examples(
        calibration_chains,
        args.variants,
        args.seed + 500_000,
        priors_v1,
        priors_v2,
    )
    validation_examples = build_examples(
        validation_chains, args.variants, args.seed + 1_000_000, priors_v1, priors_v2
    )
    mean, std = feature_normalization(train_examples)
    normalize_examples(train_examples, mean, std)
    normalize_examples(calibration_examples, mean, std)
    normalize_examples(validation_examples, mean, std)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        ExampleDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_examples,
    )
    calibration_loader = DataLoader(
        ExampleDataset(calibration_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    validation_loader = DataLoader(
        ExampleDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    gate_config = GateConfig(hidden_channels=args.hidden_channels)
    model = ConfidenceGate1D(cfg=gate_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = []
        for batch in train_loader:
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
            train_loss.append(float(loss.detach()))
        calibration_metrics = evaluate(model, calibration_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_loss)),
            "calibration_loss": calibration_metrics["weighted_bce"],
            "calibration_auc": calibration_metrics["roc_auc"],
        }
        history.append(row)
        print(
            f"[epoch {epoch:02d}] train={row['train_loss']:.4f} "
            f"cal={row['calibration_loss']:.4f} auc={row['calibration_auc']:.4f}",
            flush=True,
        )
        if row["calibration_loss"] < best_loss:
            best_loss = row["calibration_loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    decision_threshold, calibration_metrics = calibrate_threshold(
        model, calibration_loader, device
    )
    validation_metrics = evaluate(
        model, validation_loader, device, decision_threshold=decision_threshold
    )
    validation_metrics["n_parameters"] = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    checkpoint = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "feature_mean": mean,
        "feature_std": std,
        "gate_config": asdict(gate_config),
        "state_dict": best_state,
        "training": {
            "seed": args.seed,
            "corruption_version": 2,
            "variants": args.variants,
            "n_train_targets": len(train_chains),
            "n_calibration_targets": len(calibration_chains),
            "n_validation_targets": len(validation_chains),
            "train_last_date": str(train_chains["date"].max().date()),
            "calibration_first_date": str(calibration_chains["date"].min().date()),
            "calibration_last_date": str(calibration_chains["date"].max().date()),
            "validation_first_date": str(validation_chains["date"].min().date()),
            "decision_threshold": decision_threshold,
            "train_target_digest": hashlib.sha256(
                "\n".join(train_chains["target_id"]).encode()
            ).hexdigest(),
            "validation_target_digest": hashlib.sha256(
                "\n".join(validation_chains["target_id"]).encode()
            ).hexdigest(),
            "history": history,
        },
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    write_report(
        Path(args.report),
        train_chains,
        calibration_chains,
        validation_chains,
        calibration_metrics,
        validation_metrics,
        history,
        args,
        time.time() - started,
    )
    print(pd.Series(validation_metrics).round(5).to_string())
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[report] {args.report}")


def write_report(
    path: Path,
    train_chains: pd.DataFrame,
    calibration_chains: pd.DataFrame,
    validation_chains: pd.DataFrame,
    calibration_metrics: dict,
    metrics: dict,
    history: list[dict],
    args: argparse.Namespace,
    seconds: float,
) -> None:
    metric_table = pd.Series(metrics, name="value").to_frame()
    gate_passed = metrics["learned_gate_error"] < min(
        metrics["gap_rule_error"], metrics["confidence_rule_error"]
    )
    gap_improvement = metrics["gap_rule_error"] - metrics["learned_gate_error"]
    lines = [
        "# GeoFuse Phase D — synthetic confidence gate",
        "",
        "This bootstrap experiment trains a tiny 1D residue gate only on temporal-safe "
        "`train_v2` native structures corrupted into template-like and pretrained-like "
        "sources. The newest targets are held out as a time-ordered validation split.",
        "",
        f"- Train targets: {len(train_chains)} ({train_chains['date'].min().date()} to "
        f"{train_chains['date'].max().date()})",
        f"- Calibration targets: {len(calibration_chains)} "
        f"({calibration_chains['date'].min().date()} to "
        f"{calibration_chains['date'].max().date()})",
        f"- Held-out targets: {len(validation_chains)} "
        f"({validation_chains['date'].min().date()} to {validation_chains['date'].max().date()})",
        f"- Synthetic variants per target: {args.variants}",
        f"- Sampling cap / length range: {args.max_targets} targets / "
        f"{args.min_len}-{args.max_len} residues",
        f"- Seed: {args.seed}",
        f"- Epochs: {args.epochs}",
        f"- Runtime: {seconds:.1f} seconds",
        f"- Synthetic held-out gate: **{'pass' if gate_passed else 'fail'}**",
        "",
        "## Held-out metrics",
        "",
        metric_table.round(5).to_markdown(),
        "",
        f"Decision threshold {metrics['decision_threshold']:.3f} was selected only on "
        f"the calibration split (calibration gate error "
        f"{calibration_metrics['learned_gate_error']:.4f} Å).",
        f"Learned-gate improvement over gap rule: {gap_improvement:+.4f} Å "
        "(positive means lower error).",
        "",
        "Lower error is better. `oracle_residue_error` chooses the lower-error source with "
        "native knowledge and is only a ceiling; the learned gate never sees native at inference.",
        "",
        "## Training history",
        "",
        pd.DataFrame(history).round(5).to_markdown(index=False),
        "",
        "## Validity boundary",
        "",
        "Synthetic corruption is bootstrap supervision, not sufficient evidence that confidence "
        "is calibrated for DRfold2/Boltz/TBM outputs. The checkpoint must next be frozen and "
        "tested on real out-of-fold template/model candidates; CASP15 labels must not be used "
        "to retrain or tune it.",
        "Corruption v2 randomizes model confidence scales because Phase B found raw source "
        "confidence uncalibrated; absolute global-confidence features are intentionally omitted.",
        "The frozen real-candidate transfer pilot is reported separately in "
        "`geofuse_phase_d_transfer.md`; synthetic gate success is not presented as real-domain "
        "fusion success.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-targets", type=int, default=600)
    parser.add_argument("--min-len", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint", default=processed() / "geofuse_confidence_gate_v1.pt"
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT / "reports" / "thesis_notes" / "geofuse_phase_d_gate.md",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
