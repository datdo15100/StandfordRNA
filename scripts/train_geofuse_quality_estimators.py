#!/usr/bin/env python
"""Compare real-OOF local quality estimators under a frozen 60/20/20 protocol."""
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
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rna3d.eval.statistics import paired_target_summary
from rna3d.geofuse.phase_d import FEATURE_NAMES, ConfidenceGate1D, GateConfig
from rna3d.paths import cache, processed
from train_geofuse_phase_d import (
    ExampleDataset,
    collate_examples,
    feature_normalization,
    load_priors,
    normalize_examples,
)
from train_geofuse_real_gate import build_examples


SUPERVISION = {
    "aligned_point": {
        "target": "target",
        "weight": "weight",
        "mask": "resolved_mask",
        "template": "aligned_template_error",
        "pretrained": "aligned_pretrained_error",
        "unit": "angstrom",
    },
    "c1_lddt": {
        "target": "lddt_target",
        "weight": "lddt_weight",
        "mask": "lddt_resolved_mask",
        "template": "template_lddt",
        "pretrained": "pretrained_lddt",
        "unit": "1-minus-lddt",
    },
    "window15_rmsd": {
        "target": "window_target",
        "weight": "window_weight",
        "mask": "window_resolved_mask",
        "template": "template_window_rmsd",
        "pretrained": "pretrained_window_rmsd",
        "unit": "angstrom",
    },
}


def configure_supervision(examples: list[dict], name: str) -> None:
    """Map one frozen label definition to the generic training/evaluation fields."""
    schema = SUPERVISION[name]
    for example in examples:
        example["target"] = example[schema["target"]].astype(np.float32)
        example["weight"] = example[schema["weight"]].astype(np.float32)
        example["resolved_mask"] = np.asarray(example[schema["mask"]], dtype=bool)
        if name == "c1_lddt":
            example["template_error"] = (
                1.0 - example[schema["template"]]
            ).astype(np.float32)
            example["pretrained_error"] = (
                1.0 - example[schema["pretrained"]]
            ).astype(np.float32)
        else:
            example["template_error"] = example[schema["template"]].astype(np.float32)
            example["pretrained_error"] = example[schema["pretrained"]].astype(np.float32)


def subset_targets(examples: list[dict], target_ids: set[str]) -> list[dict]:
    return [example for example in examples if example["target_id"] in target_ids]


def flatten_examples(examples: list[dict]) -> dict:
    rows = {
        "features": [],
        "target": [],
        "weight": [],
        "template_error": [],
        "pretrained_error": [],
        "gap_rule": [],
        "confidence_rule": [],
        "target_id": [],
    }
    for example in examples:
        mask = np.asarray(example["resolved_mask"], dtype=bool)
        count = int(mask.sum())
        rows["features"].append(example["features"][mask])
        for name in (
            "target",
            "weight",
            "template_error",
            "pretrained_error",
            "gap_rule",
            "confidence_rule",
        ):
            rows[name].append(np.asarray(example[name])[mask])
        rows["target_id"].extend([example["target_id"]] * count)
    return {
        name: np.concatenate(values, axis=0)
        if name != "target_id"
        else np.asarray(values)
        for name, values in rows.items()
    }


def _target_errors(flat: dict, decision: np.ndarray) -> pd.Series:
    selected = np.where(
        decision, flat["pretrained_error"], flat["template_error"]
    )
    return pd.DataFrame(
        {"target_id": flat["target_id"], "error": selected}
    ).groupby("target_id")["error"].mean()


def prediction_metrics(
    flat: dict, probability: np.ndarray, threshold: float
) -> tuple[dict, pd.Series]:
    hard_target = flat["target"] >= 0.5
    decision = np.asarray(probability) >= threshold
    target_error = _target_errors(flat, decision)
    try:
        auc = float(roc_auc_score(hard_target, probability))
    except ValueError:
        auc = float("nan")
    result = {
        "target_mean_error": float(target_error.mean()),
        "residue_mean_error": float(
            np.where(
                decision, flat["pretrained_error"], flat["template_error"]
            ).mean()
        ),
        "accuracy": float((decision == hard_target).mean()),
        "roc_auc": auc,
        "pretrained_fraction": float(decision.mean()),
        "threshold": float(threshold),
        "targets": int(target_error.size),
        "residues": int(len(decision)),
    }
    return result, target_error


def fixed_rule_metrics(flat: dict) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    decisions = {
        "always_tbm": np.zeros(len(flat["target"]), dtype=bool),
        "always_pretrained": np.ones(len(flat["target"]), dtype=bool),
        "gap_rule": flat["gap_rule"] >= 0.5,
        "confidence_rule": flat["confidence_rule"] >= 0.5,
        "oracle_residue": flat["pretrained_error"] < flat["template_error"],
    }
    rows = []
    target_errors = {}
    for name, decision in decisions.items():
        probability = decision.astype(float)
        metrics, target_error = prediction_metrics(flat, probability, 0.5)
        rows.append({"model": name, **metrics})
        target_errors[name] = target_error
    return pd.DataFrame(rows), target_errors


def calibrate_threshold(flat: dict, probability: np.ndarray) -> tuple[float, dict]:
    choices = []
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics, _ = prediction_metrics(flat, probability, float(threshold))
        choices.append((metrics["target_mean_error"], abs(threshold - 0.5), threshold, metrics))
    _, _, threshold, metrics = min(choices)
    return float(threshold), metrics


def _fit_tabular(
    kind: str,
    train: dict,
    calibration: dict,
    seed: int,
) -> dict:
    configurations = (
        [{"C": value} for value in (0.1, 1.0, 10.0)]
        if kind == "logistic"
        else [
            {"max_leaf_nodes": leaves, "l2_regularization": regularization}
            for leaves in (7, 15)
            for regularization in (0.0, 1.0)
        ]
    )
    trials = []
    for config in configurations:
        if kind == "logistic":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=config["C"], max_iter=500, random_state=seed
                ),
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=config["max_leaf_nodes"],
                l2_regularization=config["l2_regularization"],
                random_state=seed,
            )
        model.fit(
            train["features"],
            train["target"] >= 0.5,
            **{"logisticregression__sample_weight": train["weight"]}
            if kind == "logistic"
            else {"sample_weight": train["weight"]},
        )
        probability = model.predict_proba(calibration["features"])[:, 1]
        threshold, metrics = calibrate_threshold(calibration, probability)
        trials.append((metrics["target_mean_error"], config, threshold, model, metrics))
    _, config, threshold, model, metrics = min(trials, key=lambda row: row[0])
    return {
        "model": model,
        "config": config,
        "threshold": threshold,
        "calibration_metrics": metrics,
    }


def _conv_probabilities(
    model: ConfidenceGate1D,
    examples: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            features = (example["features"] - mean) / std
            tensor = torch.as_tensor(
                features[None], dtype=torch.float32, device=device
            )
            probability = torch.sigmoid(model(tensor))[0].cpu().numpy()
            values.append(probability[np.asarray(example["resolved_mask"], dtype=bool)])
    return np.concatenate(values)


def _fit_conv(
    train_examples: list[dict],
    calibration_examples: list[dict],
    calibration_flat: dict,
    args: argparse.Namespace,
) -> dict:
    mean, std = feature_normalization(train_examples)
    normalized_train = [
        {**example, "features": example["features"].copy()}
        for example in train_examples
    ]
    normalize_examples(normalized_train, mean, std)
    loader = DataLoader(
        ExampleDataset(normalized_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_examples,
    )
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    config = GateConfig(hidden_channels=args.hidden_channels)
    model = ConfidenceGate1D(cfg=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features)
            per_residue = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (per_residue[mask] * weight[mask]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        probability = _conv_probabilities(
            model, calibration_examples, mean, std, device
        )
        threshold, metrics = calibrate_threshold(calibration_flat, probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "calibration_error": metrics["target_mean_error"],
                "threshold": threshold,
            }
        )
        candidate = (
            metrics["target_mean_error"],
            epoch,
            threshold,
            {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            metrics,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, epoch, threshold, state, metrics = best
    model.load_state_dict(state)
    return {
        "model": model,
        "config": asdict(config),
        "threshold": threshold,
        "calibration_metrics": metrics,
        "feature_mean": mean,
        "feature_std": std,
        "state_dict": state,
        "best_epoch": epoch,
        "history": history,
        "device": device,
    }


def _model_probability(
    name: str, fitted: dict, examples: list[dict], flat: dict
) -> np.ndarray:
    if name == "conv1d":
        return _conv_probabilities(
            fitted["model"],
            examples,
            fitted["feature_mean"],
            fitted["feature_std"],
            fitted["device"],
        )
    return fitted["model"].predict_proba(flat["features"])[:, 1]


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    manifest = pd.read_csv(args.manifest, dtype={"target_id": str})
    priors_v1, priors_v2 = load_priors()
    grouped, failures = build_examples(args, manifest, priors_v1, priors_v2)
    for values in grouped.values():
        configure_supervision(values, args.supervision)
    counts = {split: len({x["target_id"] for x in values}) for split, values in grouped.items()}
    expected = {
        "train": args.expected_train_targets,
        "calibration": args.expected_calibration_targets,
        "validation": args.expected_validation_targets,
    }
    if counts != expected:
        raise RuntimeError(f"frozen target counts not ready: {counts}, expected {expected}")

    train_order = (
        manifest[manifest["split"] == "train"]
        .sort_values(["date", "target_id"])["target_id"]
        .tolist()
    )
    calibration_examples = grouped["calibration"]
    validation_examples = grouped["validation"]
    calibration_flat = flatten_examples(calibration_examples)
    validation_flat = flatten_examples(validation_examples)
    learning_rows = []
    full_models = {}
    requested_sizes = sorted(
        {
            min(size, len(train_order))
            for size in (10, 25, 40, 60)
            if min(size, len(train_order)) > 0
        }
    )
    sizes = requested_sizes
    for size in sizes:
        chosen = set(train_order[:size])
        train_examples = subset_targets(grouped["train"], chosen)
        train_flat = flatten_examples(train_examples)
        print(f"[learning] train targets={size}, examples={len(train_examples)}", flush=True)
        fitted = {
            "logistic": _fit_tabular(
                "logistic", train_flat, calibration_flat, args.seed
            ),
            "gradient_boosting": _fit_tabular(
                "gradient_boosting", train_flat, calibration_flat, args.seed
            ),
            "conv1d": _fit_conv(
                train_examples, calibration_examples, calibration_flat, args
            ),
        }
        for name, result in fitted.items():
            learning_rows.append(
                {
                    "train_targets": size,
                    "model": name,
                    "calibration_target_error": result["calibration_metrics"][
                        "target_mean_error"
                    ],
                    "calibration_auc": result["calibration_metrics"]["roc_auc"],
                    "threshold": result["threshold"],
                    "config": json.dumps(result["config"], sort_keys=True),
                }
            )
        if size == len(train_order):
            full_models = fitted

    calibration_baselines, calibration_baseline_targets = fixed_rule_metrics(
        calibration_flat
    )
    feasible_calibration = calibration_baselines[
        calibration_baselines["model"] != "oracle_residue"
    ]
    strongest_baseline = str(
        feasible_calibration.sort_values("target_mean_error").iloc[0]["model"]
    )
    selected_model = min(
        full_models,
        key=lambda name: full_models[name]["calibration_metrics"][
            "target_mean_error"
        ],
    )
    # The choice above is frozen on calibration. Validation is first read below.
    validation_baselines, validation_baseline_targets = fixed_rule_metrics(validation_flat)
    validation_rows = validation_baselines.to_dict("records")
    validation_target_errors = dict(validation_baseline_targets)
    for name, fitted in full_models.items():
        probability = _model_probability(
            name, fitted, validation_examples, validation_flat
        )
        metrics, target_error = prediction_metrics(
            validation_flat, probability, fitted["threshold"]
        )
        validation_rows.append({"model": name, **metrics})
        validation_target_errors[name] = target_error
    validation_results = pd.DataFrame(validation_rows)
    selected_error = validation_target_errors[selected_model]
    baseline_error = validation_target_errors[strongest_baseline]
    shared = selected_error.index.intersection(baseline_error.index)
    bootstrap = paired_target_summary(
        baseline_error.loc[shared].to_numpy(),
        selected_error.loc[shared].to_numpy(),
        higher_is_better=False,
    )
    feasible_validation = validation_results[
        ~validation_results["model"].isin(["oracle_residue", selected_model])
        & validation_results["model"].isin(
            ["always_tbm", "always_pretrained", "gap_rule", "confidence_rule"]
        )
    ]
    selected_value = float(
        validation_results.loc[
            validation_results["model"] == selected_model, "target_mean_error"
        ].iloc[0]
    )
    passed = bool(selected_value < feasible_validation["target_mean_error"].min())

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    learning = pd.DataFrame(learning_rows)
    learning.to_csv(output / f"{args.supervision}_learning_curves.csv", index=False)
    calibration_baselines.to_csv(
        output / f"{args.supervision}_calibration_baselines.csv", index=False
    )
    validation_results.to_csv(
        output / f"{args.supervision}_validation_results.csv", index=False
    )
    target_table = pd.DataFrame(validation_target_errors)
    target_table.index.name = "target_id"
    target_table.to_csv(output / f"{args.supervision}_validation_targets.csv")

    selected = full_models[selected_model]
    if selected_model == "conv1d":
        checkpoint = {
            "schema_version": 1,
            "feature_names": FEATURE_NAMES,
            "feature_mean": selected["feature_mean"],
            "feature_std": selected["feature_std"],
            "gate_config": selected["config"],
            "state_dict": selected["state_dict"],
            "training": {
                "supervision": args.supervision,
                "seed": args.seed,
                "decision_threshold": selected["threshold"],
                "manifest_sha256": hashlib.sha256(
                    Path(args.manifest).read_bytes()
                ).hexdigest(),
                "selected_on": "calibration_target_mean_error",
            },
        }
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        selected_checkpoint = checkpoint_path
    else:
        sklearn_path = Path(args.sklearn_checkpoint)
        sklearn_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": 1,
                "feature_names": FEATURE_NAMES,
                "model_name": selected_model,
                "model": selected["model"],
                "decision_threshold": selected["threshold"],
                "supervision": args.supervision,
                "manifest_sha256": hashlib.sha256(
                    Path(args.manifest).read_bytes()
                ).hexdigest(),
            },
            sklearn_path,
        )
        selected_checkpoint = sklearn_path
    selection_path = Path(args.selection)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "supervision": args.supervision,
                "model_name": selected_model,
                "checkpoint": str(selected_checkpoint.resolve()),
                "decision_threshold": float(selected["threshold"]),
                "manifest_sha256": hashlib.sha256(
                    Path(args.manifest).read_bytes()
                ).hexdigest(),
                "selected_on": "calibration_target_mean_error",
                "strongest_fixed_baseline": strongest_baseline,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    lines = [
        f"# GeoFuse quality estimators — {args.supervision}",
        "",
        "This is E12 under the frozen 60/20/20 protocol. Learning curves and all "
        "hyperparameters/thresholds use train+calibration only. The confirmatory "
        "estimator and strongest fixed baseline are selected on calibration before "
        "the 20 newest validation targets are evaluated once.",
        "",
        f"- Ready target counts: `{counts}`",
        f"- Rejected pair attempts: {len(failures)}",
        f"- Supervision/evaluation unit: {SUPERVISION[args.supervision]['unit']}",
        f"- Calibration-selected estimator: **{selected_model}**",
        f"- Calibration-selected strongest baseline: **{strongest_baseline}**",
        f"- Confirmatory router gate: **{'pass' if passed else 'fail'}**",
        f"- Runtime: {time.time() - started:.1f} seconds",
        "",
        "## Calibration learning curves",
        "",
        learning.round(6).drop(columns="config").to_markdown(index=False),
        "",
        "## Final newest-target results",
        "",
        validation_results.sort_values("target_mean_error")
        .round(6)
        .to_markdown(index=False),
        "",
        "## Target bootstrap: selected estimator versus calibration-selected baseline",
        "",
        pd.Series(bootstrap, name="value").to_frame().round(6).to_markdown(),
        "",
        "Positive bootstrap delta means the selected estimator has lower error. The "
        "pass criterion requires it to beat always-TBM, always-pretrained, gap and "
        "raw-confidence rules on the equal-weight target mean.",
        "",
        "## Interpretation",
        "",
        (
            "- The learned router passes the frozen real-OOF gate."
            if passed
            else "- The learned router fails the frozen real-OOF gate. It must not "
            "control final fusion as though calibrated; selective fusion should "
            "abstain or remain an upper-bound experiment."
        ),
        "- The per-residue oracle is non-deployable and appears only to quantify "
        "remaining headroom.",
        "",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    print(validation_results.sort_values("target_mean_error").round(6).to_string(index=False))
    print(f"selected={selected_model} gate={'pass' if passed else 'fail'}")
    print(f"[report] {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=processed() / "geofuse_real_oof_v2" / "medium_manifest.csv",
    )
    parser.add_argument("--cache-root", default=cache() / "geofuse_candidates")
    parser.add_argument("--max-templates", type=int, default=2)
    parser.add_argument("--max-pretrained", type=int, default=2)
    parser.add_argument("--expected-train-targets", type=int, default=60)
    parser.add_argument("--expected-calibration-targets", type=int, default=20)
    parser.add_argument("--expected-validation-targets", type=int, default=20)
    parser.add_argument(
        "--supervision", choices=sorted(SUPERVISION), default="c1_lddt"
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        default=processed() / "geofuse_quality_gate_real_oof_v2.pt",
    )
    parser.add_argument(
        "--sklearn-checkpoint",
        default=processed() / "geofuse_quality_estimator_real_oof_v2.joblib",
    )
    parser.add_argument(
        "--selection",
        default=processed() / "geofuse_quality_estimator_selection_v2.json",
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "reports" / "tables" / "geofuse_quality_estimators",
    )
    parser.add_argument(
        "--report",
        default=REPO_ROOT
        / "reports"
        / "thesis_notes"
        / "geofuse_quality_estimators.md",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
