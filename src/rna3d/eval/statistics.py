"""Target-level statistical summaries for paired structure experiments."""
from __future__ import annotations

import numpy as np


def paired_target_summary(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    higher_is_better: bool,
    seed: int = 2025,
    bootstrap_samples: int = 10_000,
    tie_tolerance: float = 1e-12,
) -> dict:
    """Summarize paired target deltas with a target bootstrap confidence interval.

    The returned delta is always oriented so positive means the method is better.
    """
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)
    if baseline.shape != method.shape or baseline.ndim != 1:
        raise ValueError("baseline and method must be same-length one-dimensional arrays")
    finite = np.isfinite(baseline) & np.isfinite(method)
    raw_delta = method[finite] - baseline[finite]
    delta = raw_delta if higher_is_better else -raw_delta
    if len(delta) == 0:
        return {
            "n_targets": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "improved": 0,
            "tied": 0,
            "regressed": 0,
        }
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    tied = np.abs(delta) <= tie_tolerance
    return {
        "n_targets": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "improved": int((delta > tie_tolerance).sum()),
        "tied": int(tied.sum()),
        "regressed": int((delta < -tie_tolerance).sum()),
    }
