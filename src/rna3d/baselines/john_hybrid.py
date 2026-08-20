"""Executable control-flow port of the publicly released John hybrid notebook.

The public notebook couples Boltz-1, a Boltz-restraint-conditioned DRfold2 run,
and the public template fallback.  V4 does not have the original Boltz checkpoint,
so this module deliberately separates *routing* from the injected model runners.
It can reproduce and test the captured control flow without pretending that the
unavailable structural branch has been reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


StructureRunner = Callable[[], Sequence[np.ndarray]]


@dataclass(frozen=True)
class HybridRouteResult:
    structures: tuple[np.ndarray, ...]
    requested_route: str
    executed_route: str
    fallback_reason: str | None


def public_drfold_index_range(target_count: int) -> tuple[int, int]:
    """Return the exact retained-index range in the captured public notebook."""
    if target_count > 15:
        return 14, target_count - 1
    return 0, 0


def pad_or_trim(structures: Sequence[np.ndarray], n: int = 5) -> tuple[np.ndarray, ...]:
    """Match the public notebook: duplicate the final model until N is reached."""
    values = [np.asarray(value, dtype=float) for value in structures]
    if not values:
        return ()
    while len(values) < n:
        values.append(values[-1].copy())
    return tuple(values[:n])


def run_public_hybrid_route(
    *,
    retained_dataframe_index: int,
    target_count: int,
    elapsed_seconds: float,
    drfold_time_limit_seconds: float,
    boltz_conditioned_drfold_runner: StructureRunner | None,
    template_runner: StructureRunner,
    n: int = 5,
) -> HybridRouteResult:
    """Execute captured routing while exposing unavailable model artifacts honestly."""
    start, end = public_drfold_index_range(target_count)
    use_drfold = (
        start <= retained_dataframe_index <= end
        and elapsed_seconds < drfold_time_limit_seconds
    )
    if use_drfold and boltz_conditioned_drfold_runner is not None:
        try:
            generated = pad_or_trim(boltz_conditioned_drfold_runner(), n=n)
        except Exception as error:  # public behavior is a template fallback
            generated = ()
            failure = f"predictor_error:{type(error).__name__}"
        else:
            failure = None if generated else "predictor_returned_no_structures"
        if generated:
            return HybridRouteResult(generated, "boltz_conditioned_drfold2", "boltz_conditioned_drfold2", None)
    elif use_drfold:
        failure = "boltz_conditioned_drfold2_artifacts_unavailable"
    elif elapsed_seconds >= drfold_time_limit_seconds:
        failure = "drfold_time_limit"
    else:
        failure = "retained_index_outside_drfold_range"

    fallback = pad_or_trim(template_runner(), n=n)
    if not fallback:
        raise RuntimeError("public hybrid template fallback returned no structure")
    return HybridRouteResult(
        fallback,
        "boltz_conditioned_drfold2" if use_drfold else "template",
        "template",
        failure,
    )
