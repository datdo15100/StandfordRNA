# GeoFuse confirmatory protocol (frozen before the new results)

Protocol version: 1.0
Frozen: 2026-07-28
Random seed: 2025
Scope: phases 1–7. Phase 8 (all-atom INF/DI/MolProbity validation) is excluded at
the author's request.

## Research question

The main pipeline searches for several plausible global RNA folds using temporal-safe
templates and frozen pretrained predictors. The confirmatory side question is:

> Can native-blind local quality estimates identify which source is more reliable in
> each region, and can conservative geometry-aware fusion improve local C1′ accuracy
> without losing global TM-score?

This separates three claims which must not be conflated:

1. candidate diversity increases fold coverage;
2. geometry projection improves physical-looking local geometry;
3. a router/fuser improves native local or global accuracy.

## Data boundary

- Development geometry ablation: the existing 12 validation targets, with R1128 also
  reported separately because of known pretrained-training overlap.
- Real OOF experiment: 60 train, 20 calibration and 20 final validation targets.
- Targets are chronological and family-disjoint at 80% identity and 80% coverage.
- Pretrained candidates must come from a frozen model whose structural-training cutoff
  predates every target. TBM templates must predate each target and direct target PDB
  identifiers are excluded.
- Native coordinates are unavailable to candidate generation, feature extraction,
  clustering, source selection and fusion. They are read only for supervision on the
  designated training split, threshold/model choice on calibration, and one final
  validation evaluation.
- Residues are not treated as independent samples for uncertainty estimates. The
  bootstrap unit is the target.

## Independent accuracy metrics

All metrics use the C1′ atom because this is the competition representation.

1. **TM-score**: global fold accuracy and the Kaggle-aligned primary endpoint.
2. **Global C1′ RMSD**: Kabsch-aligned RMSD over shared resolved residues. Lower is
   better.
3. **C1′-lDDT**: superposition-free preservation of native inter-residue distances.
   Native C1′ pairs within 15 Å are evaluated at 0.5, 1, 2 and 4 Å tolerances. Higher
   is better.
4. **Sliding-window C1′ RMSD**: local Kabsch RMSD in full windows of 9, 15 and 31
   residues. A sequence shorter than a requested window is evaluated once using the
   complete sequence. Lower is better.

For multiple native conformations, one reference is selected per candidate by highest
TM-score, then every local metric is computed against that same reference. This avoids
choosing a different favourable native for every metric.

The existing sharp-kink fraction, C1′ clash proxy, adjacent-distance deviation,
radius-of-gyration error, angle NLL and torsion NLL are project diagnostics. They are
not presented as official biological validation metrics. Angle/torsion/kink terms used
by geometry v2 are optimization endpoints, not independent evidence.

## Phase 1 — metric verification

Synthetic unit tests cover rigid-transform invariance, exact structures, controlled
local perturbations, unresolved residues, short sequences and invalid inputs.

Pass criterion: all tests pass and exact/rigidly transformed structures attain zero
RMSD and one lDDT within numerical tolerance.

## Phase 2 — geometry v2 re-evaluation

Raw, geometry v1 and geometry v2 use the identical candidate set. No native value is
used by either refiner.

Geometry-v2 local-accuracy gate:

- mean C1′-lDDT delta over raw is positive;
- sliding-window RMSD improves for at least two of the three frozen window sizes;
- global best-of-five TM loss is no worse than 0.005;
- target-bootstrap 95% intervals and improved/regressed target counts are reported,
  even when the interval includes zero.

Optimization diagnostics are reported separately and cannot make this gate pass.

## Phase 3 — supervision audit

For every real TBM/pretrained pair, compare three residue labels:

- the existing error label after independent whole-structure alignment;
- per-residue C1′-lDDT;
- sliding-window RMSD at 15 residues.

Report their target-centred rank agreement, source prevalence and disagreement cases.
The primary quality target is local C1′-lDDT. Sliding-window RMSD is an independent
continuous secondary target. The old globally aligned point-error label remains an
ablation only.

## Phase 4 — real OOF scale-up

Generate and audit candidates for exactly 100 targets: 60/20/20 by split. Selection is
deterministic within each split, spans its chronological range and is limited to
tractable sequence lengths before any native scoring.

Required audit: both source types present; provenance valid; no family crosses a split;
candidate counts, failures and length/date distributions logged.

## Phase 5 — quality estimators

Frozen feature set contains inference-available source confidence, support/gap mask,
pair disagreement after native-blind robust alignment, and project geometry
diagnostics. It never contains native-derived values.

Baselines/models:

- always TBM;
- always pretrained;
- template-gap rule;
- raw-confidence rule;
- logistic regression;
- gradient-boosted trees;
- tiny 1D convolutional gate.

Learning curves use 10, 25, 40 and 60 training targets. Hyperparameters and thresholds
are selected only on the 20 calibration targets. The final 20 targets are evaluated
once after freezing.

Router pass criterion: on final validation, the learned decision has lower target-mean
local error than every fixed baseline. Target-bootstrap 95% intervals are reported for
the delta to the strongest baseline.

## Phase 6 — global and local clustering

Global complete-link self-TM thresholds 0.35, 0.45 and 0.55 are compared using
calibration only. Within mixed-source global clusters, local disagreement is summarized
in 15-residue windows. This tests whether two models share a fold but disagree in a
localized region.

The threshold/configuration is frozen from calibration before final validation.

## Phase 7 — selective fusion with abstention

Variants:

- F0: raw parents only;
- F1: existing heuristic fusion;
- F2: quality-gated fusion;
- F3: F2 followed by geometry v2 projection;
- F4: native oracle upper bound, never deployable.

Fusion is allowed only for parents in the same global cluster. Switching requires a
calibrated quality margin, persists for a contiguous segment of at least seven
residues, and abstains elsewhere. Both raw parents always remain in the candidate bank;
fusion only adds candidates.

Fusion pass criterion:

- augmented oracle increases over raw oracle;
- native-blind selected best-of-five TM and C1′-lDDT do not regress;
- no material target regression is hidden by an aggregate mean;
- F4 is labelled only as headroom, never as a method result.

## Statistical reporting

- Primary aggregation: equal-weight target mean.
- Uncertainty: deterministic target bootstrap, seed 2025, 10,000 resamples, percentile
  95% confidence interval.
- Also report median delta and numbers of improved/tied/regressed targets.
- This is a modest confirmatory dataset, so confidence intervals and failure cases take
  precedence over residue-level p-values.
