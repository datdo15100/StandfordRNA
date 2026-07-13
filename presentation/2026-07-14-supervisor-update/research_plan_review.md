# Research plan review: current pipeline vs GeoFuse-RNA

## Bottom line

GeoFuse-RNA is not a different thesis topic. It is an extension of the same Kaggle-centred
pipeline intended to make the integration layer—not the borrowed candidate generators—the
primary research contribution.

## What the current pipeline already implements

| Capability | Current status |
|---|---|
| Temporal-safe template filtering | implemented and evaluated |
| MMseqs search | implemented |
| Exhaustive composite search | implemented; largest measured TM gain |
| C1′ coordinate transfer and gap fill | implemented |
| De novo hedge | implemented |
| Best-of-five output | implemented |
| Rule and gradient refiners | implemented and compared |
| Leakage/oracle diagnostics | implemented |
| Pretrained candidate generation | not integrated locally yet |
| Per-residue source confidence | only rudimentary gap/template confidence |
| Fold clustering before fusion | not implemented |
| Segment-wise TBM/pretrained fusion | not implemented |
| Motif-conditioned geometry | not implemented |
| Quality-diversity selector | not implemented as a complete method |

## What GeoFuse adds

1. A candidate bank containing both TBM and pretrained folds.
2. Fold clustering before any coordinate fusion.
3. Reliability estimates at residue/segment level rather than one global score.
4. Segment-wise fusion with smoothing and boundary/seam control.
5. Geometry v2 conditioned on simple RNA context (paired/unpaired, stem/loop/junction).
6. A quality-diversity selector for the final five predictions.

The most defensible primary contributions are **segment-wise confidence-aware fusion** and
**motif-conditioned geometry projection**. Clustering, evaluation and selection support
those claims but should not all be presented as independent novel methods.

## Methodological corrections before implementation

### 1. Torsion terminology

The dihedral of four consecutive C1′ coordinates is a **C1′ chain dihedral**. Standard
RNA pseudo-torsion definitions use phosphorus plus C1′ or C4′ atoms. Do not call a
C1′-only dihedral a standard RNA pseudo-torsion unless the parser/representation is
extended to retain the required atoms.

### 2. Selection regret

Use an additive gap:

```text
selection_regret = oracle_pool_TM - selected_best5_TM
```

The ratio currently sketched in `PLAN.md` is harder to interpret and behaves poorly for
low scores.

### 3. Pretrained temporal safety

Template release filtering is insufficient if pretrained weights were trained on
post-cutoff structures. Record model training-data cutoff and report any model with an
unknown/post-CASP15 cutoff as competition-oriented rather than temporal-safe evidence.

### 4. Confidence-gate training data

Synthetic corruptions can bootstrap a gate but cannot be the only training evidence.
Use out-of-fold, family-held-out or time-held-out predictions from the actual TBM and
pretrained sources so the gate learns real source failure patterns.

### 5. Per-residue supervision

Per-residue error after one global Kabsch alignment mixes local error with domain motion.
Compare robust/anchor alignment and a local lDDT-like label. State exactly which error the
confidence gate is trained to predict.

### 6. Small validation set

Repeatedly tuning on 12 CASP15 targets risks adapting to the benchmark. Freeze a final
subset or use family/time-split development data, report paired per-target changes and
bootstrap intervals, and treat Kaggle private score as external validation rather than an
ablation set.

## Experiment gates

1. **Candidate gate:** pretrained raw union must increase oracle-pool TM on at least some
   target regimes. If not, do not build a complicated fusion model around it.
2. **Fusion gate:** fused candidates must beat choosing the best whole-source candidate
   under a leakage-free selection protocol.
3. **Geometry gate:** v2 must retain v1 clash/backbone gains while keeping sharp-kink rate
   at or below no refinement.
4. **Selection gate:** final-five selection must reduce additive selection regret while
   retaining multiple fold clusters.
5. **Production gate:** the exact Kaggle notebook version must finish offline, validate
   `submission.csv`, and stay inside the competition runtime limit.

## Recommended thesis claim

> GeoFuse-RNA is a model-agnostic integration framework that clusters RNA fold candidates,
> estimates source reliability at residue/segment level, fuses complementary template and
> pretrained regions, and projects the result onto context-dependent RNA geometry. It is
> evaluated for both fold accuracy and structural validity under temporal-safe controls.

