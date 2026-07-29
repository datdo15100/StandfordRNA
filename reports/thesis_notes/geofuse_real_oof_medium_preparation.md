# GeoFuse real-OOF medium preparation

## Frozen cohort

The confirmatory cohort contains exactly 100 post-2023 `train_v2` targets selected
before native scoring. Within every chronological/family-disjoint split, eligible
targets of at most 100 nt are ordered by date, length and ID, then sampled at equally
spaced indices. This covers each full time band without selecting by model accuracy.

| split | targets | families | min length | median length | max length | first date | last date |
|:--|--:|--:|--:|--:|--:|:--|:--|
| train | 60 | 54 | 30 | 74 | 100 | 2024-01-10 | 2024-12-04 |
| calibration | 20 | 17 | 30 | 57 | 91 | 2024-12-11 | 2025-01-08 |
| validation | 20 | 16 | 40 | 64 | 96 | 2025-01-15 | 2025-03-26 |

Audit result: zero 80%-identity/80%-coverage families cross a split, and zero exact
sequence groups cross a split.

## Candidate generation

Three temporal-safe composite-TBM candidates have been generated for all 100 targets:

| split | TBM-ready targets | TBM candidates |
|:--|--:|--:|
| train | 60/60 | 180 |
| calibration | 20/20 | 60 |
| validation | 20/20 | 60 |

The private Kaggle GPU kernel
`datdo151000/geofuse-real-oof-drfold2-medium` reads only competition sequences and
exports the two highest-confidence direct DRfold2 candidates from 20 cfg97 checkpoints
per target. It never reads `train_labels.v2.csv`.

Version 1 completed in 2.573 GPU-hours and produced two candidates for 99/100 frozen
targets. Neural inference also completed for `8YUR_X`, but Arena segfaulted while
converting its highest-confidence structure. Version 2 tried all 20 confidence-ranked
checkpoint structures without native coordinates; Arena segfaulted on every one.

Before any native score was calculated, the failed train target was replaced by
`9EY0_T`, the first unused later eligible target in the same deterministic train-split
ordering. This logged technical replacement:

- keeps the split sizes at 60/20/20;
- keeps zero family and exact-sequence groups crossing splits;
- does not select by model accuracy;
- does not weaken the prediction to a raw-coordinate or C4′ proxy.

Version 3 predicted only the reserve target. It produced two valid structures from
20 checkpoint outputs in 88.6 inference seconds with no Arena failure.

After download and import with structural-training cutoff `2023-12-31`, the final
candidate audit is:

| split | ready targets | TBM candidates | DRfold2 candidates |
|:--|--:|--:|--:|
| train | 60/60 | 180 | 120 |
| calibration | 20/20 | 60 | 40 |
| validation | 20/20 | 60 | 40 |

Thus all 100 targets have exactly three temporal-safe TBM and two frozen DRfold2
candidates before the quality estimator sees native supervision.

## Resource boundary

TBM search and transfer ran locally on CPU. The pretrained inference is isolated on a
Kaggle GPU because the local RTX 3060 Ti has only 8 GB VRAM. Large templates, candidates,
model output and checkpoints remain ignored cache artifacts; Git stores the exact
selection/generation code, protocol and compact result reports.
