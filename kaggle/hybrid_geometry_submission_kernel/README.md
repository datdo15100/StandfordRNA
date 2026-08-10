# Kaggle hybrid + Geometry v2 submission kernel

This private, offline GPU kernel runs the native-blind pre-GeoFuse pipeline on
the runtime competition sequences (including Kaggle's hidden rerun):

1. temporal-safe composite TBM candidate generation;
2. DRfold2 cfg97 direct candidate generation for targets up to 600 nt;
3. fixed 3-TBM + 2-DRfold2 assembly, with TBM fallback on model failure;
4. Geometry v2 refinement with the frozen 300-step configuration;
5. strict schema, row-order and finite-coordinate checks.

The inference manifest records that native labels are not loaded. GeoFuse is not
part of this ablation. The kernel uses a separate ID from the scored TBM-only
baseline.
