# GeoFuse Phase A — findings and next decision

## What was tested

The experiment asks whether pretrained inference contributes folds that the
current temporal-safe TBM search does not contain. Each validation target has
10 TBM candidates. The pretrained bank contains five direct cfg97 DRfold2
checkpoint hypotheses for each target up to 374 nt, the earlier optimized
DRfold candidates for the two pilot targets, and one Boltz-1 hypothesis for the
720-nt `R1138`. Native coordinates are used only after candidate generation to
measure TM-score and oracle ceilings.

## Main result

| Evaluation slice | Targets | TBM selected | Union selected | TBM oracle | Union oracle | Oracle gain |
|:--|--:|--:|--:|--:|--:|--:|
| Competition-style, all validation targets | 12 | 0.3113 | 0.4713 | 0.3143 | 0.5123 | +0.1981 |
| Exact-overlap sensitivity (`R1128` excluded) | 11 | 0.3155 | 0.4245 | 0.3188 | 0.4693 | +0.1505 |

The Phase-A gate passes in both views. In the sensitivity slice, pretrained
candidates improve the union oracle on 10/11 targets and tie on one. Thus the
new branch adds useful fold hypotheses even after removing the largest known
DRfold2 training overlap.

This does **not** mean pretrained is uniformly better. `R1117v2` has TBM oracle
0.4616 while pretrained reaches only 0.2799; the union stays at the TBM result.
Conversely, on temporally suspect `R1128`, DRfold2 reaches 0.9857 versus TBM
0.2642, illustrating exactly why the overlap audit must accompany the headline
competition result.

## Why routing/fusion is now the bottleneck

Across all 55 direct DRfold2 candidates, global confidence and TM-score have a
moderate correlation (`Spearman rho = 0.486`). That aggregate relationship is
misleading for model selection: within each target's five hypotheses, the mean
Spearman correlation is `-0.036` (median `-0.100`). Examples:

- `R1156`: confidence-top TM 0.3364, pretrained oracle 0.6997;
- `R1136`: confidence-top TM 0.2272, pretrained oracle 0.3814;
- `R1149`: confidence-top TM 0.5702, pretrained oracle 0.7325.

The current native-blind union selection averages TM 0.4713 versus oracle
0.5123. Mean selection regret is 0.0411 and reaches 0.2297 on `R1156`.
Candidate generation is therefore good enough to continue; ranking and
cross-source calibration are the next measurable bottleneck.

## Decision for the next phase

Proceed with the GeoFuse plan, in this order:

1. compute fold-level and residue-level geometry diagnostics for every cached
   candidate without looking at native labels;
2. calibrate TBM, DRfold2 and Boltz confidence onto comparable scales;
3. learn or validate a native-blind router that preserves the complementary
   TBM fold on cases such as `R1117v2`;
4. apply geometry refinement only after routing, and evaluate both TM-score and
   physical-validity metrics;
5. keep the full competition view and temporal/overlap sensitivity view as
   separate result tables throughout the thesis.

The contribution is therefore not “TBM plus a copied pretrained notebook.” It
is the audited candidate contract, evidence that candidate sources are
complementary, and a geometry-aware confidence/fusion method designed to close
the observed selection gap.
