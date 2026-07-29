# GeoFuse experiment log

This is the append-only narrative log for the thesis experiments. Every entry records
the idea, frozen comparison, data boundary, result and conclusion. Generated tables
are stored under `reports/tables/`; large candidates and checkpoints remain ignored
cache artifacts.

## E00 — prior pipeline baseline

- **Idea:** temporal-safe template search should provide a reproducible RNA 3D
  baseline; adding complementary template databases should improve fold coverage.
- **Comparison:** dummy, TBM top-1/top-5, geometry refinement, reproduced leaderboard
  TBM and composite template search.
- **Result:** the early dummy/TBM top-1/TBM top-5/refined scores were
  0.0689/0.1544/0.1584/0.1612. Reproduced temporal-safe top-1 reached 0.2983, while a
  leaked reproduction reached 0.9355. Composite search improved 0.2117 to 0.3072.
  The temporal-safe composite TBM submission scored 0.60084 public and 0.60175 private.
- **Conclusion:** source coverage and leakage control dominate the early gains. The
  Kaggle submission was TBM-only; it was not evidence for pretrained fusion.

## E01 — pretrained candidate diversity

- **Idea:** frozen pretrained candidates may cover folds missed by templates.
- **Comparison:** template bank versus source-balanced template+pretrained bank.
- **Result:** on all 12 development targets, selected-union/oracle TM were
  0.4713/0.5123 versus TBM oracle 0.3143. Excluding overlap target R1128, they were
  0.4245/0.4693 versus TBM 0.3188.
- **Conclusion:** source complementarity is real on this development set, but the gap
  between selected and oracle scores shows a quality-selection bottleneck.

## E02 — geometry v1/v2

- **Idea:** an empirical geometry projection may remove locally implausible C1′
  configurations while preserving the fold.
- **Comparison:** same raw candidates, geometry v1 and geometry v2.
- **Result before independent local metrics:** raw/v2 TM were 0.471268/0.471308.
  Geometry v2 reduced clash proxy 0.0950 to 0.0466, adjacent-distance deviation 1.298
  to 1.076 and sharp-kink fraction 0.0481 to 0.0436. V1 worsened the kink fraction to
  0.0758.
- **Conclusion:** v2 is a safe projection by its own diagnostic objectives, not yet an
  accuracy improvement. E10 below re-evaluates this claim with C1′-lDDT and local RMSD.

## E03 — native-blind diagnostics

- **Idea:** source confidence and geometry features may rank candidate quality.
- **Comparison:** pooled, target-centred and within-source correlations with native TM.
- **Result:** pair-like fraction had target-centred Spearman rho about +0.736 and angle
  NLL about -0.593, but associations weakened or reversed within individual sources.
- **Conclusion:** the features mostly distinguish generators and are not calibrated
  enough to select among candidates from a single generator.

## E04 — heuristic clustering and fusion

- **Idea:** cluster global folds, fuse agreeing TBM/pretrained pairs and select for
  quality plus diversity.
- **Comparison:** source-balanced raw, quality raw, clustered raw, clustered augmented
  and raw/augmented oracle.
- **Result:** source-balanced/quality/clustered-augmented TM were
  0.4713/0.4972/0.4818; augmented oracle remained 0.5123. R1117 regressed materially.
- **Conclusion:** heuristic selection sometimes helps, but coordinate fusion created no
  oracle headroom and did not generalize safely.

## E05 — synthetic learned gate

- **Idea:** synthetic corruptions can bootstrap a small residue router.
- **Comparison:** learned gate versus gap rule on chronological synthetic validation.
- **Result:** 188/28/56 train/calibration/validation examples, 8,737 parameters,
  validation AUC 0.9869; learned error 0.8819 Å versus gap-rule 0.8958 Å.
- **Conclusion:** the implementation can learn the synthetic task. This does not prove
  transfer to real TBM/pretrained errors.

## E06 — frozen synthetic-to-real transfer

- **Idea:** apply the frozen synthetic gate to real candidates without retraining.
- **Comparison:** heuristic and learned fusion against raw oracle on three targets.
- **Result:** raw oracle 0.6170, heuristic fusion 0.5373, learned fusion 0.4462.
- **Conclusion:** strong synthetic-to-real domain gap; the synthetic gate failed.

## E07 — real OOF pilot

- **Idea:** supervise routing with actual, provenance-audited TBM and frozen DRfold2
  errors.
- **Comparison:** learned gate versus always-TBM, always-pretrained, gap and confidence
  rules on a 5/5/5 chronological/family-disjoint pilot.
- **Result:** held-out oracle residue error 3.2967 Å. Always pretrained was best among
  feasible rules at 6.9689 Å; learned/always-TBM/confidence/gap were
  7.6027/7.7180/7.8380/7.8780 Å. Gate AUC was 0.4815.
- **Conclusion:** strict failure. Real labels or sample size differ too much from the
  synthetic setting; five held-out targets are insufficient for a stable conclusion.

## E08 — confirmatory protocol

- **Idea:** replace project-only geometry evidence and the five-target pilot with
  recognized independent local metrics and a larger frozen OOF evaluation.
- **Protocol:** `geofuse_confirmatory_protocol.md`, frozen before E10 onward.
- **Status:** phase 8/all-atom validation intentionally excluded; phases 1–7 active.

## E09 — medium real-OOF cohort preparation

- **Hypothesis:** the five-target pilot is too small to distinguish router failure
  from sampling noise; a 60/20/20 real-source experiment is feasible without violating
  the pretrained structural cutoff.
- **Frozen selection:** exactly 100 targets of at most 100 nt, equally spaced across
  each chronological split after 80%-identity/80%-coverage family grouping.
- **Audit:** 60/20/20 targets, 54/17/16 families, zero family or exact-sequence groups
  crossing splits. The length ranges are 30–100, 30–91 and 40–96 nt.
- **TBM result:** all 100 targets ready with 300 temporal-safe TBM candidates.
- **Pretrained result:** private Kaggle kernel version 1 used 2.573 GPU-hours and
  returned two DRfold2 candidates for 99/100 targets. Arena segfaulted when converting
  all 20 checkpoint structures of `8YUR_X`, despite successful neural inference.
- **Technical replacement:** before native scoring, `8YUR_X` was replaced by
  `9EY0_T`, the first unused later eligible target in the same train-split ordering.
  Version 3 produced two candidates for the reserve from 20 checkpoints in 88.6
  inference seconds. The replacement preserves 60/20/20 and zero cross-split family
  or exact-sequence groups; it was not chosen by accuracy.
- **Final candidate audit:** all 100 targets are ready: train/calibration/validation
  contain 180/60/60 TBM candidates and 120/40/40 DRfold2 candidates, respectively.
- **Conclusion:** the data path is scaled, provenance-audited and frozen. No native
  result was read before candidate generation and technical repair finished.

## E10 onward — new confirmatory runs

Results are appended here only after the corresponding code, tests and frozen inputs
have been saved. A failed hypothesis remains a result and is not silently retuned on
final validation.

## E10 — independent local-metric geometry ablation

- **Hypothesis:** geometry v2 improves local native accuracy while preserving the
  global fold; its earlier improvement was not only an artefact of measuring its own
  optimization losses.
- **Frozen comparison:** the same 60 candidates across 12 targets, raw versus geometry
  v1 versus geometry v2. Native conformation is chosen by per-candidate TM first; C1′
  RMSD, C1′-lDDT and window RMSD all use that same conformation. Target bootstrap:
  10,000 samples, seed 2025.
- **Result:** raw/v2 best-of-five TM was 0.471268/0.471308. Mean C1′-lDDT increased
  0.472117 to 0.481823 (delta +0.009706, target-bootstrap 95% CI
  [+0.007403, +0.012617]); all 12 targets improved and 56/60 candidates improved.
  Window RMSD improved at 9 residues by 0.0572 Å (CI [+0.0351, +0.0917]) and at
  15 residues by 0.0236 Å (CI [-0.0129, +0.0662]), but the 31-residue result was
  essentially unchanged/slightly worse by 0.0049 Å (CI [-0.0685, +0.0518]).
  Global C1′ RMSD was also essentially unchanged.
- **Gate:** pass under the pre-registered rule: positive lDDT, improvement at two of
  three local scales and TM loss no worse than 0.005.
- **Conclusion:** geometry v2 now supports a narrow independent claim: it makes local
  native distance patterns modestly more accurate at short scales while preserving
  TM-score. It does not improve the long-window/global structure. Geometry v1 produced
  larger local gains but lost 0.0031 mean TM and has the known project-diagnostic kink
  regression, so it is not the safer final projection.
- **Artifacts:** `geofuse_independent_metrics.md` and
  `reports/tables/geofuse_independent_metrics/`.

## E11 — supervision-definition audit

- **Hypothesis:** the original residue label based on displacement after one global
  alignment is not equivalent to recognized local-structure metrics, and this label
  mismatch may contribute to poor routing.
- **Comparison:** pretrained-versus-TBM choice according to aligned point error,
  per-residue C1′-lDDT and 15-residue sliding-window RMSD on the 15-target real OOF
  pilot (jointly valid rows only).
- **Pilot result:** aligned point error agreed with C1′-lDDT on 68.3% of 3,760
  residue rows (target-centred Spearman rho 0.367) and with window RMSD on 72.0%
  (rho 0.493). C1′-lDDT and window RMSD agreed on 70.0% (rho 0.475).
- **Confirmatory result:** across all 100 targets, 352 source-pair examples and
  24,876 jointly valid residue rows, the corresponding agreement/rho values were
  69.7%/0.319, 72.7%/0.350 and 70.9%/0.418.
- **Conclusion:** the labels share signal but are not interchangeable. A displacement
  label after a whole-fold fit can call a residue bad because a remote domain is
  misplaced, while lDDT/window RMSD focus on local relationships. Subsequent
  confirmatory routing uses C1′-lDDT as primary supervision and retains the other two
  as ablations.
- **Status:** the larger audit confirms the pilot finding.

## E12 — real-OOF quality estimator benchmark

- **Hypothesis:** with more real TBM/pretrained errors and locally meaningful
  supervision, a small native-blind model can identify which source is locally better
  on unseen, newer targets.
- **Frozen design:** 60 train, 20 calibration and 20 newest validation targets;
  candidate generation and provenance were completed before scoring. Logistic,
  histogram gradient boosting and a small Conv1D were compared with always-TBM,
  always-pretrained, template-gap and raw-confidence rules. All configuration and
  threshold choices used calibration only. Target means weight each RNA equally.
- **Primary C1′-lDDT result:** calibration selected Conv1D at threshold 0.75. On
  validation it reduced mean `1-lDDT` error to 0.238994, versus
  always-pretrained 0.269575, gap 0.322453, confidence 0.326173 and always-TBM
  0.347735. ROC-AUC was 0.729868. Against the calibration-selected gap baseline,
  target-bootstrap mean improvement was +0.083458, 95% CI
  [+0.023419, +0.155562], with 14/20 targets improved.
- **Label ablations:** Conv1D also beat every fixed baseline in mean aligned-point
  error (6.5897 Å; strongest baseline 7.2129 Å) and 15-residue window RMSD
  (3.1310 Å; strongest baseline 3.1812 Å). Their paired bootstrap CIs crossed zero:
  [-1.1637, +2.5769] Å and [-0.1610, +0.2435] Å, so those two supporting effects
  are less stable than the primary C1′-lDDT result.
- **Learning-curve finding:** calibration error did not improve monotonically from
  10 to 60 targets. More real data enabled the confirmatory pass, but model capacity,
  label noise and distribution shift remain limiting factors rather than sample count
  alone.
- **Conclusion:** the five-target pilot failure was not the final answer. A compact
  real-OOF router can generalize enough to select a locally better parent source under
  the primary recognized metric. This establishes a local source-selection result,
  not yet a coordinate-fusion or TM-score improvement.

## E13 — native-blind global clustering ablation

- **Hypothesis:** self-TM clustering can prevent coordinate fusion between different
  global folds while retaining mixed-source pairs that plausibly describe the same
  fold.
- **Frozen comparison:** self-TM thresholds 0.35, 0.45 and 0.55 on the 20 calibration
  targets; selection maximizes F2 selected C1′-lDDT, then selected TM, before opening
  validation.
- **Result:** all three thresholds gave the same calibration aggregates; threshold
  0.35 was frozen by deterministic ordering. Mixed TBM/pretrained clusters occurred
  in 3/20 calibration targets and 12/20 validation targets.
- **Conclusion:** clustering supplies a valid native-blind safety boundary and shows
  that the newer validation set has more opportunities for same-fold cross-source
  comparison. This cohort does not identify a uniquely superior threshold.

## E14 — selective fusion with abstention

- **Hypothesis:** a calibrated quality router, same-fold constraint, seven-residue
  persistence and disagreement bounds can turn local parent-selection skill into a
  better coordinate candidate without risking the raw parents.
- **Frozen variants:** F0 raw; F1 old heuristic; F2 quality-gated selective fusion;
  F3 F2 plus geometry v2; F4 native-guided diagnostic. Raw parents are retained in
  every augmented bank.
- **Result:** calibration selected F2, but its threshold 0.75 plus 0.15 confidence
  margin and segment/disagreement conditions caused it to abstain on every mixed pair.
  Therefore final F2/F3 exactly matched F0: selected/oracle TM 0.588304 and
  selected/oracle C1′-lDDT 0.795059, with 20/20 target ties and no material
  regressions. F1 added 24 candidates on 12 validation targets and raised selected TM
  to 0.589985 and oracle TM to 0.592826, but lowered selected lDDT to 0.793841.
  F1's selected-TM delta CI crossed zero [-0.004908, +0.008495], while its oracle-TM
  headroom was +0.004522 with CI [+0.000795, +0.010107]; native-blind selection did
  not reliably realize that headroom. Non-deployable F4 reached TM 0.589158 and lDDT
  0.795648.
- **Gate:** fail, because quality-gated augmentation created no oracle TM headroom.
- **Conclusion:** improved local source classification does not automatically produce
  useful Cartesian fusion. Under the frozen conservative rule, the scientifically
  correct behavior is abstention and retention of raw parents. The heuristic's small
  TM gain trades against local accuracy and is not a robust confirmatory win.
