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

## E11 — supervision-definition audit (pilot)

- **Hypothesis:** the original residue label based on displacement after one global
  alignment is not equivalent to recognized local-structure metrics, and this label
  mismatch may contribute to poor routing.
- **Comparison:** pretrained-versus-TBM choice according to aligned point error,
  per-residue C1′-lDDT and 15-residue sliding-window RMSD on the 15-target real OOF
  pilot (jointly valid rows only).
- **Result:** aligned point error agreed with C1′-lDDT on 68.3% of 3,760 residue rows
  (target-centred Spearman rho 0.367) and with window RMSD on 72.0% (rho 0.493).
  C1′-lDDT and window RMSD agreed on 70.0% (rho 0.475).
- **Conclusion:** the labels share signal but are not interchangeable. A displacement
  label after a whole-fold fit can call a residue bad because a remote domain is
  misplaced, while lDDT/window RMSD focus on local relationships. Subsequent
  confirmatory routing uses C1′-lDDT as primary supervision and retains the other two
  as ablations.
- **Status:** pilot finding; the report will be regenerated on the frozen 100-target
  set after pretrained candidates finish.
