# GeoFuse Phase D — frozen synthetic-gate transfer pilot

All clustering, fusion weights, geometry quality scores, and final-five choices are fixed before native labels are read. Native TM is joined only post hoc.

- Targets: 3
- Fold threshold: 0.35 complete-link self-TM
- Generated fused/projected candidates: 12
- Final sets containing a fusion: 0/3
- Gate: **fail**
- Sensitivity gate excluding R1128: **fail**

Gate requires both positive native-blind final-five gain and positive augmented oracle gain. A selector improvement alone does not prove coordinate fusion works.

## Full aggregate

|                             |   mean |
|:----------------------------|-------:|
| source_balanced_raw_tm      | 0.5274 |
| quality_raw_tm              | 0.617  |
| cluster_raw_tm              | 0.5557 |
| cluster_augmented_tm        | 0.5557 |
| raw_oracle_tm               | 0.617  |
| augmented_oracle_tm         | 0.617  |
| heuristic_fusion_oracle_tm  | 0.5373 |
| learned_fusion_oracle_tm    | 0.4462 |
| augmented_oracle_gain       | 0      |
| augmented_selection_regret  | 0.0614 |
| source_balanced_raw_self_tm | 0.2913 |
| quality_raw_self_tm         | 0.5123 |
| cluster_raw_self_tm         | 0.3701 |
| cluster_augmented_self_tm   | 0.3696 |
| refinement_seconds          | 5.3018 |

Selector gain over source-balanced raw: +0.028280 TM.
Quality-only gain over source-balanced raw: +0.089650 TM.

## Sensitivity excluding R1128

|                             |   mean |
|:----------------------------|-------:|
| source_balanced_raw_tm      | 0.5274 |
| quality_raw_tm              | 0.617  |
| cluster_raw_tm              | 0.5557 |
| cluster_augmented_tm        | 0.5557 |
| raw_oracle_tm               | 0.617  |
| augmented_oracle_tm         | 0.617  |
| heuristic_fusion_oracle_tm  | 0.5373 |
| learned_fusion_oracle_tm    | 0.4462 |
| augmented_oracle_gain       | 0      |
| augmented_selection_regret  | 0.0614 |
| source_balanced_raw_self_tm | 0.2913 |
| quality_raw_self_tm         | 0.5123 |
| cluster_raw_self_tm         | 0.3701 |
| cluster_augmented_self_tm   | 0.3696 |
| refinement_seconds          | 5.3018 |

Sensitivity selector gain: +0.028280 TM.
Sensitivity quality-only gain: +0.089650 TM.

## Interpretation

- `raw_oracle` vs `augmented_oracle` isolates whether fusion creates a better fold hypothesis; selected scores measure the native-blind routing problem.
- Heuristic fusion did not raise oracle TM. It should remain an experimental candidate generator, not replace either parent source.
- The selected set did not regress on this target subset; fusion quality is assessed separately by its oracle ceiling.
- The next justified step is real out-of-fold gate supervision from homolog pairs/model predictions. Further synthetic or heuristic tuning on these native-scored targets would overfit the development set.
- Frozen learned-vs-heuristic fusion oracle delta: -0.0911 TM. A negative value is evidence that the synthetic gate did not transfer to real candidate errors.
- Lower selected self-TM means more fold diversity. Diversity is useful only if the selected best-of-five TM is preserved or improved.
- This is development-set evidence. R1128 is reported separately because of the known exact pretrained-training overlap; pretrained cutoffs remain distinct from the temporal-safe TBM/prior claim.

## Reproducibility

- Max mixed clusters fused per target: 3
- Geometry projection steps: 100
- Fusion config: `{"alignment_iterations": 3, "alignment_trim_fraction": 0.8, "max_supported_disagreement": 12.0, "pretrained_heavy_floor": 0.7, "reliable_template_confidence": 0.5, "reliable_template_partner_cap": 0.15, "smoothing_radius": 2, "unsupported_partner_weight": 0.9}`
- Selection config: `{"cluster_support_weight": 0.1, "diversity_weight": 0.25, "new_cluster_bonus": 0.2}`
- Learned gate checkpoint digest: `341b9408778d6221`
- The 0.35 threshold was chosen from a native-blind cross-source self-TM audit on the pilot targets; it was not selected from fusion/native TM outcomes.

## Per-target

| target_id   |   seq_len |   n_raw |   n_fused |   n_raw_clusters |   n_augmented_clusters |   n_mixed_source_clusters |   raw_oracle_tm |   augmented_oracle_tm |   fusion_oracle_tm |   heuristic_fusion_oracle_tm |   learned_fusion_oracle_tm |   refinement_seconds |   source_balanced_raw_tm |   source_balanced_raw_self_tm |   quality_raw_tm |   quality_raw_self_tm |   cluster_raw_tm |   cluster_raw_self_tm |   cluster_augmented_tm |   cluster_augmented_self_tm |   augmented_oracle_gain |   augmented_selection_regret |
|:------------|----------:|--------:|----------:|-----------------:|-----------------------:|--------------------------:|----------------:|----------------------:|-------------------:|-----------------------------:|---------------------------:|---------------------:|-------------------------:|------------------------------:|-----------------:|----------------------:|-----------------:|----------------------:|-----------------------:|----------------------------:|------------------------:|-----------------------------:|
| R1107       |        69 |      16 |         6 |                7 |                      7 |                         1 |          0.6136 |                0.6136 |             0.5448 |                       0.5448 |                     0.4026 |               9.4843 |                   0.5973 |                        0.2896 |           0.6136 |                0.4016 |           0.5973 |                0.322  |                 0.5973 |                      0.3205 |                       0 |                       0.0162 |
| R1116       |       157 |      15 |         6 |                8 |                      8 |                         1 |          0.5379 |                0.5379 |             0.5298 |                       0.5298 |                     0.4898 |               6.421  |                   0.5149 |                        0.3709 |           0.5379 |                0.7135 |           0.5149 |                0.5221 |                 0.5149 |                      0.5221 |                       0 |                       0.023  |
| R1156       |       135 |      15 |         0 |               12 |                     12 |                         0 |          0.6997 |                0.6997 |           nan      |                     nan      |                   nan      |               0      |                   0.47   |                        0.2135 |           0.6997 |                0.422  |           0.5548 |                0.2663 |                 0.5548 |                      0.2663 |                       0 |                       0.1449 |
