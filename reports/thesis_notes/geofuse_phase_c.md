# GeoFuse Phase C — fold clustering and heuristic fusion

All clustering, fusion weights, geometry quality scores, and final-five choices are fixed before native labels are read. Native TM is joined only post hoc.

- Targets: 12
- Fold threshold: 0.35 complete-link self-TM
- Generated fused/projected candidates: 12
- Final sets containing a fusion: 1/12
- Gate: **fail**
- Sensitivity gate excluding R1128: **fail**

Gate requires both positive native-blind final-five gain and positive augmented oracle gain. A selector improvement alone does not prove coordinate fusion works.

## Full aggregate

|                             |   mean |
|:----------------------------|-------:|
| source_balanced_raw_tm      | 0.4713 |
| quality_raw_tm              | 0.4972 |
| cluster_raw_tm              | 0.4818 |
| cluster_augmented_tm        | 0.4818 |
| raw_oracle_tm               | 0.5123 |
| augmented_oracle_tm         | 0.5123 |
| augmented_oracle_gain       | 0      |
| augmented_selection_regret  | 0.0305 |
| source_balanced_raw_self_tm | 0.2941 |
| quality_raw_self_tm         | 0.4681 |
| cluster_raw_self_tm         | 0.3137 |
| cluster_augmented_self_tm   | 0.3133 |
| refinement_seconds          | 4.1361 |

Selector gain over source-balanced raw: +0.010577 TM.
Quality-only gain over source-balanced raw: +0.025919 TM.

## Sensitivity excluding R1128

|                             |   mean |
|:----------------------------|-------:|
| source_balanced_raw_tm      | 0.4245 |
| quality_raw_tm              | 0.4528 |
| cluster_raw_tm              | 0.436  |
| cluster_augmented_tm        | 0.436  |
| raw_oracle_tm               | 0.4693 |
| augmented_oracle_tm         | 0.4693 |
| augmented_oracle_gain       | 0      |
| augmented_selection_regret  | 0.0333 |
| source_balanced_raw_self_tm | 0.2924 |
| quality_raw_self_tm         | 0.4246 |
| cluster_raw_self_tm         | 0.3191 |
| cluster_augmented_self_tm   | 0.3187 |
| refinement_seconds          | 4.5121 |

Sensitivity selector gain: +0.011538 TM.
Sensitivity quality-only gain: +0.028275 TM.

## Interpretation

- `raw_oracle` vs `augmented_oracle` isolates whether fusion creates a better fold hypothesis; selected scores measure the native-blind routing problem.
- Heuristic fusion did not raise oracle TM. It should remain an experimental candidate generator, not replace either parent source.
- The largest selected-set regression is R1117v2 (-0.1817 TM), showing that the hand-built cross-source confidence score is not calibrated.
- The next justified step is a leakage-safe learned confidence gate. Further weight tuning on these 12 native-scored targets would overfit the development set.
- Lower selected self-TM means more fold diversity. Diversity is useful only if the selected best-of-five TM is preserved or improved.
- This is development-set evidence. R1128 is reported separately because of the known exact pretrained-training overlap; pretrained cutoffs remain distinct from the temporal-safe TBM/prior claim.

## Reproducibility

- Max mixed clusters fused per target: 3
- Geometry projection steps: 300
- Fusion config: `{"alignment_iterations": 3, "alignment_trim_fraction": 0.8, "max_supported_disagreement": 12.0, "pretrained_heavy_floor": 0.7, "reliable_template_confidence": 0.5, "reliable_template_partner_cap": 0.15, "smoothing_radius": 2, "unsupported_partner_weight": 0.9}`
- Selection config: `{"cluster_support_weight": 0.1, "diversity_weight": 0.25, "new_cluster_bonus": 0.2}`
- The 0.35 threshold was chosen from a native-blind cross-source self-TM audit on the pilot targets; it was not selected from fusion/native TM outcomes.

## Per-target

| target_id   |   seq_len |   n_raw |   n_fused |   n_raw_clusters |   n_augmented_clusters |   n_mixed_source_clusters |   raw_oracle_tm |   augmented_oracle_tm |   fusion_oracle_tm |   refinement_seconds |   source_balanced_raw_tm |   source_balanced_raw_self_tm |   quality_raw_tm |   quality_raw_self_tm |   cluster_raw_tm |   cluster_raw_self_tm |   cluster_augmented_tm |   cluster_augmented_self_tm |   augmented_oracle_gain |   augmented_selection_regret |
|:------------|----------:|--------:|----------:|-----------------:|-----------------------:|--------------------------:|----------------:|----------------------:|-------------------:|---------------------:|-------------------------:|------------------------------:|-----------------:|----------------------:|-----------------:|----------------------:|-----------------------:|----------------------------:|------------------------:|-----------------------------:|
| R1107       |        69 |      16 |         4 |                7 |                      7 |                         1 |          0.6136 |                0.6136 |             0.5448 |              14.875  |                   0.5973 |                        0.2896 |           0.6136 |                0.4016 |           0.5973 |                0.322  |                 0.5973 |                      0.3205 |                       0 |                       0.0162 |
| R1108       |        69 |      15 |         4 |                5 |                      5 |                         1 |          0.6019 |                0.6019 |             0.5982 |              14.9114 |                   0.6019 |                        0.3113 |           0.6019 |                0.4685 |           0.6019 |                0.4685 |                 0.6019 |                      0.4656 |                       0 |                       0      |
| R1116       |       157 |      15 |         4 |                8 |                      8 |                         1 |          0.5379 |                0.5379 |             0.5298 |              19.8463 |                   0.5149 |                        0.3709 |           0.5379 |                0.7135 |           0.5149 |                0.5221 |                 0.5149 |                      0.5221 |                       0 |                       0.023  |
| R1117v2     |        30 |      20 |         0 |               14 |                     14 |                         0 |          0.4616 |                0.4616 |           nan      |               0      |                   0.4616 |                        0.348  |           0.2799 |                0.2649 |           0.2799 |                0.1696 |                 0.2799 |                      0.1696 |                       0 |                       0.1817 |
| R1126       |       363 |      15 |         0 |               12 |                     12 |                         0 |          0.3593 |                0.3593 |           nan      |               0      |                   0.3405 |                        0.1805 |           0.3593 |                0.2975 |           0.3593 |                0.231  |                 0.3593 |                      0.231  |                       0 |                       0      |
| R1128       |       238 |      15 |         0 |                9 |                      9 |                         0 |          0.9857 |                0.9857 |           nan      |               0      |                   0.9857 |                        0.312  |           0.9857 |                0.9472 |           0.9857 |                0.2545 |                 0.9857 |                      0.2545 |                       0 |                       0      |
| R1136       |       374 |      15 |         0 |               12 |                     12 |                         0 |          0.3814 |                0.3814 |           nan      |               0      |                   0.2444 |                        0.1937 |           0.3814 |                0.2797 |           0.3814 |                0.2797 |                 0.3814 |                      0.2797 |                       0 |                       0      |
| R1138       |       720 |      11 |         0 |                5 |                      5 |                         0 |          0.2751 |                0.2751 |           nan      |               0      |                   0.2751 |                        0.6421 |           0.2751 |                0.5999 |           0.2751 |                0.3859 |                 0.2751 |                      0.3859 |                       0 |                       0      |
| R1149       |       124 |      15 |         0 |               11 |                     11 |                         0 |          0.7325 |                0.7325 |           nan      |               0      |                   0.6645 |                        0.2699 |           0.7325 |                0.7359 |           0.7325 |                0.3776 |                 0.7325 |                      0.3776 |                       0 |                       0      |
| R1156       |       135 |      15 |         0 |               12 |                     12 |                         0 |          0.6997 |                0.6997 |           nan      |               0      |                   0.47   |                        0.2135 |           0.6997 |                0.422  |           0.5548 |                0.2663 |                 0.5548 |                      0.2663 |                       0 |                       0.1449 |
| R1189       |       118 |      15 |         0 |               14 |                     14 |                         0 |          0.2289 |                0.2289 |           nan      |               0      |                   0.2289 |                        0.1985 |           0.2289 |                0.2435 |           0.2289 |                0.2435 |                 0.2289 |                      0.2435 |                       0 |                       0      |
| R1190       |       118 |      15 |         0 |               14 |                     14 |                         0 |          0.2703 |                0.2703 |           nan      |               0      |                   0.2703 |                        0.1985 |           0.2703 |                0.2435 |           0.2703 |                0.2435 |                 0.2703 |                      0.2435 |                       0 |                       0      |
