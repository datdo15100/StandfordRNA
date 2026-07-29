# GeoFuse confirmatory clustering and selective fusion

This combines E13 (global/local clustering) and E14 (selective fusion). Threshold and deployable variant are selected only on calibration. The newest 20 targets are evaluated once after freezing. Every augmented bank retains all raw parents.

- Quality estimator: conv1d (c1_lddt)
- Router decision threshold: 0.7500
- Calibration-selected fold threshold: 0.35
- Calibration-selected deployable variant: **F2_quality**
- Confirmatory fusion gate: **fail**
- Validation material regressions (>0.05 TM or lDDT): 0

## Calibration: global-cluster threshold ablation

|   fold_threshold | variant             |   selected_tm |   selected_lddt |   oracle_tm |   oracle_lddt |   mixed_clusters |
|-----------------:|:--------------------|--------------:|----------------:|------------:|--------------:|-----------------:|
|             0.35 | F0_raw              |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.35 | F1_heuristic        |      0.407436 |        0.591713 |    0.407436 |      0.591713 |             0.15 |
|             0.35 | F2_quality          |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.35 | F3_quality_geometry |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.35 | F4_oracle           |      0.407436 |        0.590646 |    0.407436 |      0.590646 |             0.15 |
|             0.45 | F0_raw              |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.45 | F1_heuristic        |      0.407436 |        0.591713 |    0.407436 |      0.591713 |             0.15 |
|             0.45 | F2_quality          |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.45 | F3_quality_geometry |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.45 | F4_oracle           |      0.407436 |        0.590646 |    0.407436 |      0.590646 |             0.15 |
|             0.55 | F0_raw              |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.55 | F1_heuristic        |      0.407436 |        0.591713 |    0.407436 |      0.591713 |             0.15 |
|             0.55 | F2_quality          |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.55 | F3_quality_geometry |      0.407436 |        0.589968 |    0.407436 |      0.589968 |             0.15 |
|             0.55 | F4_oracle           |      0.407436 |        0.590646 |    0.407436 |      0.590646 |             0.15 |

Threshold selection maximizes F2 selected C1′-lDDT, then selected TM. F2 versus F3 is then frozen by the same calibration ordering.

## Final newest-target variants

|   fold_threshold | variant             |   selected_tm |   selected_lddt |   oracle_tm |   oracle_lddt |   mixed_clusters |
|-----------------:|:--------------------|--------------:|----------------:|------------:|--------------:|-----------------:|
|             0.35 | F0_raw              |      0.588304 |        0.795059 |    0.588304 |      0.795059 |              0.6 |
|             0.35 | F1_heuristic        |      0.589984 |        0.793841 |    0.592826 |      0.795828 |              0.6 |
|             0.35 | F2_quality          |      0.588304 |        0.795059 |    0.588304 |      0.795059 |              0.6 |
|             0.35 | F3_quality_geometry |      0.588304 |        0.795059 |    0.588304 |      0.795059 |              0.6 |
|             0.35 | F4_oracle           |      0.589158 |        0.795648 |    0.58916  |      0.795648 |              0.6 |

F0 is raw parents; F1 is the old heuristic; F2 is quality-gated fusion with abstention; F3 projects F2 with geometry v2; F4 reads native local lDDT and is a non-deployable native-guided diagnostic. The exact source-selection error lower bound is the `oracle_residue` row in E12.

- Selected TM delta over F0: +0.000000
- Selected C1′-lDDT delta over F0: +0.000000
- Augmented oracle TM gain over F0: +0.000000

## Target bootstrap versus F0

### Selected TM

|              |   value |
|:-------------|--------:|
| n_targets    |      20 |
| mean_delta   |       0 |
| median_delta |       0 |
| ci_low       |       0 |
| ci_high      |       0 |
| improved     |       0 |
| tied         |      20 |
| regressed    |       0 |

### Selected C1′-lDDT

|              |   value |
|:-------------|--------:|
| n_targets    |      20 |
| mean_delta   |       0 |
| median_delta |       0 |
| ci_low       |       0 |
| ci_high      |       0 |
| improved     |       0 |
| tied         |      20 |
| regressed    |       0 |

## Supporting F1 heuristic diagnostic

F1 was not the calibration-selected confirmatory method. These paired target summaries
are descriptive and show whether its extra candidates created headroom that
native-blind final-five selection could realize.

### F1 selected TM versus F0

|              |     value |
|:-------------|----------:|
| n_targets    | 20        |
| mean_delta   |  0.001681 |
| median_delta |  0        |
| ci_low       | -0.004908 |
| ci_high      |  0.008495 |
| improved     |  5        |
| tied         | 12        |
| regressed    |  3        |

### F1 selected C1′-lDDT versus F0

|              |     value |
|:-------------|----------:|
| n_targets    | 20        |
| mean_delta   | -0.001218 |
| median_delta |  0        |
| ci_low       | -0.004476 |
| ci_high      |  0.001237 |
| improved     |  3        |
| tied         | 15        |
| regressed    |  2        |

### F1 oracle TM versus F0

|              |    value |
|:-------------|---------:|
| n_targets    | 20       |
| mean_delta   | 0.004522 |
| median_delta | 0        |
| ci_low       | 0.000795 |
| ci_high      | 0.010107 |
| improved     | 5        |
| tied         | 15       |
| regressed    | 0        |

## Interpretation

- Selective fusion fails the frozen gate. Raw parents remain the deployable choice; F4 is diagnostic headroom but is not a method.
- Local disagreement is descriptive and native-blind. A same-fold cluster does not imply that either source is locally correct.
