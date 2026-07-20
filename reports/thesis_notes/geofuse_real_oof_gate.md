# GeoFuse real-OOF confidence gate

This experiment uses actual temporal-safe TBM candidates and frozen pretrained-model predictions. Native coordinates are used only to create residue labels and metrics; candidate selection and all gate features are native-blind.

- Gate: **fail**
- Runtime: 12.3 seconds
- Pair counts: `{'train': 20, 'calibration': 20, 'validation': 20}`
- Target counts: `{'train': 5, 'calibration': 5, 'validation': 5}`
- Rejected pair attempts: 0
- Initialized from synthetic gate: `True`

## Held-out newest-target metrics

|                            |      value |
|:---------------------------|-----------:|
| weighted_bce               |    0.86342 |
| accuracy                   |    0.49434 |
| balanced_accuracy          |    0.51464 |
| roc_auc                    |    0.48152 |
| brier                      |    0.31981 |
| ece_10bin                  |    0.20907 |
| template_error             |    7.71795 |
| pretrained_error           |    6.96894 |
| oracle_residue_error       |    3.29674 |
| learned_gate_error         |    7.60272 |
| gap_rule_error             |    7.87834 |
| confidence_rule_error      |    7.83797 |
| n_residues                 | 1236       |
| pretrained_better_fraction |    0.52265 |
| decision_threshold         |    0.675   |
| n_parameters               | 8737       |

The decision threshold (0.675) was selected only on calibration data; its calibration gate error was 15.3137 Å.
The pass criterion requires lower held-out error than always-template, always-pretrained, gap-rule, and confidence-rule baselines.

## Training history

|   epoch |   train_loss |   calibration_loss |   calibration_auc |
|--------:|-------------:|-------------------:|------------------:|
|       1 |      1.16138 |            1.48081 |           0.6369  |
|       2 |      1.12554 |            1.41297 |           0.637   |
|       3 |      1.11202 |            1.35029 |           0.63654 |
|       4 |      1.02479 |            1.2903  |           0.63574 |
|       5 |      1.00087 |            1.23305 |           0.63546 |
|       6 |      0.86174 |            1.17993 |           0.63561 |
|       7 |      0.82373 |            1.13153 |           0.63522 |
|       8 |      0.85045 |            1.08791 |           0.63492 |
|       9 |      0.82443 |            1.04599 |           0.63471 |
|      10 |      0.74674 |            1.00734 |           0.63409 |
|      11 |      0.77724 |            0.97118 |           0.63372 |
|      12 |      0.70759 |            0.93673 |           0.63325 |

## Leakage boundary

The manifest groups >=80% identity sequences when prepared with MMseqs. Every pretrained candidate must declare either a structural-training cutoff older than its target or an explicit exclusion manifest. Every TBM template must predate its target and direct target PDB IDs are rejected. Sequence-language-model pretraining overlap is not claimed absent.
