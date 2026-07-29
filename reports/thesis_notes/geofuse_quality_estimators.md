# GeoFuse quality estimators — c1_lddt

This is E12 under the frozen 60/20/20 protocol. Learning curves and all hyperparameters/thresholds use train+calibration only. The confirmatory estimator and strongest fixed baseline are selected on calibration before the 20 newest validation targets are evaluated once.

- Ready target counts: `{'train': 60, 'calibration': 20, 'validation': 20}`
- Rejected pair attempts: 0
- Supervision/evaluation unit: 1-minus-lddt
- Calibration-selected estimator: **conv1d**
- Calibration-selected strongest baseline: **gap_rule**
- Confirmatory router gate: **pass**
- Runtime: 132.6 seconds

## Calibration learning curves

|   train_targets | model             |   calibration_target_error |   calibration_auc |   threshold |
|----------------:|:------------------|---------------------------:|------------------:|------------:|
|              10 | logistic          |                   0.436239 |          0.719689 |       0.825 |
|              10 | gradient_boosting |                   0.444874 |          0.674203 |       0.9   |
|              10 | conv1d            |                   0.436279 |          0.72859  |       0.575 |
|              25 | logistic          |                   0.435007 |          0.736259 |       0.725 |
|              25 | gradient_boosting |                   0.443247 |          0.691237 |       0.75  |
|              25 | conv1d            |                   0.435755 |          0.736664 |       0.575 |
|              40 | logistic          |                   0.436743 |          0.72756  |       0.725 |
|              40 | gradient_boosting |                   0.441791 |          0.703497 |       0.775 |
|              40 | conv1d            |                   0.435681 |          0.740113 |       0.725 |
|              60 | logistic          |                   0.435074 |          0.747502 |       0.675 |
|              60 | gradient_boosting |                   0.44014  |          0.723847 |       0.75  |
|              60 | conv1d            |                   0.433555 |          0.736342 |       0.75  |

## Final newest-target results

| model             |   target_mean_error |   residue_mean_error |   accuracy |   roc_auc |   pretrained_fraction |   threshold |   targets |   residues |
|:------------------|--------------------:|---------------------:|-----------:|----------:|----------------------:|------------:|----------:|-----------:|
| oracle_residue    |            0.193424 |             0.179536 |   0.923691 |  0.934561 |              0.506751 |       0.5   |        20 |       4888 |
| conv1d            |            0.238994 |             0.224007 |   0.650573 |  0.729868 |              0.378069 |       0.75  |        20 |       4888 |
| logistic          |            0.247937 |             0.232183 |   0.628682 |  0.687387 |              0.641367 |       0.675 |        20 |       4888 |
| gradient_boosting |            0.250951 |             0.232962 |   0.643208 |  0.731747 |              0.41653  |       0.75  |        20 |       4888 |
| always_pretrained |            0.269575 |             0.253345 |   0.583061 |  0.5      |              1        |       0.5   |        20 |       4888 |
| gap_rule          |            0.322453 |             0.314133 |   0.458265 |  0.534949 |              0.04419  |       0.5   |        20 |       4888 |
| confidence_rule   |            0.326173 |             0.317661 |   0.452332 |  0.530001 |              0.037439 |       0.5   |        20 |       4888 |
| always_tbm        |            0.347735 |             0.337613 |   0.416939 |  0.5      |              0        |       0.5   |        20 |       4888 |

## Target bootstrap: selected estimator versus calibration-selected baseline

|              |     value |
|:-------------|----------:|
| n_targets    | 20        |
| mean_delta   |  0.083458 |
| median_delta |  0.01256  |
| ci_low       |  0.023419 |
| ci_high      |  0.155562 |
| improved     | 14        |
| tied         |  0        |
| regressed    |  6        |

Positive bootstrap delta means the selected estimator has lower error. The pass criterion requires it to beat always-TBM, always-pretrained, gap and raw-confidence rules on the equal-weight target mean.

## Interpretation

- The learned router passes the frozen real-OOF gate.
- The per-residue oracle is non-deployable and appears only to quantify remaining headroom.
