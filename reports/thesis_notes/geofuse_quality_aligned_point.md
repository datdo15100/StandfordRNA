# GeoFuse quality estimators — aligned_point

This is E12 under the frozen 60/20/20 protocol. Learning curves and all hyperparameters/thresholds use train+calibration only. The confirmatory estimator and strongest fixed baseline are selected on calibration before the 20 newest validation targets are evaluated once.

- Ready target counts: `{'train': 60, 'calibration': 20, 'validation': 20}`
- Rejected pair attempts: 0
- Supervision/evaluation unit: angstrom
- Calibration-selected estimator: **conv1d**
- Calibration-selected strongest baseline: **always_pretrained**
- Confirmatory router gate: **pass**
- Runtime: 100.4 seconds

## Calibration learning curves

|   train_targets | model             |   calibration_target_error |   calibration_auc |   threshold |
|----------------:|:------------------|---------------------------:|------------------:|------------:|
|              10 | logistic          |                    14.2677 |          0.572584 |       0.275 |
|              10 | gradient_boosting |                    15.3683 |          0.476436 |       0.25  |
|              10 | conv1d            |                    14.2069 |          0.605021 |       0.6   |
|              25 | logistic          |                    14.7225 |          0.573848 |       0.725 |
|              25 | gradient_boosting |                    15.3304 |          0.525166 |       0.5   |
|              25 | conv1d            |                    14.5141 |          0.606245 |       0.525 |
|              40 | logistic          |                    14.497  |          0.583405 |       0.65  |
|              40 | gradient_boosting |                    14.9505 |          0.577332 |       0.225 |
|              40 | conv1d            |                    14.4477 |          0.592055 |       0.6   |
|              60 | logistic          |                    14.3766 |          0.609748 |       0.575 |
|              60 | gradient_boosting |                    15.0155 |          0.560615 |       0.625 |
|              60 | conv1d            |                    14.1997 |          0.608548 |       0.625 |

## Final newest-target results

| model             |   target_mean_error |   residue_mean_error |   accuracy |   roc_auc |   pretrained_fraction |   threshold |   targets |   residues |
|:------------------|--------------------:|---------------------:|-----------:|----------:|----------------------:|------------:|----------:|-----------:|
| oracle_residue    |             4.134   |              3.75396 |   1        |  1        |              0.481792 |       0.5   |        20 |       4888 |
| conv1d            |             6.58968 |              5.95221 |   0.601678 |  0.621638 |              0.371522 |       0.625 |        20 |       4888 |
| logistic          |             6.7649  |              6.06457 |   0.529869 |  0.580254 |              0.649141 |       0.575 |        20 |       4888 |
| always_pretrained |             7.21289 |              6.55309 |   0.481792 |  0.5      |              1        |       0.5   |        20 |       4888 |
| gradient_boosting |             8.0097  |              7.5447  |   0.571399 |  0.579745 |              0.386661 |       0.625 |        20 |       4888 |
| gap_rule          |            10.2126  |             10.3861  |   0.530892 |  0.514312 |              0.04419  |       0.5   |        20 |       4888 |
| confidence_rule   |            10.2656  |             10.4287  |   0.529051 |  0.512222 |              0.037439 |       0.5   |        20 |       4888 |
| always_tbm        |            10.294   |             10.5385  |   0.518208 |  0.5      |              0        |       0.5   |        20 |       4888 |

## Target bootstrap: selected estimator versus calibration-selected baseline

|              |     value |
|:-------------|----------:|
| n_targets    | 20        |
| mean_delta   |  0.623212 |
| median_delta |  0.268336 |
| ci_low       | -1.16369  |
| ci_high      |  2.57688  |
| improved     | 13        |
| tied         |  0        |
| regressed    |  7        |

Positive bootstrap delta means the selected estimator has lower error. The pass criterion requires it to beat always-TBM, always-pretrained, gap and raw-confidence rules on the equal-weight target mean.

## Interpretation

- The learned router passes the frozen real-OOF gate.
- The per-residue oracle is non-deployable and appears only to quantify remaining headroom.
