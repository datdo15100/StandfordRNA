# GeoFuse quality estimators — window15_rmsd

This is E12 under the frozen 60/20/20 protocol. Learning curves and all hyperparameters/thresholds use train+calibration only. The confirmatory estimator and strongest fixed baseline are selected on calibration before the 20 newest validation targets are evaluated once.

- Ready target counts: `{'train': 60, 'calibration': 20, 'validation': 20}`
- Rejected pair attempts: 0
- Supervision/evaluation unit: angstrom
- Calibration-selected estimator: **conv1d**
- Calibration-selected strongest baseline: **always_pretrained**
- Confirmatory router gate: **pass**
- Runtime: 104.6 seconds

## Calibration learning curves

|   train_targets | model             |   calibration_target_error |   calibration_auc |   threshold |
|----------------:|:------------------|---------------------------:|------------------:|------------:|
|              10 | logistic          |                    6.22665 |          0.552041 |       0.075 |
|              10 | gradient_boosting |                    6.20108 |          0.565736 |       0.2   |
|              10 | conv1d            |                    6.16945 |          0.640072 |       0.475 |
|              25 | logistic          |                    6.14035 |          0.60867  |       0.4   |
|              25 | gradient_boosting |                    6.2012  |          0.558427 |       0.1   |
|              25 | conv1d            |                    5.95055 |          0.71167  |       0.55  |
|              40 | logistic          |                    6.1616  |          0.589964 |       0.225 |
|              40 | gradient_boosting |                    6.14884 |          0.57441  |       0.375 |
|              40 | conv1d            |                    6.02147 |          0.681435 |       0.525 |
|              60 | logistic          |                    6.11206 |          0.621058 |       0.45  |
|              60 | gradient_boosting |                    6.17913 |          0.541355 |       0.325 |
|              60 | conv1d            |                    5.99411 |          0.701581 |       0.5   |

## Final newest-target results

| model             |   target_mean_error |   residue_mean_error |   accuracy |   roc_auc |   pretrained_fraction |   threshold |   targets |   residues |
|:------------------|--------------------:|---------------------:|-----------:|----------:|----------------------:|------------:|----------:|-----------:|
| oracle_residue    |             2.59323 |              2.32202 |   1        |  1        |              0.496113 |       0.5   |        20 |       4888 |
| gradient_boosting |             3.12723 |              2.82228 |   0.544394 |  0.66907  |              0.944354 |       0.325 |        20 |       4888 |
| conv1d            |             3.13097 |              2.80974 |   0.613543 |  0.698761 |              0.62275  |       0.5   |        20 |       4888 |
| logistic          |             3.15618 |              2.84117 |   0.537439 |  0.608261 |              0.907938 |       0.45  |        20 |       4888 |
| always_pretrained |             3.18115 |              2.86956 |   0.496113 |  0.5      |              1        |       0.5   |        20 |       4888 |
| gap_rule          |             4.16531 |              3.9793  |   0.534984 |  0.531442 |              0.04419  |       0.5   |        20 |       4888 |
| confidence_rule   |             4.19627 |              4.00787 |   0.530687 |  0.527093 |              0.037439 |       0.5   |        20 |       4888 |
| always_tbm        |             4.2956  |              4.11688 |   0.503887 |  0.5      |              0        |       0.5   |        20 |       4888 |

## Target bootstrap: selected estimator versus calibration-selected baseline

|              |     value |
|:-------------|----------:|
| n_targets    | 20        |
| mean_delta   |  0.050186 |
| median_delta |  0.063435 |
| ci_low       | -0.160998 |
| ci_high      |  0.243457 |
| improved     | 12        |
| tied         |  1        |
| regressed    |  7        |

Positive bootstrap delta means the selected estimator has lower error. The pass criterion requires it to beat always-TBM, always-pretrained, gap and raw-confidence rules on the equal-weight target mean.

## Interpretation

- The learned router passes the frozen real-OOF gate.
- The per-residue oracle is non-deployable and appears only to quantify remaining headroom.
