# GeoFuse Phase D — synthetic confidence gate

This bootstrap experiment trains a tiny 1D residue gate only on temporal-safe `train_v2` native structures corrupted into template-like and pretrained-like sources. The newest targets are held out as a time-ordered validation split.

- Train targets: 188 (1995-01-26 to 2019-01-16)
- Calibration targets: 28 (2019-04-24 to 2020-06-24)
- Held-out targets: 56 (2020-07-08 to 2022-05-18)
- Synthetic variants per target: 2
- Sampling cap / length range: 600 targets / 30-600 residues
- Seed: 2025
- Epochs: 8
- Runtime: 15.0 seconds
- Synthetic held-out gate: **pass**

## Held-out metrics

|                            |       value |
|:---------------------------|------------:|
| weighted_bce               |     0.27317 |
| accuracy                   |     0.96682 |
| balanced_accuracy          |     0.94908 |
| roc_auc                    |     0.98689 |
| brier                      |     0.06252 |
| ece_10bin                  |     0.16585 |
| template_error             |     2.60566 |
| pretrained_error           |     2.85577 |
| oracle_residue_error       |     0.83954 |
| learned_gate_error         |     0.88191 |
| gap_rule_error             |     0.89575 |
| confidence_rule_error      |     1.46528 |
| n_residues                 | 12928       |
| pretrained_better_fraction |     0.16576 |
| decision_threshold         |     0.525   |
| n_parameters               |  8737       |

Decision threshold 0.525 was selected only on the calibration split (calibration gate error 0.8852 Å).
Learned-gate improvement over gap rule: +0.0138 Å (positive means lower error).

Lower error is better. `oracle_residue_error` chooses the lower-error source with native knowledge and is only a ceiling; the learned gate never sees native at inference.

## Training history

|   epoch |   train_loss |   calibration_loss |   calibration_auc |
|--------:|-------------:|-------------------:|------------------:|
|       1 |      0.38336 |            0.30644 |           0.96694 |
|       2 |      0.30096 |            0.29084 |           0.96486 |
|       3 |      0.29209 |            0.28741 |           0.96634 |
|       4 |      0.28768 |            0.28451 |           0.96843 |
|       5 |      0.28512 |            0.28286 |           0.96786 |
|       6 |      0.28155 |            0.27951 |           0.96936 |
|       7 |      0.2782  |            0.27743 |           0.96984 |
|       8 |      0.27624 |            0.27665 |           0.97004 |

## Validity boundary

Synthetic corruption is bootstrap supervision, not sufficient evidence that confidence is calibrated for DRfold2/Boltz/TBM outputs. The checkpoint must next be frozen and tested on real out-of-fold template/model candidates; CASP15 labels must not be used to retrain or tune it.
Corruption v2 randomizes model confidence scales because Phase B found raw source confidence uncalibrated; absolute global-confidence features are intentionally omitted.
The frozen real-candidate transfer pilot is reported separately in `geofuse_phase_d_transfer.md`; synthetic gate success is not presented as real-domain fusion success.
