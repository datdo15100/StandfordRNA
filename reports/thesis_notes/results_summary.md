# CASP15 validation results

Best-of-5 TM-score (US-align), 12 CASP15 targets, temporal-safe templates.

## Mean TM by method

|                 |   mean_TM |
|:----------------|----------:|
| A_no_gapweights |    0.162  |
| B4_tbm_refined  |    0.1612 |
| A_no_rg         |    0.1612 |
| A_no_clash      |    0.1586 |
| B2_tbm_top5     |    0.1584 |
| B1_tbm_top1     |    0.1544 |
| B0_dummy        |    0.0689 |

## Refinement (B4 = TBM+refine) vs TBM-only (B2)

- Overall mean gain: **+0.0028** (B2 0.1584 -> B4 0.1612)
- Per-target: 8 improved, 0 unchanged, 4 worse

## Stratified by template confidence

| conf_bin   |   n |   mean_best_conf |   B2_tbm_top5 |   B4_tbm_refined |   refine_gain |
|:-----------|----:|-----------------:|--------------:|-----------------:|--------------:|
| high       |   1 |           0.6624 |        0.4534 |           0.4689 |        0.0155 |
| medium     |   2 |           0.5797 |        0.3238 |           0.3192 |       -0.0045 |
| low        |   2 |           0.1439 |        0.1626 |           0.1608 |       -0.0018 |
| none       |   7 |           0      |        0.0679 |           0.0723 |        0.0044 |

## Ablations (mean TM)

- B4_tbm_refined: 0.1612
- A_no_clash: 0.1585
- A_no_rg: 0.1612
- A_no_gapweights: 0.1620

## Best-of-5 diversity
- mean pairwise self-TM (B4): 0.7006 (lower = more diverse)

## Per-target detail

| target_id   |   seq_len |   best_conf |   B1_tbm_top1 |   B2_tbm_top5 |   B4_tbm_refined |   refine_gain |
|:------------|----------:|------------:|--------------:|--------------:|-----------------:|--------------:|
| R1107       |        69 |      0.5797 |        0.3144 |        0.3292 |           0.326  |       -0.0032 |
| R1108       |        69 |      0.5797 |        0.308  |        0.3183 |           0.3124 |       -0.0059 |
| R1116       |       157 |      0.6624 |        0.4328 |        0.4534 |           0.4689 |        0.0155 |
| R1117v2     |        30 |      0      |        0.0966 |        0.1084 |           0.1074 |       -0.001  |
| R1126       |       363 |      0.1488 |        0.167  |        0.1676 |           0.1616 |       -0.006  |
| R1128       |       238 |      0      |        0.0505 |        0.0444 |           0.0488 |        0.0044 |
| R1136       |       374 |      0.139  |        0.1575 |        0.1575 |           0.16   |        0.0025 |
| R1138       |       720 |      0      |        0.0343 |        0.0341 |           0.0377 |        0.0036 |
| R1149       |       124 |      0      |        0.0659 |        0.0652 |           0.0697 |        0.0045 |
| R1156       |       135 |      0      |        0.0693 |        0.0693 |           0.0767 |        0.0074 |
| R1189       |       118 |      0      |        0.0765 |        0.0709 |           0.0789 |        0.008  |
| R1190       |       118 |      0      |        0.0794 |        0.0831 |           0.0867 |        0.0036 |