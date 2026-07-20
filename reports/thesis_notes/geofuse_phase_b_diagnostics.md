# GeoFuse Phase B — native-blind geometry diagnostics

Candidates: 182 across 12 validation targets.
Native TM-score is joined only for this post-hoc signal audit; none of the features reads validation labels.

Positive rho means larger feature values associate with better TM. Geometry violations are expected to have negative rho if they are useful for routing.
Target-centered rho removes between-target difficulty/length effects.

| feature                 |   n |   pooled_rho |   pooled_p |   target_centered_rho |   mean_within_target_rho |   n_within_targets |
|:------------------------|----:|-------------:|-----------:|----------------------:|-------------------------:|-------------------:|
| pair_like_fraction      | 182 |       0.5662 |     0      |                0.7357 |                   0.7117 |                 12 |
| support_fraction        | 182 |       0.5735 |     0      |                0.5323 |                   0.5916 |                 12 |
| sharp_kinks             | 182 |      -0.3458 |     0      |               -0.3039 |                  -0.3335 |                 12 |
| clash_per_res           | 182 |      -0.3528 |     0      |               -0.3566 |                  -0.3226 |                 12 |
| rg_err                  | 182 |      -0.3905 |     0      |               -0.4005 |                  -0.4231 |                 12 |
| global_confidence       | 182 |      -0.1658 |     0.0253 |               -0.4345 |                  -0.3851 |                 12 |
| torsion_nll             | 182 |      -0.4215 |     0      |               -0.4465 |                  -0.4403 |                 12 |
| bb_dev                  | 182 |      -0.4395 |     0      |               -0.5197 |                  -0.449  |                 12 |
| mean_residue_confidence | 182 |      -0.2629 |     0.0003 |               -0.5236 |                  -0.468  |                 12 |
| angle_nll               | 182 |      -0.6699 |     0      |               -0.5929 |                  -0.6053 |                 12 |

## Sensitivity excluding R1128

| feature                 |   n |   pooled_rho |   pooled_p |   target_centered_rho |   mean_within_target_rho |   n_within_targets |
|:------------------------|----:|-------------:|-----------:|----------------------:|-------------------------:|-------------------:|
| pair_like_fraction      | 167 |       0.5086 |     0      |                0.7097 |                   0.6893 |                 11 |
| support_fraction        | 167 |       0.5355 |     0      |                0.5033 |                   0.5744 |                 11 |
| sharp_kinks             | 167 |      -0.3439 |     0      |               -0.3074 |                  -0.3182 |                 11 |
| clash_per_res           | 167 |      -0.302  |     0.0001 |               -0.3229 |                  -0.2982 |                 11 |
| torsion_nll             | 167 |      -0.3695 |     0      |               -0.3556 |                  -0.4297 |                 11 |
| rg_err                  | 167 |      -0.3081 |     0.0001 |               -0.4041 |                  -0.3872 |                 11 |
| global_confidence       | 167 |      -0.1594 |     0.0397 |               -0.4683 |                  -0.4162 |                 11 |
| bb_dev                  | 167 |      -0.3872 |     0      |               -0.4706 |                  -0.434  |                 11 |
| mean_residue_confidence | 167 |      -0.2365 |     0.0021 |               -0.5376 |                  -0.459  |                 11 |
| angle_nll               | 167 |      -0.6302 |     0      |               -0.547  |                  -0.5827 |                 11 |

## Source-specific correlations

These expose whether a feature ranks candidates within a generator or mainly calibrates differences between generators.

| source      | feature                 |   n |   pooled_rho |   target_centered_rho |
|:------------|:------------------------|----:|-------------:|----------------------:|
| drfold2     | global_confidence       |   6 |      -0.6547 |              nan      |
| drfold2     | mean_residue_confidence |   6 |      -0.6547 |              nan      |
| drfold2     | support_fraction        |   6 |     nan      |              nan      |
| drfold2     | clash_per_res           |   6 |     nan      |              nan      |
| drfold2     | bb_dev                  |   6 |      -0.7143 |               -0.9429 |
| drfold2     | rg_err                  |   6 |       0.0857 |                0.9429 |
| drfold2     | sharp_kinks             |   6 |      -0.6547 |              nan      |
| drfold2     | angle_nll               |   6 |      -0.3714 |                0.2571 |
| drfold2     | torsion_nll             |   6 |      -0.8286 |               -0.6    |
| drfold2     | pair_like_fraction      |   6 |       0.3086 |               -0.1543 |
| drfold2_e2e | global_confidence       |  55 |       0.4859 |               -0.0929 |
| drfold2_e2e | mean_residue_confidence |  55 |       0.4859 |               -0.0929 |
| drfold2_e2e | support_fraction        |  55 |     nan      |              nan      |
| drfold2_e2e | clash_per_res           |  55 |      -0.4332 |               -0.1071 |
| drfold2_e2e | bb_dev                  |  55 |      -0.6927 |                0.0506 |
| drfold2_e2e | rg_err                  |  55 |      -0.0087 |               -0.1349 |
| drfold2_e2e | sharp_kinks             |  55 |       0.1166 |               -0.1558 |
| drfold2_e2e | angle_nll               |  55 |      -0.5986 |               -0.1909 |
| drfold2_e2e | torsion_nll             |  55 |      -0.3624 |               -0.0549 |
| drfold2_e2e | pair_like_fraction      |  55 |       0.481  |                0.1691 |
| tbm         | global_confidence       | 120 |       0.4724 |                0.2549 |
| tbm         | mean_residue_confidence | 120 |       0.3563 |                0.1105 |
| tbm         | support_fraction        | 120 |       0.355  |                0.1102 |
| tbm         | clash_per_res           | 120 |      -0.0399 |                0.1657 |
| tbm         | bb_dev                  | 120 |      -0.1284 |               -0.0308 |
| tbm         | rg_err                  | 120 |      -0.3972 |               -0.3671 |
| tbm         | sharp_kinks             | 120 |      -0.2172 |               -0.0768 |
| tbm         | angle_nll               | 120 |      -0.4865 |               -0.156  |
| tbm         | torsion_nll             | 120 |      -0.1931 |                0.0681 |
| tbm         | pair_like_fraction      | 120 |       0.2694 |                0.5433 |

## Interpretation

- Angle NLL is the strongest violation signal after target centering (rho=-0.593; excluding the overlap target: -0.547).
- Pair-like fraction has the strongest positive association (rho=0.736), but it is a candidate-derived topology proxy, not true secondary-structure accuracy.
- Source-specific target-centered correlations are weaker or can reverse. These features currently calibrate heterogeneous generators better than they rank samples from one generator.
- Raw confidence is not calibrated across sources, so it is unsafe as a single whole-bank ranking score.

This table is a gate, not a trained selector. Features with weak or reversed within-target signal must not be assigned large heuristic routing weights.
