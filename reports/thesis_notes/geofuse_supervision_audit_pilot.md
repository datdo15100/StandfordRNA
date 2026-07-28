# GeoFuse real-OOF supervision audit

This is E11 under the frozen confirmatory protocol. Native coordinates are used only here to compare training labels; no native-derived column is available to the inference-time gate.

- Manifest targets: 15
- Ready targets: 15
- Pair examples: 60
- Rejected pair attempts: 0
- Jointly valid residue rows: 3760

Positive advantage always means the pretrained source is locally better.

## Pairwise label agreement

| label_a       | label_b       |   decision_agreement |   pooled_spearman |   target_centered_spearman |   n_residues |
|:--------------|:--------------|---------------------:|------------------:|---------------------------:|-------------:|
| aligned_point | c1_lddt       |              0.68324 |           0.47333 |                    0.36714 |         3760 |
| aligned_point | window15_rmsd |              0.72048 |           0.56155 |                    0.49325 |         3760 |
| c1_lddt       | window15_rmsd |              0.7     |           0.57112 |                    0.4751  |         3760 |

## Source summaries

| split       | label         | direction   |   template_mean |   pretrained_mean |   pretrained_better_fraction |   targets |   pairs |   residues |
|:------------|:--------------|:------------|----------------:|------------------:|-----------------------------:|----------:|--------:|-----------:|
| calibration | aligned_point | lower       |        15.8844  |          18.6391  |                      0.45124 |         5 |      20 |       1128 |
| calibration | c1_lddt       | higher      |         0.45915 |           0.44267 |                      0.45745 |         5 |      20 |       1128 |
| calibration | window15_rmsd | lower       |         8.39065 |           7.634   |                      0.6977  |         5 |      20 |       1128 |
| train       | aligned_point | lower       |         8.20798 |           6.99766 |                      0.55802 |         5 |      20 |       1396 |
| train       | c1_lddt       | higher      |         0.63057 |           0.73082 |                      0.56375 |         5 |      20 |       1396 |
| train       | window15_rmsd | lower       |         4.37617 |           3.02607 |                      0.68123 |         5 |      20 |       1396 |
| validation  | aligned_point | lower       |         7.71795 |           6.96894 |                      0.52265 |         5 |      20 |       1236 |
| validation  | c1_lddt       | higher      |         0.64682 |           0.76207 |                      0.58091 |         5 |      20 |       1236 |
| validation  | window15_rmsd | lower       |         3.92173 |           2.52287 |                      0.63835 |         5 |      20 |       1236 |

## Interpretation

- The old globally aligned point-error choice agrees with C1′-lDDT on 68.3% of jointly valid residue rows (target-centred rho=0.367).
- It agrees with 15-residue window RMSD on 72.0% (target-centred rho=0.493).
- Disagreement is expected because global point error assigns displacement after one whole-fold fit, lDDT measures local distance preservation without superposition, and window RMSD refits each local segment.
- C1′-lDDT is the primary gate supervision in subsequent confirmatory models. Aligned point error and window RMSD remain explicit ablations.
