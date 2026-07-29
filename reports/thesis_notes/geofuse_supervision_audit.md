# GeoFuse real-OOF supervision audit

This is E11 under the frozen confirmatory protocol. Native coordinates are used only here to compare training labels; no native-derived column is available to the inference-time gate.

- Manifest targets: 100
- Ready targets: 100
- Pair examples: 352
- Rejected pair attempts: 0
- Jointly valid residue rows: 24876

Positive advantage always means the pretrained source is locally better.

## Pairwise label agreement

| label_a       | label_b       |   decision_agreement |   pooled_spearman |   target_centered_spearman |   n_residues |
|:--------------|:--------------|---------------------:|------------------:|---------------------------:|-------------:|
| aligned_point | c1_lddt       |              0.69686 |           0.52711 |                    0.31928 |        24876 |
| aligned_point | window15_rmsd |              0.7266  |           0.60589 |                    0.35048 |        24876 |
| c1_lddt       | window15_rmsd |              0.70912 |           0.62135 |                    0.41777 |        24876 |

## Source summaries

| split       | label         | direction   |   template_mean |   pretrained_mean |   pretrained_better_fraction |   targets |   pairs |   residues |
|:------------|:--------------|:------------|----------------:|------------------:|-----------------------------:|----------:|--------:|-----------:|
| calibration | aligned_point | lower       |        16.3409  |          16.3967  |                      0.44475 |        20 |      72 |       4308 |
| calibration | c1_lddt       | higher      |         0.518   |           0.49428 |                      0.40854 |        20 |      72 |       4308 |
| calibration | window15_rmsd | lower       |         7.13194 |           6.15607 |                      0.54178 |        20 |      72 |       4308 |
| train       | aligned_point | lower       |        10.3483  |           7.91908 |                      0.55287 |        60 |     214 |      15680 |
| train       | c1_lddt       | higher      |         0.59751 |           0.70373 |                      0.58941 |        60 |     214 |      15680 |
| train       | window15_rmsd | lower       |         5.23976 |           3.78764 |                      0.65332 |        60 |     214 |      15680 |
| validation  | aligned_point | lower       |        10.5386  |           6.55309 |                      0.48179 |        20 |      74 |       4888 |
| validation  | c1_lddt       | higher      |         0.66239 |           0.74666 |                      0.50675 |        20 |      74 |       4888 |
| validation  | window15_rmsd | lower       |         4.11688 |           2.86956 |                      0.49611 |        20 |      74 |       4888 |

## Interpretation

- The old globally aligned point-error choice agrees with C1′-lDDT on 69.7% of jointly valid residue rows (target-centred rho=0.319).
- It agrees with 15-residue window RMSD on 72.7% (target-centred rho=0.350).
- Disagreement is expected because global point error assigns displacement after one whole-fold fit, lDDT measures local distance preservation without superposition, and window RMSD refits each local segment.
- C1′-lDDT is the primary gate supervision in subsequent confirmatory models. Aligned point error and window RMSD remain explicit ablations.
