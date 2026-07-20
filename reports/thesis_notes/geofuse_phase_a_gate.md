# GeoFuse-RNA Phase A — pretrained candidate gate

This gate asks whether pretrained candidates add a useful fold hypothesis before confidence fusion or geometry v2 is implemented. Candidate selection never sees native labels; the oracle columns use labels only to measure the candidate-generation ceiling.

This 12-target table is the **competition-style** result. `R1128` has an exact
sequence match (`8BTZ_A`) in the DRfold2 training FASTA, so the full number must
not be presented as a temporally held-out thesis estimate. See
`geofuse_pretrained_overlap.md` and the overlap-excluded sensitivity report.

## Result

- Gate status: **pass**
- Targets with TBM candidates: 12
- Targets with pretrained candidates: 12
- Paired targets used by the gate: 12
- Mean paired TBM oracle TM: 0.3143
- Mean paired pretrained oracle TM: 0.4972
- Mean paired union oracle TM: 0.5123
- Mean oracle gain over TBM: +0.1981
- Improved/tied paired targets: 11/1

## Per-target metrics

| target_id   |   seq_len |   n_tbm |   n_pretrained |   tbm_selected_tm |   pretrained_selected_tm |   union_selected_tm |   tbm_oracle_tm |   pretrained_oracle_tm |   union_oracle_tm |   oracle_gain_over_tbm |   selection_regret |
|:------------|----------:|--------:|---------------:|------------------:|-------------------------:|--------------------:|----------------:|-----------------------:|------------------:|-----------------------:|-------------------:|
| R1107       |        69 |      10 |              6 |            0.3821 |                   0.6136 |              0.5973 |          0.3821 |                 0.6136 |            0.6136 |                 0.2315 |             0.0162 |
| R1108       |        69 |      10 |              5 |            0.4938 |                   0.6019 |              0.6019 |          0.4938 |                 0.6019 |            0.6019 |                 0.1081 |             0      |
| R1116       |       157 |      10 |              5 |            0.4534 |                   0.5379 |              0.5149 |          0.4534 |                 0.5379 |            0.5379 |                 0.0845 |             0.023  |
| R1117v2     |        30 |      10 |             10 |            0.4616 |                   0.2799 |              0.4616 |          0.4616 |                 0.2799 |            0.4616 |                 0      |             0      |
| R1126       |       363 |      10 |              5 |            0.1824 |                   0.3593 |              0.3405 |          0.2044 |                 0.3593 |            0.3593 |                 0.1549 |             0.0188 |
| R1128       |       238 |      10 |              5 |            0.2642 |                   0.9857 |              0.9857 |          0.2642 |                 0.9857 |            0.9857 |                 0.7215 |             0      |
| R1136       |       374 |      10 |              5 |            0.2316 |                   0.3814 |              0.2444 |          0.2316 |                 0.3814 |            0.3814 |                 0.1498 |             0.137  |
| R1138       |       720 |      10 |              1 |            0.2243 |                   0.2751 |              0.2751 |          0.2243 |                 0.2751 |            0.2751 |                 0.0508 |             0      |
| R1149       |       124 |      10 |              5 |            0.3214 |                   0.7325 |              0.6645 |          0.3214 |                 0.7325 |            0.7325 |                 0.4111 |             0.0681 |
| R1156       |       135 |      10 |              5 |            0.2763 |                   0.6997 |              0.47   |          0.2763 |                 0.6997 |            0.6997 |                 0.4234 |             0.2297 |
| R1189       |       118 |      10 |              5 |            0.2198 |                   0.2289 |              0.2289 |          0.2198 |                 0.2289 |            0.2289 |                 0.0091 |             0      |
| R1190       |       118 |      10 |              5 |            0.2241 |                   0.2703 |              0.2703 |          0.2381 |                 0.2703 |            0.2703 |                 0.0322 |             0      |

## Interpretation

- **Pass**: the pretrained branch raises mean oracle-pool TM above the configured threshold; proceed to geometry v2 and fold-aware fusion.
- **Fail**: candidate generation is still the bottleneck; improve model coverage, sampling, or checkpoints before building a more complex refiner.
- **Not evaluable**: at least one target needs both a TBM and a pretrained candidate.
