# GeoFuse-RNA Phase A — pretrained candidate gate

This gate asks whether pretrained candidates add a useful fold hypothesis before confidence fusion or geometry v2 is implemented. Candidate selection never sees native labels; the oracle columns use labels only to measure the candidate-generation ceiling.

## Result

- Gate status: **fail**
- Targets with TBM candidates: 1
- Targets with pretrained candidates: 1
- Paired targets used by the gate: 1
- Mean paired TBM oracle TM: 0.4616
- Mean paired pretrained oracle TM: 0.2734
- Mean paired union oracle TM: 0.4616
- Mean oracle gain over TBM: +0.0000
- Improved/tied paired targets: 0/1

## Per-target metrics

| target_id   |   seq_len |   n_tbm |   n_pretrained |   tbm_selected_tm |   pretrained_selected_tm |   union_selected_tm |   tbm_oracle_tm |   pretrained_oracle_tm |   union_oracle_tm |   oracle_gain_over_tbm |   selection_regret |
|:------------|----------:|--------:|---------------:|------------------:|-------------------------:|--------------------:|----------------:|-----------------------:|------------------:|-----------------------:|-------------------:|
| R1117v2     |        30 |      10 |              5 |            0.4616 |                   0.2734 |              0.4616 |          0.4616 |                 0.2734 |            0.4616 |                      0 |                  0 |

## Interpretation

- **Pass**: the pretrained branch raises mean oracle-pool TM above the configured threshold; proceed to geometry v2 and fold-aware fusion.
- **Fail**: candidate generation is still the bottleneck; improve model coverage, sampling, or checkpoints before building a more complex refiner.
- **Not evaluable**: at least one target needs both a TBM and a pretrained candidate.
