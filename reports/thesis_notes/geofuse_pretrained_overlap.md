# GeoFuse pretrained exact-sequence overlap audit

Evaluation split: `validation`.

This is a conservative exact normalized-sequence audit (DNA `T` and RNA `U` are treated as equivalent). It does not test structural or remote-homology overlap, and it only covers the supplied training manifests.

- Target/model pairs checked: 12
- Exact target/model overlaps: 2

## Exact matches

| target_id   |   seq_len | model   | matching_training_ids   |
|:------------|----------:|:--------|:------------------------|
| R1128       |       238 | drfold2 | 8BTZ_A                  |
| R1138       |       720 | drfold2 | 7PTK_B;7PTL_B           |

## Evaluation rule

Report the full pretrained result as competition-style evidence. For the thesis's retrospective validation claim, also report a sensitivity analysis that excludes every exact-overlap target for the relevant model.
