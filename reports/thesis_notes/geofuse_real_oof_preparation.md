# GeoFuse Phase E — real out-of-fold gate preparation

## Outcome

The real-OOF training path and its 15-target pilot are complete. The pilot has
temporal-safe TBM candidates plus frozen DRfold2 predictions generated on Kaggle.

DRfold2's official paper states that its structural training set uses RNA
structures released before 2024. We therefore use `2023-12-31` as the conservative
structural-training cutoff and admit only targets released later. The paper also
reports removing >80% sequence-identity overlap for its own independent test set;
our split independently clusters all eligible targets at 80% identity and 80%
coverage with MMseqs. See the [DRfold2 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12931758/).

## Prepared data

After the model-cutoff, length, native-resolution, chronology, and family guards:

| split | targets |
|---|---:|
| train | 206 |
| calibration | 63 |
| newest held-out validation | 85 |
| total | 354 |

The eligible length range is 30–396 nt. Families crossing a temporal boundary are
assigned to the later split and their older members are excluded, preserving both
strict chronology and family disjointness.

The pilot samples five short RNAs from each split (15 total, 30–96 nt). It currently
contains 45 audited TBM candidates: three per target. Its audit result is:

| split | targets | valid TBM | valid pretrained | ready pairs |
|---|---:|---:|---:|---:|
| train | 5 | 15 | 10 | 5 |
| calibration | 5 | 15 | 10 | 5 |
| validation | 5 | 15 | 10 | 5 |

All 15 targets pass the provenance audit. Each DRfold2 prediction was generated
from 20 cfg97 checkpoints, with two candidates retained by model confidence.

## Implemented safeguards

- Every pretrained candidate must declare a structural-training cutoff older than
  its target, or point to an explicit target/family exclusion manifest.
- Every TBM template must have a release date older than its target.
- Direct target PDB IDs are excluded from template search and checked again before
  training.
- Template/pretrained pairs are ranked only by inference-time confidence, never by
  native error.
- Native coordinates enter only after pair selection, to build residue targets and
  held-out metrics.
- Unresolved native residues are masked from both loss and evaluation.
- Calibration selects the decision threshold; the newest validation split remains
  untouched until final evaluation.

## Execution path

The private Kaggle GPU kernel `datdo151000/geofuse-real-oof-drfold2-pilot` generated
two DRfold2 candidates for each of the 15 pilot targets. It read only
`train_sequences.v2.csv`, never native labels. The outputs were imported with
explicit cutoff provenance before the native-supervised trainer was run.

## Pilot result

The synthetic-initialized gate was fine-tuned on 5 train targets, calibrated on 5,
and evaluated once on 5 newest targets:

| rule | held-out residue error (Å) |
|---|---:|
| oracle per-residue source | 3.2967 |
| always DRfold2 | **6.9689** |
| always TBM | 7.7180 |
| learned gate | 7.6027 |
| confidence rule | 7.8380 |
| TBM-gap rule | 7.8783 |

The learned gate beats both earlier residue heuristics but loses to always using
DRfold2 by 0.6338 Å. Held-out ROC AUC is 0.4815. Under the strict criterion that a
gate must beat both whole-source baselines as well as the heuristic rules, the
pilot **fails**.

This is still useful evidence: real errors differ materially from the synthetic
corruption model, and five training targets are too few to learn robust routing.
The failed gate must not be used in a Kaggle submission. A larger real-OOF run is
justified only as a data-scaling experiment, not as confirmation of the method.

This pilot is a pipeline and domain-transfer check, not by itself the thesis result.
