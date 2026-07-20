# GeoFuse Phase E — real out-of-fold gate preparation

## Outcome

The real-OOF training path is implemented and its 15-target pilot has a complete
temporal-safe TBM bank. Pretrained predictions are the only missing input, so the
gate has deliberately not been trained yet.

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
| train | 5 | 15 | 0 | 0 |
| calibration | 5 | 15 | 0 | 0 |
| validation | 5 | 15 | 0 | 0 |

This zero is expected and useful: the hard provenance gate prevents validation
DRfold2 files or synthetic corruptions from silently entering real-OOF training.

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

The private Kaggle GPU kernel `datdo151000/geofuse-real-oof-drfold2-pilot` is prepared
to generate two DRfold2 candidates for each of the 15 pilot targets. It reads only
`train_sequences.v2.csv`, never native labels. After downloading and importing its
outputs with cutoff provenance, run the audit and then `train_geofuse_real_gate.py`.

This pilot is a pipeline and domain-transfer check. A positive pilot justifies a
larger family-disjoint run; it is not by itself the thesis result.
