# Competition data audit

Generated: `2026-07-13T21:30:45.374713+00:00`

Data path: `/mnt/c/M2Prj/StandfordRNA/data/stanford-rna-3d-folding`

## Sequence tables

| file | targets | residues | min | median | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| train_sequences.csv | 844 | 137,095 | 3 | 39.5 | 162.43 | 4298 |
| train_sequences.v2.csv | 5,135 | 3,677,095 | 10 | 98 | 716.08 | 4417 |
| validation_sequences.csv | 12 | 2,515 | 30 | 129.5 | 209.58 | 720 |
| test_sequences.csv | 12 | 2,515 | 30 | 129.5 | 209.58 | 720 |

## Label tables

| file | residue rows | coordinate/reference sets |
|---|---:|---:|
| train_labels.csv | 137,095 | 1 |
| train_labels.v2.csv | 3,677,095 | 1 |
| validation_labels.csv | 2,515 | 40 |

## Large-file inventory

| component | files | size of matched files |
|---|---:|---:|
| MSA/*.fasta | 856 | 445.37 MiB |
| MSA_v2/*.fasta | 2,534 | 3.14 GiB |
| PDB_RNA/*.cif | 8,670 | 56.89 GiB |

Validation and local test sequence tables identical: **True**.

## Interpretation

- The local test table is public CASP15 development data, not the hidden Kaggle private set.
- Length variation makes memory/runtime strongly target-dependent; the 720-nt target is a stress case.
- PDB_RNA is a search/template resource. Release dates must be filtered for temporal-safe local evaluation.
- MSA depth varies substantially, so pretrained methods should retain a single-sequence/TBM fallback.

Historical full-parse result recorded in the repository: 23,869 RNA/hybrid chains, 10.86M residues, 99.9% modelled C1′, zero parser errors.
