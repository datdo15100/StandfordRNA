# Clean-machine run status

Run date: 14 July 2026 (local time)

## Environment and data

- Conda environment: `rna-fold`, Python 3.12.13.
- PyTorch 2.13.0 + CUDA 13.0 sees the RTX 3060 Ti (8 GB).
- MMseqs2 18.8cc5c available.
- Raw Kaggle data: 61 GiB, 8,670 CIFs, 856 MSA and 2,534 MSA_v2 files.
- Sample submission: 2,515 × 18 and passes repository validation.

## Artifact rebuild

The full rebuild ran with six CIF workers:

- 8,670/8,670 CIFs parsed in 273.7 seconds;
- 23,869 RNA/hybrid chains;
- 10,870,310 residues, 10,861,955 with resolved C1′;
- zero parse errors and release dates found for every chain;
- 20,844-chain MMseqs FASTA/DB;
- 7,155 unique-sequence composite-search library;
- template coordinate cache: 179 MiB.

Peak observed WSL memory remained well below the current 11 GiB allocation and swap was
not used materially.

## Scoring verification

US-align was built from upstream commit `177cc8a2bbd3e2a6e9c5faaaa4ff5dfa1e6048f7`
(reported version 20260527).

| sanity case | TM |
|---|---:|
| native vs native | 1.00000 |
| native vs rigidly rotated | 1.00000 |
| native vs mirrored | 0.22387 |

The 12-target extended-chain dummy baseline reproduced **0.0687** mean best-of-five TM.

## Fresh headline reproductions

| Experiment | Mean temporal-safe best-of-five TM |
|---|---:|
| MMseqs-only + de novo/refine | 0.2117 |
| current + composite search | **0.3072** |
| reproduced top-1 method | 0.2983 |
| top-1 full-PDB leaked diagnostic | 0.9355 |

Composite search improved 11/12 targets, mean ΔTM **+0.0955**, and beat the freshly
reproduced top-1 baseline on 9/12 targets.

The refinement truthfulness result also reproduced:

| setting | TM | clashes/res | backbone dev | sharp kinks |
|---|---:|---:|---:|---:|
| none | 0.3092 | 0.1635 | 1.4579 | 0.0536 |
| rule | 0.3098 | 0.0992 | 1.0258 | 0.0944 |
| gradient v1 | 0.3072 | 0.0935 | 0.7768 | 0.1025 |

## Reproducibility bug found and fixed

A three-target smoke run wrote a subset `validation_query.fasta`. Several full-run scripts
previously reused that file whenever it existed, silently omitting MMseqs queries for the
remaining targets. The scripts now always materialise the exact query set before search.
After the fix, the historical 0.2117 → 0.3072 ablation reproduced exactly.

## Kaggle external validation

- Private artifact dataset `datdo151000/rna3d-thesis-inference-artifacts`, version 2,
  is ready and contains the code, temporal-safe artifacts, MMseqs runtime and offline
  Biopython wheel.
- Private CPU notebook `datdo151000/rna3d-thesis-composite-tbm-baseline`, version 4,
  completed successfully without internet access.
- The downloaded notebook output was checked against the competition sample: 2,515 × 18,
  exact columns and ID order, unique IDs, no missing values and finite coordinates.
- Late submission **54662648** was accepted by Kaggle at 2026-07-13 22:32:44 UTC with
  description `Temporal-safe composite TBM thesis baseline`; scoring is currently pending.

## Still pending

- Record the hidden-set public/private score when Kaggle finishes the late-submission run.
- Pretrained candidate generation and GeoFuse fusion/geometry v2 are not implemented yet.
