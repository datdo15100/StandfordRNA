# V5 CASP15 local-validation sandbox

## Role

CASP15 is the only local model-development and validation benchmark in V5. Its 12
targets may be inspected and used to understand components. It is not called an
untouched or confirmatory test. All claims explicitly acknowledge the small sample.

The train-derived 97-target V4 cohort is absent from this sandbox and from V5 evidence.

## Target manifest

| Target | Length | Strict temporal cutoff | Excluded native PDB | Native conformations |
|---|---:|---|---|---:|
| R1107 | 69 | 2022-05-28 | 7QR4 | 1 |
| R1108 | 69 | 2022-05-27 | 7QR3 | 2 |
| R1116 | 157 | 2022-06-04 | 8S95 | 1 |
| R1117v2 | 30 | 2022-06-03 | 8FZA | 2 |
| R1126 | 363 | 2022-06-11 | 8TVZ | 1 |
| R1128 | 238 | 2022-06-10 | 8BTZ | 1 |
| R1136 | 374 | 2022-06-18 | 7ZJ4 | 1 |
| R1138 | 720 | 2022-06-24 | 7PTK | 2 |
| R1149 | 124 | 2022-07-02 | 8UYS | 10 |
| R1156 | 135 | 2022-07-07 | 8UYE | 40 |
| R1189 | 118 | 2022-08-11 | 7YR7 | 1 |
| R1190 | 118 | 2022-08-11 | 7YR6 | 1 |

The machine-readable manifest is `casp15_target_manifest.csv`, SHA-256
`f482a3e2...`. Sequence and label files have SHA-256 `42dc2e35...` and
`5a9d6a71...`. The local `test_sequences.csv` is byte-identical to
`validation_sequences.csv`; it is the public CASP15 set, not Kaggle's inaccessible
hidden scoring set.

R1189 and R1190 share an exact sequence but have different native PDB entries. This
dependence is retained in target-level Kaggle-style means and grouped in uncertainty
resampling.

## Common structural universe

Every method receives the same frozen snapshot:

- 23,869 full chain rows from 8,613 PDB entries;
- 7,155 exact-sequence representatives from 4,881 PDB entries as a derived view;
- full metadata/coordinate hashes `80463138...` and `d5ce6232...`;
- derived-view metadata/coordinate hashes `8e96ffb4...` and `b419405d...`.

An algorithm may use a derived view because database view is itself a method component,
but it cannot use a different underlying structural snapshot without an explicitly
labelled universe-sensitivity analysis.

For target `q`, the allowed full table is defined before retrieval as:

`release_date < q.temporal_cutoff AND PDB_ID not in q.excluded_native_pdb_ids`.

Missing or malformed release dates fail closed. The filtered row order is frozen because
stable sorting, tie resolution and KMeans can depend on input order.

## Candidate and failure contract

- Every bank returns exactly N finite `L x 3` C1-prime structures for its declared
  candidate budget.
- The main competition bank has N=5.
- Failed or unavailable candidates trigger the method's preregistered deterministic
  fallback and remain in the score; no target is silently dropped.
- Candidate IDs, sources, confidence, fallback events and coordinate hashes are stored
  before native scoring.
- DRfold2 candidates use the audited direct `cfg_97` outputs and are ranked by mean
  pLDDT. Source availability, including the 720-nt resource case, is reported for every
  allocation.

## Evaluator

The evaluator writes C1-prime-only PDBs, calls the frozen US-align binary SHA-256
`7fb696c3...`, and reads the TM-score normalized by the native/reference structure. For
one target it takes the maximum over every submitted prediction and every nonempty
native conformation. The headline score is the equal-weight mean of the 12 target
best-of-bank scores.

The following are reported for each comparison:

- per-target score and paired delta;
- mean and median paired delta;
- wins, ties and losses;
- a 95% exact-sequence-cluster bootstrap interval with 10,000 replicates;
- an exact or enumerated cluster sign-flip result when useful;
- target length, template availability, fallback use and candidate diversity.

There are 11 exact-sequence clusters because R1189/R1190 form one two-target cluster.
Intervals and p-values are descriptive support for a 12-target development benchmark,
not a confirmatory gate. No small negative mean automatically deletes a component.

## Metric boundaries

- **RQ1/RQ2 primary outcome:** raw best-of-bank TM-score at the declared fixed N.
- **Candidate diversity:** mean pairwise self-TM, where a lower value means less
  redundant candidates.
- **RQ3 global outcomes:** locked-reference candidate TM and bank best-of-five TM.
- **RQ3 local outcomes:** SW-RMSD9, SW-RMSD15, C1-prime lDDT adaptation and global
  Kabsch-aligned C1-prime RMSD.
- **Diagnostics:** backbone spacing, clashes, kink fraction, radius-of-gyration error,
  angle/torsion likelihood and pair-like fraction. Diagnostics that appear in a Geometry
  objective are not presented as independent validation.

All input hashes are recorded in `v5_design_input_audit.json`. No prediction method has
been run by the design-audit script.

