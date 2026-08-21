# Audited reconstruction of the historical thesis submissions

## Evidence boundary

This document identifies the exact thesis pipelines that produced the two strong
historical Kaggle submissions. The reconstruction is based on archived notebooks,
runtime bundles, code, artifact manifests, checkpoint hashes, Kaggle output records and
submission identifiers. Later simplified implementations are not substituted for the
historical methods.

The hidden target sequences and native structures used by Kaggle are not available
locally. Therefore, the algorithms and deployed artifacts can be reconstructed, while
their hidden per-target scores cannot. Historical leaderboard scores are whole-pipeline
context only.

## Submission mapping

| Submission | Kaggle kernel | Public TM | Private TM | Pipeline identity |
|---|---|---:|---:|---|
| `54662648` | `datdo151000/rna3d-thesis-composite-tbm-baseline`, version 4 | 0.60084 | 0.60175 | composite/MMseqs TBM plus historical gradient refinement |
| `55393315` | `datdo151000/rna3d-thesis-hybrid-geometry-v2`, version 10 | 0.62809 | 0.61390 | raw historical TBM bank, direct DRfold2 allocation and Geometry Refinement |

The absolute leaderboard difference between these submissions is +0.02725 public and
+0.01215 private. It is not a component attribution because candidate allocation and
refinement both change.

## Historical TBM submission

### Data universe

The submission used one frozen structural snapshot through two method-specific views:

| View | Rows | Distinct PDB entries | Role |
|---|---:|---:|---|
| Full template table | 23,869 chains | 8,613 | MMseqs2 target database and coordinate source |
| Deduplicated composite table | 7,155 exact-sequence representatives | 4,881 | exhaustive composite retrieval |

The second table is a derived, sequence-deduplicated view of the first snapshot. Its
chain keys are a subset of the full table. The full metadata and coordinate hashes are
`80463138...` and `d5ce6232...`; the composite hashes are `8e96ffb4...` and
`b419405d...`. Earlier notes stating approximately 20,844 full chains are superseded by
the audited 23,869-row artifact.

### Retrieval and reconstruction algorithm

1. **MMseqs2 retrieval.** All query RNAs are searched against the full nucleotide
   FASTA with k-mer length 13, sensitivity 7.5, maximum 300 hits, E-value 100 and a
   3 GB split-memory limit. Hits are sorted by bitscore, duplicate chain IDs are
   removed, temporal/reference filters are applied, and at most 40 valid templates are
   realigned.
2. **Exhaustive composite retrieval.** This branch always runs; it is not only a
   fallback. Templates pass a relative-length rule: at most 0.6 difference when either
   sequence is shorter than 50 nt, 0.2 when either is longer than 1,000 nt, and 0.4
   otherwise. The score is
   `0.4 global alignment + 0.3 local alignment + 0.2 RNA feature cosine + 0.1 3-mer
   Jaccard`. Positive hits are ranked; when more than ten exist, candidates at or above
   the 80th score percentile are kept and capped at 50. Sequence-feature clustering
   then returns up to eight representatives.
3. **Branch merge.** Valid MMseqs2 candidates are inserted first. A composite candidate
   with the same chain ID is not inserted twice.
4. **Coordinate transfer.** Query and template are globally realigned with match 2,
   mismatch -1, gap opening -6, gap extension -0.5 and unpenalised terminal gaps.
   Resolved template C1-prime coordinates are copied to aligned query positions.
5. **Ranking variables.** Identity is the fraction of identical aligned query-template
   bases. Query coverage is the fraction of query positions receiving a resolved
   template coordinate. Template completeness is the fraction of template positions
   with a resolved C1-prime coordinate. Candidates are ranked by
   `identity x query coverage x template completeness`. Multiplication acts as a soft
   AND: a candidate must be similar, cover the query and have usable coordinates;
   weakness in any factor lowers the product.
6. **Candidate selection.** Candidates are sorted by that product. Up to five different
   PDB entries are selected first, then remaining chains backfill unused slots.
7. **Gap reconstruction.** Short internal gaps use linear interpolation. Internal gaps
   of at least three residues receive a sinusoidal perpendicular displacement. Terminal
   gaps extend in the nearest resolved backbone direction using the learned adjacent
   C1-prime distance.
8. **Fallback.** Missing candidate slots are filled with deterministic sequence-only
   de-novo structures using base seed 0 and global confidence 0.1.
9. **Historical refinement.** Each of five candidates is optimized with Adam for 300
   steps at learning rate 0.05. The loss weights are source/template 1.0, backbone 1.0,
   clash 0.5, radius of gyration 0.05 and predicted-distance 0. The geometry strength is
   `0.2 + 0.8(1-confidence)`. Candidate index is the refinement seed.
10. **Submission fallback and validation.** Targets longer than 1,000 nt use the
    notebook dummy fallback. The output is forced to exactly five finite `L x 3`
    structures in sample-submission order.

### Temporal and self-exclusion behaviour

The deployed code requires `template release_date < target temporal_cutoff`. The hidden
Kaggle path did not pass an explicit target-PDB exclusion list, because hidden targets
were expected to be protected by temporal separation. A labelled local reproduction
must additionally exclude the target's declared PDB IDs. This is a controlled safety
adapter, not part of the historical hidden invocation.

### Exact archived evidence

- archived artifact-bundle creation: `2026-07-13T22:22:43Z`;
- recorded bundle commit: `01f47fe87987db6b0da797c1329664a1c5b1b925`;
- archived inference runner SHA-256: `af98f4de...`;
- archived TBM module SHA-256: `beda3515...`;
- archived P0 prior SHA-256: `6d75eadd...`.

The bundle was made from a working tree whose line endings and repository cleanliness
were not fully represented by the recorded commit. Consequently, the archived file
hashes and bundle contents, rather than the commit alone, define the deployed method.

## Historical hybrid submission

### TBM branch

The hybrid starts from the same raw candidate generation just described: MMseqs2 plus
the deduplicated composite scan, post-transfer identity-coverage-completeness ranking,
distinct-PDB-first allocation, curved gaps and deterministic de-novo fallback. It does
not first apply the older TBM gradient refiner. Refinement occurs after source
allocation.

### Pretrained branch

1. DRfold2 configuration `cfg_97` runs 20 checkpoints directly.
2. The 20 direct outputs are ranked by mean DRfold2 pLDDT and the top two are converted
   to C1-prime structures with Arena.
3. PotentialFold/potential-energy optimization is not run.
4. The model source, all 20 `cfg_97` checkpoints and the RCLM checkpoint have been
   hash-audited. Every historical checkpoint is byte-identical to the frozen local V5
   resource. The RCLM checkpoint SHA-256 is
   `2b8d62e48d080f7fdff5eb03897cb91c214c53bcfddfa7171d1ae1a66bc055f5`.
5. Structural training and language-model pretraining provenance remain distinct. Weight
   identity does not establish complete time-safety of RCLM pretraining membership.

### Allocation and resource routing

- Targets are processed shortest-first for DRfold2.
- DRfold2 is attempted only for sequences of at most 600 nt and while the cumulative
  DRfold budget remains below 6.5 hours.
- With two valid pretrained structures the bank is `3T + 2D`.
- With fewer than two, the next ranked TBM candidates fill the unused slots, so the bank
  always has five structures.
- On the captured 12-target public execution, two DRfold2 candidates were produced for
  11 targets; the 720-nt R1138 target used `5T`.

### Geometry Refinement

All five allocated candidates receive 300 Adam steps with learning rate 0.04. Default
weights are source 3.0, backbone 1.0, clash 0.3, radius of gyration 0.02, angle 0.30,
torsion 0.15 and kink 20.0. Refinement strength is adaptive, and angle/torsion context is
candidate-derived. A failed refinement returns its raw candidate and records the
failure. The public execution completed all 60 refinements.

### Output and score evidence

The scored output has 2,515 rows, 18 columns, exact sample-ID order, unique IDs and
finite coordinates. The manifest records `native_labels_used: false`. The exact scored
CSV SHA-256 is
`54fe6e6598a1c2924ef7cbf03482f7ecf3c2e52e8726a1609569f777b1ac8c9a`.
The local file `data/interim/kaggle_hybrid_geometry_v2/submission.csv` has a different
hash and is an older cached export; it must not be presented as the scored submission.

The historical runtime runner SHA-256 is
`4c5cb367d36c9a2450e7c03bd8752b8a45c340e77fad4bcd24ea7ffda35c5d09`.

## What is exact, adapted and unknown

| Item | Status |
|---|---|
| TBM code, database views, priors and deployed hyperparameters | Exact archived artifacts |
| Hybrid code, DRfold2/RCLM weights, allocation and Geometry configuration | Exact code and weights |
| Hybrid scored final CSV | Exact output hash and Kaggle record |
| Hidden target sequences, native structures and per-target scores | Unavailable |
| Raw `.ret` and intermediate PDB files from the scored hybrid execution | Not preserved; regenerable, but cross-GPU output need not be byte-identical |
| Explicit self-PDB exclusion on local CASP15 | Required controlled adapter, not historical hidden behaviour |
| DRfold2 RCLM pretraining membership provenance | Unresolved |

No Boltz model, GeoFuse learned fusion or native-guided selection belonged to either
historical thesis submission above.

