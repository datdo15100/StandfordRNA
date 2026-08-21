# Same-sandbox John baseline design

## Baseline identity

The inherited baseline is the publicly captured John TBM-only notebook, not an assumed
exact winning submission. The capture is `utilities/top1_tbm.py`, SHA-256
`c8eb61c...`, associated with public notebook version 242152007 and a reported score of
0.59298. Its attached dataset version, row order, package versions and byte-equivalence
to any final winning submission are unavailable.

V5 therefore distinguishes:

- **J-public-captured:** contextual description of the public code and reported score;
- **J-SS-RAW:** John candidate generation on the common V5 CASP15 sandbox, before its
  rule refiner and jitter; this is the main raw RQ1 baseline;
- **J-SS-COMPLETE:** J-SS-RAW followed by the public John rule refiner and deterministic
  legacy-compatible jitter; this is the inherited complete-pipeline baseline.

## Captured public algorithm

### Data and safety

The capture combines 844 original training rows with 18,102 new rows from its attached
RNA CIF dataset, yielding 18,946 sequence rows and 18,815 coordinate groups. The exact
attached dataset version and hashes are unknown. The public code has neither temporal
filtering nor explicit target-PDB exclusion.

### Retrieval and selection

1. Exhaustively scan a single template table; MMseqs2 is not used.
2. Apply the same relative-length thresholds later inherited by the thesis: 0.6 if
   either sequence is shorter than 50 nt, 0.2 if either is longer than 1,000 nt, and
   0.4 otherwise.
3. Score each template as `0.4 global + 0.3 local + 0.2 RNA feature cosine + 0.1 3-mer
   Jaccard`. Global and local alignments use match 2.9, mismatch -1, gap opening -10 and
   gap extension -0.5, then normalize by twice the shorter sequence length.
4. The feature vector contains A/U/G/C fractions, ten dinucleotide fractions, GC/AU and
   purine/pyrimidine fractions, capped length, normalized entropy and repeated-3-mer
   content.
5. Keep positive scores. If more than ten hits remain, keep those at or above the 80th
   percentile and cap the shortlist at 50.
6. Cluster shortlisted sequence-feature vectors. With at least 15 hits use KMeans with
   five clusters, random state 42 and ten initializations; otherwise use deterministic
   farthest-first labels. Select the highest composite-score template in each cluster,
   sort representatives by composite score and keep five.

This is sequence-feature diversity, not structural clustering. John does not perform a
second identity-coverage-completeness ranking and does not prefer distinct PDB IDs.

### Transfer, gaps and completion

The query and template are globally aligned with 2.9/-1/-10/-0.5 scores, and template
C1-prime coordinates are copied by aligned residue index. Missing internal positions use
linear interpolation except when the anchor distance is below 0.7 times the expected
5.9-Angstrom span; then John uses a stretched line plus a perpendicular sine wave of
amplitude 2 Angstrom. Terminal gaps extend directionally. Any unresolved remainder is
converted to the origin.

The one-pass rule refiner uses
`strength = 0.8 x (1 - min(composite confidence, 0.8))`. It sequentially corrects
adjacent distances outside 5.5-6.5 Angstrom, pushes nonadjacent pairs below 3.8
Angstrom using a distance matrix computed once, and, at low confidence, moves the first
eligible A-U or G-C complement at sequence separation 3-19 and current distance 8-14
Angstrom toward 10.5 Angstrom.

Template candidates then receive independent Gaussian jitter with
`sigma = max(0.05, 0.8 - composite score)`. If fewer than five representatives exist,
de-novo fallback fills the bank, receives the John refiner at confidence 0.2 and is not
jittered.

## Same-sandbox adapters

J-SS preserves the algorithm but changes conditions required for a fair labelled local
comparison:

- use the exact V3 full snapshot of 23,869 chains as John's single exhaustive table;
- filter each target by the frozen strict date and explicit target-PDB exclusion before
  search;
- fail closed on invalid dates;
- preserve the frozen row order and dependency versions;
- use an explicitly recorded deterministic legacy MT19937 stream for de-novo fallback
  and Gaussian jitter;
- always return five finite structures and record fallback rather than dropping a
  target;
- use the common CASP15 evaluator and target ordering.

The 7,155-sequence V3 composite view is a secondary universe-sensitivity analysis. It
must not silently replace the main J-SS full-table baseline. The historical local result
0.2983 used that smaller view and is contextual, not the new V5 result.

## Experimental boundaries

| Boundary | Candidate generation | Post-processing | Purpose |
|---|---|---|---|
| J-SS-RAW | faithful John composite retrieval, clustering, transfer and gaps | none | fair raw RQ1 baseline |
| J-SS-REFINER | J-SS-RAW | one-pass John rules, no jitter | isolate refiner effect |
| J-SS-COMPLETE | J-SS-RAW | one-pass John rules plus legacy-compatible jitter | inherited complete pipeline |
| J-SS-DEDUP | same as J-SS-RAW | 7,155-sequence derived view | universe sensitivity only |

Applying the John refiner to V3 or DRfold2 candidates is labelled a **controlled John
refiner**, because their confidence is not John's composite retrieval score.

## Components that must not be added to primary J-SS

MMseqs2, identity-coverage-completeness reranking, distinct-PDB selection and V3 curved
gap logic do not belong to John. Adding them would erase the contrast rather than make
the baseline fair.

## Remaining unknowns

1. exact attached public dataset version, hashes and input row order;
2. exact public runtime package versions;
3. original Python hash salt and global RNG state;
4. equivalence between the captured notebook and the final winning submission;
5. byte-for-byte mapping of reported 0.59298 to the captured code and dataset;
6. sensitivity to duplicate rows, database view and KMeans/pairwise-alignment tie cases.

These unknowns prevent the phrase "exact John winning pipeline reproduction". They do
not prevent a controlled same-sandbox implementation whose deviations are explicit.

