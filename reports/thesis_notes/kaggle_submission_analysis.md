# Kaggle late-submission analysis

Submission `54662648` (`datdo151000/rna3d-thesis-composite-tbm-baseline`, kernel
version 4) completed with:

| split | TM score |
|---|---:|
| public | **0.60084** |
| private | **0.60175** |

The submitted method was the repository's **temporal-safe composite TBM baseline**,
not either copied `utilities/top1_*` notebook and not a pretrained hybrid. The artifact
manifest identifies the inference bundle as commit `01f47fe`; the notebook then calls
`kaggle.inference_pipeline.run_inference(..., steps=300, max_len=1000)`.

## Exact submitted path

For every hidden test sequence, the kernel:

1. searches roughly 20,844 parsed RNA chains with MMseqs2;
2. independently scans a 7,155-unique-sequence library with the top-1-style composite
   score (global alignment, local alignment, RNA composition/features and 3-mer overlap);
3. drops templates whose PDB release date is not strictly before the target's
   `temporal_cutoff`;
4. globally realigns the union and ranks it by
   `identity × target coverage × template completeness`;
5. selects up to five candidates, preferring distinct PDB entries;
6. transfers C1′ coordinates, fills gaps with explicit low confidence, and fills spare
   best-of-five slots with de-novo candidates;
7. applies 300 Adam steps of confidence-weighted C1′ geometry refinement; and
8. validates exact sample ID order, uniqueness, dimensions and finite coordinates.

Lengths above 1,000 nt use the dummy fallback. There is no Boltz, DRfold2, AF3/Chai,
MSA model or learned fusion in this submission.

## Why 0.60175 is plausible rather than obviously anomalous

The private score is only **+0.00877** above the reported TBM-only notebook score
0.59298 (about +1.48% relative). This almost exactly mirrors the controlled local
comparison: current pipeline 0.3072 versus reproduced top-1 temporal-safe 0.2983,
or **+0.0089**. The local pipeline beat that reproduction on 9/12 targets.

The likely sources of the small gain are:

- the union of fast MMseqs retrieval and exhaustive composite retrieval, instead of one
  search/ranking route;
- confidence ranking based on identity, coordinate coverage and completeness;
- explicit preference for five distinct PDB sources, which helps the best-of-five metric;
- partial-template alignment with unpenalised terminal gaps and confidence-aware gap fill;
- deterministic gradient refinement, which preserves confident transferred residues and
  mostly moves gaps/weak regions.

The much larger difference between local 0.3072 and Kaggle private 0.60175 is not a valid
same-target comparison. Local evaluation consists of 12 particularly difficult CASP15
targets under 2022 cutoffs; Kaggle scores a different hidden set. The late score is an
external validation result, not evidence that local TM was calculated incorrectly.

## Comparison with the copied hybrid V4

`utilities/top1_4_4_hybrid_final_take.py` is a captured Kaggle notebook transcript,
not a directly runnable Python module. Its final prediction path is approximately:

- generate one Boltz structure;
- run DRfold2 with the Boltz structure as an AF3-style restraint for a selected subset;
- use a template fallback for the remaining targets;
- for sequences longer than 600 nt, replace the first final conformation with Boltz and
  replace detected extreme placeholders in the other conformations.

Important implementation details explain why adding pretrained components need not beat
the simpler TBM baseline:

1. A later `find_similar_sequences` definition overrides the earlier enhanced composite
   search with a global-alignment-only, top-five search.
2. `test_sequences` is sorted by length without resetting its index; DRfold eligibility
   is then tested using the original DataFrame index. Therefore the code does not reliably
   run DRfold on the intended shortest/ordered subset.
3. Missing DRfold residues are filled with `(0, 0, 0)`, and captured output contains very
   large finite placeholder coordinates for some TBM conformations. Finite values pass a
   NaN check but can still be structurally invalid.
4. Boltz directly replaces only conformation 1 for sequences longer than 600 nt. It is not
   a confidence-aware per-target or per-segment fusion rule.
5. Pretrained candidates can be worse than a close template. Best-of-five helps only when
   the new candidate is added without displacing a better/diverse candidate.

Thus the reported hybrid V4 score 0.57631 being below both TBM-only 0.59298 and our
0.60175 is consistent with its implemented selection/fusion logic; it is not evidence
that pretrained models are generally inferior.

## Leakage audit

Evidence against accidental label leakage in the submitted artifact:

- the manifest contains no label CSV, validation data, hidden test table or prebuilt
  submission;
- its only derived data artifacts are template coordinates/metadata, template FASTA,
  MMseqs index, the deduplicated composite library and aggregate geometry priors;
- all 23,869 main-template rows and all 7,155 composite rows have known release dates;
- both retrieval branches require `release_date < temporal_cutoff`;
- the composite deduplication keeps the earliest deposited coordinate representative for
  an identical sequence.

This makes the score credible, but not yet a proof of per-target causality. Kaggle does
not expose hidden labels or per-target TM, so we cannot identify which private targets
produced the gain. Two hardening changes are advisable before calling the implementation
fully audited:

- make missing/malformed target cutoffs fail closed rather than relying on string
  comparison; and
- parse the target's declared PDB/reference IDs, when present, into an explicit self-PDB
  exclusion list in addition to the date gate.

Do not tune repeated variants against the private leaderboard. Attribute improvements
with the frozen temporal/family-aware local set, and use the 0.60175 private score once as
external confirmation.

## Score references supplied with the comparison

- TBM-only notebook V1, reported 0.59298:
  <https://www.kaggle.com/code/jaejohn/rna-3d-folds-tbm-only-approach?scriptVersionId=242152007>
- Hybrid final-take V4, reported 0.57631:
  <https://www.kaggle.com/code/jaejohn/sub-1-4-4-hybrid-final-take?scriptVersionId=242575996>
