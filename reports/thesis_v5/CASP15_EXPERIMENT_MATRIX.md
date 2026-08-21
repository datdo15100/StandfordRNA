# V5 CASP15 experiment matrix

## Status

This matrix is fixed before the full V5 CASP15 run. CASP15 is a 12-target development
benchmark, so the matrix supports scientific comparison and pipeline selection rather
than confirmatory declarations. The exact historical TBM remains a complete contender
throughout, irrespective of any isolated ablation.

## Five-question component registry

| Component | Origin | Intended problem | Exact algorithm | Expected effect before evaluation | CASP15 test |
|---|---|---|---|---|---|
| John composite retrieval | Public John notebook | Find templates without a high-identity homolog | exhaustive global/local/feature/3-mer score plus feature clustering | high recall and candidate diversity; potentially slower and sensitive to database view | J-SS full vs deduplicated view; component score family |
| MMseqs2 branch | thesis adaptation using standard homology search | recover strong alignment hits efficiently and independently of the composite shortlist | k=13, sensitivity 7.5, max 300, E-value 100; realign top 40 valid hits | occasionally contributes a high-quality candidate absent from composite search; insufficient alone on remote targets | MMseqs-only, composite-only and their fixed-budget union |
| Composite feature terms | inherited from John and adapted to V3 | distinguish RNA similarity not captured by one global alignment | historical `0.4G+0.3L+0.2F+0.1K3` | improve remote/partial retrieval and reduce redundant global-only candidates | G-only; alignment-only `4/7G+3/7L`; full historical score |
| Identity ranking | standard alignment interpretation | prefer sequence-consistent coordinate transfer | identical aligned bases divided by aligned base pairs | useful but may select short or structurally incomplete templates | I-only rank on one frozen candidate pool |
| Query coverage | thesis adaptation | penalize a good local match that leaves much of the query unresolved | `Cq = transferred resolved query positions / Lq`; rank `I x Cq` | improve whole-query fold coverage | I versus I x Cq |
| Template completeness | thesis adaptation motivated by missing coordinates | penalize structurally incomplete templates | `Ct = resolved template C1-prime positions / Lt`; rank `I x Cq x Ct` | reduce gaps without changing high-quality complete templates | I x Cq versus I x Cq x Ct |
| Multiplicative ranking | thesis design | require simultaneous identity, coverage and coordinate support | product of bounded factors, acting as a soft AND | avoid a candidate compensating one near-zero factor with another high factor | compare the three nested ranks and inspect selected candidates |
| Distinct-PDB selection | thesis candidate-diversity safeguard | prevent five chains from one PDB consuming the bank | select different PDB IDs first, then backfill | lower self-TM/redundancy; TM may be neutral or improve best-of-five coverage | same ranked pool with distinct-PDB off/on |
| John gap reconstruction | public John notebook | complete missing residues while reacting to compressed anchors | linear or stretched sinusoidal internal completion, 5.9-Angstrom terminal extension | repair compressed gaps but may distort long spans | locked transfer maps: John vs linear vs V3 curved |
| V3 curved gaps | thesis adaptation | avoid straight rods in long unresolved segments | linear interpolation plus length-dependent perpendicular sine displacement for gaps >=3 | improve local windows in some long gaps; global TM expected mostly stable | same locked transfer maps and same selected templates |
| De-novo fallback | thesis engineering safeguard | always satisfy the five-prediction contract when retrieval fails | deterministic sequence-only candidate, base seed 0 | increase availability, not claimed as an accuracy contribution unless measured | forced-deficit validation plus natural fallback ledger |
| John rule refiner | inherited public method | fix adjacent spacing, clashes and one complement proxy cheaply | faithful one-pass rules with adaptive strength | improve some local geometry, with possible global-fold disturbance | same-candidate Raw vs John refiner; jitter isolated |
| Simple smoothing | controlled baseline | test whether generic source anchoring/backbone smoothing explains Geometry effects | source plus backbone terms, fixed strength, all additional geometry terms off | small local repair with fold preservation | same-candidate Raw/Simple/Geometry |
| Geometry Refinement | thesis-specific extension inspired by geometric and physics-informed constraints | improve local C1-prime plausibility without rewriting the candidate fold | source anchor, backbone, clash, optional Rg, angle, torsion and kink terms with gradient clipping | metric-specific local changes; TM should be preserved, but lDDT and SW-RMSD may disagree | full same-candidate global/local factorial; historical and training-motivated configurations |
| Direct DRfold2 source | learned RNA predictor from prior work | add folds that template retrieval may miss | 20 cfg_97 checkpoints, top two direct Arena structures by mean pLDDT; no PotentialFold | improve best-of-five when structurally complementary, not necessarily when individually stronger | fixed-N allocations on John and V3 TBM banks |
| Fixed candidate allocation | competition objective | distinguish source complementarity from simply adding more predictions | compare banks with identical N and frozen source order | 3T+2D or 4T+1D helps only when D replaces redundant/weaker T | 5T, 4T+1D, 3T+2D and matched-N 2T/1T+1D/2D |

## RQ1 - controlled TBM

### Headline and complete boundaries

| ID | Method | N | Refinement | Purpose |
|---|---|---:|---|---|
| RQ1-J-RAW | J-SS-RAW over full 23,869-chain allowed table | 5 | none | inherited raw baseline |
| RQ1-V3-RAW | exact reconstructed historical V3 TBM | 5 | none | headline thesis TBM contender |
| RQ1-J-COMPLETE | J-SS rule refiner plus legacy-compatible jitter | 5 | John | inherited complete TBM |
| RQ1-V3-DEPLOYED | exact V3 raw plus historical 300-step TBM gradient | 5 | historical TBM gradient | reproduce deployed TBM boundary |

The primary raw contrast is RQ1-V3-RAW minus RQ1-J-RAW. Complete boundaries are
reported separately so retrieval is not confounded with post-processing.

### Retrieval and score family

- John exhaustive composite on full allowed table;
- John exhaustive composite on the 7,155-sequence derived view, sensitivity only;
- V3 composite-only;
- V3 MMseqs2-only;
- V3 merged composite plus MMseqs2;
- global-only exhaustive baseline;
- alignment-only composite `4/7 global + 3/7 local`;
- full historical composite `0.4/0.3/0.2/0.1`.

The three composite score settings are a small nested family, not an unconstrained
hyperparameter search. The historical weights are preserved as one complete contender.

### Ranking, diversity and reconstruction

On frozen candidate pools, evaluate:

- identity;
- identity x query coverage;
- identity x query coverage x template completeness;
- final nested rank with distinct-PDB off/on;
- John, linear and V3 curved gap reconstruction on locked transfer maps;
- exact raw V3 versus historical gradient-refined V3.

At least one interaction table crosses retrieval pool `{composite, merged}` with ranking
`{I x Cq, I x Cq x Ct}`. This checks whether a ranking term appears useful only after a
retrieval source changes the candidate pool.

## RQ2 - fixed-budget source allocation

Run the following raw banks separately on the J-SS template order and exact V3 template
order:

| Bank | Total N | Templates | DRfold2 | Question |
|---|---:|---:|---:|---|
| 5T | 5 | 5 | 0 | template-only competition bank |
| 4T+1D | 5 | 4 | 1 | conservative source replacement |
| 3T+2D | 5 | 3 | 2 | historical hybrid allocation |
| 2T | 2 | 2 | 0 | matched-N template mechanism control |
| 1T+1D | 2 | 1 | 1 | matched-N complementarity test |
| 2D | 2 | 0 | 2 | learned-source strength control |

For unavailable D slots, the next T candidate fills the slot and the realized allocation
is recorded. The analysis reports candidate-level source oracle, source winner counts,
mean pairwise self-TM, realized allocation and compute asymmetry. DRfold2 chooses two
outputs from 20 checkpoint predictions, whereas T candidates are ranked from a much
larger template pool; this is fixed-output-budget, not compute-matched, comparison.

The key interaction is whether `3T+2D minus 5T` has the same sign on John and V3 banks.
If it helps only the weaker bank, the conclusion is compensation for template coverage,
not universal source complementarity.

## RQ3 - same-candidate refinement

Apply every setting to identical raw candidates in these banks where technically
meaningful: J-SS 5T, V3 5T, J-SS 3T+2D and V3 3T+2D.

| Setting | Interpretation |
|---|---|
| Raw | no coordinate modification |
| John faithful | public one-pass John refiner; for J candidates only, native composite confidence |
| John controlled | same rule with fixed strength or applied to non-John sources; explicitly non-faithful mechanism comparator |
| Simple | source anchoring and backbone smoothing only |
| Geometry historical | exact historical candidate-derived context and Rg=0.02 configuration |
| Geometry global | training-motivated unconditional context with Rg off; retained as a complete contender, not assumed superior |

For John complete-pipeline reproduction, Gaussian jitter is a separate post-refiner
factor rather than being hidden inside the refiner comparison.

For each cell report:

- candidate-level locked-reference TM;
- bank best-of-five TM;
- SW-RMSD9 and SW-RMSD15;
- C1-prime lDDT adaptation;
- global Kabsch-aligned C1-prime RMSD;
- failure/fallback rate;
- backbone, clash, kink, Rg, angle/torsion and pair-like diagnostics without treating
  optimized objective terms as independent proof.

Conflicting metrics are retained. No Geometry configuration is called generally better
from an SW-RMSD change if lDDT or TM moves in the opposite direction.

## Complete-pipeline CASP15 table

The final local table contains at least:

1. J-SS-COMPLETE, 5T;
2. exact reconstructed V3 raw TBM, 5T;
3. historically deployed V3 TBM including its gradient refiner;
4. historical V3 hybrid: normally 3T+2D plus historical Geometry, with frozen resource
   fallback;
5. raw J-SS 3T+2D and raw V3 3T+2D;
6. V3 3T+2D plus Simple;
7. V3 3T+2D plus each predeclared Geometry contender;
8. any final V5 contender assembled from the predeclared components.

No component is silently removed because one isolated mean delta is slightly negative.
Selection of the Kaggle-bound complete system considers headline TM, per-target failure,
source availability, structural diversity, metric trade-offs, runtime and historical
deployment evidence. The final choice and hashes are recorded before requesting its new
Kaggle score.

## Statistical and reporting plan

- Equal-weight target mean reproduces the competition objective.
- Paired target effects, medians and wins/ties/losses are mandatory.
- R1189/R1190 form one exact-sequence dependence cluster; other targets are singleton.
- Use 10,000 exact-sequence-cluster bootstrap samples for 95% intervals.
- For the three headline RQ contrasts only, an exact cluster sign-flip distribution is
  enumerated where feasible; p-values are descriptive because CASP15 is development
  data.
- Full per-target tables accompany every aggregate.
- Means are reported to three decimals; paired effects and intervals to four decimals;
  decisions use unrounded values.
- Absolute TM differences are primary. Relative percentages are not used to inflate
  small score changes.
- With N=12, conclusions use language such as "observed on CASP15" rather than
  universal superiority.

## Artifact order

1. generate and hash all raw John, V3 and DRfold2 candidate banks;
2. freeze candidate IDs, source ordering, confidence and fallback records;
3. generate and hash all refinement cells;
4. run the evaluator and write raw per-target tables;
5. update the evidence ledger and select complete contenders;
6. freeze the Kaggle-bound pipeline before external scoring.

