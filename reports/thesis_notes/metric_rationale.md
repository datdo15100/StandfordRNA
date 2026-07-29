# Metric rationale and claim boundary

## What is independently recognized?

### TM-score

TM-score is the competition-aligned global-fold endpoint. It is length-normalized and
less dominated by a few large coordinate errors than RMSD. Best-of-five TM is the main
pipeline/Kaggle result.

### C1′ RMSD

RMSD after optimal rigid superposition is a standard structural-comparison metric. In
this project it is computed only on shared resolved C1′ positions. It answers whether
the complete trace is closer in Cartesian space, but can be dominated by a misplaced
domain or hinge.

### C1′-lDDT

lDDT is a published, superposition-free measure of local distance preservation. The
original definition evaluates native pairs within an inclusion radius and averages
the fractions preserved at 0.5, 1, 2 and 4 Å tolerances
([Mariani et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3799472/)).

The thesis uses those published thresholds and a 15 Å inclusion radius, but evaluates
one C1′ atom per nucleotide because that is the competition representation. Therefore
the exact name must be **C1′-lDDT adaptation**, not full-atom lDDT and not an official
Kaggle metric.

### Sliding-window C1′ RMSD

Local/windowed RMSD follows the recognized RMSD operation but repeats the fit on
consecutive RNA segments. Windows 9, 15 and 31 are pre-registered analysis scales, not
universally standardized biological cutoffs. It is independently interpretable as
short-, medium- and longer-range trace accuracy, but should not be called an official
named score.

RNA-Puzzles established the need to evaluate RNA models with several complementary
structure metrics rather than a single global number
([RNA-Puzzles evaluation](https://rnajournal.cshlp.org/content/18/4/610.full);
[RNA-Puzzles toolkit](https://pmc.ncbi.nlm.nih.gov/articles/PMC7145511/)).
Recent CASP RNA assessment likewise reports complementary global, local and interaction
metrics
([CASP16 RNA assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC12750035/)).

## What is custom/project-specific?

| metric | status | defensible use |
|:--|:--|:--|
| sharp-kink fraction below 70° | custom thresholded pseudo-angle diagnostic | detect a specific failure mode; not independent evidence when optimized by v2 |
| C1′ clash count | coarse trace-distance proxy | compare C1′ self-overlap only; never call it MolProbity clashscore |
| adjacent-distance deviation | empirical C1′ backbone diagnostic | check continuity against train-derived priors |
| radius-of-gyration error | project size prior | detect collapse/over-expansion |
| angle/torsion NLL | train-derived optimization objective | optimizer endpoint and ablation diagnostic |
| pair-like fraction | candidate-derived topology proxy | native-blind feature; not base-pair accuracy |

These diagnostics remain useful for debugging and constrained optimization. They cannot
replace native TM/RMSD/lDDT/window-RMSD evidence, and they are never described as
official biology metrics.

## What cannot be claimed after excluding phase 8?

The experiment does not compute all-atom interaction-network fidelity, deformation
index, steric validation or MolProbity-style nucleic-acid validation. Consequently:

- it can claim improved **C1′-trace local distance accuracy** if C1′-lDDT passes;
- it can claim improved project geometry diagnostics if those objectives improve;
- it cannot claim corrected base pairing, stacking, sugar pucker, bond stereochemistry
  or full physical validity;
- it cannot call the C1′ clash proxy a biological clashscore.

This boundary keeps the thesis contribution proportional to the data representation
and to the experiments actually run.
