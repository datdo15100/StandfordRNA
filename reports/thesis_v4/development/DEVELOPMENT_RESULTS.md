# V4 development results and component decisions

## Evidence scope

All results in this report come from 20 previously exposed development RNAs in 17
MMseqs sequence-similarity clusters. They support preregistered component selection,
not confirmatory claims or component attribution on the hidden Kaggle set.

The three primary hypotheses use 10,000 cluster-bootstrap replicates, 100,000 cluster
sign flips, PCG64 seed 20260819, and Holm step-down correction.

## RQ1: template-based modelling

J-controlled TBM reaches mean best-of-five TM 0.389, while retained thesis TBM reaches
0.395. The paired delta is +0.0056, or +1.45%, with 95% interval
[-0.0030, 0.0170] and raw one-sided p=0.1558. Ten targets improve, two tie, and eight
regress. H1 does not pass, so V4 does not claim a demonstrated TBM accuracy improvement
over John.

The component results are more specific:

- Adding MMseqs to composite produces delta -0.0030 and no availability gain. MMseqs
  is removed from the final method in this scope.
- Full composite is 0.0057 below global-only, with an interval fully below zero. The
  local-alignment, feature, and 3-mer terms are removed.
- Identity multiplied by query coverage has point estimate +0.0206 versus identity,
  but interval [-0.0024, 0.0522] crosses zero. Coverage remains an engineering
  heuristic rather than a demonstrated contribution.
- Template completeness produces exactly zero incremental effect beyond coverage and
  is removed from ranking.
- Distinct-PDB selection has delta -0.0013, within tolerance 0.005. It remains only as
  a redundancy safeguard.
- Curved gap completion does not beat linear completion in any preregistered gap bin.
  Linear is retained as the simplest noninferior method.
- Intentionally unsafe templates produce delta +0.0315 with interval
  [0.0086, 0.0569]. This is evidence of leakage sensitivity, not a deployable gain.

The retained TBM is global-only exhaustive retrieval, identity-by-coverage ranking,
distinct-PDB safeguarding, and linear gap completion.

## RQ2: candidate-source allocation

Bank 5T reaches mean best-of-five TM 0.395. Replacing two template slots with direct
DRfold2 candidates raises the 3T+2D mean to 0.411. The paired delta is +0.0157, or
+3.97%, with interval [-0.0056, 0.0419] and raw p=0.1371. Eight targets improve, 11
tie, and one regresses. H2 is positive but inconclusive.

The mechanism evidence is consistent with independent-source diversity. Mean self-TM
falls from 0.453 for 5T to 0.380 for 3T+2D, a 16.05% reduction. Near-duplicate pair
fraction falls from 8% to 2%. However, 2D reaches only 0.288 compared with 0.381 for
2T. DRfold2 is not uniformly stronger than templates. Its plausible value lies in
complementary fold coverage.

The preregistered decision retains 3T+2D for confirmatory evaluation while limiting
the development claim to source diversification. Boltz remains excluded because the
required checkpoint and provenance are unavailable.

## RQ3: geometric refinement

Production Geometry initially beats Simple on SW-RMSD9 by 0.0220 Angstrom, with
interval [0.0060, 0.0388]. Mechanism gates retain adaptive strength, replace
candidate-derived angle and torsion context with the unconditional global prior, and
disable the radius-of-gyration term. Fixed strength improves SW-RMSD9 but its TM
interval relative to production Geometry reaches -0.00526, just beyond the -0.005
preservation margin.

The combined selected configuration is frozen before scoring. Relative to Simple on
the same 100 thesis-bank candidates, it reduces SW-RMSD9 from 4.555 to 4.532. The
effect is 0.02345 Angstrom, or 0.515%, with interval [0.00689, 0.04062], raw
p=0.00769, and Holm-adjusted p=0.02307. Fifteen targets improve and five regress.

Mean best-of-five TM changes from 0.40805 for Simple to 0.40884 for selected Geometry.
The +0.000787 delta, or +0.193%, has interval [-0.00318, 0.00433]. This interval does
not demonstrate a TM increase, but its lower bound remains above -0.005, so the
preservation safeguard passes. No selected-refinement execution fails.

Raw 3T+2D has mean TM 0.41068, approximately 0.00184 above selected Geometry. The
supported development claim is precise: Geometry improves local native accuracy beyond
Simple and preserves TM relative to Simple. It does not demonstrate a TM gain relative
to Raw. John fixed has the lowest SW-RMSD9, 4.444, but reduces mean TM to 0.398,
showing why a local metric cannot replace the global objective.

## Holm family and final development decisions

H3 passes the first Holm threshold with adjusted p=0.02307. H2 fails the next step,
and Holm step-down stops. H1 is also not rejected. H3 is the only primary development
hypothesis that passes the multiplicity-controlled family.

The frozen system entering the confirmatory checkpoint uses P0-production, retained
thesis TBM, a 3T+2D candidate bank, and selected Geometry. Unsupported components are
removed. Coverage and distinct PDB are labelled accurately as a heuristic and a
safeguard.
