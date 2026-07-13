# Refinement ablation — is the geometry refinement truthful? (CASP15, temporal-safe)

New pipeline (MMseqs + composite search + de novo hedge). Aux metrics averaged over all 5 predicted structures per target, then over targets.

**OPTIMIZED by gradient** = clash / bb_dev / rg_err (drops are expected by construction). **INDEPENDENT** = TM and sharp_kinks (not in any objective) — the real truthfulness test.

| setting   |     tm |   clash |   bb_dev |   rg_err |   kinks |
|:----------|-------:|--------:|---------:|---------:|--------:|
| none      | 0.3092 |  0.1635 |   1.4579 |  14.6856 |  0.0536 |
| rule      | 0.3098 |  0.0992 |   1.0258 |  14.6312 |  0.0944 |
| gradient  | 0.3072 |  0.0935 |   0.7768 |  13.7503 |  0.1025 |

- **TM (independent accuracy)**: none 0.309 -> gradient 0.307 (-0.002), rule 0.310.
- **clash/res (optimized)**: none 0.164 -> gradient 0.094 / rule 0.099.
- **backbone dev Å (optimized)**: none 1.458 -> gradient 0.777 / rule 1.026.
- **sharp kinks (INDEPENDENT)**: none 0.054 -> gradient 0.102 / rule 0.094.

## Verdict — partly truthful, with an honest caveat

Reading the two axes the refiner does not optimise:

1. **TM is preserved, not improved.** None 0.309 → gradient 0.307 (−0.002), while
   rule-based refinement is 0.310. With the stronger composite candidate pool,
   refinement is effectively TM-neutral.
2. **The clash/backbone gains are real but not free.** Gradient refinement cuts
   clashes by about 43% and backbone deviation by about 47%, but the independent
   sharp-kink rate nearly doubles (0.054 → 0.103). A distance-only objective can
   satisfy spacing constraints by bending the chain too sharply.

Therefore v1 supports only a scoped claim about clashes and adjacent spacing, not
overall physical plausibility. Geometry v2 should add context-conditioned angle or
curvature terms and must keep the kink rate at or below the no-refinement baseline
while retaining the clash/backbone gains and TM.
