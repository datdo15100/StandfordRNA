# Refinement ablation — is the geometry refinement truthful? (CASP15, temporal-safe)

New pipeline (MMseqs + composite search + de novo hedge). Aux metrics averaged over all 5 predicted structures per target, then over targets.

**OPTIMIZED by gradient** = clash / bb_dev / rg_err (drops are expected by construction). **INDEPENDENT** = TM and sharp_kinks (not in any objective) — the real truthfulness test.

| setting   |     tm |   clash |   bb_dev |   rg_err |   kinks |
|:----------|-------:|--------:|---------:|---------:|--------:|
| none      | 0.3092 |  0.1634 |   1.4576 |  14.6928 |  0.0536 |
| rule      | 0.3098 |  0.0991 |   1.0255 |  14.6385 |  0.0944 |
| gradient  | 0.3072 |  0.0935 |   0.7766 |  13.7568 |  0.1025 |

- **TM (independent accuracy)**: none 0.309 -> gradient 0.307 (-0.002), rule 0.310.
- **clash/res (optimized)**: none 0.163 -> gradient 0.094 / rule 0.099.
- **backbone dev Å (optimized)**: none 1.458 -> gradient 0.777 / rule 1.026.
- **sharp kinks (INDEPENDENT)**: none 0.054 -> gradient 0.102 / rule 0.094.

## Verdict — partly truthful, with an honest caveat

Reading the two axes the refiner does NOT optimize:

1. **TM is preserved, not gamed.** none 0.309 → gradient 0.307 (−0.002), rule 0.310. On
   this new pipeline the candidates (composite-search real folds + gap-fill) are already
   good, so refinement is essentially **TM-neutral** — it does not inflate the score, and
   does not need to. (On the older, weaker candidate pool it was slightly TM-positive.)
2. **The clash / backbone gains are real but NOT a free lunch.** Gradient cuts clashes
   −42 % and backbone deviation −47 % (rule: −39 % / −30 %) — but the **independent
   sharp-kink rate nearly doubles** (0.054 → 0.102; rule 0.094). The v1 objective
   constrains *distances* (adjacent spacing + clash) with **no angle term**, so it
   satisfies those distances by bending the chain more sharply. It is **trading
   distance/clash error for pseudo-bond-angle error**, not uniformly improving geometry.

**Conclusion**: the gradient refinement is honest about the metrics it optimizes (and
beats rule-based on them) and it does not distort TM — but it is **not** a truthful
improvement of *overall* physical plausibility: an un-optimized geometric property (angle
kinks) degrades. Claiming "refinement improves physical validity" is only defensible
scoped to clash + backbone spacing, and must be reported alongside the kink regression.

**v2 fix (clear next step)**: add a soft pseudo-bond-angle / curvature penalty to the
refinement energy so it cannot buy distance compliance with sharp kinks. Expected to
drive clash + backbone down **and** keep kinks ≤ the no-refine baseline — then the
"improves physical validity" claim is truthful across the board, at unchanged TM.

**For the competition metric (TM) specifically**: refinement is optional now — it neither
helps nor hurts best-of-5 TM on the new pipeline. Its value is downstream (cleaner local
geometry for full-atom reconstruction / docking), which is exactly why the kink caveat
matters and the v2 angle term is worth adding.