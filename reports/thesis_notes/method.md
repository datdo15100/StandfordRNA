# Method — Temporal-Safe Hybrid TBM + Geometry-Informed Refinement

This note documents the implemented pipeline at C1' resolution for the Stanford
RNA 3D Folding benchmark. It is the reference for the thesis methodology chapter.

## 1. Problem & metric
Predict five C1'-only 3D structures per RNA sequence; scored by best-of-5 TM-score
(US-align, sequence-independent, normalised by the reference length), averaged over
targets. TM-score is length-normalised and robust to local error, so the design
prioritises getting the **global fold** right over atomic precision.

## 2. Pipeline overview
```
sequence --> MMseqs2 nucleotide search --> temporal/leakage filter
         --> Biopython global align --> C1' coordinate transfer (+ transfer mask)
         --> geometry-aware gap fill --> confidence-weighted geometry refinement
         --> best-of-5 selection --> submission.csv
```

## 3. Temporal safety & leakage control (central to a credible thesis)
- **Geometry priors** estimated only from train chains with `temporal_cutoff < 2022-05-27`.
- **Template search** filters every hit to `release_date < target.temporal_cutoff`
  (strict). Verified: for R1107 the top sequence hit is its own native structure
  7QR4 (100 % id), released 2022-10-26 — correctly dropped by the gate.
- **Self-leakage**: the target's own PDB ids (parsed from `all_sequences` headers)
  are additionally excluded.
- No network calls during prediction; US-align is only used Kaggle-side for scoring.

## 4. Template database (Phase 3)
All 8,670 `PDB_RNA/*.cif` parsed with gemmi → 23,869 RNA/RNA-DNA-hybrid chains,
10.86 M residues (99.9 % with a modeled C1'). RNA vs DNA decided by gemmi polymer
type; modified bases mapped to canonical A/C/G/U via gemmi's table + an explicit
fallback (≈60 entries). C1' altlocs resolved by occupancy; missing C1' kept as NaN
so real gaps survive to the gap-filler.

## 5. Geometry priors (Phase 2)
Data-driven, not hard-coded:
- adjacent C1'-C1' distance: mean 6.09 Å, std 1.11 Å (cf. the ~5.9 Å folklore value);
- clash radius r_min = 4.18 Å (1st-pct of nearest non-adjacent neighbour, floored at
  a 4 Å physical minimum so sub-Å data artifacts cannot collapse it);
- radius of gyration scaling Rg = 5.18 · L^0.346 (≈ globular L^{1/3}).

## 6. Coordinate transfer + gap fill (Phase 5)
Global align (match +2 / mismatch −1 / gap −6,−0.5, free end gaps). Resolved template
C1' atoms are copied to matched target positions; a boolean transfer mask records
which positions are template-derived. Gaps are reconstructed geometrically:
short internal → linear interpolation; long internal → interpolation + perpendicular
sinusoidal curvature at ~6 Å spacing; termini → extension along the backbone
direction; fully-missing → extended chain. Each residue carries a confidence
(transferred = 1.0; filled decays into the gap).

## 7. Geometry-informed refinement (Phase 7 — the contribution)
Minimise, with Adam on C1' coordinates,
```
E(X) = w_tpl·L_tpl + s·( w_bb·L_bb + w_clash·L_clash + w_rg·L_rg ) [+ w_dist·L_dist]
```
- `L_tpl` = confidence-weighted pull to transferred coordinates → **trust the
  template where reliable, free the gaps**;
- `L_bb` keeps consecutive C1' spacing near the prior mean/std;
- `L_clash` penalises non-adjacent pairs closer than r_min;
- `L_rg` keeps the radius of gyration length-appropriate;
- `L_dist` optional pairwise/distogram prior (off in v1; the hook for a pretrained
  predictor when GPU permits).
- **Adaptive strength** `s = 0.2 + 0.8·(1 − template_confidence)`: confident templates
  are barely perturbed; weak candidates get assertive geometry repair.

## 8. Best-of-5 + no-template fallback (Phase 8)
Five structures from the top distinct-PDB templates (refined); padded by perturbation
when fewer than five templates exist. Diversity measured by mean pairwise self-TM.

For targets with **no temporal-safe template** (7/12 CASP15), a sequence-only **de novo
generator** replaces the bare extended chain: it detects complementary stem candidates
and lays residues along helix/loop/single paths with stochastic backbone steps (adapted
from the 1st-place TBM notebook), giving five diverse folds that refinement then cleans
up. `src/rna3d/geometry/denovo.py`.

## 8b. Refiner comparison (thesis contribution vs the 1st place)
Two refinement strategies are compared head-to-head:
- **Gradient geometry-energy (ours)** — `refine.optimizer`: minimises a global
  differentiable energy (template + backbone + clash + Rg) with Adam.
- **Rule-based nudging (1st-place baseline)** — `refine.rule_based`: single-pass greedy
  corrections (sequential distance, clash push-apart, light base-pairing), scaled by
  `0.8·(1 − min(conf, 0.8))`.
See `reports/thesis_notes/refiner_comparison.md` for the numbers.

## 9. Known limitations / future work
- 295 rare modified-residue codes (<0.1 % of residues, e.g. Y5P, P5P, 4AC) currently
  map to 'N'; coordinates still transfer, only sequence identity dips marginally.
  Cheap to extend the map and re-parse for the final run.
- MMseqs2 nucleotide search uses k=13 (k=15 default exceeds the 5 GB dev box); hits
  are re-aligned downstream so this only widens the candidate net.
- Pretrained-DL candidate branch (DRfold2/Boltz/Chai) deferred — GTX 1650 (4 GB) is
  marginal; `L_dist` and the candidate-pool interface are already in place. The de novo
  generator is the current no-template stand-in until the DL branch lands.
```
