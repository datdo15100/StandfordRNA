---
marp: true
theme: default
paginate: true
title: GeoFuse-RNA — Supervisor Progress Update
---

# GeoFuse-RNA

## Supervisor progress update

Confidence-aware fusion and motif-conditioned geometric refinement for RNA 3D prediction

**14 July 2026**

---

# Executive summary

- Built a reproducible, temporal-safe RNA 3D evaluation pipeline around the Stanford Kaggle challenge.
- Current strongest local result: **0.3072 mean best-of-5 TM** on 12 CASP15 targets.
- Main empirical win is improved template recall: **+0.0955 TM** from composite search.
- Geometry refinement v1 improves clashes/backbone spacing but is **TM-neutral** and increases sharp kinks.
- Thesis extension: fuse TBM and pretrained predictions by residue/segment, then apply motif-conditioned geometry v2.
- Immediate external validation: package a reproducible Kaggle notebook and make a late submission.

---

# Problem and benchmark

- Input: RNA sequence, MSA, template database and release-date metadata.
- Output: **five** C1′ coordinate structures per target, each shaped `[L, 3]`.
- Competition metric: best TM-score among five predictions, averaged over targets/references.
- TM-score rewards the correct global fold and is relatively tolerant of local geometric error.
- Thesis therefore evaluates both **fold accuracy** and **structural validity**.

---

# Data and EDA snapshot

- Train v1: **844 / 137k residues**; train v2: **5,135 / 3.68M residues**.
- Local validation/test: **12 CASP15 targets**, 2,515 residues total.
- Validation lengths: **30–720 nt**, median **129.5 nt**; one 720-nt stress case.
- Structural library: **8,670 RNA CIF files**, 56.89 GiB.
- Historical full parse: **23,869 chains**, 10.86M residues, 99.9% modelled C1′, zero parse errors.
- MSA and MSA_v2 provide target-specific evolutionary context.

---

# Evaluation design: leakage is the central risk

- For a CASP15 target, templates must predate its cutoff (about 27 May 2022).
- The target’s own PDB identifiers are explicitly excluded.
- Geometry priors are estimated only from pre-cutoff training structures.
- Diagnostic on the earlier TBM pipeline:
  - temporal-safe: **0.1612**
  - ignore cutoff but exclude native: **0.6388**
  - native/oracle allowed: **0.9566**
- Conclusion: local scores without temporal controls are scientifically misleading.

---

# Implemented pipeline

```text
sequence + cutoff
      ↓
MMseqs search + exhaustive composite search
      ↓
temporal/self-leakage filtering
      ↓
alignment → C1′ transfer → gap fill
      ↓
TBM candidates + de novo hedge
      ↓
optional geometry refinement
      ↓
quality/diversity choice → five structures
```

One code path is reused for controlled local evaluation and Kaggle inference.

---

# How performance evolved

| Stage | Temporal-safe mean best-of-5 TM |
|---|---:|
| Dummy extended-chain floor | 0.069 |
| MMseqs TBM + refinement | 0.161 |
| Add de novo fallback | 0.212 |
| Add composite template search | **0.307** |
| Reproduced top-1 method | 0.298 |

The largest gain came from finding better candidate folds, not stronger coordinate optimisation.

---

# Bottleneck diagnosis: candidate recall

- Only 5/12 targets originally had an MMseqs temporal-safe hit; 7/12 fell back to weak de novo structures.
- Reproduced top-1 uses an exhaustive composite similarity scan and scored **0.2983** temporal-safe.
- This diagnosed search/recall—not refinement—as the main accuracy bottleneck.
- Adding composite search improved our pipeline from **0.2117 to 0.3072**.
- Improved 11/12 targets and beat the reproduced top-1 on 9/12.

---

# Composite-search ablation

| Target | MMseqs only | + composite | ΔTM |
|---|---:|---:|---:|
| R1117v2 | 0.100 | 0.420 | +0.320 |
| R1108 | 0.312 | 0.489 | +0.177 |
| R1149 | 0.164 | 0.324 | +0.160 |
| R1156 | 0.173 | 0.280 | +0.106 |
| Mean (12) | 0.212 | **0.307** | **+0.096** |

Cost is practical: about 8 s/target, roughly five minutes for a 40-target run.

---

# Geometry refinement v1: honest result

| Method | TM ↑ | clashes/res ↓ | backbone dev ↓ | sharp kinks ↓ |
|---|---:|---:|---:|---:|
| No refinement | **0.3092** | 0.1634 | 1.4576 | **0.0536** |
| Rule-based | 0.3098 | 0.0991 | 1.0255 | 0.0944 |
| Gradient v1 | 0.3072 | **0.0935** | **0.7766** | 0.1025 |

- Gradient v1 cuts clashes by 42% and backbone deviation by 47%.
- It does not improve TM and nearly doubles the independent sharp-kink rate.
- Distance-only constraints can satisfy bond lengths by bending the chain too sharply.

---

# What is already a contribution?

- A reproducible temporal-safe benchmark and explicit leakage quantification.
- A search diagnosis and composite-recall improvement over the reproduced top-1 baseline.
- An adversarial refinement evaluation that reports an unoptimised failure metric.
- Reusable evaluation across TM, clashes, backbone deviation, Rg and diversity.

But “TBM + pretrained + generic refinement” alone is too close to leaderboard practice.

---

# Proposed thesis extension: GeoFuse-RNA

```text
TBM candidates ─┐
                ├─ fold clustering ─ confidence per residue/segment
pretrained ─────┘                         ↓
                              confidence-aware segment fusion
                                           ↓
                           motif-conditioned geometry projection
                                           ↓
                              quality-diversity final five
```

Core claim: contribution lies in the **adaptive integration layer**, not in training another large RNA model.

---

# Geometry v2

- Keep adjacent-distance, clash and size terms from v1.
- Add angle/curvature distributions to prevent kink trading.
- Add signed pseudo-torsion only when the required atom representation is available.
- Condition priors on simple structural context: paired/unpaired, stem/loop/junction.
- Use stage-wise optimisation: stitch/fuse → repair local geometry → weak global prior.
- Success condition: preserve TM while reducing clash/backbone error **without exceeding the no-refine kink rate**.

---

# Confidence-aware fusion

- Estimate reliability for each source at each residue or contiguous segment.
- TBM features: identity, coverage, completeness, gap mask, template date and local agreement.
- Pretrained features: model confidence plus agreement among candidate folds.
- Fuse only after rigid alignment within a fold cluster.
- Avoid residue-wise “Frankenstein” switching through segment smoothing and seam penalties.
- Train/evaluate the gate on real held-out predictions, not synthetic corruptions alone.

---

# Experiments that test the contribution

| Experiment | Question answered |
|---|---|
| B0 current TBM/composite | strong baseline |
| B1 + raw pretrained union | do new sources improve oracle candidate TM? |
| B2 + confidence-aware fusion | does fusion beat whole-candidate selection? |
| B3 + geometry v2 | does projection repair without artifacts? |
| B4 + fold-aware final-five selection | does quality-diversity reduce selection regret? |

Report paired per-target ΔTM, oracle-pool TM, selection regret, kink/clash metrics, runtime and VRAM.

---

# Current engineering status

- New WSL Conda environment `rna-fold` installed and GPU verified on RTX 3060 Ti.
- PyTorch CUDA, MMseqs2 and four core tests pass.
- Clean rebuild: 23,869 chains / 10.87M residues / zero parser errors in 4.6 minutes.
- Clean rerun reproduced dummy 0.0687, top-1 0.2983 and current pipeline 0.3072.
- Kaggle token and CLI work; account currently has no competition submission.
- WSL cap configured for 18 GB RAM + 8 GB swap; restart required to apply.

---

# Compute strategy

- RTX 3060 Ti, 8 GB: main local development, DRfold2/RibonanzaNet-scale experiments.
- GTX 1650, 4 GB / 16 GB RAM laptop: CPU tests, analysis, cached-candidate evaluation and slide/thesis work.
- Do not rebuild 57 GB CIF library or run AF3-style models on the laptop.
- Kaggle GPU: final offline notebook and heavier Boltz/Chai candidates where feasible.
- Cache candidate outputs so fusion/refinement experiments are cheap and reproducible.

---

# Immediate next steps and decision gates

1. Finish data audit and rebuild reusable template artifacts on the 3060 Ti machine.
2. Reproduce B0 from scratch, then freeze its artifact bundle.
3. Create and run a Kaggle baseline notebook; validate `submission.csv`; late-submit its exact version.
4. Add DRfold2/pretrained candidates and first measure **oracle-pool TM**.
5. Only proceed to learned fusion if the candidate pool actually improves.
6. Implement geometry v2 and require the kink regression to disappear.

---

# Questions for supervisor

1. Is the primary thesis claim best framed around segment fusion, with geometry v2 as its projection/repair component?
2. Is 12-target temporal-safe CASP15 evaluation acceptable if paired with Kaggle private-score validation?
3. Should the final frozen holdout be target-, family-, or time-based?
4. Is a learned confidence gate required, or is a strong heuristic gate with clear ablation sufficient?
5. Which success criterion should dominate: TM uplift, geometry validity, or a two-axis claim?

---

# Appendix: evidence map

- Pipeline results: `reports/thesis_notes/results_summary.md`
- Composite ablation: `reports/thesis_notes/composite_ablation.md`
- Refinement truthfulness: `reports/thesis_notes/refine_ablation.md`
- Top-1 reproduction: `reports/thesis_notes/reproduce_top1.md`
- Leakage diagnostic: `reports/thesis_notes/leakage_demo.md`
- Full experiment chronology: `LOG.md`
- GeoFuse design and review: `PLAN.md` + this folder’s `research_plan_review.md`
- External papers and Kaggle workflow: `sources.md`
