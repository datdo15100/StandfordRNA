# Implementation Log — Temporal-Safe Hybrid TBM + Geometry-Informed Refinement

Tracking progress on the thesis pipeline described in `PLAN.md`. Each entry records
what was built, key decisions, leakage safeguards, and results.

Working environment:
- conda env `rna-fold` (Python 3.12): numpy 2.4, pandas 3.0, scipy, gemmi 0.7.5, torch 2.12+cu130, sklearn, biopython 1.87, matplotlib, **mmseqs** on PATH.
- GPU: NVIDIA GTX 1650 (4 GB VRAM) — small; heavy DL predictors (DRfold2/Boltz/Chai) are a stretch locally, so core = TBM + geometry refinement (CPU + light GPU).
- Interpreter used in scripts: `/home/datdo/miniconda3/envs/rna-fold/bin/python`.

Data facts established up front:
- `test_sequences.csv` is **identical** to `validation_sequences.csv` → both are the 12 CASP15 public targets (temporal_cutoff 2022-05-27/28). This is our local eval set; the real private test is hidden on Kaggle.
- train v1 = 844 sequences / 137k residue rows; train v2 is larger.
- Label coordinate sentinel for unresolved residues = `-1e18`; official "resolved" rule is `x > -1e17`.
- `PDB_RNA/` = 8,672 `.cif` files (57 GB) + `pdb_seqres_NA.fasta` + `pdb_release_dates_NA.csv`.
- Internet available locally (used once to fetch + compile US-align; **not** used for any per-target structural info → no temporal leakage).

---

## Repo scaffold — DONE

- Created `src/rna3d/` package: `data/ cif/ template/ geometry/ refine/ eval/ pipeline/`.
- `configs/paths.yaml` + `src/rna3d/paths.py`: single source of truth for all paths, the CASP15 safe cutoff (`2022-05-27`), and the coord sentinel.
- Rewrote `.gitignore` (it previously contained a literal heredoc command, not ignore rules). Now ignores `.env`, raw 57 GB data, derived caches, binaries, large artifacts.
- `src/rna3d/data/io.py`: load sequences/labels, multi-reference coordinate extraction with sentinel→NaN, submission build/validate/write helpers.
- `src/rna3d/geometry/transforms.py`: rotation, mirror, Kabsch, RMSD, Rg, extended-chain baseline.

## Phase 1 — Scoring + sanity — DONE

- Compiled **US-align** (v20260527) from source → `external/binaries/USalign`.
- `src/rna3d/eval/usalign.py`: C1'-only PDB writer + TM-score parser that mirror the official scorer exactly (TM normalised by the **reference** structure; best over 5 preds × all reference conformations; averaged over targets).
- `scripts/run_phase1_scoring.py` results:

| check | TM | expected |
|---|---|---|
| native vs native | 1.000 | ≈1 ✓ |
| native vs rotated+translated | 1.000 | ≈1 ✓ (rotation invariant) |
| native vs mirrored | 0.224 | <1 ✓ (chirality penalised → validates the mirror-hedge idea) |

- **B0 dummy baseline** (extended chain, 5 jittered copies) on the 12 CASP15 targets: **mean best-of-5 TM = 0.0687**. This is the floor every real method must clear.
- Multi-reference scoring confirmed working (R1156 has 40 refs, R1149 has 10).
- Artifacts: `reports/tables/phase1_sanity.csv`, `reports/tables/phase1_dummy_baseline.csv`.

### Leakage notes (Phase 1)
- Validation = the 12 CASP15 targets. Any template DB used to predict these must be filtered to `release_date <= temporal_cutoff` (≈2022-05-27), and the targets' own later PDB depositions must be excluded (self-leakage guard) — enforced in Phase 3/4.

## Phase 2 — Geometry priors — DONE

- `src/rna3d/geometry/priors.py` + `scripts/run_phase2_priors.py`.
- **Temporal-safe**: estimated only from train_v2 chains with `temporal_cutoff < 2022-05-27` → 3,397 / 5,135 chains. Priors are physical/statistical (not target-specific) so they generalise to CASP15.
- Results (`data/processed/geometry_priors.json`):
  - **Adjacent C1'-C1'**: mean **6.09 Å**, std 1.11, median 5.66 (matches the ~5.9 Å figure top solutions cite). 2.1M adjacent pairs.
  - **Clash radius r_min = 4.18 Å** — derived as `max(4.0 Å physical floor, 1st-pct of nearest non-adjacent C1' neighbour)`. Chose nearest-neighbour basis (not raw all-pairs) because a naive low percentile of all pairs (7.66 Å) would wrongly penalise legitimate stacked/paired tertiary contacts; sub-Å data artifacts are floored out.
  - **Rg power law**: `Rg = 5.18 · L^0.346` (exponent ≈ 1/3 → compact globular scaling) + per-length-bin median table.
- Plots: `reports/figures/adjacent_distance_distribution.png`, `rg_by_length.png`.

## Phase 3 — PDB_RNA parser + template DB — IN PROGRESS

- `src/rna3d/cif/nucleotide_map.py`: canonical-base mapping. Trusts gemmi's tabulated one-letter code (covers most modified bases: PSU→U, 1MA→A, OMG→G, H2U→U…) with an explicit fallback table for ~60 modifications gemmi leaves blank (e.g. 5MC→C, LNA series, inosine→G). RNA vs DNA decided by gemmi polymer-type, not residue name.
- `src/rna3d/cif/parser.py`: per-RNA-chain extraction of canonical seq + C1' coords (highest-occupancy altloc; NaN for missing C1' so true gaps survive). First model only.
- `scripts/build_template_db.py`: parallel parse (8 workers, ~24 files/s → ~6 min for 8,670 CIFs). Robust release-date parser (the CSV concatenates several download blocks with repeated headers + trailing commas). On a 300-file pilot: 480 chains, 99.9% residues resolved, all release dates matched, 21 rare unmapped codes.
- Outputs: `data/processed/template_meta.parquet`, `data/cache/template_coords.pkl`, `data/processed/pdb_parse_report.json`.
- Full 8,670-file parse running in background (I/O-bound on the /mnt/d Windows mount reading 57 GB → ~15-20 min). Results table appended once it lands.

## Phase 4/5/7 — code complete, eval pending DB — IN PROGRESS

All modules written and import-clean; will run end-to-end as soon as the template DB finishes building.

- **Phase 4 (search)** `template/mmseqs_search.py` + `template/confidence.py`: MMseqs2 nucleotide `easy-search` (`--search-type 3`) over the full template FASTA, built once; temporal/leakage filtering applied on the hit list (release_date < target cutoff, plus the target's own PDB ids — parsed from the `all_sequences` headers — excluded). Confidence = identity × coverage × completeness.
- **Phase 5 (TBM)** `template/align.py` (Biopython global align + C1' transfer with an explicit transfer mask), `template/gap_fill.py` (linear interp for short gaps; interp + perpendicular sinusoidal curvature for long gaps; backbone-direction extension for termini; extended-chain fallback), `pipeline/tbm.py` (assembles confidence-ranked, PDB-diverse candidates).
- **Phase 7 (refinement — core contribution)** `refine/losses.py` + `refine/optimizer.py`: confidence-weighted template pull + backbone distance + clash + Rg terms, optimised with Adam. Two confidence signals: per-residue weights free the gaps; scalar template confidence sets overall refinement strength (`s = 0.2 + 0.8·(1−conf)` → confident templates barely move, weak ones get assertive geometry).
  - **Synthetic validation** (native CASP15 R1149 + 3 Å noise + 25 % residues dropped → gap-filled → refined): best-of-5 TM **0.489 → 0.547**. Confirms loss signs/scales and that refinement repairs a degraded template. Real-template numbers pending DB.
- **Phase 8/9 harness** `pipeline/methods.py` + `scripts/run_eval.py`: method registry (B0 dummy, B1 top-1, B2 top-5, B4 top-5+refine) and ablations (no-clash, no-Rg, no-gap-weights), scored best-of-5 on the 12 CASP15 targets with self-TM diversity logging.
- **Kaggle inference** `kaggle/inference_pipeline.py`: production loop (load precomputed bundle → search → TBM → refine → submission.csv), reusing the exact thesis code paths.

## Phase 3 — template DB — DONE (full build)

- Parsed all 8,670 CIFs → **23,869 RNA / RNA-DNA-hybrid chains, 10.86 M residues, 99.9 % with a modeled C1'**, 0 errors, all release dates matched. Took 35 min (I/O-bound). coords pickle = 179 MB.
- 295 rare modified-residue codes still map to 'N' (top: Y5P 1075, P5P 784, 4AC 512) — **< 0.1 % of residues**; coordinates still transfer, only identity dips marginally. Logged as a cheap future fix (extend map + re-parse).

## Phase 4 — MMseqs2 search — DONE

- **Memory hurdle solved**: mmseqs nucleotide search default `k=15` builds an offset table too large for this 7.7 GB box (`Cannot fit databases into 6G`); splitting doesn't help (the table is per-split). Fix: search on the fly with **`k=13`** + `--split-memory-limit 3G`, target DB built once with `createdb` (`--dbtype 2`). Hits are re-aligned downstream, so lower k only widens the net. ~3 s for 12 queries.
- **Leakage guard verified end-to-end**: for R1107 the top sequence hit is its own native `7QR4_B` (100 % id), released **2022-10-26** > cutoff 2022-05-28 → correctly dropped. Same for R1149/R1156/R1189 natives (released 2023-24).

## Phase 5/7/8/9 — TBM + refinement + ablation on CASP15 — DONE (first full run)

`scripts/run_eval.py` (300 refine steps) over all 12 CASP15 targets.

**Mean best-of-5 TM:** B0 dummy 0.069 · B1 top-1 0.154 · B2 top-5 0.158 · **B4 TBM+refine 0.161**. Refinement net **+0.0028** (8/12 targets improved, 4 slightly worse).

**Critical finding — template availability dominates.** Only **5/12** CASP15 targets have any temporal-safe template; the other 7 (mostly *designed* RNAs — origami R1126, PXT triangle R1128, aptamer R1136, 6-helix bundle R1138 — plus novel virals R1149/R1156/R1189/R1190) have **no pre-cutoff homolog**: the only sequence-similar PDB entries ARE the post-cutoff natives. This is correct leakage-free behaviour and is exactly the regime where DL fallback (deferred) is needed.

**Stratified by template confidence** (`reports/tables/eval_by_confidence.csv`):

| bin | n | B2 top5 | B4 refined | refine gain |
|---|---|---|---|---|
| high (R1116) | 1 | 0.453 | **0.469** | **+0.0155** |
| medium (R1107/8) | 2 | 0.324 | 0.319 | −0.0045 |
| low (R1126/36) | 2 | 0.163 | 0.161 | −0.0018 |
| none (7 targets) | 7 | 0.068 | 0.072 | +0.0044 |

Refinement helps most exactly where PLAN predicted (high-confidence template: +0.0155; clean physical repair of dummy fallbacks: +0.004). On medium templates the v1 strength is a touch aggressive (small regressions) — tuning candidate for v2. Best-of-5 diversity: mean self-TM 0.70 (genuinely diverse, not 5 copies).

Artifacts: `reports/tables/eval_methods.csv`, `eval_summary.csv`, `eval_by_confidence.csv`, `reports/thesis_notes/results_summary.md`, `method.md`.

## Leakage demonstration — DONE (centerpiece result)

`scripts/run_leakage_demo.py` runs the **same** TBM+refinement pipeline under three regimes:

| regime | mean best-of-5 TM | meaning |
|---|---|---|
| **temporal_safe (honest)** | **0.1612** | release_date < cutoff, native PDB excluded — the thesis number |
| no_temporal (ignore cutoff) | 0.6388 | post-CASP15 homologs leak in (+0.478) |
| oracle_leak (native allowed) | **0.9566** | upper bound — pipeline reaches near-perfect TM when the true template is present |

**Why this matters for the thesis:**
1. **The pipeline is correct and powerful** — oracle 0.957 proves search→align→transfer→gap-fill→refine→best-of-5 yields near-perfect structures *when a template exists*. The honest 0.161 is bounded by **template availability**, not pipeline quality.
2. **Quantifies the exact leakage the competition warns about**: ignoring `temporal_cutoff` inflates TM by **+0.48** (e.g. R1138 0.038→1.000, R1136 0.16→0.97, R1149 0.07→0.84) because the natives / near-natives now sit in the PDB. Many leaderboard solutions legitimately used these (CASP15 targets are "burned"), which is part of why a temporally-honest score looks lower than the public leaderboard.
3. Gives the thesis a rigorous, defensible framing: *temporal-safe* TBM + geometry refinement, with leakage explicitly measured rather than hidden.

Artifacts: `reports/tables/leakage_demo.csv`, `reports/thesis_notes/leakage_demo.md`.

## Refinement isolated — physical validity (best-of-1) — DONE

`scripts/run_refine_analysis.py` refines the single best template per target (the 5 with templates) and measures accuracy + physical validity before/after:

| metric | before | after |
|---|---|---|
| TM-score | 0.2759 | 0.2756 (neutral — TM is robust to local error) |
| **clashes / residue** | 0.057 | **0.038 (−33 %)** |
| **backbone deviation (Å)** | 0.774 | **0.518 (−33 %)** |
| Rg error (Å) | 167.5 | 161.3 |

On well-covered templates the effect is clearest (R1116: backbone 1.63→0.72 Å, clash 0.29→0.19, TM +0.015). **Conclusion**: the geometry-informed refinement consistently produces physically more plausible structures (fewer steric clashes, tighter backbone geometry) without sacrificing TM, and adds TM where the template is strong — a clean, defensible thesis result that does not depend on the leaked/unavailable templates.

Artifacts: `reports/tables/refine_geometry.csv`, `reports/thesis_notes/refine_geometry.md`.

---

## Status summary

**Done & verified, end-to-end, leakage-free:** scoring harness · geometry priors · 23.8 k-chain template DB · MMseqs2 temporal-safe search · TBM transfer + gap-fill · geometry-informed refinement · best-of-5 · ablations · leakage demonstration · Kaggle inference loop. All wired through one codebase (`src/rna3d/`), reproducible via `scripts/`.

**Headline numbers (12 CASP15 public targets, best-of-5 TM):**
- Honest temporal-safe TBM+refine: **0.161** (dummy floor 0.069).
- Pipeline ceiling when a template exists (oracle): **0.957** → the machinery is sound; the gap is template availability (7/12 targets have no pre-cutoff homolog).
- Refinement: net +0.0028 TM (best-of-5), +0.0155 on high-confidence templates, and **−33 % clashes / −33 % backbone deviation** (physical validity).

**Next steps (v2, optional / GPU-permitting):**
1. Per-residue geometry weighting `(1 − conf_residue)` so refinement never distorts well-transferred regions (fixes the small medium-confidence regressions).
2. DL fallback branch (DRfold2/Boltz/Chai) for the 7 no-template targets — the `L_dist` hook + candidate-pool interface are already in place; GTX 1650 (4 GB) is the constraint.
3. Extend the modified-nucleotide map (Y5P/P5P/4AC/…) and re-parse for the final template DB.
4. Package the precomputed bundle + `kaggle_submission.ipynb` for a Kaggle late submission.

## Production submission — DONE

`scripts/make_submission.py` → `kaggle/inference_pipeline.run_inference` on `test_sequences.csv` produced a **valid `data/processed/submission.csv`**: 2,515 rows, 12 targets, 18 columns (ID, resname, resid + x/y/z × 5), format validation PASSED. Confirms the full production path (search → TBM → gap-fill → refine → 5 structures → validated CSV) runs offline end-to-end with the same code as the experiments.

---

## Adopting ideas from the 1st-place TBM notebook (`utilities/top1_tbm.py`)

Read the 1st-place TBM-only notebook; it shares our architecture (same TBM skeleton, near-identical gap-fill, confidence-adaptive refinement) but does **no temporal filtering** (so on CASP15 it runs in our "no_temporal/oracle" regime). Ported two low-risk, high-value pieces:

### #1 De novo fallback — `src/rna3d/geometry/denovo.py` — DONE
Sequence-only heuristic fold (stem detection → helix/loop/single with stochastic backbone steps; seeds give diversity). Replaces the bare extended-chain fallback for the 7 no-template CASP15 targets. Now the default no-template branch in `kaggle/inference_pipeline.py` (`m_tbm_grad`).

### #2 Rule-based refiner baseline — `src/rna3d/refine/rule_based.py` — DONE
Faithful port of the 1st-place `adaptive_rna_constraints` (single-pass greedy nudging: sequential distance, clash push-apart, light base-pairing; strength `0.8·(1−min(conf,0.8))`). Exists purely to benchmark against our gradient geometry-energy refinement.

### Comparison result (`scripts/run_compare_refiners.py`, `reports/thesis_notes/refiner_comparison.md`)

| | no-template (n=7) | templated (n=5) | all 12 |
|---|---|---|---|
| extended chain (old floor) | 0.068 | — | 0.069 |
| **de novo (ported)** | **0.154** | — | 0.163 |
| + gradient refine (ours) | 0.159 | 0.286 | **0.212** |
| + rule-based refine (1st) | **0.163** | 0.285 | **0.214** |

**Physical validity** (representative structure, before → after): clashes/res 0.330 → **grad 0.155** / rule 0.241; backbone dev 2.207 Å → **grad 0.308** / rule 1.028.

**Findings:**
1. **De novo fallback is the big win** — more than doubles TM on no-template targets (0.068→0.154) and **lifts the honest overall mean from 0.161 → ~0.213** (the 7 no-template targets were the drag).
2. **Refiners near-tied on TM** (TM robust to local error): rule 0.214 vs gradient 0.212 overall (rule edges ahead only on rough de novo inits; gradient ahead on templated).
3. **Our gradient refinement wins decisively on physical validity**: −53 % clashes vs −27 %, −86 % backbone deviation vs −53 %. Equal TM, far more plausible geometry → the clean thesis differentiator (TM alone doesn't reward valid geometry).
4. **v2 insight**: gradient strength is a touch aggressive on de novo inits; softening it there should close the small TM gap.

**New honest headline: ~0.21 best-of-5 TM** on the 12 CASP15 targets (was 0.161 before the de novo fallback), fully temporal-safe.

Attribution: de novo generator and rule-based refiner are adapted from the 1st-place notebook (credited in module docstrings); the gradient geometry-energy refinement, temporal-safety, MMseqs pipeline, and evaluation harness are ours.
