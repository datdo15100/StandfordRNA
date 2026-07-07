# 1st-place TBM-only — faithful reproduction on the 12 CASP15 targets

Best-of-5 TM (US-align). Same method (composite similarity + KMeans diversity + transfer + rule-based refine + de novo), scored under two template regimes.

- **full_pdb (their setup — no temporal filter, LEAKED): 0.9403**
- **temporal_safe (honest): 0.2973**

Leakage on CASP15 = **+0.6430** TM. Their public 0.593 is a *private-set* score (≈40 hidden targets), NOT reproducible on these 12 public targets; full_pdb here is the local leaked proxy.

## Head-to-head vs our pipeline (temporal-safe, honest)

| | all 12 | templated (5) | no-template (7) |
|---|---|---|---|
| **top-1 reproduced** | **0.297** | **0.348** | **0.261** |
| ours (de novo + gradient) | 0.212 | 0.286 | 0.159 |
| ours (de novo + rule) | 0.214 | — | — |

**The reproduced 1st-place method beats ours on every one of the 12 targets, temporal-safe.**
The gap is largest exactly on the targets where **our MMseqs search returns nothing** and
we fall back to de novo (R1117 +0.30, R1149 +0.13, R1156 +0.09, R1190 +0.08).

### Diagnosis: the weak link is SEARCH, not refinement
Their `find_similar_sequences` does an **exhaustive composite-similarity scan** of every
template — global + local Smith-Waterman + RNA k-mer/feature similarity — so it returns
*some* real RNA template for essentially every target, even non-homologous ones. Copying a
compositionally/​locally-similar **real** fold (TM ≈ 0.2–0.4: right size, plausible global
shape) beats our **de novo** heuristic (TM ≈ 0.15) on no-homolog targets. Our **MMseqs k=13
nucleotide** prefilter is far less sensitive and misses these weak/partial matches → 0
candidates → de novo. This matches the 2nd-place team's point that plain nucleotide search
misses remote RNA homologs (they used RibonanzaNet representations instead).

Our gradient refinement is **not** the problem — on physical validity it still beats their
rule-based nudging (−53 % clashes vs −27 %, −86 % backbone dev vs −53 %; see
`refiner_comparison.md`). The refinement is a good contribution; the **template search is
where we lose TM**, and it is the highest-value thing to improve next.

### What this does and does not show
- ✅ Faithfully reproduces the 1st-place *method*; honest temporal-safe score on CASP15 ≈ **0.30**.
- ✅ Quantifies the leakage the method is exposed to on CASP15 (**+0.64** if the cutoff is ignored — their natives sit in the PDB dump).
- ❌ Does **not** reproduce their **0.593** — that is a *private-set* number (≈40 hidden post-competition RNAs), unmeasurable locally. Only a Kaggle late submission can produce a comparable figure.

### Caveats on fidelity
- Template library = our (superset, gemmi-parsed) DB reduced to 7,155 unique sequences,
  not their exact 18,881 Biopython extraction. Being a cleaner superset, if anything this
  *helps* the reproduced baseline — it does not explain its lead.
- `full_pdb` includes the native depositions (their notebook does no date filtering), so
  those columns are near-1.0 by construction on the targets whose native is a clean copy.

| target_id   |   seq_len |   full_pdb |   full_pdb_sec |   temporal_safe |   temporal_safe_sec |
|:------------|----------:|-----------:|---------------:|----------------:|--------------------:|
| R1107       |        69 |     0.995  |            6.5 |          0.3709 |                 3.9 |
| R1108       |        69 |     0.9947 |            6.1 |          0.4749 |                 4.5 |
| R1116       |       157 |     0.9949 |            9   |          0.5262 |                 5.4 |
| R1117v2     |        30 |     0.9395 |            8.9 |          0.3946 |                 2.9 |
| R1126       |       363 |     0.9885 |           12.6 |          0.18   |                 6.9 |
| R1128       |       238 |     0.9995 |            8.4 |          0.1995 |                 5.4 |
| R1136       |       374 |     0.9689 |           15.6 |          0.1902 |                 8   |
| R1138       |       720 |     0.9999 |           46.8 |          0.2173 |                32.7 |
| R1149       |       124 |     0.9342 |            9.4 |          0.2957 |                 6.5 |
| R1156       |       135 |     0.8895 |           16.3 |          0.2622 |                12.1 |
| R1189       |       118 |     0.7058 |            9.1 |          0.2255 |                 5.6 |
| R1190       |       118 |     0.8734 |            8.2 |          0.2304 |                 6.1 |