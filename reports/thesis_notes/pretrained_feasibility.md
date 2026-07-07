# Pretrained RNA model feasibility on the new box (RTX 3060 Ti, 8 GB)

Target machine: **i7-9700F (8c), RTX 3060 Ti 8 GB (Ampere sm_86), 24 GB RAM, 1 TB NVMe.**
Big jumps vs the old box (GTX 1650 4 GB / 7.7 GB RAM / Windows mount): **2× VRAM,
3× RAM, fast local disk.** The question: which Source-B pretrained models from PLAN.md
can actually run here, and which need Kaggle / a bigger GPU.

## Verdict per model

| Model | Type | VRAM need (≈, ≤300 nt) | On 8 GB 3060 Ti | Where to run |
|---|---|---|---|---|
| **RibonanzaNet / RibonanzaNet2** | single-seq transformer (2D / distance / reactivity prior) | ~1–3 GB | ✅ **Comfortable** | **Local** — do first |
| **DRfold2** | RNA-specific end-to-end + potential optimisation (LBFGS) | ~4–7 GB (≤~300 nt) | ✅ **Fits** for typical lengths | **Local** — priority for 3D candidates |
| **Boltz-1** | AF3-style (pairformer + diffusion), O(N²) pair rep | ~8–12 GB @150 nt, 16 GB+ @300 nt | ⚠️ **Marginal** — only short RNA, reduced settings | Small local / **Kaggle** |
| **Boltz-2 / Chai-1** | AF3-style, heavier triangle attention | 12–24 GB+ | ❌ **OOM** beyond very short | **Kaggle (16 GB)** or rented 24 GB GPU |

Rule: **the O(N²) triangle-attention models (Boltz/Chai) are the VRAM problem, not the
RNA-specific ones.** Long targets make it worse — e.g. CASP15 **R1138 = 720 nt** OOMs
AF3-style models even on 16 GB; those must fall back to TBM / de novo regardless.

## So: is the 3060 Ti enough?

- **Yes** for the two models the plan actually prioritises: **RibonanzaNet2** (single
  sequence → feeds the refinement `L_dist` distance prior directly) and **DRfold2**
  (the 1st place's choice; produces 3D C1' candidates for weak/no-template targets).
  These realise Source-B locally without a bigger machine.
- **No** for AF3-style Boltz/Chai on medium–long RNA. Options there, in order:
  1. **Kaggle GPU** (T4×2 or P100, **16 GB**) — where final inference must run anyway
     (offline, ≤8 h). 2× the VRAM of the 3060 Ti; slower per-step but enough memory.
  2. A rented/cloud 24 GB GPU (3090/4090/A10/L4) for local-scale Boltz/Chai experiments.
  3. Restrict Boltz to short targets locally, TBM/de novo for the rest.

## Practical enablers (already true / easy on the new box)
- **MSAs are provided**: the competition `MSA/` + `MSA_v2/` folders are precomputed RNA
  MSAs; Boltz/Chai accept custom MSA (a3m) input → the models can run **offline** on
  Kaggle. RibonanzaNet needs no MSA (single-sequence) — another reason it's the easy win.
- **24 GB RAM** lifts the MMseqs `k=13` memory workaround → can use default `k=15`.
- **NVMe** makes the 61 GB CIF parse a few minutes instead of ~35.

## Recommended sequence on the new machine
1. **RibonanzaNet2** → distance/contact prior → wire into `refine.losses.loss_distance`
   (the `L_dist` hook + `w_dist` are already in `RefineConfig`). Cheapest, highest
   leverage: improves refinement on *all* targets, especially no-template ones.
2. **DRfold2** offline → 3D candidates for weak/no-template targets → add to the
   candidate pool (`pipeline/methods._candidate_pool`) alongside TBM + de novo.
3. Benchmark on the 12 CASP15 targets with the existing harness (`run_eval.py`,
   `run_compare_refiners.py`) — same temporal-safe, best-of-5 evaluation.
4. Only if time permits: Boltz-1 on short targets locally / on Kaggle for the rest.

Bottom line: **the 3060 Ti box is enough to add the pretrained branch the thesis needs
(RibonanzaNet2 + DRfold2). Reserve Kaggle's 16 GB for AF3-style Boltz/Chai and for the
final offline submission run.**
