# Evidence and sources

## Repository evidence used by the deck

- `LOG.md` — chronological experiment record and dataset inventory.
- `reports/thesis_notes/composite_ablation.md` — 0.2117 → 0.3072 composite-search result.
- `reports/thesis_notes/refine_ablation.md` — TM/clash/backbone/kink truthfulness table.
- `reports/thesis_notes/reproduce_top1.md` — temporal-safe and deliberately leaked top-1 reproduction.
- `reports/thesis_notes/leakage_demo.md` — three-regime leakage diagnostic.
- `reports/thesis_notes/results_summary.md` — initial TBM/refinement results by target.
- `reports/thesis_notes/pretrained_feasibility.md` — hardware-oriented model triage.

## Kaggle workflow

- Kaggle CLI competition commands and code-competition submission options:
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md>
- Official code-competition CLI tutorial:
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md>
- Official kernel/notebook metadata and push workflow:
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md>
- Official KaggleHub download/authentication documentation:
  <https://github.com/Kaggle/kagglehub>

For a code competition, the late-submission command refers to an output filename produced
by a specific notebook/kernel slug and version. Therefore the notebook version must finish
successfully and expose `submission.csv` before submission.

## Research context

- RhoFold+: language-model-based RNA 3D prediction and intermediate structural outputs:
  <https://arxiv.org/abs/2207.01586>
- RNA-FrameFlow: RNA-specific global and local structural-validity evaluation context:
  <https://arxiv.org/abs/2406.13839>
- AMIGOS III: standard RNA pseudo-torsion definitions and conformational analysis:
  <https://academic.oup.com/bioinformatics/article/38/10/2937/6564222>
- RiboSphere: motif-enriched RNA structure representations (recent context, not yet an
  implemented dependency): <https://arxiv.org/abs/2603.19636>

External papers motivate design choices; the deck’s numeric results come from the checked-in
repository reports listed above.

