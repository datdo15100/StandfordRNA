# Thesis V4 working draft

V4 is maintained in English only. The active manuscript is `thesis_en.tex`; no
Vietnamese V4 manuscript is maintained.

The scientific spine is a controlled decomposition of the pipeline under the
best-of-five TM-score objective:

1. template retrieval, ranking, and reconstruction;
2. fixed-budget candidate-source allocation;
3. same-candidate geometric refinement;
4. confirmatory local evaluation and a complete-system external check.

## Evidence state

- Preregistration: commit `7739ff6`, tag `v4-preregistration-2026-08-20`.
- Master train-V2 ledger: 5,135 RNA records; CASP15 is managed in a separate
  development ledger.
- P0-production reproduction: PASS with maximum absolute numeric error zero.
- Development RQ1, RQ2, and RQ3 experiments: complete.
- Final method and manifest: frozen at 97 targets in 86 sequence-similarity clusters.
- Native-blind confirmatory output: 97/97 targets complete and hash-frozen.
- Final native performance: opened once after the explicit receipt; H1 failed
  negatively, while H2 and H3 passed without post-opening retuning.
- Exact frozen-V4 Kaggle external check: running as a separate whole-pipeline
  deployment benchmark.

V3 is used only as a reference for verified background, citation material, and writing
style. V3 results and claims are not automatically migrated into V4.

## Build

The manuscript uses XeLaTeX and Times New Roman. Under WSL, the preamble falls back to
the Windows font directory at `/mnt/c/Windows/Fonts`. Tectonic is installed in the
`rna-fold` environment and has compiled the 41-page English manuscript successfully.

```text
cd reports/thesis_v4
conda run -n rna-fold tectonic thesis_en.tex --outdir .
```
