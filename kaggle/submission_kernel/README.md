# Kaggle production kernel

This private notebook consumes the private artifact dataset
`datdo151000/rna3d-thesis-inference-artifacts` and the competition source. It writes
`/kaggle/working/submission.csv` after strict shape, ID-order and NaN validation.

Push a new version:

```bash
kaggle kernels push -p kaggle/submission_kernel
```

After the version finishes successfully, inspect/download its output and submit that
exact version:

```bash
kaggle competitions submit -c stanford-rna-3d-folding \
  -f submission.csv \
  -k datdo151000/rna3d-thesis-composite-tbm-baseline \
  -v <VERSION> \
  -m "Temporal-safe composite TBM thesis baseline"
```

Do not guess `<VERSION>`: use the number returned by `kaggle kernels push` and verify
kernel status/output first.

## Verified deployment

- Artifact dataset: `datdo151000/rna3d-thesis-inference-artifacts`, version 2.
- Kernel: `datdo151000/rna3d-thesis-composite-tbm-baseline`, version 4.
- Kernel state: complete; CPU/offline runtime 119.1 seconds for 12 targets.
- Output validation: 2,515 × 18, exact sample ID order, unique IDs, no NaNs and all
  coordinates finite.
- Competition submission: `54662648`, submitted 2026-07-13 22:32:44 UTC.
