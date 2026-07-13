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
  -k datdo151000/rna3d-thesis-composite-baseline \
  -v <VERSION> \
  -m "Temporal-safe composite TBM thesis baseline"
```

Do not guess `<VERSION>`: use the number returned by `kaggle kernels push` and verify
kernel status/output first.

