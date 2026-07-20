# Kaggle fallback for the GeoFuse Phase-A DRfold2 bank

This private GPU kernel runs only the 720-nt validation target `R1138`, which
exceeds the local RTX 3060 Ti's 8 GB gate. It uses the same official cfg97 code
and 20 checkpoints as the local runner, selects five direct E2E hypotheses by
model-side confidence, fills C1' with Arena, and exports pickle-free confidence
and distance-prior sidecars.

The kernel does not read validation labels and does not submit to the
competition. Its output is an experiment artifact to download and import with:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split validation --target-ids R1138 \
  --source drfold2_e2e --model cfg97_20ckpt_e2e \
  --root /path/to/downloaded/output \
  --glob '**/{target_id}/e2e_relax/model_*.pdb'
```

Push a new private version with Kaggle CLI:

```bash
kaggle kernels push -p kaggle/geofuse_phase_a_drfold2
```
