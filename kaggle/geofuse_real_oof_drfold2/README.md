# GeoFuse real-OOF DRfold2 pilot

This private GPU kernel predicts 15 post-2023 `train_v2` targets: five from each
time/family-disjoint train, calibration, and held-out split. DRfold2 reports that
its structural training set contains PDB releases before 2024, so these targets
are date-auditable out of fold for structural supervision.

The kernel never reads `train_labels.v2.csv`. It exports only model predictions,
confidence sidecars, and a status file. After downloading the output, import it
with explicit provenance:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split train_v2 \
  --target-file data/processed/geofuse_real_oof_v2/pilot_targets.txt \
  --source drfold2_e2e --model cfg97_20ckpt_e2e \
  --model-training-cutoff 2023-12-31 \
  --model-training-data "DRfold2 official structural training set: PDB releases before 2024" \
  --root /path/to/geofuse_drfold2_real_oof \
  --glob '**/{target_id}/e2e_relax/model_*.pdb'
```

This 15-target run is a pipeline pilot, not the final statistical experiment.
