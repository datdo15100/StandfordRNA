# GeoFuse real-OOF DRfold2 medium run

This private GPU kernel predicts the frozen 100 post-2023 `train_v2` targets:
60 train, 20 calibration and 20 held-out validation targets from the
time/family-disjoint manifest. DRfold2 reports that
its structural training set contains PDB releases before 2024, so these targets
are date-auditable out of fold for structural supervision.

The kernel never reads `train_labels.v2.csv`. It exports only model predictions,
confidence sidecars, and a status file. After downloading the output, import it
with explicit provenance:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split train_v2 \
  --target-file data/processed/geofuse_real_oof_v2/medium_manifest_targets.txt \
  --source drfold2_e2e --model cfg97_20ckpt_e2e \
  --model-training-cutoff 2023-12-31 \
  --model-training-data "DRfold2 official structural training set: PDB releases before 2024" \
  --root /path/to/geofuse_drfold2_real_oof \
  --glob '**/{target_id}/e2e_relax/model_*.pdb'
```

The target IDs are frozen from `medium_manifest.csv` before the kernel is launched.

Version 2 is a one-target repair for `8YUR_X`: neural inference succeeded in version 1,
but Arena segfaulted on the highest-confidence checkpoint output. The repair keeps the
same model/checkpoints and deterministically tries the next confidence-ranked output
when Arena fails; it does not inspect native coordinates.

Arena failed for all 20 outputs of `8YUR_X` in version 2. Before any native scoring,
version 3 therefore predicts `9EY0_T`, the first unused later target in the same frozen
train-split ordering. This is a logged technical replacement, not score-based selection.
