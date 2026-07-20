# Kaggle Boltz fallback for GeoFuse Phase A

This private GPU kernel runs only validation target `R1138` (720 nt). DRfold2
exhausted 16 GB on this target, while the top-1 hybrid explicitly routes RNA
longer than 600 nt to Boltz-1. The kernel therefore reproduces the top-1 Boltz
settings: one diffusion sample, 10 recycling steps, 500 sampling steps, and
seed 42. It uses sequence only and never reads the validation labels.

The output includes a validated 720-residue mmCIF, residue pLDDT, confidence
JSON, a status file, and a ZIP archive. After downloading, normalize it with:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split validation --target-ids R1138 \
  --source boltz --model boltz1_top1_seed42 \
  --root /path/to/geofuse_boltz \
  --glob '**/{target_id}/boltz/model_*.cif' \
  --max-candidates 1 --overwrite
```

Push a new private version with:

```bash
kaggle kernels push -p kaggle/geofuse_phase_a_boltz
```
