# GeoFuse-RNA Phase A: pretrained candidate gate

Phase A answers one question before confidence fusion or geometry v2 is built:

> Does a pretrained predictor add a useful fold hypothesis beyond the current
> temporal-safe TBM candidate pool?

The implementation normalizes every model output to the same cache contract:

```text
target + sequence hash
candidate/source/model provenance
C1' coordinates + valid mask
per-residue confidence + direct-support mask
optional pairwise priors
```

The cache is stored under `data/cache/geofuse_candidates/` and is intentionally
gitignored. It uses compressed NumPy files with pickle disabled.

## 1. Cache the current TBM pool

```bash
conda activate rna-fold
python scripts/run_geofuse_phase_a.py build-tbm \
  --split validation --max-candidates 10
```

This reuses the existing MMseqs hit table when it covers the requested targets.
Temporal release filtering and native-PDB exclusion remain active inside the
existing TBM builder.

## 2. Import pretrained outputs

The repository includes a resumable DRfold2 runner. Keep the upstream checkout
and its 1.3 GB weights outside Git, then run shortest targets first:

```bash
python scripts/run_drfold2_candidates.py \
  --repo /path/to/DRfold2 \
  --output-root /path/to/drfold2_outputs \
  --split validation --mode cfg97 --limit 1 --no-cluster
```

`cfg97` matches the pretrained configuration used by the first-place hybrid
notebook. `official` uses all four upstream configurations. The runner resumes
completed targets, keeps one log per target, stops at the first failure, and
converts trusted DRfold `.ret` pickles into safe pLDDT/distogram NPZ sidecars.
It also applies the minimal compatibility fix needed by new SciPy versions that
removed the old `iprint` argument. The patch is confined to the external,
gitignored DRfold checkout; the `rna-fold` environment is not downgraded.

Use `--no-cluster` for the first gate/smoke test (one optimized structure). The
default `--cluster` follows the top-1 notebook and may optimize up to five fold
clusters; keep that more expensive setting for the final candidate-bank run.

For broad Phase-A coverage, direct end-to-end checkpoint candidates can be
exported without the expensive CPU potential optimization:

```bash
python scripts/run_drfold2_candidates.py \
  --repo /path/to/DRfold2 \
  --output-root /path/to/drfold2_outputs \
  --split validation --mode cfg97 --e2e-only --e2e-candidates 5
```

These are explicitly a separate source (`cfg97_e2e`), not a replacement for the
official optimized result. They are useful for measuring the raw pretrained
candidate ceiling before spending CPU time on selected targets.

DRfold2's standard output layout is recognized directly:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split validation \
  --source drfold2 \
  --model official_ensemble \
  --root /path/to/drfold2_outputs
```

Boltz output and pLDDT/confidence sidecars are also recognized:

```bash
python scripts/run_geofuse_phase_a.py import \
  --split validation \
  --source boltz \
  --model boltz1_conf \
  --root /path/to/boltz_outputs
```

The checked-in private Kaggle runners are the reproducible fallback when local
VRAM is insufficient:

- `kaggle/geofuse_phase_a_drfold2/` runs raw cfg97 checkpoint inference;
- `kaggle/geofuse_phase_a_boltz/` follows the top-1 Boltz settings for `R1138`.

The 720-nt `R1138` exhausted a Kaggle P100's 16 GB under DRfold2, so retrying
the same model is not the chosen fallback. This also matches the top-1 hybrid's
explicit routing rule: sequences longer than 600 nt use Boltz-1.

For another layout, repeat `--glob` with a pattern containing `{target_id}`.
Only a structure with an unambiguous C1' chain matching the target length is
accepted. A rejection log is written to `data/cache/geofuse_phase_a/`.

## 3. Run the gate

```bash
python scripts/run_geofuse_phase_a.py evaluate --split validation
```

Outputs:

- `reports/tables/geofuse_phase_a/candidate_scores.csv`: native TM per raw candidate;
- `reports/tables/geofuse_phase_a/target_pool_metrics.csv`: selected and oracle metrics;
- `reports/tables/geofuse_phase_a/summary.json`: machine-readable gate result;
- `reports/thesis_notes/geofuse_phase_a_gate.md`: thesis-facing summary.

`*_selected_tm` uses only model-side confidence. The union selector round-robins
sources because their confidence scales are not calibrated yet. `*_oracle_tm`
uses the native label only after candidate generation to measure the ceiling.
The central gate statistic is paired per target:

```text
union oracle TM - TBM oracle TM
```

If the mean gain is positive, proceed to geometry v2 and fold-aware fusion. If
it is not, improve pretrained coverage/checkpoints/sampling first; a downstream
refiner cannot recover a fold absent from the candidate bank.

## 4. Audit pretrained overlap before making thesis claims

Competition-valid pretrained models can post-date the retrospective validation
split. Audit every available model-training manifest before interpreting the
gate as temporally held-out evidence:

```bash
python scripts/audit_pretrained_overlap.py \
  --split validation \
  --model-fasta drfold2=/path/to/DRfold2/data/train.fasta
```

For the current official DRfold2 checkout, `R1128` and `R1138` have exact
normalized sequence matches in `data/train.fasta`. Report the full gate as the
competition-style result, then report a sensitivity analysis excluding exact
overlaps as the defensible retrospective result. Exact-match absence is not a
complete leakage guarantee; structural/homology overlap and Boltz's training
set still require separate provenance checks.

Boltz-1 needs a model-specific interpretation rather than inheriting DRfold2's
overlap flag. Its paper states a structural-training cutoff of 2021-09-30,
whereas the local competition release-date table lists `7PTK`/`7PTL` (the
R1138 structures) as 2022-10-05. That supports temporal separation for the
Boltz branch, while the exact input/checkpoint version and any earlier sequence
homologs should still be recorded. See the
[Boltz-1 paper](https://gcorso.github.io/assets/boltz1.pdf) and the
[official training documentation](https://github.com/jwohlwend/boltz/blob/main/docs/training.md).

Inspect cache coverage at any point with:

```bash
python scripts/run_geofuse_phase_a.py status --split validation
```
