# Setup & machine migration guide

How to bring the pipeline up on a new machine (target: i7-9700F / RTX 3060 Ti 8 GB /
24 GB RAM / 1 TB NVMe). Everything here is Linux-first (native Linux or WSL2).

## What moves vs what rebuilds

| Item | Size | Move or rebuild? |
|---|---|---|
| Source (`src/`, `scripts/`, `configs/`, `kaggle/`, `reports/thesis_notes/`) | small | **git** (move) |
| Raw competition data `data/stanford-rna-3d-folding/` | **61 GB** | **copy** (not in git) |
| Template DB `data/cache/template_coords.pkl` + `data/processed/template_meta.parquet` | ~180 MB | copy **or** rebuild |
| Geometry priors, MMseqs DB, hits | small | rebuild (cheap) |
| `external/binaries/USalign` | ~1 MB | recompile (1 command) |
| `.env` (Kaggle token) | tiny | copy by hand (never in git) |

Rule of thumb: **git-clone the code, copy the 61 GB data + the 180 MB template DB, rebuild the rest.**

## 1. Code
```bash
git clone <your remote>  rna3d-thesis      # or copy the repo folder
cd rna3d-thesis
```

## 2. Environment
```bash
conda env create -f environment.yml
conda activate rna-fold
```
Verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
→ expect `True NVIDIA GeForce RTX 3060 Ti`. If PyTorch can't see the GPU, reinstall the
torch wheel matching your CUDA/driver (see the note in `environment.yml`).

## 3. Data placement
Put the 61 GB competition folder anywhere on the NVMe and point the env var at it
(no config edit needed):
```bash
export RNA3D_DATA=/data/stanford-rna-3d-folding      # add to ~/.bashrc
```
Placing it on the **native NVMe (ext4), not a Windows `/mnt` mount**, is the single
biggest speedup: the CIF parse drops from ~35 min (Windows mount) to a few minutes.

## 4. USalign (scoring)
```bash
git clone --depth 1 https://github.com/pylelab/USalign /tmp/USalign
g++ -O3 -ffast-math -o external/binaries/USalign /tmp/USalign/USalign.cpp
```

## 5. Rebuild derived artifacts
Either copy `data/processed/template_meta.parquet` + `data/cache/template_coords.pkl`
from the old machine, **or** rebuild everything:
```bash
bash scripts/rebuild_artifacts.sh          # priors -> template DB -> MMseqs DB
```

## 6. Sanity check
```bash
python scripts/run_phase1_scoring.py       # US-align sanity + dummy baseline
python scripts/run_eval.py --targets 3     # end-to-end on 3 CASP15 targets
```

## 7. New-hardware wins to apply
- **24 GB RAM** removes the MMseqs memory limit → you can drop the `k=13` workaround
  and use default `k=15` for higher search specificity (edit `mmseqs_search.search`
  default or pass `kmer=15`). Test it; revert if RAM-tight.
- **8 cores** → `build_template_db.py --workers 8` and MMseqs `--threads 8`.
- **NVMe** → parsing / search I/O no longer the bottleneck.

## 8. GPU / pretrained models
See `reports/thesis_notes/pretrained_feasibility.md` for what fits in 8 GB VRAM
(RibonanzaNet2, DRfold2) vs what belongs on Kaggle (Boltz/Chai).
