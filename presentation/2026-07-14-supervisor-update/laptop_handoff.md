# Laptop handoff — GTX 1650 / 16 GB RAM / Windows

## What to pull

```powershell
git clone https://github.com/datdo15100/StandfordRNA.git
cd StandfordRNA
git pull origin master
```

The raw competition data and derived artifacts are intentionally not in Git.

## Recommended work on the laptop

- Edit the thesis, plan, slide source and speaker notes.
- Run the four data-free unit tests.
- Develop confidence/fusion code against small synthetic fixtures.
- Analyse copied/cached candidate coordinates and experiment tables.
- Review Kaggle notebook source and submission validation logic.

## Avoid on the laptop

- Do not download/rebuild the full 57–61 GB structural library unless there is ample disk.
- Do not parse all CIFs with eight workers; 16 GB shared with Windows is too tight.
- Do not expect Boltz/Chai or medium/long DRfold2 inference to fit 4 GB VRAM.
- Do not copy `.env` into Git. Add the Kaggle token locally.

## Lightweight setup

WSL is still recommended. Recreate the environment from `environment.yml`; if the
PyTorch CUDA 13 wheel does not match the laptop driver, install a compatible official
PyTorch build after creating the environment. For CPU-only documentation and tests,
CUDA is not required.

```bash
conda env create -f environment.yml
conda activate rna-fold
PYTHONPATH=src python -m unittest discover -s tests -v
```

If disk or environment creation is inconvenient, use a small Python environment with
NumPy, pandas, SciPy, scikit-learn and PyYAML for analysis-only work.

## Moving experiment artifacts safely

Copy only the compact reusable products from the main machine, not the raw PDB dump:

```text
data/processed/geometry_priors.json
data/processed/template_meta.parquet
data/processed/top1_library.pkl
data/cache/template_coords.pkl
cached pretrained candidate coordinates
reports/tables/*.csv
```

The template coordinate pickle is roughly hundreds of MB, while the raw structural
library is tens of GB. Use `RNA3D_DATA` and `RNA3D_CACHE` to point the code to external
locations without editing tracked config.

## Meeting files

Open `supervisor_update.pptx`. Keep `speaker_notes_vi.md` beside it. If a result changes,
edit `slides.md` and `build_presentation.py`, then regenerate the deck.

